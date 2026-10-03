"""Durable acknowledgement and corruption invariants; no original receipts modified."""

from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from rosclaw.sim.store import PARTITIONS, SimStore


@pytest.mark.parametrize("partition", PARTITIONS)
@pytest.mark.parametrize("payload", [{"receipt": "complete"}, b"binary evidence"])
def test_ack_requires_file_sync_then_atomic_replace_then_directory_sync(
    tmp_path, monkeypatch, payload, partition
):
    store = SimStore(tmp_path)
    (store.root / partition).mkdir(parents=True)
    store.put(partition, {"prime": True})
    events = []
    real_sync, real_replace = os.fsync, os.replace

    def sync(fd):
        events.append("directory_sync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_sync")
        real_sync(fd)

    def replace(source, target):
        events.append("replace")
        real_replace(source, target)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    ref = store.put(partition, payload)
    assert events[-3:] == ["file_sync", "replace", "directory_sync"]
    assert events[:-3].count("file_sync") == 1  # registered intent is synced on retry
    assert store.get(ref) == payload


@pytest.mark.parametrize("damage", [b"", b'{"receipt":'])
def test_corrupt_existing_reference_is_not_success_or_repaired(tmp_path, damage):
    store = SimStore(tmp_path)
    payload = {"receipt": "complete"}
    ref = store.put("experiments", payload)
    path = store.resolve(ref)
    path.write_bytes(damage)
    assert not store.exists(ref)
    with pytest.raises(ValueError, match="STORE_(IMMUTABLE_VIOLATION|DIGEST_MISMATCH)"):
        store.put("experiments", payload)
    with pytest.raises(ValueError, match="STORE_DIGEST_MISMATCH"):
        store.get(ref)
    with pytest.raises(ValueError, match="STORE_DIGEST_MISMATCH"):
        store.resolve(ref)
    assert path.read_bytes() == damage


@pytest.mark.parametrize("failure", ["file_sync", "replace", "directory_sync"])
def test_io_failure_never_acknowledges_and_retry_preserves_evidence(tmp_path, monkeypatch, failure):
    store = SimStore(tmp_path)
    folder = store.root / "traces"
    folder.mkdir(parents=True)
    payload = {"qpos": [1, 2, 3]}
    real_sync, real_replace = os.fsync, os.replace

    def sync(fd):
        kind = "directory_sync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_sync"
        if kind == failure and (
            kind == "file_sync" or os.readlink(f"/proc/self/fd/{fd}") == str(folder)
        ):
            raise OSError("injected durable write failure")
        real_sync(fd)

    def replace(source, target):
        if failure == "replace":
            raise OSError("injected durable write failure")
        real_replace(source, target)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "fsync", sync)
        scoped.setattr(os, "replace", replace)
        with pytest.raises(OSError, match="injected durable write failure"):
            store.put("traces", payload)
    assert not list(folder.glob(".tmp_*"))
    targets = list(folder.glob("simtrc_*.json"))
    assert len(targets) == (1 if failure == "directory_sync" else 0)
    original = targets[0].read_bytes() if targets else None
    ref = store.put("traces", payload)
    assert store.get(ref) == payload
    if original is not None:
        assert store.resolve(ref).read_bytes() == original


def test_idempotent_ack_still_syncs_existing_content_and_parent(tmp_path, monkeypatch):
    store = SimStore(tmp_path)
    ref = store.put("states", {"state": "valid"})
    synced = []
    real = os.fsync

    def sync(fd):
        synced.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real(fd)

    monkeypatch.setattr(os, "fsync", sync)
    assert store.put("states", {"state": "valid"}) == ref
    assert synced[-2:] == ["file", "dir"]
    assert synced[:-2].count("file") == 1  # registered namespace intent


def test_new_directory_links_are_synced_before_object_ack(tmp_path, monkeypatch):
    store = SimStore(tmp_path / "new-task")
    synced = []
    real = os.fsync

    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        real(fd)

    monkeypatch.setattr(os, "fsync", sync)
    store.put("models", {"model": "valid"})
    assert str(tmp_path) in synced
    assert str(store.root.parent) in synced
    assert str(store.root) in synced
    assert str(store.root / "models") in synced


def test_concurrent_cooperating_writers_return_one_verified_object(tmp_path):
    def write(_):
        return SimStore(tmp_path).put("models", {"model": "same"})

    with ThreadPoolExecutor(max_workers=8) as executor:
        refs = list(executor.map(write, range(24)))
    assert len(set(refs)) == 1
    assert SimStore(tmp_path).get(refs[0]) == {"model": "same"}
    assert len(SimStore(tmp_path).list_children("models")) == 1


def test_cross_suffix_same_ref_cannot_change_payload_type(tmp_path):
    store = SimStore(tmp_path)
    ref = store.put("models", {"model": "same"})
    original = store.resolve(ref).read_bytes()
    with pytest.raises(ValueError, match="STORE_IMMUTABLE_VIOLATION"):
        store.put("models", original)
    assert store.get(ref) == {"model": "same"}


def test_directory_creation_failed_sync_is_completed_on_retry(tmp_path, monkeypatch):
    store = SimStore(tmp_path / "new-task")
    real = os.fsync
    with monkeypatch.context() as scoped:

        def fail(fd):
            raise OSError("directory creation sync failed")

        scoped.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="directory creation sync failed"):
            store.put("models", {"model": "retry"})
    synced = []

    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        real(fd)

    monkeypatch.setattr(os, "fsync", sync)
    ref = store.put("models", {"model": "retry"})
    assert store.get(ref) == {"model": "retry"}
    assert str(tmp_path) in synced
    assert str(store.root.parent) in synced
    assert str(store.root) in synced


def test_concurrent_digest_prefix_collision_preserves_first_bytes(tmp_path, monkeypatch):
    import rosclaw.sim.store as module
    from rosclaw.sim.refs import make_ref

    ref = make_ref("simmdl", "1" * 64)
    monkeypatch.setattr(module, "make_ref", lambda *_: ref)

    def write(value):
        try:
            SimStore(tmp_path).put("models", {"writer": value})
            return value
        except ValueError as error:
            assert "STORE_IMMUTABLE_VIOLATION" in str(error)
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, [1, 2]))
    winners = [value for value in results if value is not None]
    assert len(winners) == 1
    from rosclaw.contracts.common import canonical_json

    target = tmp_path / "sim" / "models" / f"{ref}.json"
    assert target.read_bytes() == canonical_json({"writer": winners[0]}).encode()


def test_deep_directory_creation_retry_retains_original_ancestor_barriers(tmp_path, monkeypatch):
    store = SimStore(tmp_path / "new1" / "new2" / "new3")
    real = os.fsync
    with monkeypatch.context() as scoped:

        def fail_first(fd):
            if (
                os.readlink(f"/proc/self/fd/{fd}") == str(tmp_path)
                and (store.root / "models").is_dir()
            ):
                raise OSError("first ancestor sync failed")
            real(fd)

        scoped.setattr(os, "fsync", fail_first)
        with pytest.raises(OSError, match="first ancestor sync failed"):
            store.put("models", {"model": "deep retry"})
    synced = []

    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        real(fd)

    monkeypatch.setattr(os, "fsync", sync)
    ref = store.put("models", {"model": "deep retry"})
    assert store.get(ref) == {"model": "deep retry"}
    assert synced[-6:] == [
        str(tmp_path),
        str(tmp_path / "new1"),
        str(tmp_path / "new1" / "new2"),
        str(store.root.parent),
        str(store.root),
        str(store.root / "models"),
    ]
    assert str(tmp_path.parent) not in synced


def test_new_instance_deep_retry_cannot_forget_original_ancestor_barriers(tmp_path, monkeypatch):
    task_root = tmp_path / "new1" / "new2" / "new3"
    store = SimStore(task_root)
    real = os.fsync
    with monkeypatch.context() as scoped:

        def fail_after_creation(fd):
            path = os.readlink(f"/proc/self/fd/{fd}")
            if path == str(tmp_path) and (store.root / "models").is_dir():
                raise OSError("original ancestor sync failed")
            real(fd)

        scoped.setattr(os, "fsync", fail_after_creation)
        with pytest.raises(OSError, match="original ancestor sync failed"):
            store.put("models", {"model": "new instance retry"})
    synced = []

    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        real(fd)

    monkeypatch.setattr(os, "fsync", sync)
    replacement = SimStore(task_root)
    ref = replacement.put("models", {"model": "new instance retry"})
    assert replacement.get(ref) == {"model": "new instance retry"}
    assert str(tmp_path) in synced
    assert str(tmp_path / "new1") in synced
    assert str(tmp_path.parent) not in synced


@pytest.mark.parametrize("failure", ["file", "directory"])
def test_intent_sync_failure_precedes_directory_creation(tmp_path, monkeypatch, failure):
    task = tmp_path / "uncreated" / "nested"
    real = os.fsync
    with monkeypatch.context() as scoped:

        def fail(fd):
            kind = "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
            if kind == failure:
                raise OSError("intent sync failure")
            real(fd)

        scoped.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="intent sync failure"):
            SimStore(task).put("traces", {"trace": "pending"})
    assert not (tmp_path / "uncreated").exists()
    replacement = SimStore(task)
    ref = replacement.put("traces", {"trace": "pending"})
    assert replacement.get(ref) == {"trace": "pending"}


def test_corrupt_namespace_intent_is_not_reconstructed(tmp_path):
    store = SimStore(tmp_path)
    ref = store.put("models", {"model": "complete"})
    target = store.resolve(ref)
    original = target.read_bytes()
    (intent,) = tmp_path.glob(".rosclaw-sim-directory-*.json")
    intent.write_bytes(b"")
    with pytest.raises(ValueError, match="STORE_DIRECTORY_INTENT_INVALID"):
        SimStore(tmp_path).put("models", {"model": "complete"})
    assert intent.read_bytes() == b""
    assert target.read_bytes() == original


def test_cross_process_retry_recovers_exact_registered_boundary(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    task = tmp_path / "new1" / "new2" / "new3"
    source = Path(__file__).resolve().parents[2] / "src"
    script = """
import os, stat, sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from rosclaw.sim.store import SimStore
anchor, task = Path(sys.argv[2]), Path(sys.argv[3])
store = SimStore(task)
real = os.fsync
synced = []
def sync(fd):
    path = os.readlink(f"/proc/self/fd/{fd}")
    if stat.S_ISDIR(os.fstat(fd).st_mode):
        synced.append(path)
    if sys.argv[4] == "fail" and path == str(anchor) and (store.root / "models").is_dir():
        raise OSError("injected ancestor sync failure")
    real(fd)
os.fsync = sync
if sys.argv[4] == "fail":
    try:
        store.put("models", {"model": "cross process"})
    except OSError:
        assert (store.root / "models").is_dir()
        assert not list((store.root / "models").glob("simmdl_*.json"))
    else:
        raise AssertionError("failure was acknowledged")
else:
    ref = store.put("models", {"model": "cross process"})
    assert store.get(ref) == {"model": "cross process"}
    assert str(anchor) in synced
    assert str(anchor / "new1") in synced
    assert str(anchor.parent) not in synced
"""
    for mode in ("fail", "retry"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(source), str(tmp_path), str(task), mode],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr


def test_first_writer_registration_race_cannot_create_two_namespace_anchors(tmp_path, monkeypatch):
    import hashlib
    import threading
    from pathlib import Path

    task = tmp_path / "new1" / "new2" / "new3"
    name = (
        ".rosclaw-sim-directory-"
        + hashlib.sha256(str(task.resolve()).encode()).hexdigest()
        + ".json"
    )
    old_intent = tmp_path / name
    scanned, proceed = threading.Event(), threading.Event()
    real_exists = Path.exists

    def exists(path):
        answer = real_exists(path)
        if (
            path == old_intent
            and threading.current_thread().name == "delayed"
            and not scanned.is_set()
        ):
            assert not answer
            scanned.set()
            assert proceed.wait(10)
        return answer

    monkeypatch.setattr(Path, "exists", exists)
    results, failures = [], []

    def delayed():
        try:
            results.append(SimStore(task).put("models", {"model": "concurrent first writers"}))
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=delayed, name="delayed")
    thread.start()
    try:
        assert scanned.wait(10)
        first = SimStore(task).put("models", {"model": "concurrent first writers"})
    finally:
        proceed.set()
        thread.join(10)
    assert not thread.is_alive()
    assert not failures
    assert results == [first]
    intents = [parent / name for parent in (task, *task.parents) if real_exists(parent / name)]
    assert intents == [old_intent]
    assert SimStore(task).put("models", {"model": "concurrent first writers"}) == first
