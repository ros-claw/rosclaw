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
    assert all(event == "directory_sync" for event in events[:-3])
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
    assert all(event == "dir" for event in synced[:-2])


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
