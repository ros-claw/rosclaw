"""Filesystem-only evidence-store acknowledgement and corruption checks."""

import os
import stat
from pathlib import Path

import pytest

from rosclaw.practice.artifact_store import ArtifactStore
from rosclaw.sim.plan_store import PersistentPlanStore


def test_practice_corrupt_registered_artifact_is_never_recreated(tmp_path):
    store = ArtifactStore(tmp_path)
    record = store.write_jsonl("events", [{"event": 1}], session_id="session")
    target = Path(record.path)
    target.write_bytes(b"")
    with pytest.raises(ValueError, match="ARTIFACT_(CORRUPT|INTEGRITY)"):
        store.write_jsonl("events", [{"event": 1}], session_id="session")
    assert target.read_bytes() == b""


@pytest.mark.parametrize("damage", [b"", b"["])
def test_practice_corrupt_manifest_is_not_replaced(tmp_path, damage):
    store = ArtifactStore(tmp_path)
    store.write_jsonl("events", [{"event": 1}], session_id="session")
    manifest = store.manifest_path("session")
    manifest.write_bytes(damage)
    with pytest.raises(ValueError, match="ARTIFACT_MANIFEST_CORRUPT"):
        store.write_jsonl("new_events", [{"event": 2}], session_id="session")
    assert manifest.read_bytes() == damage


def test_practice_artifact_and_manifest_both_have_durable_replace_barriers(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)
    events = []
    real_sync, real_replace = os.fsync, os.replace

    def sync(fd):
        events.append(("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file", None))
        real_sync(fd)

    def replace(source, target):
        events.append(("replace", str(target)))
        real_replace(source, target)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    record = store.write_jsonl("events", [{"event": 1}], session_id="session")
    for target in (record.path, str(store.manifest_path("session"))):
        index = events.index(("replace", target))
        assert events[index - 1][0] == "file"
        assert events[index + 1][0] == "dir"


def test_corrupt_plan_is_not_recreated_or_consumed(tmp_path, monkeypatch):
    import uuid
    from types import SimpleNamespace

    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex="1" * 32))
    store = PersistentPlanStore(tmp_path)
    store.put({"points": [], "hash": "digest"}, "fixture")
    target = tmp_path / "plan_1111111111111111.json"
    target.write_bytes(b"")
    with pytest.raises(ValueError, match="PLAN_STORE_CORRUPT"):
        store.put({"points": [], "hash": "digest"}, "fixture")
    with pytest.raises(ValueError, match="PLAN_STORE_CORRUPT"):
        store.consume("plan_1111111111111111")
    assert target.read_bytes() == b""


def test_plan_create_and_consume_ack_have_file_then_replace_then_dir_sync(tmp_path, monkeypatch):
    store = PersistentPlanStore(tmp_path)
    events = []
    real_sync, real_replace = os.fsync, os.replace

    def sync(fd):
        events.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real_sync(fd)

    def replace(source, target):
        if Path(target).name.startswith("plan_") and Path(target).suffix == ".json":
            events.append("replace")
        real_replace(source, target)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    created = store.put({"points": [], "hash": "digest"}, "fixture")
    index = events.index("replace")
    assert events[index - 1 : index + 2] == ["file", "replace", "dir"]
    events.clear()
    store.consume(created["plan_id"])
    index = events.index("replace")
    assert events[index - 1 : index + 2] == ["file", "replace", "dir"]


def test_manifest_directory_sync_failure_has_no_ack_and_preserves_complete_bytes(
    tmp_path, monkeypatch
):
    store = ArtifactStore(tmp_path)
    manifest = store.manifest_path("session")
    real = os.fsync
    with monkeypatch.context() as scoped:

        def fail(fd):
            if (
                stat.S_ISDIR(os.fstat(fd).st_mode)
                and os.readlink(f"/proc/self/fd/{fd}") == str(manifest.parent)
                and manifest.exists()
            ):
                raise OSError("manifest directory sync failure")
            real(fd)

        scoped.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="manifest directory sync failure"):
            store.write_jsonl("events", [{"event": 1}], session_id="session")
    original = manifest.read_bytes()
    replacement = ArtifactStore(tmp_path)
    record = replacement.write_jsonl("events", [{"event": 1}], session_id="session")
    assert manifest.read_bytes() == original
    assert replacement.verify_artifact(record.artifact_id, "session")[0]


def test_plan_consumed_after_failed_ack_never_returns_to_planned(tmp_path, monkeypatch):
    store = PersistentPlanStore(tmp_path)
    created = store.put({"points": [], "hash": "digest"}, "fixture")
    target = tmp_path / f"{created['plan_id']}.json"
    real = os.fsync
    with monkeypatch.context() as scoped:

        def fail(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode) and b'"CONSUMED"' in target.read_bytes():
                raise OSError("consume directory sync failure")
            real(fd)

        scoped.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="consume directory sync failure"):
            store.consume(created["plan_id"])
    original = target.read_bytes()
    replacement = PersistentPlanStore(tmp_path)
    with pytest.raises(ValueError, match="already consumed"):
        replacement.get_for_execute(created["plan_id"])
    assert target.read_bytes() == original


@pytest.mark.parametrize("owner", ["practice", "plan"])
def test_fresh_owner_deep_directory_retry_recovers_registered_anchor(tmp_path, monkeypatch, owner):
    root = tmp_path / "new1" / "new2" / "new3"

    def make():
        return ArtifactStore(root) if owner == "practice" else PersistentPlanStore(root)

    real = os.fsync
    with monkeypatch.context() as scoped:

        def fail(fd):
            if os.readlink(f"/proc/self/fd/{fd}") == str(tmp_path) and root.is_dir():
                raise OSError("deep owner directory sync failure")
            real(fd)

        scoped.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="deep owner directory sync failure"):
            make()
    synced = []

    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        real(fd)

    monkeypatch.setattr(os, "fsync", sync)
    store = make()
    if owner == "practice":
        store.write_yaml("snapshot", {"state": "valid"}, session_id="session")
    else:
        store.put({"points": [], "hash": "digest"}, "fixture")
    assert str(tmp_path) in synced
    assert str(tmp_path / "new1") in synced
    assert str(tmp_path.parent) not in synced


def test_directory_helper_rejects_escape_before_creating_outside_owner(tmp_path):
    from rosclaw.storage.durable import DurableNamespace

    owner = DurableNamespace(tmp_path / "owner", kind="practice")
    with pytest.raises(ValueError, match="DURABLE_PATH_ESCAPE"):
        owner.ensure_directory(tmp_path / "outside")
    assert not (tmp_path / "outside").exists()
    assert not (tmp_path / "owner").exists()
