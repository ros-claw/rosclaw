"""Forced cooperative transactions across real processes; no robot execution."""

from __future__ import annotations

import multiprocessing
from contextlib import suppress
from queue import Empty

import pytest

from rosclaw.practice.artifact_store import ArtifactStore
from rosclaw.sim.plan_store import PersistentPlanStore


def _worker(owner, root, identity, plan_id, paused, release, begun, results, previews):
    try:
        if owner == "plan":
            store = PersistentPlanStore(root)
            snapshot = store.get_for_execute(plan_id)
            previews.put((identity, snapshot["status"]))
            original = store._write

            def write(record):
                if identity == "first":
                    paused.set()
                    assert release.wait(10)
                original(record)

            store._write = write
            begun.set()
            store.consume(plan_id)
        else:
            store = ArtifactStore(root)
            original = store._save_manifest

            def save(session, episode, manifest):
                if identity == "first":
                    paused.set()
                    assert release.wait(10)
                original(session, episode, manifest)

            store._save_manifest = save
            begun.set()
            store.write_jsonl(identity, [{"writer": identity}], session_id="session")
        results.put((identity, "ACK", ""))
    except BaseException as error:
        results.put((identity, "ERROR", str(error)))


@pytest.mark.parametrize("owner", ["plan", "practice"])
def test_cooperating_processes_do_not_double_consume_or_drop_manifest_updates(tmp_path, owner):
    # Pause first writer immediately before its actual mutation. Second writer
    # can observe the original PLANNED snapshot, but may not authorize twice.
    context = multiprocessing.get_context("fork")
    paused, release = context.Event(), context.Event()
    begun_first, begun_second = context.Event(), context.Event()
    results, previews = context.Queue(), context.Queue()
    if owner == "plan":
        store = PersistentPlanStore(tmp_path)
        plan_id = store.put({"points": [], "hash": "fixture"}, "fixture")["plan_id"]
    else:
        ArtifactStore(tmp_path)
        plan_id = "unused"
    first = context.Process(
        target=_worker,
        args=(owner, tmp_path, "first", plan_id, paused, release, begun_first, results, previews),
    )
    second = context.Process(
        target=_worker,
        args=(owner, tmp_path, "second", plan_id, paused, release, begun_second, results, previews),
    )
    collected = []
    first.start()
    try:
        assert paused.wait(10)
        second.start()
        assert begun_second.wait(10)
        # Corrected second writer waits for the transaction lock.
        with suppress(Empty):
            collected.append(results.get(timeout=2))
        release.set()
        first.join(10)
        second.join(10)
        assert not first.is_alive() and not second.is_alive()
        while len(collected) < 2:
            collected.append(results.get(timeout=2))
        if owner == "plan":
            assert sorted(previews.get(timeout=2)[1] for _ in range(2)) == ["PLANNED", "PLANNED"]
            assert sum(result[1] == "ACK" for result in collected) == 1, collected
            denied = [result for result in collected if result[1] == "ERROR"]
            assert len(denied) == 1 and "already consumed" in denied[0][2], collected
            with pytest.raises(ValueError, match="already consumed"):
                PersistentPlanStore(tmp_path).get_for_execute(plan_id)
        else:
            assert all(result[1] == "ACK" for result in collected), collected
            assert {
                record.artifact_id for record in ArtifactStore(tmp_path).list_artifacts("session")
            } == {"first", "second"}
    finally:
        release.set()
        for process in (first, second):
            if process.pid is not None:
                process.join(1)
                if process.is_alive():
                    process.terminate()
                    process.join(2)


def test_native_raw_plan_is_durably_consumed_exactly_once(tmp_path):
    import json

    plan_id = "plan_" + "a" * 16
    (tmp_path / f"{plan_id}.json").write_text(json.dumps({"points": [], "hash": "fixture"}))
    store = PersistentPlanStore(tmp_path)
    assert store.get_for_execute(plan_id)["status"] == "PLANNED"
    store.consume(plan_id)
    replacement = PersistentPlanStore(tmp_path)
    with pytest.raises(ValueError, match="already consumed"):
        replacement.consume(plan_id)
    with pytest.raises(ValueError, match="already consumed"):
        replacement.get_for_execute(plan_id)


def test_missing_consume_is_not_acknowledged(tmp_path):
    with pytest.raises(ValueError, match="REF_NOT_FOUND"):
        PersistentPlanStore(tmp_path).consume("plan_" + "b" * 16)


def test_plan_clear_synchronizes_deletions(tmp_path, monkeypatch):
    import os
    import stat

    store = PersistentPlanStore(tmp_path)
    store.put({"points": [], "hash": "fixture"}, "fixture")
    synced = []
    real = os.fsync

    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        real(fd)

    monkeypatch.setattr(os, "fsync", sync)
    store.clear()
    assert not list(tmp_path.glob("plan_*.json"))
    assert synced[-1] == str(tmp_path)


def test_transaction_symlink_is_rejected_without_changing_target(tmp_path):
    from rosclaw.storage.durable import DurableNamespace

    outside = tmp_path / "outside"
    outside.write_bytes(b"original")
    namespace = DurableNamespace(tmp_path / "owner", kind="practice")
    namespace.ensure_directory(tmp_path / "owner")
    (tmp_path / "owner" / ".rosclaw-transaction.lock").symlink_to(outside)
    with (
        pytest.raises(ValueError, match="DURABLE_PATH_ESCAPE"),
        namespace.transaction(tmp_path / "owner"),
    ):
        raise AssertionError("symlink lock entered")
    assert outside.read_bytes() == b"original"


def test_valid_practice_update_keeps_manifest_bound_to_new_actual_bytes(tmp_path):
    store = ArtifactStore(tmp_path)
    first = store.write_jsonl("events", [{"event": 1}], session_id="session")
    second = store.write_jsonl("events", [{"event": 2}], session_id="session")
    assert first.sha256 != second.sha256
    assert store.get_artifact("events", "session").sha256 == second.sha256
    assert store.verify_artifact("events", "session")[0]
