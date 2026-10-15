"""Owned ordinary payloads, contextual replay and fail-closed export regressions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from rosclaw.agentd.cli import cmd_artifact_export
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.projection import project_deliverables
from rosclaw.task_kernel.service import TaskKernel, read_artifact_payload


@pytest.fixture
def store(tmp_path):
    home = tmp_path / "home"
    (home / "agentd").mkdir(parents=True)
    conn = sqlite3.connect(home / "agentd" / "missions.db")
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    kernel = TaskKernel(conn, home)
    bound = kernel.bind_message(
        mission_id="history",
        session_ref="s",
        backend_native_id="s",
        message_id="first",
        text="write report",
        cwd=str(tmp_path),
    )
    yield kernel, str(bound["task_id"]), home
    conn.close()


def register(store, path, **kwargs):
    kernel, task, _ = store
    record = kernel.register_artifact(
        task_id=task, path=str(path), media_type="application/octet-stream", **kwargs
    )
    kernel._conn.commit()
    return record


def export(store, record, dest):
    return cmd_artifact_export(
        argparse.Namespace(
            home=str(store[2]),
            artifact_id=record["artifact_id"],
            dest=str(dest),
        )
    )


def assert_payload(record, expected):
    assert record["sha256"] == hashlib.sha256(expected).hexdigest()
    assert record["size_bytes"] == len(expected)
    assert (
        read_artifact_payload(Path(record["path"]), record["sha256"], record["size_bytes"])
        == expected
    )


@pytest.mark.parametrize("mutation", ["inplace", "replace", "delete", "detach"])
@pytest.mark.parametrize("payload", [b"\x00\xffABC", "回球A".encode()])
def test_historical_bytes_survive_source_mutation(store, tmp_path, mutation, payload):
    source = tmp_path / "video.mp4"
    source.write_bytes(payload)
    old = register(store, source)
    changed = payload[:-1] + b"B"
    if mutation in {"replace", "detach"}:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(payload if mutation == "detach" else changed)
        os.replace(replacement, source)
    if mutation in {"inplace", "detach"}:
        source.write_bytes(changed)
    if mutation == "delete":
        source.unlink()
    assert_payload(old, payload)
    assert export(store, old, tmp_path / "old-export") == 0
    assert (tmp_path / "old-export").read_bytes() == payload
    if mutation != "delete":
        new = register(store, source)
        assert new["artifact_id"] != old["artifact_id"]
        assert_payload(new, changed)
        assert export(store, new, tmp_path / "new-export") == 0
        assert (tmp_path / "new-export").read_bytes() == changed
        assert_payload(old, payload)


def test_projection_is_independent_and_validates_before_copy(store, tmp_path):
    kernel, task, home = store
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"AAAA")
    old = register(store, source)
    projected = home / "runs" / task / "r1" / "outputs" / source.name
    assert projected.read_bytes() == b"AAAA"
    assert projected.stat().st_ino != Path(old["path"]).stat().st_ino
    projected.write_bytes(b"BBBB")
    assert_payload(old, b"AAAA")
    projected.unlink()
    Path(old["path"]).chmod(0o600)
    Path(old["path"]).write_bytes(b"CCCC")
    assert project_deliverables(kernel, task) == "DEGRADED"
    assert not projected.exists()


def test_complete_context_idempotency_and_distinct_role_producer_operation(store, tmp_path):
    source = tmp_path / "data.bin"
    source.write_bytes(b"same bytes")
    first = register(store, source, metadata={"role": "diagnostic_source"})
    for _ in range(10):
        replay = register(store, source, metadata={"role": "diagnostic_source"})
        assert replay["artifact_id"] == first["artifact_id"]
    refs = {first["artifact_id"]}
    for kwargs in [
        {"metadata": {"role": "report"}},
        {"metadata": {"role": "diagnostic_source"}, "producer": "kernel:test"},
        {"metadata": {"role": "diagnostic_source"}, "producer_operation_id": "op_other"},
    ]:
        record = register(store, source, **kwargs)
        assert record["artifact_id"] not in refs
        refs.add(record["artifact_id"])
    assert len(store[0].artifacts_for_revision(store[1], 1)) == 4
    meta = json.loads(first["metadata_json"])
    assert meta["role"] == "diagnostic_source"
    assert meta["evidence_tier"] == "EXPERIMENTAL"


def test_same_bytes_new_actual_revision_gets_current_ref(store, tmp_path):
    kernel, task, _ = store
    source = tmp_path / "report.bin"
    source.write_bytes(b"same")
    old = register(store, source)
    kernel.bind_message(
        mission_id="history",
        session_ref="s",
        backend_native_id="s",
        message_id="second",
        text="revise report",
        cwd=str(tmp_path),
    )
    assert kernel.get_task(task)["active_revision"] == 2
    new = register(store, source)
    assert new["artifact_id"] != old["artifact_id"]
    assert kernel.artifacts_for_revision(task, 2)[0]["artifact_id"] == new["artifact_id"]
    assert_payload(old, b"same")


@pytest.mark.parametrize("kind", ["tamper", "missing", "symlink", "parent_symlink", "legacy"])
def test_export_rejects_unavailable_or_corrupt_payload_without_destination(store, tmp_path, kind):
    source = tmp_path / "original.bin"
    source.write_bytes(b"AAAA")
    record = register(store, source)
    payload = Path(record["path"])
    if kind == "legacy":
        store[0]._conn.execute(
            "UPDATE artifacts SET path = ?, metadata_json = '{}' WHERE artifact_id = ?",
            (str(source), record["artifact_id"]),
        )
        store[0]._conn.commit()
        source.write_bytes(b"BBBB")
    elif kind == "tamper":
        payload.chmod(0o600)
        payload.write_bytes(b"BBBB")
    elif kind == "parent_symlink":
        moved = payload.parent.with_name("moved")
        payload.parent.rename(moved)
        payload.parent.symlink_to(moved, target_is_directory=True)
    else:
        payload.unlink()
        if kind == "symlink":
            payload.symlink_to(source)
    dest = tmp_path / "absent-parent" / "out.bin"
    assert export(store, record, dest) == 3
    assert not dest.exists()
    assert not dest.parent.exists()


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling", "parent_symlink"])
def test_export_never_overwrites_or_follows_destination_symlink(store, tmp_path, kind):
    source = tmp_path / "source.bin"
    source.write_bytes(b"AAAA")
    record = register(store, source)
    protected = tmp_path / "protected"
    protected.write_bytes(b"BBBB")
    dest = tmp_path / "out.bin"
    if kind == "file":
        dest.write_bytes(b"KEEP")
    elif kind == "parent_symlink":
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        dest = link / "out.bin"
    else:
        dest.symlink_to(protected if kind == "symlink" else tmp_path / "missing")
    assert export(store, record, dest) == 4
    assert protected.read_bytes() == b"BBBB"
    if kind == "file":
        assert dest.read_bytes() == b"KEEP"
    elif kind == "parent_symlink":
        assert not dest.exists()


def test_large_binary_basename_zone_and_directory_export(store):
    kernel, task, home = store
    source = home / "runs" / task / "r1" / "outputs" / "movie.mp4"
    source.write_bytes(b"\x00\xff" * 262145)
    record = kernel.register_artifact(
        task_id=task,
        path=str(source),
        media_type="video/mp4",
        metadata={"source_path": "forged", "payload_storage": "forged"},
    )
    kernel._conn.commit()
    assert record["size_bytes"] > 262144
    assert record["media_type"] == "video/mp4"
    assert Path(record["path"]).name == "movie.mp4"
    meta = json.loads(record["metadata_json"])
    assert meta["zone"] == "outputs"
    assert meta["source_path"] == str(source)
    assert meta["payload_storage"] == "owned_snapshot_v1"
    dest = home / "export"
    dest.mkdir()
    assert export(store, record, dest) == 0
    assert (dest / "movie.mp4").read_bytes() == source.read_bytes()


def test_scratch_and_model_evidence_remain_rejected(store):
    kernel, task, home = store
    for zone, error in [
        ("scratch", "SCRATCH_NOT_DELIVERABLE"),
        ("evidence", "EVIDENCE_KERNEL_ONLY"),
    ]:
        source = home / "runs" / task / "r1" / zone / "fake.bin"
        source.write_bytes(b"bytes")
        with pytest.raises(ValueError, match=error):
            register(store, source)
    assert not kernel.artifacts_for_revision(task, 1)


def test_corrupt_snapshot_replay_rejects_instead_of_reusing_ref(store, tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"AAAA")
    record = register(store, source)
    payload = Path(record["path"])
    payload.chmod(0o600)
    payload.write_bytes(b"BBBB")
    with pytest.raises(ValueError, match="ARTIFACT_PAYLOAD_CHANGED"):
        register(store, source)
