"""Real-kernel regressions for snapshot integrity and authoritative provenance."""

import json
import sqlite3
from pathlib import Path

from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.service import TaskKernel
from rosclaw.task_kernel.verifier import verdict_for
from rosclaw.task_kernel.verifier_plugins import FixtureProhibitionVerifier


def _registered(tmp_path, relative, metadata=None):
    conn = sqlite3.connect(tmp_path / "kernel.db")
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    kernel = TaskKernel(conn, tmp_path)
    task = kernel.bind_message(
        mission_id="m1",
        session_ref="s1",
        backend_native_id="n1",
        message_id="msg1",
        text="Produce a report",
        cwd=str(tmp_path),
    )
    source = Path(task["workspace_path"]) / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"original payload\n")
    ref = kernel.register_artifact(
        task_id=task["task_id"],
        path=str(source),
        media_type="text/plain",
        metadata=metadata,
    )
    row = dict(
        conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (ref["artifact_id"],)
        ).fetchone()
    )
    assert json.loads(row["metadata_json"])["source_path"] == str(source.resolve())
    assert Path(row["path"]) != source
    assert Path(row["path"]).read_bytes() == b"original payload\n"
    return conn, kernel, task, source, row


def _verdicts(row, workspace):
    direct = verdict_for(artifacts=[row], acceptance={}, workspace=workspace, summary="done")
    plugin = FixtureProhibitionVerifier().check({"artifacts": [row]})
    provenance = [f for f in direct["failures"] if "RESOURCE_PROVENANCE" in f]
    assert provenance == plugin
    return direct, plugin


def _finish(kernel, task, row):
    return kernel.finish_task(
        task_id=task["task_id"], summary="done", artifact_ids=[row["artifact_id"]]
    )


def test_fixture_origin_survives_snapshot_isolation(tmp_path):
    conn, kernel, task, source, row = _registered(tmp_path, "tests/fixtures/data.txt")
    source.unlink()
    direct, plugin = _verdicts(row, tmp_path)
    assert plugin and direct["status"] == "REPAIR_REQUIRED"
    result = _finish(kernel, task, row)
    assert result["status"] == "REPAIR_REQUIRED"
    assert any("RESOURCE_PROVENANCE" in f for f in result["failures"])
    conn.close()


def test_caller_cannot_hide_fixture_origin(tmp_path):
    conn, kernel, task, source, row = _registered(
        tmp_path,
        "tests/fixtures/data.txt",
        {"source_path": "/production/report.txt", "payload_storage": "fake"},
    )
    direct, plugin = _verdicts(row, tmp_path)
    assert plugin and direct["status"] == "REPAIR_REQUIRED"
    assert source.exists()
    result = _finish(kernel, task, row)
    assert result["status"] == "REPAIR_REQUIRED"
    assert any("RESOURCE_PROVENANCE" in f for f in result["failures"])
    conn.close()


def test_fake_fixture_metadata_cannot_condemn_production(tmp_path):
    conn, kernel, task, source, row = _registered(
        tmp_path,
        "report.txt",
        {"source_path": "/tests/fixtures/fake.txt", "payload_storage": "fake"},
    )
    source.write_bytes(b"canonical has changed")
    direct, plugin = _verdicts(row, tmp_path)
    assert not plugin and direct["status"] == "PASS"
    source.unlink()
    direct, plugin = _verdicts(row, tmp_path)
    assert not plugin and direct["status"] == "PASS"
    assert _finish(kernel, task, row)["status"] == "SUCCEEDED"
    conn.close()


def test_owned_payload_tamper_still_fails(tmp_path):
    conn, kernel, task, source, row = _registered(tmp_path, "report.txt")
    source.unlink()
    payload = Path(row["path"])
    payload.chmod(0o600)
    payload.write_bytes(b"tampered payload\n")
    direct, plugin = _verdicts(row, tmp_path)
    assert not plugin and direct["status"] == "REPAIR_REQUIRED"
    assert any("hash" in f for f in direct["failures"])
    result = _finish(kernel, task, row)
    assert result["status"] == "REPAIR_REQUIRED"
    assert any("hash" in f for f in result["failures"])
    conn.close()


def test_legacy_fixture_payload_remains_fail_closed(tmp_path):
    conn, kernel, task, source, row = _registered(tmp_path, "tests/fixtures/data.txt")
    # Reproduce a pre-snapshot row in the real ledger, preserving real bytes/hash.
    conn.execute(
        "UPDATE artifacts SET path = ?, metadata_json = '{}' WHERE artifact_id = ?",
        (str(source), row["artifact_id"]),
    )
    row = dict(
        conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (row["artifact_id"],)
        ).fetchone()
    )
    direct, plugin = _verdicts(row, tmp_path)
    assert plugin and direct["status"] == "REPAIR_REQUIRED"
    result = _finish(kernel, task, row)
    assert result["status"] == "REPAIR_REQUIRED"
    assert any("RESOURCE_PROVENANCE" in f for f in result["failures"])
    conn.close()
