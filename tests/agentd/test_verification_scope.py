"""Actual completion authority distinguishes configured checks from delivery."""

import json
from pathlib import Path

import pytest

from rosclaw.task_kernel.coordinator import TaskCoordinator
from tests.agentd.test_p0d_coordinator import _kernel, _make_task, _register_file


@pytest.mark.parametrize("acceptance", [{}, {"unknown_condition": True}])
def test_integrity_completion_is_not_semantic_acceptance(tmp_path, acceptance):
    kernel, conn = _kernel(tmp_path)
    task_id = _make_task(kernel, tmp_path)
    kernel.set_acceptance(task_id, acceptance)
    _register_file(kernel, tmp_path, task_id)
    coordinator = TaskCoordinator(kernel)
    outcome = coordinator.consider(task_id)
    assert outcome["verification"] == "PASS"
    assert outcome["verification_scope"] == "artifact_integrity_only"
    assert outcome["task_semantic_verification"] == "UNVERIFIED"
    assert outcome["delivery"] == "DELIVERED"
    assert kernel.get_task(task_id)["state"] == "SUCCEEDED"
    checks = json.loads(conn.execute("SELECT checks_json FROM verifications").fetchone()[0])
    assert checks["verification_scope"] == outcome["verification_scope"]
    events = [
        json.loads(r[0])
        for r in conn.execute(
            "SELECT payload_json FROM task_events WHERE event_type='verification.completed'"
        )
    ]
    assert events[-1]["task_semantic_verification"] == "UNVERIFIED"
    assert coordinator.consider(task_id) == outcome


def test_required_file_contract_still_really_checks(tmp_path):
    kernel, _ = _kernel(tmp_path)
    task_id = _make_task(kernel, tmp_path)
    kernel.set_acceptance(task_id, {"required_files": ["gate.txt"]})
    _register_file(kernel, tmp_path, task_id)
    assert TaskCoordinator(kernel).consider(task_id)["verification"] == "FAIL"
    work = Path(kernel.get_task(task_id)["workspace_path"])
    (work / "gate.txt").write_text("ready")
    outcome = TaskCoordinator(kernel).consider(task_id)
    assert outcome["verification"] == "PASS"
    assert outcome["verification_scope"] == "configured_acceptance"
    assert outcome["task_semantic_verification"] == "CONFIGURED_CHECKS_ONLY"


def test_direct_finish_scope_and_repeat_do_not_reverify(tmp_path):
    kernel, conn = _kernel(tmp_path)
    task_id = _make_task(kernel, tmp_path)
    result = kernel.finish_task(task_id=task_id, summary="A response", artifact_ids=[])
    assert result["verification_scope"] == "summary_nonempty_only"
    assert result["task_semantic_verification"] == "UNVERIFIED"
    before = conn.execute("SELECT checks_json FROM verifications").fetchone()[0]
    repeated = kernel.finish_task(task_id=task_id, summary="", artifact_ids=[])
    assert repeated["already_terminal"]
    assert repeated["verification_scope"] == result["verification_scope"]
    assert conn.execute("SELECT checks_json FROM verifications").fetchone()[0] == before


def test_legacy_terminal_scope_not_backfilled(tmp_path):
    kernel, conn = _kernel(tmp_path)
    task_id = _make_task(kernel, tmp_path)
    kernel.finish_task(task_id=task_id, summary="A response", artifact_ids=[])
    conn.execute("UPDATE verifications SET checks_json=?", ('{"checks":1}',))
    result = kernel.finish_task(task_id=task_id, summary="", artifact_ids=[])
    assert "verification_scope" not in result
    assert conn.execute("SELECT checks_json FROM verifications").fetchone()[0] == '{"checks":1}'


def test_progress_stays_active(tmp_path):
    kernel, conn = _kernel(tmp_path)
    task_id = _make_task(kernel, tmp_path)
    work = Path(kernel.get_task(task_id)["workspace_path"])
    p = work / "progress.txt"
    p.write_text("in progress")
    kernel.register_artifact(
        task_id=task_id, path=str(p), media_type="text/plain", metadata={"role": "progress_report"}
    )
    assert TaskCoordinator(kernel).consider(task_id) is None
    assert kernel.get_task(task_id)["state"] == "RUNNING"
    assert conn.execute("SELECT COUNT(*) FROM verifications").fetchone()[0] == 0


def test_configured_run_checks_fail_then_pass(tmp_path):
    kernel, _ = _kernel(tmp_path)
    task_id = _make_task(kernel, tmp_path)
    _register_file(kernel, tmp_path, task_id)
    kernel.set_acceptance(task_id, {"run": {"argv": ["python", "-c", "raise SystemExit(2)"]}})
    assert TaskCoordinator(kernel).consider(task_id)["verification"] == "FAIL"
    kernel.set_acceptance(task_id, {"run": {"argv": ["python", "-c", "pass"]}})
    result = TaskCoordinator(kernel).consider(task_id)
    assert result["verification"] == "PASS"
    assert result["task_semantic_verification"] == "CONFIGURED_CHECKS_ONLY"


def test_frozen_required_deliverable_and_historical_outcome(tmp_path):
    kernel, conn = _kernel(tmp_path)
    task_id = _make_task(kernel, tmp_path)
    _register_file(kernel, tmp_path, task_id)
    row = conn.execute(
        "SELECT task_spec_json FROM task_revisions WHERE task_id=?", (task_id,)
    ).fetchone()
    spec = json.loads(row[0])
    spec["deliverables"] = [{"kind": "data", "required": True}]
    conn.execute(
        "UPDATE task_revisions SET task_spec_json=? WHERE task_id=?", (json.dumps(spec), task_id)
    )
    coordinator = TaskCoordinator(kernel)
    result = coordinator.consider(task_id)
    assert result["verification_scope"] == "declared_deliverables"
    assert result["task_semantic_verification"] == "CONFIGURED_CHECKS_ONLY"
    result.pop("verification_scope")
    result.pop("task_semantic_verification")
    old_json = json.dumps(result)
    conn.execute("UPDATE task_outcomes SET outcome_json=? WHERE task_id=?", (old_json, task_id))
    assert coordinator.consider(task_id) == result
    assert (
        conn.execute(
            "SELECT outcome_json FROM task_outcomes WHERE task_id=?", (task_id,)
        ).fetchone()[0]
        == old_json
    )


def test_optional_deliverables_do_not_invent_acceptance(tmp_path):
    kernel, conn = _kernel(tmp_path)
    task_id = _make_task(kernel, tmp_path)
    _register_file(kernel, tmp_path, task_id)
    row = conn.execute(
        "SELECT task_spec_json FROM task_revisions WHERE task_id=?", (task_id,)
    ).fetchone()
    spec = json.loads(row[0])
    spec["deliverables"] = [{"kind": "data", "required": False}]
    conn.execute(
        "UPDATE task_revisions SET task_spec_json=? WHERE task_id=?", (json.dumps(spec), task_id)
    )
    assert TaskCoordinator(kernel).consider(task_id)["task_semantic_verification"] == "UNVERIFIED"
