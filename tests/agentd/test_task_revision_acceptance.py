"""Same-root continuations must not erase a frozen completion contract."""

from pathlib import Path

from rosclaw.task_kernel.coordinator import TaskCoordinator
from tests.agentd.test_pi_tool_bridge import _kernel


def bind(kernel, home: Path, message: str, *, force_new: bool = False):
    return kernel.bind_message(
        mission_id="private_mission",
        session_ref="private_session",
        backend_native_id="private_session",
        message_id=message,
        text=f"continue source review {message}",
        cwd=str(home),
        body_id="sim/ur5e",
        force_new=force_new,
    )


def deliver_report(kernel, task_id: str):
    workspace = Path(kernel.get_task(task_id)["workspace_path"])
    path = workspace / "source_report.json"
    path.write_text('{"scope":"SOURCE_ONLY","physics_pass":false}')
    kernel.register_artifact(
        task_id=task_id,
        path=str(path),
        media_type="application/json",
        metadata={"role": "report"},
    )
    return TaskCoordinator(kernel).consider(task_id)


def test_continuation_keeps_required_file_and_real_coordinator_rejects(tmp_path):
    kernel, conn = _kernel(tmp_path)
    task_id = bind(kernel, tmp_path, "first")["task_id"]
    kernel.set_acceptance(task_id, {"required_files": ["independent_gate.json"]})
    original = tuple(
        conn.execute(
            "SELECT acceptance_json, acceptance_spec_json, task_spec_json "
            "FROM task_revisions WHERE task_id=? AND revision=1",
            (task_id,),
        ).fetchone()
    )
    old_spec = kernel.get_acceptance_spec(task_id)
    revised = bind(kernel, tmp_path, "second")
    assert revised["task_id"] == task_id and revised["revision"] == 2
    current = kernel.get_acceptance_spec(task_id)
    assert current["revision"] == 2 and current["task_id"] == task_id
    assert current["spec_id"] != old_spec["spec_id"]
    assert kernel.get_task_spec(task_id)["acceptance_spec_id"] == current["spec_id"]
    assert kernel.get_task_spec(task_id)["subjects"]["body_ref"] == "robot:ur5e"
    assert (
        tuple(
            conn.execute(
                "SELECT acceptance_json, acceptance_spec_json, task_spec_json "
                "FROM task_revisions WHERE task_id=? AND revision=1",
                (task_id,),
            ).fetchone()
        )
        == original
    )
    rejected = deliver_report(kernel, task_id)
    assert kernel.get_task(task_id)["state"] != "SUCCEEDED"
    assert rejected["verification"] == "FAIL"
    (Path(kernel.get_task(task_id)["workspace_path"]) / "independent_gate.json").write_text("pass")
    accepted = TaskCoordinator(kernel).consider(task_id)
    assert accepted["verification"] == "PASS"
    assert kernel.get_task(task_id)["state"] == "SUCCEEDED"
    conn.close()


def test_latest_explicit_replacement_is_inherited(tmp_path):
    kernel, conn = _kernel(tmp_path)
    task_id = bind(kernel, tmp_path, "first")["task_id"]
    kernel.set_acceptance(task_id, {"required_files": ["old_gate.json"]})
    bind(kernel, tmp_path, "second")
    kernel.set_acceptance(task_id, {"required_files": ["new_gate.json"]})
    assert (
        kernel.get_task_spec(task_id)["acceptance_spec_id"]
        == kernel.get_acceptance_spec(task_id)["spec_id"]
    )
    bind(kernel, tmp_path, "third")
    workspace = Path(kernel.get_task(task_id)["workspace_path"])
    (workspace / "old_gate.json").write_text("old")
    assert deliver_report(kernel, task_id)["verification"] == "FAIL"
    (workspace / "new_gate.json").write_text("new")
    assert TaskCoordinator(kernel).consider(task_id)["verification"] == "PASS"
    conn.close()


def test_force_new_root_has_no_previous_contract(tmp_path):
    kernel, conn = _kernel(tmp_path)
    old = bind(kernel, tmp_path, "first")["task_id"]
    kernel.set_acceptance(old, {"required_files": ["old_gate.json"]})
    new = bind(kernel, tmp_path, "new", force_new=True)["task_id"]
    assert new != old
    assert kernel.get_acceptance_spec(new) is None
    assert kernel.get_task_spec(new)["acceptance_spec_id"] == ""
    assert deliver_report(kernel, new)["verification"] == "PASS"
    conn.close()


def test_reopened_success_retains_contract_and_supersedes_old_verification(tmp_path):
    kernel, conn = _kernel(tmp_path)
    task_id = bind(kernel, tmp_path, "first")["task_id"]
    kernel.set_acceptance(task_id, {"required_files": ["gate.json"]})
    gate = Path(kernel.get_task(task_id)["workspace_path"]) / "gate.json"
    gate.write_text("pass")
    assert deliver_report(kernel, task_id)["verification"] == "PASS"
    prior_spec = kernel.get_acceptance_spec(task_id)
    gate.unlink()
    revised = bind(kernel, tmp_path, "correct")
    assert revised["reopened"] and revised["revision"] == 2
    assert kernel.get_acceptance_spec(task_id)["spec_id"] != prior_spec["spec_id"]
    assert (
        conn.execute("SELECT status FROM verifications WHERE task_id=?", (task_id,)).fetchone()[0]
        == "SUPERSEDED"
    )
    assert deliver_report(kernel, task_id)["verification"] == "FAIL"
    conn.close()


def test_replay_does_not_recompile_contract_and_empty_legacy_stays_empty(tmp_path):
    kernel, conn = _kernel(tmp_path)
    task_id = bind(kernel, tmp_path, "first")["task_id"]
    bind(kernel, tmp_path, "second")
    assert kernel.get_acceptance_spec(task_id) is None
    assert deliver_report(kernel, task_id)["verification"] == "PASS"
    kernel.set_acceptance(task_id, {"required_files": ["gate.json"]})
    bind(kernel, tmp_path, "third")
    compiled = kernel.get_acceptance_spec(task_id)
    assert bind(kernel, tmp_path, "third")["replayed"]
    assert kernel.get_acceptance_spec(task_id) == compiled
    conn.close()


def test_explicit_empty_replacement_is_respected(tmp_path):
    kernel, conn = _kernel(tmp_path)
    task_id = bind(kernel, tmp_path, "first")["task_id"]
    kernel.set_acceptance(task_id, {"required_files": ["gate.json"]})
    kernel.set_acceptance(task_id, {})
    bind(kernel, tmp_path, "second")
    assert kernel.get_acceptance_spec(task_id)["revision"] == 2
    assert deliver_report(kernel, task_id)["verification"] == "PASS"
    conn.close()


def test_compiled_contract_is_rebound_without_losing_declared_fields(tmp_path):
    kernel, conn = _kernel(tmp_path)
    task_id = bind(kernel, tmp_path, "first")["task_id"]
    kernel.set_acceptance(
        task_id,
        {
            "required_files": ["gate.json"],
            "required_artifacts": ["trace"],
            "numeric_thresholds": {"declared_error_m": 0.01},
            "evidence_classes": ["SIMULATED"],
            "resource_provenance_required": True,
            "required_receipt": "declared_receipt",
            "verifier_refs": ["declared_verifier"],
        },
    )
    prior = kernel.get_acceptance_spec(task_id)
    bind(kernel, tmp_path, "second")
    current = kernel.get_acceptance_spec(task_id)
    for field in prior:
        if field not in {"spec_id", "revision"}:
            assert current[field] == prior[field]
    assert current["revision"] == 2
    assert current["spec_id"] != prior["spec_id"]
    assert (
        kernel.get_task_spec(task_id)["goal"]["natural_language"] == "continue source review second"
    )
    conn.close()
