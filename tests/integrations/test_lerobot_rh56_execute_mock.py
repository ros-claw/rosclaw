"""P5 §13.5 execution Practice evidence tests (mock transport, real worker).

These tests run the full receding-horizon execute loop against the mock
Modbus transport and the real persistent policy worker with the RH56
reference policy, then check Practice evidence completeness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.body.rh56.calibration import load_rh56_calibration
from rosclaw.body.rh56.transport_profile import load_transport_profile
from rosclaw.integrations.lerobot.execution import ArmingController, PermitManager
from rosclaw.integrations.lerobot.rollout.rh56_execute import run_rh56_execute

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
POLICIES = REPO_ROOT / "policies"

pytestmark = pytest.mark.skipif(
    not (POLICIES / "rh56_reference_policy_v1" / "config.json").exists(),
    reason="rh56 reference policy artifact missing",
)


def _armed_stack(tmp_path: Path):
    profile = load_transport_profile(CONFIGS / "rh56_right_rs485_v1.yaml")
    calib = load_rh56_calibration(CONFIGS / "rh56_right_01_calibration.yaml")
    hashes = {
        "policy_contract_hash": "sha256:policy",
        "body_hash": "sha256:body",
        "calibration_hash": calib.content_hash(),
        "mapping_hash": "sha256:mapping",
        "transport_profile_hash": profile.content_hash(),
    }
    pm = PermitManager()
    permit = pm.issue(
        body_id="rh56_mock",
        **hashes,
        max_step_delta_raw=30,
        operator_armed=True,
        physical_estop_confirmed=True,
        calibration_status="validated",
        execution_mode="FIXTURE",
    )
    arming = ArmingController(pm)
    arming.begin_preflight()
    arming.mark_shadow_validated(**hashes)
    arming.arm(permit.permit_id)
    return profile, calib, pm, permit, arming


def _worker_python() -> str:
    """Resolve the LeRobot worker python (same source as the P4 smoke tests)."""
    from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime

    configured = get_configured_lerobot_runtime()
    if configured and configured.get("python_executable"):
        return str(configured["python_executable"])
    pytest.skip("LeRobot runtime not configured")


def _run_execute(tmp_path: Path, *, task: str = "hold_current", steps: int = 3):
    profile, calib, pm, permit, arming = _armed_stack(tmp_path)
    practice_root = tmp_path / "practice"
    result, report = run_rh56_execute(
        policy_path=str(POLICIES / "rh56_reference_policy_v1"),
        transport_profile_path=str(CONFIGS / "rh56_right_rs485_v1.yaml"),
        permit_id=permit.permit_id,
        permit_manager=pm,
        arming=arming,
        calibration_path=str(CONFIGS / "rh56_right_01_calibration.yaml"),
        task=task,
        steps=steps,
        control_hz=5.0,
        practice_data_root=practice_root,
        python_executable=_worker_python(),
        fixture_mode=True,
    )
    return result, report, practice_root


def test_execute_requires_explicit_fixture_mode() -> None:
    with pytest.raises(RuntimeError, match="RUNTIME_ACTION_GATEWAY_REQUIRED"):
        run_rh56_execute(
            policy_path="unused",
            transport_profile_path="unused",
            permit_id="unused",
            permit_manager=PermitManager(),
            arming=ArmingController(),
        )


def test_execution_events_complete(tmp_path: Path, real_lerobot_runtime_config) -> None:
    result, report, _ = _run_execute(tmp_path)
    assert result.stop_reason.value == "completed", result.errors
    assert result.execution_mode == "FIXTURE"
    assert result.trust_level == "SYNTHETIC"
    assert result.verified is False
    assert result.commands_executed == result.steps_completed
    assert result.fixture_actions_executed == result.steps_completed
    assert result.hardware_actions_executed == 0
    assert report.summary()["hardware_actions_executed"] == 0
    trace_events = {etype for etype, _ in report.events}
    required = {
        "execution.armed",
        "execution.command.requested",
        "execution.command.sent",
        "execution.command.protocol_acknowledged",
        "execution.feedback.verified",
        "execution.step.completed",
    }
    assert required <= trace_events
    assert "execution.command.acknowledged" not in trace_events

    trace_path = Path(result.trace_path)
    recorded_types = {
        json.loads(line)["event_type"]
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert required <= recorded_types


def test_proposed_mapped_executed_actual_distinct(
    tmp_path: Path, real_lerobot_runtime_config
) -> None:
    result, report, _ = _run_execute(tmp_path, task="micro_index_flex", steps=4)
    assert result.stop_reason.value == "completed", result.errors
    trace_path = Path(result.trace_path)
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Every executed step must expose the distinct action stages (plan §10).
    inferences = [e for e in events if e["event_type"] == "rollout.policy.inference"]
    mapped = [e for e in events if e["event_type"] == "rollout.action.mapped"]
    sandbox = [e for e in events if e["event_type"] == "rollout.sandbox.decision"]
    executed = [e for e in events if e["event_type"] == "execution.step.completed"]
    feedback = [e for e in events if e["event_type"] == "execution.feedback.verified"]
    assert len(inferences) == len(mapped) == len(sandbox) == len(executed) == len(feedback) == 4

    proposal = inferences[0]["payload"]["inference"]
    assert proposal["action"]["names"] == [
        "little",
        "ring",
        "middle",
        "index",
        "thumb",
        "thumb_rot",
    ]
    exec_result = executed[0]["payload"]["result"]
    assert exec_result["target"] == proposal["action"]["values"]
    assert len(exec_result["actual"]) == 6
    assert exec_result["verification"]["position_reached"] is True


def test_execution_practice_verify_strict(tmp_path: Path, real_lerobot_runtime_config) -> None:
    result, _, practice_root = _run_execute(tmp_path)
    assert result.practice_id
    from rosclaw.practice.verifier import PracticeVerifier

    verifier = PracticeVerifier(practice_root)
    report = verifier.verify(result.practice_id, strict=True)
    assert report.passed, report.issues


def test_execution_dataset_export(tmp_path: Path, real_lerobot_runtime_config) -> None:
    result, _, practice_root = _run_execute(tmp_path)
    frames_file = practice_root / "sessions" / result.practice_id / "frames_episode.json"
    assert frames_file.exists()
    doc = json.loads(frames_file.read_text(encoding="utf-8"))
    assert len(doc["frames"]) == 3
    frame = doc["frames"][0]
    assert frame["observation"]["state"]
    assert frame["action"]
    assert "motor_current" in frame["observation"]
    assert "joint_temperature" in frame["observation"]
    assert "force_torque" in frame["observation"]
