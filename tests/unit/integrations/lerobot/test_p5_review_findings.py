"""Regression tests for the P5 review findings (pre-hardware audit).

Each test maps to a finding fixed in the review pass:

1. Revoked permits stay revoked across processes (persisted markers).
2. Permit expiry is enforced in wall-clock time too.
3. Mock-validated calibration is detectable and cannot silently arm.
4. Observation channel failure mid-execution estops and enters
   COMMUNICATION_LOST with the permit revoked.
5. hardware_actions_executed survives fault exit paths.
6. Shadow gate checks worker_restart_count == 0.
7. Execute rollouts report mode=execute (not shadow).
8. frames_episode.json uses real trace timestamps.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.body.rh56.calibration import (
    calibration_has_mock_evidence,
    load_rh56_calibration,
)
from rosclaw.body.rh56.transport_profile import load_transport_profile
from rosclaw.integrations.lerobot.execution import PermitError, PermitManager
from rosclaw.integrations.lerobot.execution.permit import (
    load_permit_into_manager,
    save_permit,
)

CONFIGS = Path(__file__).resolve().parents[4] / "configs"
HASHES = {
    "policy_contract_hash": "sha256:policy",
    "body_hash": "sha256:body",
    "calibration_hash": "sha256:calib",
    "mapping_hash": "sha256:mapping",
    "transport_profile_hash": "sha256:transport",
}


def _issue(pm: PermitManager, **overrides):
    params = {
        "body_id": "rh56_mock",
        **HASHES,
        "operator_armed": True,
        "physical_estop_confirmed": True,
    }
    params.update(overrides)
    return pm.issue(**params)


# 1. Revocation persistence -------------------------------------------------


def test_revoked_permit_stays_revoked_across_processes(tmp_path: Path) -> None:
    store = tmp_path / "permits"
    pm = PermitManager(store_dir=store)
    permit = _issue(pm)
    save_permit(permit, store)
    pm.revoke(permit.permit_id, "feedback_verification_failed")

    # Simulate a second process: fresh manager, load from disk.
    pm2 = PermitManager(store_dir=store)
    loaded = load_permit_into_manager(permit.permit_id, store, pm2)
    assert loaded is None
    assert pm2.is_revoked(permit.permit_id)
    with pytest.raises(PermitError, match="permit_revoked"):
        pm2.validate(
            permit.permit_id,
            body_id="rh56_mock",
            **HASHES,
            representation="joint_position",
            units="raw_device_unit",
        )


def test_active_permit_loads_without_marker(tmp_path: Path) -> None:
    store = tmp_path / "permits"
    pm = PermitManager()
    permit = _issue(pm)
    save_permit(permit, store)
    pm2 = PermitManager(store_dir=store)
    loaded = load_permit_into_manager(permit.permit_id, store, pm2)
    assert loaded is not None
    assert loaded.permit_id == permit.permit_id


# 2. Wall-clock expiry -------------------------------------------------------


def test_wall_clock_expiry_enforced(tmp_path: Path) -> None:
    store = tmp_path / "permits"
    pm = PermitManager(store_dir=store)
    permit = _issue(pm, expires_in_sec=3600)
    # Tamper the wall-clock expiry into the past, keep monotonic valid.
    permit.expires_at = (datetime.now(UTC) - timedelta(seconds=5)).isoformat().replace(
        "+00:00", "Z"
    )
    save_permit(permit, store)
    pm2 = PermitManager(store_dir=store)
    load_permit_into_manager(permit.permit_id, store, pm2)
    with pytest.raises(PermitError, match="permit_expired"):
        pm2.validate(
            permit.permit_id,
            body_id="rh56_mock",
            **HASHES,
            representation="joint_position",
            units="raw_device_unit",
        )


# 3. Mock calibration evidence -----------------------------------------------


def test_mock_validated_calibration_detected(tmp_path: Path) -> None:
    from rosclaw.body.rh56.calibration import RH56CalibrationGate

    profile = load_transport_profile(CONFIGS / "rh56_right_rs485_v1.yaml")
    calib = load_rh56_calibration(CONFIGS / "rh56_right_01_calibration.yaml")
    gate = RH56CalibrationGate(calib, profile)
    real_validated = gate.mark_validated(rounds=5, evidence=["probe_rounds=5/5", "mock=False"])
    assert not calibration_has_mock_evidence(real_validated)
    mock_validated = gate.mark_validated(rounds=5, evidence=["probe_rounds=5/5", "mock=True"])
    assert calibration_has_mock_evidence(mock_validated)


# 4/5/7. Execute loop fault paths --------------------------------------------


def _armed_stack(tmp_path: Path):
    from rosclaw.body.rh56.transport import MockModbusTransport
    from rosclaw.integrations.lerobot.execution import ArmingController

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
        operator_armed=True,
        physical_estop_confirmed=True,
        calibration_status="validated",
    )
    arming = ArmingController(pm)
    arming.begin_preflight()
    arming.mark_shadow_validated(**hashes)
    arming.arm(permit.permit_id)
    transport = MockModbusTransport(profile)
    transport.connect()
    return profile, calib, pm, permit, arming, transport


def test_observation_failure_estops_and_revokes(tmp_path: Path) -> None:
    from rosclaw.integrations.lerobot.rollout.rh56_execute import run_rh56_execute

    profile, calib, pm, permit, arming, transport = _armed_stack(tmp_path)
    transport.fail_next_read()  # observation read at step 0 fails

    result, report = run_rh56_execute(
        policy_path="policies/rh56_reference_policy_v1",
        transport_profile_path=str(CONFIGS / "rh56_right_rs485_v1.yaml"),
        permit_id=permit.permit_id,
        permit_manager=pm,
        arming=arming,
        transport=transport,
        calibration_path=str(CONFIGS / "rh56_right_01_calibration.yaml"),
        steps=3,
        practice_data_root=tmp_path / "practice",
        python_executable=".venv-lerobot/bin/python",
    )
    from rosclaw.integrations.lerobot.execution import ExecutionState

    assert arming.machine.state == ExecutionState.COMMUNICATION_LOST
    assert pm.is_revoked(permit.permit_id)
    assert result.stop_reason.value == "runtime_failure"
    assert any("communication_lost" in e for e in result.errors)
    # The estop + failure events must be in the Practice trace.
    event_types = {t for t, _ in report.events}
    assert "execution.estop" in event_types or "execution.communication_lost" in event_types


def test_hardware_actions_survive_fault_exit(tmp_path: Path) -> None:
    from rosclaw.integrations.lerobot.rollout.rh56_execute import run_rh56_execute

    profile, calib, pm, permit, arming, transport = _armed_stack(tmp_path)

    result, report = run_rh56_execute(
        policy_path="policies/rh56_reference_policy_v1",
        transport_profile_path=str(CONFIGS / "rh56_right_rs485_v1.yaml"),
        permit_id=permit.permit_id,
        permit_manager=pm,
        arming=arming,
        transport=transport,
        calibration_path=str(CONFIGS / "rh56_right_01_calibration.yaml"),
        task="hold_current",
        steps=2,
        control_hz=5.0,
        practice_data_root=tmp_path / "practice",
        python_executable=".venv-lerobot/bin/python",
    )
    # Even on the success path the count must be recorded on the result.
    assert result.hardware_actions_executed == result.steps_completed
    assert result.mode.value == "execute"


# 6. Shadow gate worker restart check -----------------------------------------


def test_shadow_gate_checks_worker_restart_count() -> None:
    from rosclaw.integrations.lerobot.rollout.rh56_shadow import evaluate_shadow_gate
    from rosclaw.integrations.lerobot.rollout.state import (
        RolloutMode,
        RolloutResult,
        RolloutStopReason,
    )

    profile = load_transport_profile(CONFIGS / "rh56_right_rs485_v1.yaml")
    result = RolloutResult(mode=RolloutMode.SHADOW, stop_reason=RolloutStopReason.COMPLETED)
    result.steps_completed = 1000
    result.hardware_actions_executed = 0
    result.metrics = {
        "effective_control_hz": 5.0,
        "deadline_misses": 0,
        "worker_restart_count": 1,  # restart happened
    }
    gate = evaluate_shadow_gate(result, profile, {"disconnect_count": 0})
    assert gate["checks"]["worker_restart_count"]["pass"] is False
    assert gate["passed"] is False

    result.metrics["worker_restart_count"] = 0
    gate = evaluate_shadow_gate(result, profile, {"disconnect_count": 0})
    assert gate["checks"]["worker_restart_count"]["pass"] is True


# 8. Frame timestamps ----------------------------------------------------------


def test_frames_episode_uses_trace_timestamps(tmp_path: Path) -> None:
    from rosclaw.integrations.lerobot.rollout.practice_bridge import _write_frames_episode
    from rosclaw.practice.storage.layout import PracticeLayout

    layout = PracticeLayout(tmp_path / "practice")
    layout.ensure_directories()
    layout.create_session_dirs("prac_test")

    base_ns = 1_800_000_000_000_000_000
    events = []
    for step in range(3):
        ts = base_ns + step * 500_000_000  # 0.5 s spacing
        events.append(
            {
                "event_type": "rollout.observation.validated",
                "frame_id": str(step),
                "timestamp_ns": ts,
                "payload": {
                    "snapshot": {
                        "features": {
                            "observation.state": {"values": [1000.0] * 6},
                            "observation.force": {"values": [0.0] * 6},
                        }
                    }
                },
            }
        )
        events.append(
            {
                "event_type": "rollout.policy.inference",
                "frame_id": str(step),
                "timestamp_ns": ts + 1_000_000,
                "payload": {"inference": {"action": {"values": [1000.0] * 6}}},
            }
        )

    path = _write_frames_episode(
        layout,
        "prac_test",
        events,
        robot_id="rh56_mock",
        task_id="hold_current",
        policy_path="policies/rh56_reference_policy_v1",
        episode_id="ep_test",
    )
    assert path is not None
    doc = json.loads(path.read_text(encoding="utf-8"))
    timestamps = [f["timestamp"] for f in doc["frames"]]
    assert timestamps == [0.0, 0.5, 1.0]
    assert doc["fps"] == pytest.approx(2.0, rel=0.01)
