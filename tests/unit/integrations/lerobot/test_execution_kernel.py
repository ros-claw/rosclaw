"""P5 §13.3 execution kernel tests (mock transport)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rosclaw.body.execution.rh56_executor import RH56Executor
from rosclaw.body.rh56.calibration import load_rh56_calibration
from rosclaw.body.rh56.transport import MockModbusTransport
from rosclaw.body.rh56.transport_profile import load_transport_profile
from rosclaw.integrations.lerobot.execution import (
    ArmingController,
    ExecutionState,
    FeedbackVerifier,
    PermitManager,
    SingleStepExecutor,
)

CONFIGS = Path(__file__).resolve().parents[4] / "configs"
PROFILE_PATH = CONFIGS / "rh56_right_rs485_v1.yaml"
CALIBRATION_PATH = CONFIGS / "rh56_right_01_calibration.yaml"

HASHES = {
    "policy_contract_hash": "sha256:policy",
    "body_hash": "sha256:body",
    "calibration_hash": "sha256:calib",
    "mapping_hash": "sha256:mapping",
    "transport_profile_hash": "sha256:transport",
}
NAMES = ["little", "ring", "middle", "index", "thumb", "thumb_rot"]


def _stack(*, arm: bool = True, max_step_delta: float = 30.0, max_age_ms: float = 300.0):
    profile = load_transport_profile(PROFILE_PATH)
    calibration = load_rh56_calibration(CALIBRATION_PATH)
    transport = MockModbusTransport(profile)
    transport.connect()
    pm = PermitManager()
    permit = pm.issue(
        body_id="rh56_mock",
        **HASHES,
        max_step_delta_raw=max_step_delta,
        operator_armed=True,
        physical_estop_confirmed=True,
        calibration_status="validated",
    )
    arming = ArmingController(pm)
    events: list[tuple[str, dict]] = []
    executor = SingleStepExecutor(
        executor=RH56Executor(transport, profile),
        profile=profile,
        permit_manager=pm,
        arming=arming,
        verifier=FeedbackVerifier(profile, calibration),
        max_action_age_ms=max_age_ms,
        event_sink=lambda etype, payload: events.append((etype, payload)),
    )
    if arm:
        arming.begin_preflight()
        arming.mark_shadow_validated(**HASHES)
        arming.arm(permit.permit_id)
    return profile, transport, pm, permit, arming, executor, events


def _run(executor: SingleStepExecutor, permit_id: str, values=None, **overrides):
    params = {
        "permit_id": permit_id,
        "proposal_id": "proposal_test",
        "names": NAMES,
        "values": values if values is not None else [1000.0] * 6,
        "representation": "joint_position",
        "units": "raw_device_unit",
        "hashes": HASHES,
        "speed": 100,
        "force_limit_g": 100.0,
        "observation_timestamp_ns": time.monotonic_ns(),
    }
    params.update(overrides)
    return executor.execute_candidate(**params)


def test_disarmed_execution_blocked() -> None:
    _, transport, pm, permit, arming, executor, _ = _stack(arm=False)
    result = _run(executor, permit.permit_id)
    assert result.status == "blocked"
    assert result.error_code == "not_armed"
    assert result.command_sent is False


def test_noop_execution() -> None:
    _, transport, _, permit, _, executor, events = _stack()
    # Current position == open (1000); targeting the same must succeed.
    result = _run(executor, permit.permit_id)
    assert result.status == "completed"
    assert result.command_sent and result.command_acknowledged
    assert result.verification.position_reached
    assert executor.hardware_actions_executed == 1
    event_types = {t for t, _ in events}
    assert "execution.command.sent" in event_types
    assert "execution.feedback.verified" in event_types
    assert "execution.step.completed" in event_types


def test_single_step_only() -> None:
    _, _, _, permit, _, executor, _ = _stack()
    result = _run(executor, permit.permit_id, values=[980.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    assert result.status == "completed"
    # Exactly one command per call.
    assert executor.hardware_actions_executed == 1


def test_open_loop_chunk_blocked() -> None:
    # A "chunk" here means values that would move actuators beyond the permit's
    # step delta in one call — the executor refuses multi-waypoint dispatch.
    _, _, _, permit, _, executor, _ = _stack(max_step_delta=30)
    result = _run(
        executor,
        permit.permit_id,
        values=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0],  # 900 raw jump
    )
    assert result.status == "blocked"
    assert "step_delta_exceeded" in (result.error_code or "")


def test_stale_action_not_sent() -> None:
    _, transport, _, permit, _, executor, _ = _stack(max_age_ms=50)
    stale_ns = time.monotonic_ns() - int(1e9)  # 1 s old observation
    result = _run(executor, permit.permit_id, observation_timestamp_ns=stale_ns)
    assert result.status == "stale_action"
    assert result.command_sent is False
    assert executor.hardware_actions_executed == 0


def test_sandbox_block_not_sent() -> None:
    # Permit hash mismatch stands in for any pre-command gate failure:
    # the command must never reach the transport.
    _, transport, _, permit, _, executor, _ = _stack()
    bad_hashes = {**HASHES, "mapping_hash": "sha256:tampered"}
    result = _run(executor, permit.permit_id, hashes=bad_hashes)
    assert result.status == "blocked"
    assert result.command_sent is False
    assert executor.hardware_actions_executed == 0


def test_command_ack_required() -> None:
    profile, transport, pm, permit, arming, executor, _ = _stack()
    # Force the mock transport into estop so write_position returns False.
    transport.emergency_stop()
    transport._estopped = True
    result = _run(executor, permit.permit_id)
    assert result.command_sent is True
    assert result.command_acknowledged is False
    assert result.status == "fault"


def test_feedback_required() -> None:
    _, transport, pm, permit, arming, executor, _ = _stack()
    transport.fail_next_read()  # first read (step-delta check) fails
    result = _run(executor, permit.permit_id)
    assert result.status == "fault"
    assert result.error_code == "communication_lost"
    assert arming.machine.state == ExecutionState.COMMUNICATION_LOST
    assert pm.is_revoked(permit.permit_id)


def test_step_delta_within_limit_executes() -> None:
    _, _, _, permit, _, executor, _ = _stack(max_step_delta=30)
    result = _run(
        executor,
        permit.permit_id,
        values=[975.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
    )
    assert result.status == "completed"


def test_fault_never_auto_rearms() -> None:
    _, transport, pm, permit, arming, executor, _ = _stack()
    transport.fail_next_read()
    result = _run(executor, permit.permit_id)
    assert result.status == "fault"
    assert arming.machine.state == ExecutionState.COMMUNICATION_LOST
    # Any further execution attempt is refused: DISARMED required first.
    result2 = _run(executor, permit.permit_id)
    assert result2.status == "blocked"
    assert result2.error_code == "not_armed"
