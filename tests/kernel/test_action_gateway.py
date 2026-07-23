"""Contract tests for the truthful action gateway."""

from __future__ import annotations

from rosclaw.core.event_bus import EventBus
from rosclaw.kernel import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionGateway,
    ActionState,
    AuthorizationContext,
    EvidenceDomain,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)
from rosclaw.observability.tracer import Tracer


def _action(*, action_id: str = "action-1") -> ActionEnvelope:
    return ActionEnvelope(
        action_id=action_id,
        actor_id="test-agent",
        agent_framework="pytest",
        session_id="session-1",
        body_id="sim_ur5e",
        body_snapshot_hash="sha256:test-body",
        capability_id="sandbox.reach",
        arguments={"target": [-0.24, 0.51, 0.47]},
        execution_mode=ExecutionMode.SIMULATION,
    )


def test_action_contract_round_trips_without_losing_mode() -> None:
    action = _action()

    restored = ActionEnvelope.from_dict(action.to_dict())

    assert restored == action
    assert restored.execution_mode is ExecutionMode.SIMULATION
    assert restored.to_dict()["execution_mode"] == "SIMULATION"


def test_gateway_fails_closed_when_no_executor_is_registered() -> None:
    event_bus = EventBus()
    receipt = ActionGateway(event_bus=event_bus).submit(_action())

    assert receipt.final_state is ActionState.FAILED
    assert receipt.evidence_level is EvidenceLevel.REQUESTED
    assert receipt.verified is False
    assert receipt.errors[0]["code"] == "EXECUTOR_UNAVAILABLE"
    events = event_bus.get_history("rosclaw.runtime.action.receipt")
    assert len(events) == 1
    assert events[0].payload["action_id"] == receipt.action_id
    assert events[0].payload["final_state"] == "FAILED"


def test_gateway_returns_idempotent_receipt_for_duplicate_action_id() -> None:
    calls: list[str] = []

    def execute(action: ActionEnvelope) -> ActionExecutionResult:
        calls.append(action.action_id)
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
            dispatch_result={"accepted": True},
            verification_result={"success": True},
        )

    gateway = ActionGateway()
    gateway.register_executor(
        "sandbox.reach",
        ExecutionMode.SIMULATION,
        execute,
    )

    first = gateway.submit(_action())
    second = gateway.submit(_action())

    assert first.to_dict() == second.to_dict()
    assert calls == ["action-1"]
    assert first.verified is True
    assert first.evidence_domain is EvidenceDomain.SIMULATION
    assert first.valid_for_promotion is False


def test_gateway_rejects_evidence_domain_escalation() -> None:
    def execute(_action: ActionEnvelope) -> ActionExecutionResult:
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
            evidence_domain=EvidenceDomain.HARDWARE,
        )

    gateway = ActionGateway()
    gateway.register_executor("sandbox.reach", ExecutionMode.SIMULATION, execute)

    receipt = gateway.submit(_action())

    assert receipt.final_state is ActionState.FAILED
    assert receipt.evidence_level is EvidenceLevel.REQUESTED
    assert receipt.evidence_domain is EvidenceDomain.SIMULATION
    assert any(error["code"] == "EVIDENCE_DOMAIN_MISMATCH" for error in receipt.errors)


def test_fixture_receipt_can_never_be_verified() -> None:
    def execute(_action: ActionEnvelope) -> ActionExecutionResult:
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
        )

    gateway = ActionGateway()
    gateway.register_executor("sandbox.reach", ExecutionMode.FIXTURE, execute)
    action = _action()
    action.execution_mode = ExecutionMode.FIXTURE

    receipt = gateway.submit(action)

    assert receipt.evidence_level is EvidenceLevel.SYNTHETIC
    assert receipt.verified is False
    assert receipt.usable_for_real_execution is False


def test_gateway_fails_closed_when_executor_evidence_is_too_weak() -> None:
    def execute(_action: ActionEnvelope) -> ActionExecutionResult:
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.DISPATCH_CONFIRMED,
            dispatch_result={"accepted": True},
        )

    gateway = ActionGateway()
    gateway.register_executor("sandbox.reach", ExecutionMode.SIMULATION, execute)

    receipt = gateway.submit(_action())

    assert receipt.final_state is ActionState.FAILED
    assert receipt.verified is False
    assert receipt.errors[0]["code"] == "VERIFICATION_REQUIREMENT_NOT_MET"


def test_gateway_can_degrade_when_verification_policy_does_not_fail_closed() -> None:
    def execute(_action: ActionEnvelope) -> ActionExecutionResult:
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.DRIVER_CONFIRMED,
        )

    gateway = ActionGateway()
    gateway.register_executor("sandbox.reach", ExecutionMode.SIMULATION, execute)
    action = _action()
    action.verification_policy = VerificationPolicy(
        required_evidence=EvidenceLevel.TASK_VERIFIED,
        fail_closed=False,
    )

    receipt = gateway.submit(action)

    assert receipt.final_state is ActionState.DEGRADED
    assert receipt.errors[0]["code"] == "VERIFICATION_REQUIREMENT_NOT_MET"


def test_real_action_requires_body_snapshot_before_executor_lookup() -> None:
    action = _action()
    action.execution_mode = ExecutionMode.REAL
    action.body_snapshot_hash = ""

    receipt = ActionGateway().submit(action)

    assert receipt.final_state is ActionState.BLOCKED
    assert receipt.errors[0]["code"] == "BODY_SNAPSHOT_REQUIRED"


def test_real_action_requires_explicit_authorization() -> None:
    action = _action()
    action.execution_mode = ExecutionMode.REAL

    receipt = ActionGateway().submit(action)

    assert receipt.final_state is ActionState.BLOCKED
    assert receipt.errors[0]["code"] == "AUTHORIZATION_REQUIRED"
    assert ActionState.AUTHORIZATION_REQUIRED in [item.state for item in receipt.transitions]


def test_authorized_real_action_still_fails_without_executor() -> None:
    action = _action()
    action.execution_mode = ExecutionMode.REAL
    action.authorization = AuthorizationContext(
        principal_id="operator-1",
        approved=True,
        approval_id="approval-1",
        scopes=["sandbox.reach"],
    )

    receipt = ActionGateway().submit(action)

    assert receipt.final_state is ActionState.FAILED
    assert receipt.errors[0]["code"] == "EXECUTOR_UNAVAILABLE"


def test_real_action_rejects_incomplete_authorization_evidence() -> None:
    action = _action()
    action.execution_mode = ExecutionMode.REAL
    action.authorization = AuthorizationContext(
        approved=True,
        scopes=["sandbox.reach"],
    )

    receipt = ActionGateway().submit(action)

    assert receipt.final_state is ActionState.BLOCKED
    assert receipt.errors[0]["code"] == "AUTHORIZATION_EVIDENCE_INVALID"


def test_real_action_rejects_mismatched_authorization_scope() -> None:
    action = _action()
    action.execution_mode = ExecutionMode.REAL
    action.authorization = AuthorizationContext(
        principal_id="operator-1",
        approved=True,
        approval_id="approval-1",
        scopes=["robot.observe"],
    )

    receipt = ActionGateway().submit(action)

    assert receipt.final_state is ActionState.BLOCKED
    assert receipt.errors[0]["code"] == "AUTHORIZATION_SCOPE_MISMATCH"


# ---------------------------------------------------------------------------
# P0-5: ROBOT_ACTION / ROBOT_STATE child spans under the SKILL span
# ---------------------------------------------------------------------------


class _RecordingExporter:
    def __init__(self) -> None:
        self.records: list = []

    def export(self, record) -> bool:
        self.records.append(record)
        return True


def _real_action(**overrides) -> ActionEnvelope:
    base = {
        "action_id": "real-action-1",
        "actor_id": "test-agent",
        "agent_framework": "pytest",
        "session_id": "session-1",
        "body_id": "rh56_right_01",
        "body_snapshot_hash": "sha256:body",
        "capability_id": "rh56.single_step",
        "arguments": {
            "permit_id": "permit-1",
            "names": ["little", "ring", "middle", "index", "thumb", "thumb_rot"],
            "values": [1000.0, 1000.0, 1000.0, 410.0, 420.0, 300.0],
            "representation": "joint_position",
            "units": "raw_device_unit",
            "hashes": {"body_hash": "sha256:body"},
            "speed": 500,
            "force_limit_g": 800.0,
            "observation_timestamp_ns": 123456789,
        },
        "execution_mode": ExecutionMode.REAL,
        "verification_policy": VerificationPolicy(
            required_evidence=EvidenceLevel.PHYSICALLY_OBSERVED,
        ),
        "authorization": AuthorizationContext(
            principal_id="operator-1",
            approved=True,
            approval_id="approval-1",
            scopes=["rh56.single_step"],
        ),
    }
    base.update(overrides)
    return ActionEnvelope(**base)


def _completed_real_result(action: ActionEnvelope) -> ActionExecutionResult:
    return ActionExecutionResult(
        final_state=ActionState.COMPLETED,
        evidence_level=EvidenceLevel.PHYSICALLY_OBSERVED,
        dispatch_result={"accepted": True, "command_sent": True},
        driver_ack={"acknowledged": True},
        observations=[
            {
                "position": [1001.0, 1000.0, 999.0, 409.0, 421.0, 300.0],
                "force_g": [0.0, 0.0, 0.0, 12.0, 45.0, 0.0],
                "current_ma": [120.0, 110.0, 105.0, 130.0, 140.0, 100.0],
                "temperature_c": [38.5] * 6,
                "status_bits": [0] * 6,
                "position_error": [1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            }
        ],
        verification_result={"success": True, "position_ok": True},
    )


def _finished_spans(exporter: _RecordingExporter) -> dict:
    return {r.name: r for r in exporter.records if r.status != "RUNNING"}


def test_real_execution_emits_robot_action_and_state_spans() -> None:
    """P0-5: REAL dispatch proves physical actuation + observation in the trace."""
    exporter = _RecordingExporter()
    gateway = ActionGateway(tracer=Tracer(exporters=[exporter]))
    gateway.register_executor("rh56.single_step", ExecutionMode.REAL, _completed_real_result)

    receipt = gateway.submit(_real_action())

    assert receipt.final_state is ActionState.COMPLETED
    assert receipt.evidence_level is EvidenceLevel.PHYSICALLY_OBSERVED

    spans = _finished_spans(exporter)
    skill = spans["kernel.submit_action"]
    robot_action = spans["kernel.robot_action"]
    robot_state = spans["kernel.robot_state"]

    # Causal tree: both child spans hang under the SKILL span, same trace.
    assert skill.span_kind == "SKILL"
    assert robot_action.parent_span_id == skill.span_id
    assert robot_state.parent_span_id == skill.span_id
    assert robot_action.trace_id == skill.trace_id == robot_state.trace_id

    # ROBOT_ACTION: physical actuation with command + ACK, bounded command
    # summary (hash + count, not the raw values payload).
    assert robot_action.span_kind == "ROBOT_ACTION"
    assert robot_action.attributes["physical_actuation"] is True
    assert robot_action.input["joint_count"] == 6
    assert "values_sha256" in robot_action.input
    assert "values" not in robot_action.input
    assert robot_action.output["dispatch_result"]["accepted"] is True
    assert robot_action.output["driver_ack"]["acknowledged"] is True

    # ROBOT_STATE: physical observation with positions/forces/temps/status.
    assert robot_state.span_kind == "ROBOT_STATE"
    assert robot_state.attributes["physical_observation"] is True
    observation = robot_state.output["observations"][0]
    for key in ("position", "force_g", "current_ma", "temperature_c", "status_bits"):
        assert key in observation, key


def test_trace_uses_gateway_sanitized_evidence_domain() -> None:
    exporter = _RecordingExporter()

    def execute(_action: ActionEnvelope) -> ActionExecutionResult:
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
            evidence_domain=EvidenceDomain.HARDWARE,
            dispatch_result={"accepted": True},
            observations=[{"position": [0.0]}],
        )

    gateway = ActionGateway(tracer=Tracer(exporters=[exporter]))
    gateway.register_executor("sandbox.reach", ExecutionMode.SIMULATION, execute)

    receipt = gateway.submit(_action(action_id="mismatched-domain-trace"))

    assert receipt.final_state is ActionState.FAILED
    spans = _finished_spans(exporter)
    robot_action = spans["kernel.robot_action"]
    robot_state = spans["kernel.robot_state"]
    assert robot_action.output["evidence_level"] == "REQUESTED"
    assert robot_action.output["evidence_domain"] == "SIMULATION"
    assert robot_action.output["executor_reported_evidence_domain"] == "HARDWARE"
    assert robot_state.output["evidence_level"] == "REQUESTED"
    assert robot_state.output["evidence_domain"] == "SIMULATION"
    assert robot_state.attributes["physical_observation"] is False


def test_fixture_execution_keeps_physical_flags_false() -> None:
    """P0-5: the FIXTURE path must never claim physical actuation/observation."""
    exporter = _RecordingExporter()
    gateway = ActionGateway(tracer=Tracer(exporters=[exporter]))
    gateway.register_executor("rh56.single_step", ExecutionMode.FIXTURE, _completed_real_result)

    receipt = gateway.submit(
        _real_action(
            action_id="fixture-action-1",
            execution_mode=ExecutionMode.FIXTURE,
            authorization=AuthorizationContext(),
        )
    )

    assert receipt.final_state is ActionState.DEGRADED
    assert receipt.evidence_level is EvidenceLevel.SYNTHETIC

    spans = _finished_spans(exporter)
    robot_action = spans["kernel.robot_action"]
    robot_state = spans["kernel.robot_state"]
    assert robot_action.attributes["physical_actuation"] is False
    assert robot_state.attributes["physical_observation"] is False


def test_blocked_before_dispatch_has_no_robot_state_span() -> None:
    """P0-5: a blocked action shows ROBOT_ACTION(BLOCKED, no ACK) and, with no
    observation taken, no ROBOT_STATE span at all."""

    def blocked_executor(action: ActionEnvelope) -> ActionExecutionResult:
        return ActionExecutionResult(
            final_state=ActionState.BLOCKED,
            evidence_level=EvidenceLevel.REQUESTED,
            dispatch_result={"accepted": False, "command_sent": False},
            errors=[{"code": "stale_action", "message": "observation too old"}],
        )

    exporter = _RecordingExporter()
    gateway = ActionGateway(tracer=Tracer(exporters=[exporter]))
    gateway.register_executor("rh56.single_step", ExecutionMode.REAL, blocked_executor)

    receipt = gateway.submit(_real_action(action_id="blocked-action-1"))

    assert receipt.final_state is ActionState.BLOCKED
    spans = _finished_spans(exporter)
    robot_action = spans["kernel.robot_action"]
    assert robot_action.attributes["physical_actuation"] is True
    assert robot_action.output["dispatch_result"]["accepted"] is False
    assert robot_action.output["driver_ack"] is None
    assert robot_action.status == "BLOCKED"
    assert "kernel.robot_state" not in spans
