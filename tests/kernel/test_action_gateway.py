"""Contract tests for the truthful action gateway."""

from __future__ import annotations

from rosclaw.core.event_bus import EventBus
from rosclaw.kernel import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionGateway,
    ActionState,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)


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
