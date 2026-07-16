"""Execution state machine for P5 hardware execution (plan §7.2).

Normal path::

    DISARMED → PREFLIGHT → SHADOW_VALIDATED → ARMED → EXECUTING_STEP
      → VERIFYING_FEEDBACK → HOLD → COMPLETED

Fault states::

    BLOCKED, FAULT, COMMUNICATION_LOST, ESTOP, OPERATOR_ABORT

Every fault returns to ``DISARMED``; automatic re-arming is forbidden.
"""

from __future__ import annotations

from enum import Enum


class ExecutionState(str, Enum):
    DISARMED = "DISARMED"
    PREFLIGHT = "PREFLIGHT"
    SHADOW_VALIDATED = "SHADOW_VALIDATED"
    ARMED = "ARMED"
    EXECUTING_STEP = "EXECUTING_STEP"
    VERIFYING_FEEDBACK = "VERIFYING_FEEDBACK"
    HOLD = "HOLD"
    COMPLETED = "COMPLETED"
    # Fault states
    BLOCKED = "BLOCKED"
    FAULT = "FAULT"
    COMMUNICATION_LOST = "COMMUNICATION_LOST"
    ESTOP = "ESTOP"
    OPERATOR_ABORT = "OPERATOR_ABORT"


FAULT_STATES = frozenset(
    {
        ExecutionState.BLOCKED,
        ExecutionState.FAULT,
        ExecutionState.COMMUNICATION_LOST,
        ExecutionState.ESTOP,
        ExecutionState.OPERATOR_ABORT,
    }
)

# Allowed transitions.  Any fault state may only go back to DISARMED.
_TRANSITIONS: dict[ExecutionState, set[ExecutionState]] = {
    ExecutionState.DISARMED: {ExecutionState.PREFLIGHT},
    ExecutionState.PREFLIGHT: {
        ExecutionState.SHADOW_VALIDATED,
        ExecutionState.BLOCKED,
        ExecutionState.DISARMED,
    },
    ExecutionState.SHADOW_VALIDATED: {ExecutionState.ARMED, ExecutionState.DISARMED},
    ExecutionState.ARMED: {
        ExecutionState.EXECUTING_STEP,
        ExecutionState.DISARMED,
        ExecutionState.ESTOP,
        ExecutionState.OPERATOR_ABORT,
    },
    ExecutionState.EXECUTING_STEP: {
        ExecutionState.VERIFYING_FEEDBACK,
        ExecutionState.FAULT,
        ExecutionState.COMMUNICATION_LOST,
        ExecutionState.ESTOP,
        ExecutionState.OPERATOR_ABORT,
    },
    ExecutionState.VERIFYING_FEEDBACK: {
        ExecutionState.HOLD,
        ExecutionState.ARMED,
        ExecutionState.COMPLETED,
        ExecutionState.FAULT,
        ExecutionState.COMMUNICATION_LOST,
        ExecutionState.ESTOP,
        ExecutionState.OPERATOR_ABORT,
    },
    ExecutionState.HOLD: {
        ExecutionState.ARMED,
        ExecutionState.COMPLETED,
        ExecutionState.DISARMED,
        ExecutionState.ESTOP,
        ExecutionState.OPERATOR_ABORT,
    },
    ExecutionState.COMPLETED: {ExecutionState.DISARMED},
    ExecutionState.BLOCKED: {ExecutionState.DISARMED},
    ExecutionState.FAULT: {ExecutionState.DISARMED},
    ExecutionState.COMMUNICATION_LOST: {ExecutionState.DISARMED},
    ExecutionState.ESTOP: {ExecutionState.DISARMED},
    ExecutionState.OPERATOR_ABORT: {ExecutionState.DISARMED},
}


class IllegalTransitionError(RuntimeError):
    """Raised when an execution state transition is not allowed."""


class ExecutionStateMachine:
    """Minimal explicit state machine; never auto-recovers to ARMED."""

    def __init__(self) -> None:
        self.state = ExecutionState.DISARMED
        self.history: list[tuple[ExecutionState, str]] = [(ExecutionState.DISARMED, "init")]

    def transition(self, target: ExecutionState, reason: str = "") -> ExecutionState:
        allowed = _TRANSITIONS.get(self.state, set())
        if target not in allowed:
            raise IllegalTransitionError(
                f"illegal_state_transition: {self.state.value} -> {target.value} ({reason})"
            )
        self.state = target
        self.history.append((target, reason))
        return target

    def fault(self, target: ExecutionState, reason: str) -> ExecutionState:
        """Enter a fault state; always legal from any non-fault state."""
        if target not in FAULT_STATES:
            raise IllegalTransitionError(f"{target.value} is not a fault state")
        self.state = target
        self.history.append((target, reason))
        return target

    def disarm(self, reason: str = "disarm") -> ExecutionState:
        self.state = ExecutionState.DISARMED
        self.history.append((ExecutionState.DISARMED, reason))
        return self.state

    @property
    def is_fault(self) -> bool:
        return self.state in FAULT_STATES

    @property
    def can_execute(self) -> bool:
        return self.state == ExecutionState.ARMED
