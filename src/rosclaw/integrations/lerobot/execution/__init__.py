"""P5 hardware execution kernel (single-step, permit-gated, feedback-verified)."""

from rosclaw.integrations.lerobot.execution.arming import ArmingController, ArmingError
from rosclaw.integrations.lerobot.execution.executor import SingleStepExecutor
from rosclaw.integrations.lerobot.execution.feedback_verifier import FeedbackVerifier
from rosclaw.integrations.lerobot.execution.permit import PermitError, PermitManager
from rosclaw.integrations.lerobot.execution.report import ExecutionReport
from rosclaw.integrations.lerobot.execution.schema import (
    ActionExecutionRequest,
    ActionExecutionResult,
    ExecutionPermit,
    FeedbackVerification,
)
from rosclaw.integrations.lerobot.execution.state import (
    ExecutionState,
    ExecutionStateMachine,
    IllegalTransitionError,
)
from rosclaw.integrations.lerobot.execution.watchdog import (
    ExecutionWatchdog,
    WatchdogTrip,
)

__all__ = [
    "ActionExecutionRequest",
    "ActionExecutionResult",
    "ArmingController",
    "ArmingError",
    "ExecutionPermit",
    "ExecutionReport",
    "ExecutionState",
    "ExecutionStateMachine",
    "ExecutionWatchdog",
    "FeedbackVerification",
    "FeedbackVerifier",
    "IllegalTransitionError",
    "PermitError",
    "PermitManager",
    "SingleStepExecutor",
    "WatchdogTrip",
]
