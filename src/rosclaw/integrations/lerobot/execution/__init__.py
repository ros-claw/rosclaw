"""Permit-gated RH56 fixture kernel with lazy public exports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosclaw.integrations.lerobot.execution.arming import (
        ArmingController as ArmingController,
    )
    from rosclaw.integrations.lerobot.execution.arming import (
        ArmingError as ArmingError,
    )
    from rosclaw.integrations.lerobot.execution.executor import (
        SingleStepExecutor as SingleStepExecutor,
    )
    from rosclaw.integrations.lerobot.execution.feedback_verifier import (
        FeedbackVerifier as FeedbackVerifier,
    )
    from rosclaw.integrations.lerobot.execution.permit import (
        PermitError as PermitError,
    )
    from rosclaw.integrations.lerobot.execution.permit import (
        PermitManager as PermitManager,
    )
    from rosclaw.integrations.lerobot.execution.report import (
        ExecutionReport as ExecutionReport,
    )
    from rosclaw.integrations.lerobot.execution.schema import (
        ActionExecutionRequest as ActionExecutionRequest,
    )
    from rosclaw.integrations.lerobot.execution.schema import (
        ActionExecutionResult as ActionExecutionResult,
    )
    from rosclaw.integrations.lerobot.execution.schema import (
        ExecutionPermit as ExecutionPermit,
    )
    from rosclaw.integrations.lerobot.execution.schema import (
        FeedbackVerification as FeedbackVerification,
    )
    from rosclaw.integrations.lerobot.execution.state import (
        ExecutionState as ExecutionState,
    )
    from rosclaw.integrations.lerobot.execution.state import (
        ExecutionStateMachine as ExecutionStateMachine,
    )
    from rosclaw.integrations.lerobot.execution.state import (
        IllegalTransitionError as IllegalTransitionError,
    )
    from rosclaw.integrations.lerobot.execution.watchdog import (
        ExecutionWatchdog as ExecutionWatchdog,
    )
    from rosclaw.integrations.lerobot.execution.watchdog import (
        WatchdogTrip as WatchdogTrip,
    )

_EXPORTS = {
    "ActionExecutionRequest": "schema",
    "ActionExecutionResult": "schema",
    "ArmingController": "arming",
    "ArmingError": "arming",
    "ExecutionPermit": "schema",
    "ExecutionReport": "report",
    "ExecutionState": "state",
    "ExecutionStateMachine": "state",
    "ExecutionWatchdog": "watchdog",
    "FeedbackVerification": "schema",
    "FeedbackVerifier": "feedback_verifier",
    "IllegalTransitionError": "state",
    "PermitError": "permit",
    "PermitManager": "permit",
    "SingleStepExecutor": "executor",
    "WatchdogTrip": "watchdog",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
