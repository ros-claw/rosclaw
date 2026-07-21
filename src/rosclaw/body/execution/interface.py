"""Low-level body executor interface (plan §7.1 ``body/execution/interface.py``).

A ``BodyExecutor`` can turn an ``ActionExecutionRequest`` into a transport
command. REAL implementations must only be invoked by a registered Runtime
ActionGateway executor; this interface is not authorization. Implementations
must:

- send **exactly one** single-step position command per call
- return the physical feedback measured after the command
- raise :class:`ExecutorCommunicationError` on any transport failure
- provide :meth:`emergency_stop`
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

from rosclaw.body.rh56.transport import RH56Feedback

if TYPE_CHECKING:  # avoid a body → integrations runtime dependency
    from rosclaw.integrations.lerobot.execution.schema import (
        ActionExecutionRequest,
    )


class ExecutorCommunicationError(RuntimeError):
    """Transport-level failure during command or feedback.

    ``command_sent`` records whether the physical command had already been
    dispatched when the failure occurred, and ``driver_acknowledged``
    whether the driver confirmed it.  A fault *after* dispatch (e.g. the
    post-write feedback read fails) is evidence-wise different from a fault
    before it: the trace and the receipt must be able to distinguish
    "command sent, observation lost" from "never sent" (TRACE-03).
    """

    def __init__(
        self,
        message: str,
        *,
        command_sent: bool = False,
        driver_acknowledged: bool = False,
    ) -> None:
        super().__init__(message)
        self.command_sent = command_sent
        self.driver_acknowledged = driver_acknowledged


class ExecutorSafetyError(RuntimeError):
    """Executor-side safety refusal (stale request, out-of-range, estop)."""


class BodyExecutor(Protocol):
    """Minimal single-step transport contract; not a public execution facade."""

    def execute_step(
        self,
        request: ActionExecutionRequest,
        *,
        settle_ms: float = 0.0,
        max_step_delta_raw: float | None = None,
        hold_tolerance_raw: list[float] | None = None,
    ) -> tuple[bool, RH56Feedback]:
        """Send one transport step and return ``(acknowledged, feedback)``.

        Must raise :class:`ExecutorCommunicationError` on I/O failure and
        :class:`ExecutorSafetyError` when the request cannot be sent safely.
        When ``max_step_delta_raw`` is given, any per-actuator jump from the
        current position larger than the limit must be refused before sending.
        When ``hold_tolerance_raw`` is given, implementations may retain the
        active setpoint for an actuator already within its hold band.
        """

    def read_feedback(self) -> RH56Feedback: ...

    def emergency_stop(self) -> bool: ...

    def is_connected(self) -> bool: ...


def request_is_stale(request: ActionExecutionRequest) -> bool:
    """True when the request's validity deadline has passed."""
    return time.monotonic_ns() > request.valid_until_monotonic_ns
