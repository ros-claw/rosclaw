"""Body executor interface (plan §7.1 ``body/execution/interface.py``).

A ``BodyExecutor`` is the only component allowed to turn an
``ActionExecutionRequest`` into a physical command.  Implementations must:

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
    """Transport-level failure during command or feedback."""


class ExecutorSafetyError(RuntimeError):
    """Executor-side safety refusal (stale request, out-of-range, estop)."""


class BodyExecutor(Protocol):
    """Minimal single-step executor contract."""

    def execute_step(
        self,
        request: ActionExecutionRequest,
        *,
        settle_ms: float = 0.0,
        max_step_delta_raw: float | None = None,
    ) -> tuple[bool, RH56Feedback]:
        """Send one step command and return ``(acknowledged, feedback)``.

        Must raise :class:`ExecutorCommunicationError` on I/O failure and
        :class:`ExecutorSafetyError` when the request cannot be sent safely.
        When ``max_step_delta_raw`` is given, any per-actuator jump from the
        current position larger than the limit must be refused before sending.
        """

    def read_feedback(self) -> RH56Feedback: ...

    def emergency_stop(self) -> bool: ...

    def is_connected(self) -> bool: ...


def request_is_stale(request: ActionExecutionRequest) -> bool:
    """True when the request's validity deadline has passed."""
    return time.monotonic_ns() > request.valid_until_monotonic_ns