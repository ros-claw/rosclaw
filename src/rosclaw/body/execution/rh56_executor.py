"""RH56 single-step executor over an ``RH56Transport`` (mock or serial).

Enforces the P5 execution safety rules at the executor boundary (plan §8):

- request freshness (``valid_until_monotonic_ns``); stale → not sent
- action dimension == transport actuator count
- values clamped into the transport profile position range
- per-step delta from the current position bounded by the caller-supplied
  ``max_step_delta_raw`` (from the permit)
- exactly one command per call; then feedback is read back
- any transport I/O failure raises :class:`ExecutorCommunicationError`
"""

from __future__ import annotations

import time

from rosclaw.body.execution.interface import (
    ExecutorCommunicationError,
    ExecutorSafetyError,
    request_is_stale,
)
from rosclaw.body.rh56.transport import RH56Feedback, RH56Transport, TransportIOError
from rosclaw.body.rh56.transport_profile import TransportProfile
from rosclaw.integrations.lerobot.execution.schema import ActionExecutionRequest


class RH56Executor:
    """Single-step executor for the RH56 RS485/CAN hand."""

    def __init__(
        self,
        transport: RH56Transport,
        profile: TransportProfile,
    ):
        self.transport = transport
        self.profile = profile

    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        return self.transport.is_connected()

    def read_feedback(self) -> RH56Feedback:
        try:
            return self.transport.read_state()
        except TransportIOError as exc:
            raise ExecutorCommunicationError(f"feedback_read_failed: {exc}") from exc

    def execute_step(
        self,
        request: ActionExecutionRequest,
        *,
        settle_ms: float = 0.0,
        max_step_delta_raw: float | None = None,
    ) -> tuple[bool, RH56Feedback]:
        if request_is_stale(request):
            raise ExecutorSafetyError("stale_action: request validity window expired")

        if len(request.values) != self.profile.command.actuator_count:
            raise ExecutorSafetyError(
                f"actuator_count_mismatch: {len(request.values)} values for "
                f"{self.profile.command.actuator_count} actuators"
            )

        # Read current position for the step-delta check.
        current = self.read_feedback()

        targets: list[int] = []
        for i, value in enumerate(request.values):
            target = self.profile.clamp_position(value)
            if max_step_delta_raw is not None and i < len(current.position):
                delta = target - int(current.position[i])
                if abs(delta) > max_step_delta_raw:
                    raise ExecutorSafetyError(
                        f"step_delta_exceeded: actuator {i} delta {abs(delta):.1f} > "
                        f"{max_step_delta_raw}"
                    )
            targets.append(target)

        try:
            acknowledged = self.transport.write_position(
                targets,
                speed=int(request.speed),
                force_limit=int(request.force_limit_g),
            )
        except TransportIOError as exc:
            raise ExecutorCommunicationError(f"command_send_failed: {exc}") from exc

        if settle_ms > 0:
            time.sleep(settle_ms / 1000.0)

        feedback = self.read_feedback()
        return acknowledged, feedback

    def emergency_stop(self) -> bool:
        try:
            return self.transport.emergency_stop()
        except TransportIOError:
            return False
