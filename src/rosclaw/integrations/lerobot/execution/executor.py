"""Single-step receding-horizon executor (plan §7.1, §8.1).

Pipeline for every step::

    mapped action candidate
      → verify permit (hashes, representation, unit, expiry)
      → build request with freshness deadline (max_action_age_ms)
      → executor boundary checks (fresh, dim, range, step delta)
      → send exactly one command
      → read feedback
      → verify physical result (position/force/temp/status)
      → Practice events

Open-loop chunks, background motion and multi-step dispatch are forbidden by
construction: one call = one step = one command.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from rosclaw.body.execution.interface import (
    BodyExecutor,
    ExecutorCommunicationError,
    ExecutorSafetyError,
)
from rosclaw.body.rh56.transport_profile import TransportProfile
from rosclaw.integrations.lerobot.execution.arming import ArmingController
from rosclaw.integrations.lerobot.execution.feedback_verifier import FeedbackVerifier
from rosclaw.integrations.lerobot.execution.permit import PermitError, PermitManager
from rosclaw.integrations.lerobot.execution.schema import (
    ActionExecutionRequest,
    ActionExecutionResult,
)
from rosclaw.integrations.lerobot.execution.state import ExecutionState
from rosclaw.integrations.lerobot.execution.watchdog import (
    ExecutionWatchdog,
    WatchdogTrip,
)

# Plan §8.2: observation→command freshness budget.
DEFAULT_MAX_ACTION_AGE_MS = 300.0


class SingleStepExecutor:
    """Execute one action candidate under permit + watchdog supervision."""

    def __init__(
        self,
        *,
        executor: BodyExecutor,
        profile: TransportProfile,
        permit_manager: PermitManager,
        arming: ArmingController,
        verifier: FeedbackVerifier,
        watchdog: ExecutionWatchdog | None = None,
        max_action_age_ms: float = DEFAULT_MAX_ACTION_AGE_MS,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.executor = executor
        self.profile = profile
        self.permit_manager = permit_manager
        self.arming = arming
        self.verifier = verifier
        self.watchdog = watchdog or ExecutionWatchdog()
        self.max_action_age_ms = max_action_age_ms
        self._event_sink = event_sink or (lambda etype, payload: None)
        self.hardware_actions_executed = 0

    # ------------------------------------------------------------------

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self._event_sink(event_type, payload)

    def execute_candidate(
        self,
        *,
        permit_id: str,
        proposal_id: str,
        names: list[str],
        values: list[float],
        representation: str,
        units: str,
        hashes: dict[str, str],
        speed: int,
        force_limit_g: float,
        observation_timestamp_ns: int,
        settle_ms: float = 0.0,
    ) -> ActionExecutionResult:
        """Execute one mapped candidate.  Never sends more than one command."""
        # 1. State machine: must be ARMED.
        if not self.arming.machine.can_execute:
            return ActionExecutionResult(
                status="blocked",
                error_code="not_armed",
                message=f"state {self.arming.machine.state.value} cannot execute",
            )

        # 2. Watchdog: transport + abort before doing anything.
        try:
            self.watchdog.check_abort()
            self.watchdog.check_transport(self.executor_transport())
        except WatchdogTrip as trip:
            return self._fault_result(trip, proposal_id)

        # 3. Permit validation (hashes, representation, unit, expiry).
        try:
            permit = self.permit_manager.validate(
                permit_id,
                body_id=self.arming_permit_body(hashes),
                policy_contract_hash=hashes["policy_contract_hash"],
                body_hash=hashes["body_hash"],
                calibration_hash=hashes["calibration_hash"],
                mapping_hash=hashes["mapping_hash"],
                transport_profile_hash=hashes["transport_profile_hash"],
                representation=representation,
                units=units,
            )
        except PermitError as exc:
            self._emit("execution.step.blocked", {"reason": str(exc), "proposal_id": proposal_id})
            return ActionExecutionResult(
                status="blocked",
                error_code=str(exc).split(":", 1)[0],
                message=str(exc),
            )

        # 4. Freshness: observation → command must fit the age budget.
        age_ms = (time.monotonic_ns() - observation_timestamp_ns) / 1e6
        if age_ms > self.max_action_age_ms:
            self._emit(
                "execution.step.blocked",
                {"reason": "stale_action", "age_ms": age_ms, "proposal_id": proposal_id},
            )
            return ActionExecutionResult(
                status="stale_action",
                error_code="stale_action",
                message=f"observation age {age_ms:.1f} ms > {self.max_action_age_ms} ms",
            )

        # 5. Build the request with an absolute freshness deadline.
        request = ActionExecutionRequest(
            proposal_id=proposal_id,
            candidate_id=f"cand_{uuid.uuid4().hex[:10]}",
            permit_id=permit_id,
            body_id=permit.body_id,
            representation=representation,
            units=units,
            names=list(names),
            values=[float(v) for v in values],
            speed=min(int(speed), permit.max_speed),
            force_limit_g=min(float(force_limit_g), permit.max_force_g),
            valid_until_monotonic_ns=time.monotonic_ns()
            + int(self.max_action_age_ms * 1e6),
        )
        self._emit(
            "execution.command.requested",
            {"proposal_id": proposal_id, "request": request.to_dict()},
        )

        # 6. Executor boundary: freshness/dim/range/step-delta + one command.
        self.arming.machine.transition(ExecutionState.EXECUTING_STEP, proposal_id)
        try:
            acknowledged, feedback = self.executor.execute_step(
                request,
                settle_ms=settle_ms,
                max_step_delta_raw=permit.max_step_delta_raw,
            )
        except ExecutorSafetyError as exc:
            self.arming.machine.transition(ExecutionState.VERIFYING_FEEDBACK, "safety refusal")
            self.arming.machine.transition(ExecutionState.ARMED, "resume armed")
            self._emit("execution.step.blocked", {"reason": str(exc), "proposal_id": proposal_id})
            return ActionExecutionResult(
                status="blocked",
                error_code=str(exc).split(":", 1)[0],
                message=str(exc),
            )
        except ExecutorCommunicationError as exc:
            return self._communication_lost(exc, proposal_id)

        self.hardware_actions_executed += 1
        self._emit(
            "execution.command.sent",
            {"proposal_id": proposal_id, "acknowledged": acknowledged},
        )
        self._emit(
            "execution.command.acknowledged" if acknowledged else "execution.command.not_acknowledged",
            {"proposal_id": proposal_id},
        )

        # 7. Verify physical feedback.
        self.arming.machine.transition(ExecutionState.VERIFYING_FEEDBACK, "feedback")
        verification = self.verifier.verify(
            target=request.values,
            feedback=feedback,
            force_limit_g=request.force_limit_g,
        )
        self._emit(
            "execution.feedback.verified",
            {
                "proposal_id": proposal_id,
                "verification": verification.to_dict(),
                "feedback": {
                    "position": feedback.position,
                    "force_g": feedback.force_g,
                    "current_ma": feedback.current_ma,
                    "status_bits": feedback.status_bits,
                    "temperature_c": feedback.temperature_c,
                },
            },
        )

        position_error = [
            abs(float(feedback.position[i]) - float(request.values[i]))
            if i < len(feedback.position)
            else float("nan")
            for i in range(len(request.values))
        ]
        ok = self.verifier.is_step_ok(verification) and acknowledged
        result = ActionExecutionResult(
            status="completed" if ok else "fault",
            command_sent=True,
            command_acknowledged=acknowledged,
            target=list(request.values),
            actual=[float(p) for p in feedback.position],
            position_error=position_error,
            force=list(feedback.force_g),
            current=list(feedback.current_ma),
            temperature=list(feedback.temperature_c),
            status_bits=list(feedback.status_bits),
            verification=verification,
            message="; ".join(verification.details),
        )
        if ok:
            self.arming.machine.transition(ExecutionState.HOLD, "step ok")
            self.arming.machine.transition(ExecutionState.ARMED, "resume armed")
            self._emit(
                "execution.step.completed",
                {"proposal_id": proposal_id, "result": result.to_dict()},
            )
        else:
            self.arming.fault(
                ExecutionState.FAULT,
                f"feedback verification failed: {verification.details}",
            )
            self.permit_manager.revoke(permit_id, "feedback_verification_failed")
            self._emit(
                "execution.step.blocked",
                {"proposal_id": proposal_id, "verification": verification.to_dict()},
            )
        return result

    # ------------------------------------------------------------------

    def executor_transport(self):
        transport = getattr(self.executor, "transport", None)
        if transport is None:
            raise WatchdogTrip("executor_invalid", "executor exposes no transport")
        return transport

    def arming_permit_body(self, hashes: dict[str, str]) -> str:
        permit = self.permit_manager.get(self.arming.armed_permit_id or "")
        return permit.body_id if permit is not None else hashes.get("body_id", "")

    def _fault_result(self, trip: WatchdogTrip, proposal_id: str) -> ActionExecutionResult:
        state = (
            ExecutionState.OPERATOR_ABORT
            if trip.code == "operator_abort"
            else ExecutionState.COMMUNICATION_LOST
        )
        # Capture the armed permit BEFORE faulting (fault() clears the reference).
        permit_id = self.arming.armed_permit_id
        self.arming.fault(state, str(trip))
        if permit_id:
            self.permit_manager.revoke(permit_id, trip.code)
        self._emit(
            "execution.communication_lost" if state == ExecutionState.COMMUNICATION_LOST else "execution.estop",
            {"proposal_id": proposal_id, "reason": str(trip)},
        )
        return ActionExecutionResult(
            status="aborted" if state == ExecutionState.OPERATOR_ABORT else "fault",
            error_code=trip.code,
            message=str(trip),
        )

    def _communication_lost(
        self, exc: ExecutorCommunicationError, proposal_id: str
    ) -> ActionExecutionResult:
        permit_id = self.arming.armed_permit_id
        self.arming.fault(ExecutionState.COMMUNICATION_LOST, str(exc))
        if permit_id:
            self.permit_manager.revoke(permit_id, "communication_lost")
        self._emit(
            "execution.communication_lost",
            {"proposal_id": proposal_id, "reason": str(exc)},
        )
        # Best-effort emergency stop.
        self.executor.emergency_stop()
        return ActionExecutionResult(
            status="fault",
            error_code="communication_lost",
            message=str(exc),
        )

    def emergency_stop(self) -> bool:
        stopped = self.executor.emergency_stop()
        self.arming.fault(ExecutionState.ESTOP, "software emergency stop")
        self._emit("execution.estop", {"stopped": stopped})
        return stopped
