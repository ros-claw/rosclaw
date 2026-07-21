"""Single-step receding-horizon executor (plan §7.1, §8.1).

Pipeline for every step::

    mapped action candidate
      → verify permit (hashes, representation, unit, expiry)
      → build request with freshness deadline (max_action_age_ms)
      → executor boundary checks (fresh, dim, range, step delta)
      → send exactly one command
      → read feedback
      → verify returned feedback (position/force/temp/status)
      → Practice events

Open-loop chunks, background motion and multi-step dispatch are forbidden by
construction: one call = one step = one command.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from rosclaw.body.execution.interface import (
    BodyExecutor,
    ExecutorCommunicationError,
    ExecutorSafetyError,
)
from rosclaw.body.rh56.transport import CommandDelivery
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
from rosclaw.kernel import ExecutionMode

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
        execution_mode: ExecutionMode,
        watchdog: ExecutionWatchdog | None = None,
        max_action_age_ms: float = DEFAULT_MAX_ACTION_AGE_MS,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.executor = executor
        self.profile = profile
        self.permit_manager = permit_manager
        self.arming = arming
        self.verifier = verifier
        self.execution_mode = ExecutionMode(execution_mode)
        self.watchdog = watchdog or ExecutionWatchdog()
        self.max_action_age_ms = max_action_age_ms
        self._event_sink = event_sink or (lambda etype, payload: None)
        self.commands_executed = 0
        self.fixture_actions_executed = 0
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
        transport = getattr(self.executor, "transport", None)
        transport_mode = str(getattr(transport, "execution_mode", "UNKNOWN")).upper()
        # Mode must match the transport exactly: FIXTURE drives only synthetic
        # transports; REAL drives only a verified real transport registered
        # behind the Runtime ActionGateway.  Every other pairing fails closed
        # so a fixture can never act as hardware (or vice versa).
        mode_ok = (
            self.execution_mode is ExecutionMode.FIXTURE and transport_mode == "FIXTURE"
        ) or (self.execution_mode is ExecutionMode.REAL and transport_mode == "REAL")
        if not mode_ok:
            return self._result(
                status="blocked",
                error_code="RUNTIME_ACTION_GATEWAY_REQUIRED",
                message=(
                    "RH56 physical execution requires a verified REAL executor registered "
                    "with Runtime.submit_action(); no command was dispatched."
                ),
            )

        # 1. State machine: must be ARMED.
        if not self.arming.machine.can_execute:
            return self._result(
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
                execution_mode=self.execution_mode.value,
            )
        except PermitError as exc:
            self._emit("execution.step.blocked", {"reason": str(exc), "proposal_id": proposal_id})
            return self._result(
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
            return self._result(
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
            valid_until_monotonic_ns=time.monotonic_ns() + int(self.max_action_age_ms * 1e6),
        )
        self._emit(
            "execution.command.requested",
            {"proposal_id": proposal_id, "request": request.to_dict()},
        )

        # 6. Executor boundary: freshness/dim/range/step-delta + one command.
        # Setpoint hold band (exp3 real-hardware finding): the firmware coasts
        # one servo cycle when a setpoint CHANGES to ≈ the current position
        # (zero error → ~15-17 raw dip on gravity-loaded joints).  Rewriting
        # an UNCHANGED setpoint does not re-plan.  The band therefore covers
        # only near-zero deltas (5 raw — the dip was measured at |Δ|<=2);
        # anything larger is real motion and must re-plan.  This is a device
        # constant, deliberately NOT the position tolerance: using the
        # tolerance as the band starves small deliberate steps.
        hold_tolerance: list[float] | None = None
        calibration = getattr(self.verifier, "calibration", None)
        if calibration is not None:
            hold_tolerance = [5.0] * len(self.profile.action_order)
        self.arming.machine.transition(ExecutionState.EXECUTING_STEP, proposal_id)
        try:
            acknowledged, feedback = self.executor.execute_step(
                request,
                settle_ms=settle_ms,
                max_step_delta_raw=permit.max_step_delta_raw,
                hold_tolerance_raw=hold_tolerance,
            )
        except ExecutorSafetyError as exc:
            self.arming.machine.transition(ExecutionState.VERIFYING_FEEDBACK, "safety refusal")
            self.arming.machine.transition(ExecutionState.ARMED, "resume armed")
            self._emit("execution.step.blocked", {"reason": str(exc), "proposal_id": proposal_id})
            return self._result(
                status="blocked",
                error_code=str(exc).split(":", 1)[0],
                message=str(exc),
            )
        except ExecutorCommunicationError as exc:
            return self._communication_lost(exc, proposal_id)

        self.commands_executed += 1
        if self.execution_mode is ExecutionMode.FIXTURE:
            self.fixture_actions_executed += 1
        else:
            self.hardware_actions_executed += 1
        self._emit(
            "execution.command.sent",
            {
                "proposal_id": proposal_id,
                "acknowledged": acknowledged,
                "delivery": (delivery := self._command_delivery(acknowledged)),
            },
        )
        if acknowledged:
            delivery_event = "execution.command.protocol_acknowledged"
        elif delivery == "DELIVERY_INFERRED":
            delivery_event = "execution.command.delivery_inferred"
        else:
            delivery_event = "execution.command.not_acknowledged"
        self._emit(delivery_event, {"proposal_id": proposal_id, "delivery": delivery})

        # 7. Verify transport feedback. In the currently allowed path this is
        # synthetic fixture feedback, as recorded on the result envelope.
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
        delivery_supported = acknowledged or delivery == "DELIVERY_INFERRED"
        ok = self.verifier.is_step_ok(verification) and delivery_supported
        result = self._result(
            status="completed" if ok else "fault",
            command_sent=True,
            command_acknowledged=acknowledged,
            command_delivery=delivery,
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

    def _command_delivery(self, acknowledged: bool) -> str:
        if acknowledged:
            return "PROTOCOL_ACKNOWLEDGED"
        transport = getattr(self.executor, "transport", None)
        raw = getattr(transport, "last_command_delivery", CommandDelivery.UNCERTAIN)
        if raw == CommandDelivery.DELIVERY_INFERRED:
            return "DELIVERY_INFERRED"
        if raw == CommandDelivery.REJECTED:
            return "REJECTED"
        return "UNCERTAIN"

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
            "execution.communication_lost"
            if state == ExecutionState.COMMUNICATION_LOST
            else "execution.estop",
            {"proposal_id": proposal_id, "reason": str(trip)},
        )
        return self._result(
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
        return self._result(
            status="fault",
            error_code="communication_lost",
            message=str(exc),
        )

    def emergency_stop(self) -> bool:
        acknowledged = self.executor.emergency_stop()
        self.arming.fault(ExecutionState.ESTOP, "software emergency stop")
        self._emit(
            "execution.estop",
            {
                "request_dispatched": True,
                "driver_acknowledged": acknowledged,
                "physical_stop_observed": False,
                "stopped": False,
                "execution_mode": self.execution_mode.value,
                "trust_level": "SYNTHETIC",
            },
        )
        return acknowledged

    def _result(self, **values: Any) -> ActionExecutionResult:
        fixture = self.execution_mode is ExecutionMode.FIXTURE
        return ActionExecutionResult(
            execution_mode=self.execution_mode.value,
            evidence_level="SYNTHETIC" if fixture else "REQUESTED",
            verified=False,
            trust_level="SYNTHETIC" if fixture else "UNAVAILABLE",
            usable_for_real_execution=False,
            **values,
        )
