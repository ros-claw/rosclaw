"""Single truthful facade for action submission."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.kernel.contracts import (
    AcknowledgementStage,
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    EvidenceLevel,
    ExecutionMode,
    ExecutionReceipt,
    StateTransition,
    utc_now,
)
from rosclaw.kernel.resource_manager import ResourceManager

ActionExecutor = Callable[[ActionEnvelope], ActionExecutionResult]


class ActionGateway:
    """Validate, serialize, execute, and receipt all submitted actions.

    The first implementation is intentionally in-process.  It establishes one
    contract and one fail-closed entrance before any daemon split or migration
    of legacy callers.
    """

    def __init__(
        self,
        *,
        event_bus: Any | None = None,
        resource_manager: ResourceManager | None = None,
        tracer: Any | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._resources = resource_manager or ResourceManager()
        self._executors: dict[tuple[str, ExecutionMode], ActionExecutor] = {}
        self._receipts: dict[str, ExecutionReceipt] = {}
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        if tracer is None:
            from rosclaw.observability.tracer import get_tracer

            tracer = get_tracer(event_bus)
        self._tracer = tracer

    def register_executor(
        self,
        capability_id: str,
        mode: ExecutionMode,
        executor: ActionExecutor,
    ) -> None:
        if not capability_id.strip():
            raise ValueError("capability_id is required")
        if not callable(executor):
            raise TypeError("executor must be callable")
        with self._lock:
            self._executors[(capability_id, ExecutionMode(mode))] = executor

    @property
    def registered_executors(self) -> tuple[str, ...]:
        """Return non-sensitive executor identifiers for status reporting."""

        with self._lock:
            return tuple(
                sorted(f"{capability_id}:{mode.value}" for capability_id, mode in self._executors)
            )

    def get_receipt(self, action_id: str) -> ExecutionReceipt | None:
        """Return an already-terminal receipt without exposing mutable storage."""

        with self._lock:
            return self._receipts.get(action_id)

    def discard_receipt(self, action_id: str) -> bool:
        """Discard one terminal in-memory receipt after external retention."""

        with self._lock:
            if action_id in self._inflight:
                return False
            return self._receipts.pop(action_id, None) is not None

    def submit(self, action: ActionEnvelope) -> ExecutionReceipt:
        """Submit one action and return an idempotent evidence receipt."""

        owner = False
        with self._lock:
            existing = self._receipts.get(action.action_id)
            if existing is not None:
                return existing
            waiter = self._inflight.get(action.action_id)
            if waiter is None:
                waiter = threading.Event()
                self._inflight[action.action_id] = waiter
                owner = True

        if not owner:
            timeout = self._remaining_timeout(action)
            if waiter.wait(timeout=timeout):
                with self._lock:
                    existing = self._receipts.get(action.action_id)
                if existing is not None:
                    return existing
                return self._failure_receipt(
                    action,
                    ActionState.FAILED,
                    "INFLIGHT_ACTION_LOST",
                    "The in-flight action ended without producing a receipt.",
                )
            return self._failure_receipt(
                action,
                ActionState.TIMED_OUT,
                "DUPLICATE_ACTION_TIMEOUT",
                "Timed out waiting for the in-flight action with the same action_id.",
            )

        try:
            try:
                receipt = self._execute(action)
            except Exception as exc:  # noqa: BLE001
                receipt = self._failure_receipt(
                    action,
                    ActionState.FAILED,
                    "GATEWAY_INTERNAL_ERROR",
                    str(exc),
                )
            with self._lock:
                self._receipts[action.action_id] = receipt
            self._publish_receipt(receipt)
            return receipt
        finally:
            with self._lock:
                completed = self._inflight.pop(action.action_id, None)
                if completed is not None:
                    completed.set()

    def reject(
        self,
        action: ActionEnvelope,
        *,
        code: str,
        message: str,
        state: ActionState = ActionState.BLOCKED,
    ) -> ExecutionReceipt:
        """Record a fail-closed rejection made before executor dispatch."""

        with self._lock:
            existing = self._receipts.get(action.action_id)
            if existing is not None:
                return existing
            receipt = self._failure_receipt(action, state, code, message)
            self._receipts[action.action_id] = receipt
        self._publish_receipt(receipt)
        return receipt

    def _execute(self, action: ActionEnvelope) -> ExecutionReceipt:
        started_at = utc_now()
        trace_id = action.parent_trace_id or f"trace_{uuid.uuid4().hex[:24]}"
        transitions = [
            StateTransition(ActionState.PROPOSED, reason="action_submitted"),
            StateTransition(ActionState.REQUEST_ACCEPTED, reason="request_accepted"),
        ]
        executor = self._executors.get((action.capability_id, action.execution_mode))

        with self._tracer.start_span(
            "kernel.submit_action",
            "SKILL",
            source="action_gateway",
            operation=action.capability_id,
            trace_id=trace_id,
            attributes={
                "action.id": action.action_id,
                "execution.mode": action.execution_mode.value,
                "risk.class": action.risk_class,
            },
            robot_id=action.body_id,
            session_id=action.session_id,
        ) as span:
            span.set_input(action.to_dict())
            if (
                action.execution_mode in {ExecutionMode.SHADOW, ExecutionMode.REAL}
                and not action.body_snapshot_hash
            ):
                receipt = self._failure_receipt(
                    action,
                    ActionState.BLOCKED,
                    "BODY_SNAPSHOT_REQUIRED",
                    "Shadow and real actions require an immutable body snapshot hash.",
                    trace_id=trace_id,
                    started_at=started_at,
                    transitions=transitions,
                )
                span.set_output(receipt.to_dict())
                span.set_status("BLOCKED", "body snapshot required")
                return receipt

            if action.execution_mode is ExecutionMode.REAL and not action.authorization.approved:
                transitions.append(
                    StateTransition(
                        ActionState.AUTHORIZATION_REQUIRED,
                        reason="real_execution_requires_approval",
                    )
                )
                receipt = self._failure_receipt(
                    action,
                    ActionState.BLOCKED,
                    "AUTHORIZATION_REQUIRED",
                    "Real execution requires an approved AuthorizationContext.",
                    trace_id=trace_id,
                    started_at=started_at,
                    transitions=transitions,
                )
                span.set_output(receipt.to_dict())
                span.set_status("BLOCKED", "authorization required")
                return receipt

            if action.execution_mode is ExecutionMode.REAL and (
                not action.authorization.principal_id.strip()
                or not str(action.authorization.approval_id or "").strip()
            ):
                transitions.append(
                    StateTransition(
                        ActionState.AUTHORIZATION_REQUIRED,
                        reason="authorization_evidence_incomplete",
                    )
                )
                receipt = self._failure_receipt(
                    action,
                    ActionState.BLOCKED,
                    "AUTHORIZATION_EVIDENCE_INVALID",
                    "Real execution requires a principal_id and approval_id.",
                    trace_id=trace_id,
                    started_at=started_at,
                    transitions=transitions,
                )
                span.set_output(receipt.to_dict())
                span.set_status("BLOCKED", "authorization evidence invalid")
                return receipt

            if (
                action.execution_mode is ExecutionMode.REAL
                and action.capability_id not in action.authorization.scopes
                and "*" not in action.authorization.scopes
            ):
                transitions.append(
                    StateTransition(
                        ActionState.AUTHORIZATION_REQUIRED,
                        reason="capability_scope_not_approved",
                    )
                )
                receipt = self._failure_receipt(
                    action,
                    ActionState.BLOCKED,
                    "AUTHORIZATION_SCOPE_MISMATCH",
                    (
                        f"Approval '{action.authorization.approval_id}' does not include "
                        f"capability scope '{action.capability_id}'."
                    ),
                    trace_id=trace_id,
                    started_at=started_at,
                    transitions=transitions,
                )
                span.set_output(receipt.to_dict())
                span.set_status("BLOCKED", "authorization scope mismatch")
                return receipt

            if executor is None:
                receipt = self._failure_receipt(
                    action,
                    ActionState.FAILED,
                    "EXECUTOR_UNAVAILABLE",
                    (
                        f"No executor registered for capability '{action.capability_id}' "
                        f"in mode {action.execution_mode.value}."
                    ),
                    trace_id=trace_id,
                    started_at=started_at,
                    transitions=transitions,
                )
                span.set_output(receipt.to_dict())
                span.set_status("ERROR", "executor unavailable")
                return receipt

            if action.deadline_at is not None and action.deadline_at <= datetime.now(UTC):
                receipt = self._failure_receipt(
                    action,
                    ActionState.TIMED_OUT,
                    "ACTION_DEADLINE_EXPIRED",
                    "Action deadline expired before execution.",
                    trace_id=trace_id,
                    started_at=started_at,
                    transitions=transitions,
                )
                span.set_output(receipt.to_dict())
                span.set_status("ERROR", "deadline expired")
                return receipt

            transitions.extend(
                [
                    StateTransition(ActionState.GROUNDED, reason="executor_resolved"),
                    StateTransition(ActionState.WAITING_RESOURCE, reason=action.body_id),
                ]
            )
            lease_timeout = min(
                action.verification_policy.timeout_sec,
                self._remaining_timeout(action),
            )
            lease_handle = self._resources.acquire(
                action.body_id,
                action.action_id,
                timeout_sec=lease_timeout,
            )
            if lease_handle is None:
                receipt = self._failure_receipt(
                    action,
                    ActionState.TIMED_OUT,
                    "RESOURCE_LEASE_TIMEOUT",
                    f"Timed out waiting for exclusive resource '{action.body_id}'.",
                    trace_id=trace_id,
                    started_at=started_at,
                    transitions=transitions,
                )
                span.set_output(receipt.to_dict())
                span.set_status("ERROR", "resource lease timeout")
                return receipt

            with lease_handle:
                transitions.append(StateTransition(ActionState.SCHEDULED, reason="lease_acquired"))
                # P0-5: explicit ROBOT_ACTION / ROBOT_STATE child spans under
                # the SKILL span so a REAL trace proves physical actuation
                # (command + ACK) and physical observation (positions, forces,
                # temps, status bits) instead of only "command submitted".
                physical = action.execution_mode is ExecutionMode.REAL
                with self._tracer.start_span(
                    "kernel.robot_action",
                    "ROBOT_ACTION",
                    source="action_gateway",
                    operation=action.capability_id,
                    attributes={
                        "action.id": action.action_id,
                        "execution.mode": action.execution_mode.value,
                        "physical_actuation": physical,
                    },
                    robot_id=action.body_id,
                    session_id=action.session_id,
                ) as action_span:
                    action_span.set_input(self._action_command_summary(action))
                    try:
                        result = executor(action)
                    except Exception as exc:  # noqa: BLE001
                        action_span.set_output({"error": str(exc), "command_sent": "unknown"})
                        action_span.set_status("ERROR", str(exc))
                        receipt = self._failure_receipt(
                            action,
                            ActionState.FAILED,
                            "EXECUTOR_ERROR",
                            str(exc),
                            trace_id=trace_id,
                            started_at=started_at,
                            transitions=transitions,
                            resource_lease=lease_handle.lease.to_dict(),
                        )
                        span.set_output(receipt.to_dict())
                        span.set_status("ERROR", str(exc))
                        return receipt

                    action_span.set_output(
                        {
                            "dispatch_result": result.dispatch_result,
                            "driver_ack": result.driver_ack,
                            "final_state": result.final_state.value,
                            "evidence_level": result.evidence_level.value,
                            "errors": result.errors,
                        }
                    )
                    for artifact in result.artifacts:
                        action_span.add_evidence(artifact)
                    if result.final_state is ActionState.BLOCKED:
                        action_span.set_status("BLOCKED", self._result_reason(result))
                    elif result.final_state not in {ActionState.COMPLETED, ActionState.DEGRADED}:
                        action_span.set_status("ERROR", self._result_reason(result))

                if result.dispatch_result.get("accepted") or result.observations:
                    observed = physical and result.evidence_level in {
                        EvidenceLevel.PHYSICALLY_OBSERVED,
                        EvidenceLevel.TASK_VERIFIED,
                    }
                    with self._tracer.start_span(
                        "kernel.robot_state",
                        "ROBOT_STATE",
                        source="action_gateway",
                        operation=action.capability_id,
                        attributes={
                            "action.id": action.action_id,
                            "execution.mode": action.execution_mode.value,
                            "physical_observation": observed,
                        },
                        robot_id=action.body_id,
                        session_id=action.session_id,
                    ) as state_span:
                        state_span.set_output(
                            {
                                "observations": result.observations,
                                "verification_result": result.verification_result,
                                "evidence_level": result.evidence_level.value,
                            }
                        )
                        for artifact in result.artifacts:
                            state_span.add_evidence(artifact)
                        if result.final_state not in {ActionState.COMPLETED, ActionState.DEGRADED}:
                            state_span.set_status("ERROR", self._result_reason(result))

                evidence = result.evidence_level
                final_state = result.final_state
                errors = list(result.errors)
                if action.execution_mode is ExecutionMode.FIXTURE:
                    evidence = EvidenceLevel.SYNTHETIC
                    if final_state is ActionState.COMPLETED:
                        final_state = ActionState.DEGRADED

                if (
                    action.execution_mode is not ExecutionMode.FIXTURE
                    and final_state is ActionState.COMPLETED
                    and not self._meets_evidence_requirement(
                        evidence,
                        action.verification_policy.required_evidence,
                    )
                ):
                    final_state = (
                        ActionState.FAILED
                        if action.verification_policy.fail_closed
                        else ActionState.DEGRADED
                    )
                    errors.append(
                        {
                            "code": "VERIFICATION_REQUIREMENT_NOT_MET",
                            "message": (
                                f"Executor produced {evidence.value}; action requires "
                                f"{action.verification_policy.required_evidence.value}."
                            ),
                        }
                    )

                transitions.extend(self._evidence_transitions(result, evidence))
                transitions.append(StateTransition(final_state, reason="executor_completed"))
                acknowledgement_stage = self._acknowledgement_stage(result, evidence)
                receipt = ExecutionReceipt(
                    action_id=action.action_id,
                    trace_id=trace_id,
                    mode=action.execution_mode,
                    body_id=action.body_id,
                    body_snapshot_hash=action.body_snapshot_hash,
                    capability_id=action.capability_id,
                    policy_decision=result.policy_decision,
                    authorization_decision=result.authorization_decision,
                    resource_lease=lease_handle.lease.to_dict(),
                    simulation_result=result.simulation_result,
                    dispatch_result=result.dispatch_result,
                    driver_ack=result.driver_ack,
                    acknowledgement_stage=acknowledgement_stage,
                    observations=result.observations,
                    verification_result=result.verification_result,
                    final_state=final_state,
                    evidence_level=evidence,
                    artifacts=list(result.artifacts),
                    errors=errors,
                    transitions=transitions,
                    started_at=started_at,
                    finished_at=utc_now(),
                )
                self._persist_receipt(receipt, result.artifact_directory)
                span.set_output(receipt.to_dict())
                for artifact in receipt.artifacts:
                    span.add_evidence(artifact)
                if final_state is ActionState.BLOCKED:
                    span.set_status("BLOCKED", self._result_reason(result))
                elif final_state not in {ActionState.COMPLETED, ActionState.DEGRADED}:
                    span.set_status("ERROR", self._result_reason(result))
                return receipt

    @staticmethod
    def _meets_evidence_requirement(
        actual: EvidenceLevel,
        required: EvidenceLevel,
    ) -> bool:
        order = {
            EvidenceLevel.SYNTHETIC: 0,
            EvidenceLevel.REQUESTED: 1,
            EvidenceLevel.DISPATCH_CONFIRMED: 2,
            EvidenceLevel.DRIVER_CONFIRMED: 3,
            EvidenceLevel.PHYSICALLY_OBSERVED: 4,
            EvidenceLevel.TASK_VERIFIED: 5,
        }
        return order[actual] >= order[required]

    @staticmethod
    def _evidence_transitions(
        result: ActionExecutionResult,
        evidence: EvidenceLevel,
    ) -> list[StateTransition]:
        transitions: list[StateTransition] = []
        if result.policy_decision.get("allowed") is True:
            transitions.append(
                StateTransition(ActionState.POLICY_VALIDATED, reason="policy_allowed")
            )
        if result.simulation_result and result.simulation_result.get("has_physics") is True:
            transitions.append(
                StateTransition(
                    ActionState.SIMULATION_VALIDATED,
                    reason="physics_simulation_completed",
                )
            )
        if result.authorization_decision.get("authorized") is True:
            transitions.append(
                StateTransition(ActionState.AUTHORIZED, reason="authorization_granted")
            )
        if result.dispatch_result.get("accepted") is True:
            transitions.append(
                StateTransition(ActionState.COMMAND_DISPATCHED, reason="command_dispatched")
            )
        acknowledgement = ActionGateway._driver_acknowledgement_stage(result)
        if acknowledgement is AcknowledgementStage.PROTOCOL_ACKNOWLEDGED:
            transitions.append(
                StateTransition(
                    ActionState.PROTOCOL_ACKNOWLEDGED,
                    reason="protocol_acknowledged",
                )
            )
        elif acknowledgement is AcknowledgementStage.DELIVERY_INFERRED:
            transitions.append(
                StateTransition(ActionState.DELIVERY_INFERRED, reason="delivery_inferred")
            )
        if evidence in {EvidenceLevel.PHYSICALLY_OBSERVED, EvidenceLevel.TASK_VERIFIED}:
            transitions.append(
                StateTransition(ActionState.EFFECT_OBSERVED, reason="effect_observed")
            )
        if evidence is EvidenceLevel.TASK_VERIFIED:
            transitions.append(
                StateTransition(ActionState.TASK_VERIFIED, reason="predicate_verified")
            )
        return transitions

    @staticmethod
    def _driver_acknowledgement_stage(
        result: ActionExecutionResult,
    ) -> AcknowledgementStage | None:
        if result.acknowledgement_stage in {
            AcknowledgementStage.PROTOCOL_ACKNOWLEDGED,
            AcknowledgementStage.DELIVERY_INFERRED,
        }:
            return AcknowledgementStage(result.acknowledgement_stage)
        if result.driver_ack:
            raw_stage = result.driver_ack.get("stage")
            if isinstance(raw_stage, str):
                try:
                    parsed = AcknowledgementStage(raw_stage.upper())
                except ValueError:
                    parsed = None
                if parsed in {
                    AcknowledgementStage.PROTOCOL_ACKNOWLEDGED,
                    AcknowledgementStage.DELIVERY_INFERRED,
                }:
                    return parsed
            if result.driver_ack.get("acknowledged") is True:
                return AcknowledgementStage.PROTOCOL_ACKNOWLEDGED
            if result.driver_ack.get("delivery_inferred") is True:
                return AcknowledgementStage.DELIVERY_INFERRED
        return None

    @staticmethod
    def _acknowledgement_stage(
        result: ActionExecutionResult,
        evidence: EvidenceLevel,
    ) -> AcknowledgementStage:
        if evidence is EvidenceLevel.TASK_VERIFIED:
            return AcknowledgementStage.TASK_VERIFIED
        if evidence is EvidenceLevel.PHYSICALLY_OBSERVED:
            return AcknowledgementStage.EFFECT_OBSERVED
        driver_stage = ActionGateway._driver_acknowledgement_stage(result)
        if driver_stage is not None:
            return driver_stage
        if result.acknowledgement_stage is not None:
            return AcknowledgementStage(result.acknowledgement_stage)
        if result.dispatch_result.get("accepted") is True:
            return AcknowledgementStage.COMMAND_DISPATCHED
        return AcknowledgementStage.REQUEST_ACCEPTED

    @staticmethod
    def _result_reason(result: ActionExecutionResult) -> str:
        if result.errors:
            return str(result.errors[0].get("message", result.errors[0]))
        return str(result.policy_decision.get("reason", result.final_state.value))

    @staticmethod
    def _action_command_summary(action: ActionEnvelope) -> dict[str, Any]:
        """Bounded command summary for the ROBOT_ACTION span input.

        Full joint-value payloads are NOT inlined: the span records a hash and
        count here, while the complete envelope lives on the SKILL span input
        and (when the executor provides an artifact directory) in the
        persisted receipt referenced via span evidence.
        """
        args = action.arguments or {}
        names = list(args.get("names") or [])
        values = list(args.get("values") or [])
        summary: dict[str, Any] = {
            "capability_id": action.capability_id,
            "representation": args.get("representation"),
            "units": args.get("units"),
            "joint_count": len(names),
            "names": names[:12],
            "permit_id": args.get("permit_id"),
            "speed": args.get("speed"),
            "force_limit_g": args.get("force_limit_g"),
            "observation_timestamp_ns": args.get("observation_timestamp_ns"),
        }
        if values:
            summary["values_count"] = len(values)
            summary["values_sha256"] = hashlib.sha256(
                json.dumps(values).encode("utf-8")
            ).hexdigest()
        return summary

    @staticmethod
    def _remaining_timeout(action: ActionEnvelope) -> float:
        if action.deadline_at is None:
            return max(0.0, action.verification_policy.timeout_sec)
        return max(0.0, (action.deadline_at - datetime.now(UTC)).total_seconds())

    def _failure_receipt(
        self,
        action: ActionEnvelope,
        state: ActionState,
        code: str,
        message: str,
        *,
        trace_id: str | None = None,
        started_at: datetime | None = None,
        transitions: list[StateTransition] | None = None,
        resource_lease: dict[str, Any] | None = None,
    ) -> ExecutionReceipt:
        state_transitions = list(transitions or [StateTransition(ActionState.PROPOSED)])
        state_transitions.append(StateTransition(state, reason=code))
        return ExecutionReceipt(
            action_id=action.action_id,
            trace_id=trace_id or action.parent_trace_id or f"trace_{uuid.uuid4().hex[:24]}",
            mode=action.execution_mode,
            body_id=action.body_id,
            body_snapshot_hash=action.body_snapshot_hash,
            capability_id=action.capability_id,
            final_state=state,
            evidence_level=(
                EvidenceLevel.SYNTHETIC
                if action.execution_mode is ExecutionMode.FIXTURE
                else EvidenceLevel.REQUESTED
            ),
            acknowledgement_stage=AcknowledgementStage.REQUEST_ACCEPTED,
            resource_lease=resource_lease,
            errors=[{"code": code, "message": message}],
            transitions=state_transitions,
            started_at=started_at or utc_now(),
            finished_at=utc_now(),
        )

    @staticmethod
    def _persist_receipt(receipt: ExecutionReceipt, artifact_directory: str | None) -> None:
        if not artifact_directory:
            return
        directory = Path(artifact_directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        receipt_path = directory / "receipt.json"
        receipt_uri = receipt_path.as_uri()
        if receipt_uri not in receipt.artifacts:
            receipt.artifacts.append(receipt_uri)
        temporary = receipt_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(receipt.to_dict(), indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(receipt_path)

    def _publish_receipt(self, receipt: ExecutionReceipt) -> None:
        if self._event_bus is None:
            return
        try:
            from rosclaw.core.event_bus import Event, EventPriority

            priority = (
                EventPriority.HIGH
                if receipt.final_state in {ActionState.BLOCKED, ActionState.FAILED}
                else EventPriority.NORMAL
            )
            self._event_bus.publish(
                Event(
                    topic="rosclaw.runtime.action.receipt",
                    payload=receipt.to_dict(),
                    source="action_gateway",
                    priority=priority,
                    trace_id=receipt.trace_id,
                )
            )
        except Exception:
            return


__all__ = ["ActionExecutor", "ActionGateway"]
