"""Versioned contracts for physical action execution.

These types deliberately separate a requested action from evidence that the
action was dispatched, observed, and verified.  They are transport-neutral so
CLI, MCP, ROS, and vendor adapters can share the same semantics.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

ACTION_SCHEMA_VERSION = "rosclaw.action.v1"
RECEIPT_SCHEMA_VERSION = "rosclaw.receipt.v1"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


class ExecutionMode(StrEnum):
    """How an action is intended to execute."""

    FIXTURE = "FIXTURE"
    DRY_RUN = "DRY_RUN"
    REPLAY = "REPLAY"
    SIMULATION = "SIMULATION"
    SHADOW = "SHADOW"
    REAL = "REAL"


class ActionState(StrEnum):
    """Observable states in the physical action lifecycle."""

    PROPOSED = "PROPOSED"
    GROUNDED = "GROUNDED"
    POLICY_VALIDATED = "POLICY_VALIDATED"
    SIMULATION_VALIDATED = "SIMULATION_VALIDATED"
    AUTHORIZATION_REQUIRED = "AUTHORIZATION_REQUIRED"
    AUTHORIZED = "AUTHORIZED"
    WAITING_RESOURCE = "WAITING_RESOURCE"
    SCHEDULED = "SCHEDULED"
    DISPATCHED = "DISPATCHED"
    DRIVER_ACKNOWLEDGED = "DRIVER_ACKNOWLEDGED"
    PHYSICALLY_OBSERVED = "PHYSICALLY_OBSERVED"
    TASK_VERIFIED = "TASK_VERIFIED"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    DEGRADED = "DEGRADED"


class EvidenceLevel(StrEnum):
    """Strongest evidence obtained for an action."""

    SYNTHETIC = "SYNTHETIC"
    REQUESTED = "REQUESTED"
    DISPATCH_CONFIRMED = "DISPATCH_CONFIRMED"
    DRIVER_CONFIRMED = "DRIVER_CONFIRMED"
    PHYSICALLY_OBSERVED = "PHYSICALLY_OBSERVED"
    TASK_VERIFIED = "TASK_VERIFIED"


@dataclass
class AuthorizationContext:
    """Authorization facts attached by the caller or approval service."""

    principal_id: str = ""
    approved: bool = False
    approval_id: str | None = None
    scopes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal_id": self.principal_id,
            "approved": self.approved,
            "approval_id": self.approval_id,
            "scopes": list(self.scopes),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> AuthorizationContext:
        data = value or {}
        return cls(
            principal_id=str(data.get("principal_id", "")),
            approved=bool(data.get("approved", False)),
            approval_id=data.get("approval_id"),
            scopes=[str(item) for item in data.get("scopes", [])],
        )


@dataclass
class VerificationPolicy:
    """Minimum verification requested for the action."""

    required_evidence: EvidenceLevel = EvidenceLevel.TASK_VERIFIED
    timeout_sec: float = 30.0
    fail_closed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_evidence": self.required_evidence.value,
            "timeout_sec": self.timeout_sec,
            "fail_closed": self.fail_closed,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> VerificationPolicy:
        data = value or {}
        return cls(
            required_evidence=EvidenceLevel(
                str(data.get("required_evidence", EvidenceLevel.TASK_VERIFIED.value)).upper()
            ),
            timeout_sec=float(data.get("timeout_sec", 30.0)),
            fail_closed=bool(data.get("fail_closed", True)),
        )


@dataclass
class ActionEnvelope:
    """Canonical, versioned request for an action with physical implications."""

    actor_id: str
    agent_framework: str
    session_id: str
    body_id: str
    capability_id: str
    arguments: dict[str, Any]
    execution_mode: ExecutionMode
    schema_version: str = ACTION_SCHEMA_VERSION
    action_id: str = field(default_factory=lambda: f"action_{uuid.uuid4().hex}")
    body_snapshot_hash: str = ""
    authorization: AuthorizationContext = field(default_factory=AuthorizationContext)
    risk_class: str = "medium"
    deadline_at: datetime | None = None
    parent_trace_id: str | None = None
    expected_effect: dict[str, Any] | None = None
    verification_policy: VerificationPolicy = field(default_factory=VerificationPolicy)

    def __post_init__(self) -> None:
        self.execution_mode = ExecutionMode(str(self.execution_mode).upper())
        required = {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "actor_id": self.actor_id,
            "agent_framework": self.agent_framework,
            "session_id": self.session_id,
            "body_id": self.body_id,
            "capability_id": self.capability_id,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"ActionEnvelope requires non-empty fields: {', '.join(missing)}")
        if self.schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported action schema '{self.schema_version}'; expected {ACTION_SCHEMA_VERSION}"
            )
        if not isinstance(self.arguments, dict):
            raise TypeError("ActionEnvelope.arguments must be a dict")
        if self.deadline_at is not None and self.deadline_at.tzinfo is None:
            self.deadline_at = self.deadline_at.replace(tzinfo=UTC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "actor_id": self.actor_id,
            "agent_framework": self.agent_framework,
            "session_id": self.session_id,
            "body_id": self.body_id,
            "body_snapshot_hash": self.body_snapshot_hash,
            "capability_id": self.capability_id,
            "arguments": dict(self.arguments),
            "execution_mode": self.execution_mode.value,
            "authorization": self.authorization.to_dict(),
            "risk_class": self.risk_class,
            "deadline_at": _iso(self.deadline_at),
            "parent_trace_id": self.parent_trace_id,
            "expected_effect": self.expected_effect,
            "verification_policy": self.verification_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ActionEnvelope:
        return cls(
            schema_version=str(value.get("schema_version", ACTION_SCHEMA_VERSION)),
            action_id=str(value.get("action_id") or f"action_{uuid.uuid4().hex}"),
            actor_id=str(value.get("actor_id", "")),
            agent_framework=str(value.get("agent_framework", "")),
            session_id=str(value.get("session_id", "")),
            body_id=str(value.get("body_id", "")),
            body_snapshot_hash=str(value.get("body_snapshot_hash", "")),
            capability_id=str(value.get("capability_id", "")),
            arguments=dict(value.get("arguments", {})),
            execution_mode=ExecutionMode(str(value.get("execution_mode", "")).upper()),
            authorization=AuthorizationContext.from_dict(value.get("authorization")),
            risk_class=str(value.get("risk_class", "medium")),
            deadline_at=_parse_datetime(value.get("deadline_at")),
            parent_trace_id=value.get("parent_trace_id"),
            expected_effect=value.get("expected_effect"),
            verification_policy=VerificationPolicy.from_dict(value.get("verification_policy")),
        )


@dataclass
class StateTransition:
    """One timestamped action state transition."""

    state: ActionState
    at: datetime = field(default_factory=utc_now)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "at": _iso(self.at), "reason": self.reason}


@dataclass
class ActionExecutionResult:
    """Executor-to-gateway result before the canonical receipt is assembled."""

    final_state: ActionState
    evidence_level: EvidenceLevel
    policy_decision: dict[str, Any] = field(default_factory=dict)
    authorization_decision: dict[str, Any] = field(default_factory=dict)
    simulation_result: dict[str, Any] | None = None
    dispatch_result: dict[str, Any] = field(default_factory=dict)
    driver_ack: dict[str, Any] | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    verification_result: dict[str, Any] | None = None
    artifacts: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    artifact_directory: str | None = None


@dataclass
class ExecutionReceipt:
    """Evidence-bearing outcome of one ActionEnvelope."""

    action_id: str
    trace_id: str
    mode: ExecutionMode
    body_id: str
    body_snapshot_hash: str
    capability_id: str
    final_state: ActionState
    evidence_level: EvidenceLevel
    schema_version: str = RECEIPT_SCHEMA_VERSION
    policy_decision: dict[str, Any] = field(default_factory=dict)
    authorization_decision: dict[str, Any] = field(default_factory=dict)
    resource_lease: dict[str, Any] | None = None
    simulation_result: dict[str, Any] | None = None
    dispatch_result: dict[str, Any] = field(default_factory=dict)
    driver_ack: dict[str, Any] | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    verification_result: dict[str, Any] | None = None
    artifacts: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    transitions: list[StateTransition] = field(default_factory=list)
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime = field(default_factory=utc_now)

    @property
    def verified(self) -> bool:
        if self.mode is ExecutionMode.FIXTURE:
            return False
        return self.evidence_level in {
            EvidenceLevel.PHYSICALLY_OBSERVED,
            EvidenceLevel.TASK_VERIFIED,
        }

    @property
    def trust_level(self) -> str:
        if self.mode is ExecutionMode.FIXTURE:
            return "SYNTHETIC"
        if self.mode is ExecutionMode.REPLAY:
            return "RECORDED"
        if self.mode is ExecutionMode.SIMULATION:
            return "SIMULATED"
        if self.verified:
            return "VERIFIED"
        return "UNVERIFIED"

    @property
    def usable_for_real_execution(self) -> bool:
        return self.mode is ExecutionMode.REAL and self.verified

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "trace_id": self.trace_id,
            "mode": self.mode.value,
            "execution_mode": self.mode.value,
            "body_id": self.body_id,
            "body_snapshot_hash": self.body_snapshot_hash,
            "capability_id": self.capability_id,
            "policy_decision": self.policy_decision,
            "authorization_decision": self.authorization_decision,
            "resource_lease": self.resource_lease,
            "simulation_result": self.simulation_result,
            "dispatch_result": self.dispatch_result,
            "driver_ack": self.driver_ack,
            "observations": self.observations,
            "verification_result": self.verification_result,
            "final_state": self.final_state.value,
            "evidence_level": self.evidence_level.value,
            "verified": self.verified,
            "trust_level": self.trust_level,
            "usable_for_real_execution": self.usable_for_real_execution,
            "artifacts": list(self.artifacts),
            "errors": list(self.errors),
            "transitions": [transition.to_dict() for transition in self.transitions],
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
        }


class EmergencyStopStatus(StrEnum):
    """Final status of an emergency-stop request."""

    REQUESTED = "REQUESTED"
    DISPATCHED = "DISPATCHED"
    PARTIALLY_ACKNOWLEDGED = "PARTIALLY_ACKNOWLEDGED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PHYSICALLY_VERIFIED = "PHYSICALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    FAILED = "FAILED"


@dataclass
class EmergencyStopReceipt:
    """Truthful result of an emergency-stop fan-out."""

    request_id: str
    reason: str
    requested_at: datetime
    targets: list[str]
    request_dispatched: bool
    driver_acknowledged: bool
    acknowledged_drivers: list[str]
    unacknowledged_drivers: list[str]
    physical_stop_observed: bool
    observed_velocity: float | None
    observed_joint_velocity: list[float] | None
    verification_source: str | None
    timeout: bool
    timeout_sec: float
    final_status: EmergencyStopStatus
    execution_mode: str = "UNKNOWN"
    driver_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def stopped(self) -> bool:
        return self.final_status is EmergencyStopStatus.PHYSICALLY_VERIFIED

    def to_dict(self) -> dict[str, Any]:
        mode = self.execution_mode.upper()
        if mode == "FIXTURE":
            trust_level = "SYNTHETIC"
        elif mode == "SIMULATION":
            trust_level = "SIMULATED"
        elif mode == "REAL" and self.stopped:
            trust_level = "VERIFIED"
        else:
            trust_level = "UNVERIFIED"
        return {
            "request_id": self.request_id,
            "reason": self.reason,
            "requested_at": _iso(self.requested_at),
            "targets": list(self.targets),
            "request_dispatched": self.request_dispatched,
            "driver_acknowledged": self.driver_acknowledged,
            "acknowledged_drivers": list(self.acknowledged_drivers),
            "unacknowledged_drivers": list(self.unacknowledged_drivers),
            "physical_stop_observed": self.physical_stop_observed,
            "observed_velocity": self.observed_velocity,
            "observed_joint_velocity": self.observed_joint_velocity,
            "verification_source": self.verification_source,
            "timeout": self.timeout,
            "timeout_sec": self.timeout_sec,
            "final_status": self.final_status.value,
            "stopped": self.stopped,
            "mode": mode.lower(),
            "driver_results": self.driver_results,
            "errors": self.errors,
            "execution_mode": mode,
            "trust_level": trust_level,
            "usable_for_real_execution": False,
        }


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "ActionEnvelope",
    "ActionExecutionResult",
    "ActionState",
    "AuthorizationContext",
    "EmergencyStopReceipt",
    "EmergencyStopStatus",
    "EvidenceLevel",
    "ExecutionMode",
    "ExecutionReceipt",
    "StateTransition",
    "VerificationPolicy",
    "utc_now",
]
