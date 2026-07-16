"""Execution schemas: permit, request, result (plan §7.3-§7.5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

EXECUTION_PERMIT_SCHEMA = "rosclaw.execution_permit.v1"
ACTION_EXECUTION_REQUEST_SCHEMA = "rosclaw.action_execution_request.v1"
ACTION_EXECUTION_RESULT_SCHEMA = "rosclaw.action_execution_result.v1"


@dataclass
class ExecutionPermit:
    """``rosclaw.execution_permit.v1`` — bound to exact content hashes.

    Any change to body, calibration, policy contract, mapping or transport
    profile invalidates the permit.  Worker restarts and serial reconnects
    revoke it explicitly via the permit manager.
    """

    permit_id: str
    body_id: str
    policy_contract_hash: str
    body_hash: str
    calibration_hash: str
    mapping_hash: str
    transport_profile_hash: str
    allowed_representation: str = "joint_position"
    allowed_unit: str = "raw_device_unit"
    max_step_delta_raw: float = 30.0
    max_speed: int = 100
    max_force_g: float = 100.0
    expires_at: str = ""
    expires_at_monotonic_ns: int = 0
    operator_armed: bool = False
    physical_estop_confirmed: bool = False
    task: str = ""
    schema_version: str = EXECUTION_PERMIT_SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActionExecutionRequest:
    """``rosclaw.action_execution_request.v1`` — one single-step command."""

    proposal_id: str
    candidate_id: str
    permit_id: str
    body_id: str
    representation: str
    units: str
    names: list[str]
    values: list[float]
    speed: int = 100
    force_limit_g: float = 100.0
    valid_until_monotonic_ns: int = 0
    schema_version: str = ACTION_EXECUTION_REQUEST_SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeedbackVerification:
    position_reached: bool = False
    force_safe: bool = True
    temperature_safe: bool = True
    fault_free: bool = True
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActionExecutionResult:
    """``rosclaw.action_execution_result.v1`` — outcome + physical feedback."""

    status: str = "blocked"  # completed | blocked | fault | stale_action | aborted
    command_sent: bool = False
    command_acknowledged: bool = False
    target: list[float] = field(default_factory=list)
    actual: list[float] = field(default_factory=list)
    position_error: list[float] = field(default_factory=list)
    force: list[float] = field(default_factory=list)
    current: list[float] = field(default_factory=list)
    temperature: list[float] = field(default_factory=list)
    status_bits: list[int] = field(default_factory=list)
    verification: FeedbackVerification = field(default_factory=FeedbackVerification)
    error_code: str | None = None
    message: str = ""
    schema_version: str = ACTION_EXECUTION_RESULT_SCHEMA

    def to_dict(self) -> dict:
        out = asdict(self)
        out["verification"] = self.verification.to_dict()
        return out
