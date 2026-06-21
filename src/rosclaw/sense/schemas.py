"""ROSClaw Sense schemas.

This module defines the data structures used by the rosclaw.sense package:

- BodyState: high-frequency raw dynamic state from sensors and telemetry.
- BodyRiskSummary: normalized per-subsystem risk levels.
- BodyReadiness: per-task/capability readiness evaluation.
- BodySense: low-frequency semantic body sense for agents and humans.
- BodyEvent: discrete body events detected by estimators.

All dataclasses are intentionally tolerant of missing fields so that different
robots and different data sources can be supported without crashing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EnergyState:
    """Power and energy state."""

    battery_percent: float | None = None
    voltage: float | None = None
    current_draw: float | None = None
    estimated_runtime_min: float | None = None
    power_mode: str = "unknown"


@dataclass
class JointState:
    """State of a single joint."""

    position_rad: float | None = None
    velocity_rad_s: float | None = None
    torque_nm: float | None = None
    temperature_c: float | None = None
    tracking_error: float | None = None
    health: str = "unknown"


@dataclass
class IMUState:
    """Inertial measurement unit state."""

    pitch_deg: float | None = None
    roll_deg: float | None = None
    yaw_deg: float | None = None
    angular_velocity: list[float] | None = None


@dataclass
class FootContactState:
    """Foot/endeffector contact state."""

    contact: bool | None = None
    confidence: float | None = None
    slip_risk: str = "unknown"


@dataclass
class CommunicationState:
    """Control and data communication health."""

    dds_latency_ms: float | None = None
    packet_loss: float | None = None
    heartbeat_ok: bool | None = None
    last_heartbeat_ms: float | None = None


@dataclass
class PerceptionHealth:
    """Perception sensor and detector health."""

    front_camera_fps: float | None = None
    depth_camera_fps: float | None = None
    camera_obstructed: bool | None = None
    target_detector_confidence: float | None = None
    status: str = "unknown"


@dataclass
class BalanceState:
    """Balance and stability state."""

    support_margin: float | None = None
    com_projection: str = "unknown"
    fall_risk_raw: float | None = None
    stable_for_sec: float | None = None


@dataclass
class ComputeState:
    """On-board compute resource state."""

    cpu_usage_percent: float | None = None
    gpu_usage_percent: float | None = None
    cpu_temp_c: float | None = None
    gpu_temp_c: float | None = None
    memory_usage_percent: float | None = None


@dataclass
class BodyState:
    """High-frequency raw dynamic body state.

    This is the source-of-truth snapshot produced by collectors.  It may be
    partially populated because not every robot exposes every sensor.
    """

    robot_id: str
    timestamp: float
    source: str = "unknown"

    energy: EnergyState = field(default_factory=EnergyState)
    joints: dict[str, JointState] = field(default_factory=dict)
    imu: IMUState = field(default_factory=IMUState)
    contact: dict[str, FootContactState] = field(default_factory=dict)
    communication: CommunicationState = field(default_factory=CommunicationState)
    perception: PerceptionHealth = field(default_factory=PerceptionHealth)
    balance: BalanceState = field(default_factory=BalanceState)
    compute: ComputeState = field(default_factory=ComputeState)

    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyState:
        """Deserialize from a dictionary produced by ``to_dict``."""
        energy = EnergyState(**data.get("energy", {}))
        joints = {
            name: JointState(**fields)
            for name, fields in data.get("joints", {}).items()
        }
        imu = IMUState(**data.get("imu", {}))
        contact = {
            name: FootContactState(**fields)
            for name, fields in data.get("contact", {}).items()
        }
        communication = CommunicationState(**data.get("communication", {}))
        perception = PerceptionHealth(**data.get("perception", {}))
        balance = BalanceState(**data.get("balance", {}))
        compute = ComputeState(**data.get("compute", {}))
        return cls(
            robot_id=data.get("robot_id", "unknown"),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", "unknown"),
            energy=energy,
            joints=joints,
            imu=imu,
            contact=contact,
            communication=communication,
            perception=perception,
            balance=balance,
            compute=compute,
            raw=data.get("raw", {}),
        )


@dataclass
class BodyRiskSummary:
    """Normalized risk levels for each subsystem.

    Risk level ordering: unknown < low < medium < high < critical.
    """

    power_risk: str = "unknown"
    thermal_risk: str = "unknown"
    actuator_risk: str = "unknown"
    balance_risk: str = "unknown"
    contact_risk: str = "unknown"
    perception_risk: str = "unknown"
    communication_risk: str = "unknown"
    compute_risk: str = "unknown"
    fatigue_risk: str = "unknown"
    overall_risk: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyRiskSummary:
        return cls(**{k: data.get(k, "unknown") for k in cls.__dataclass_fields__})


@dataclass
class FailedRequirement:
    """A single failed requirement for a task/capability."""

    name: str
    current: Any
    required: Any
    severity: str = "medium"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "current": self.current,
            "required": self.required,
            "severity": self.severity,
            "reason": self.reason,
        }


@dataclass
class ReadinessItem:
    """Readiness evaluation for a single capability."""

    capability: str
    status: str = "unknown"  # ready | degraded | not_ready | unknown
    reasons: list[str] = field(default_factory=list)
    failed_requirements: list[FailedRequirement] = field(default_factory=list)
    allowed_alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "status": self.status,
            "reasons": self.reasons,
            "failed_requirements": [
                asdict(req) for req in self.failed_requirements
            ],
            "allowed_alternatives": self.allowed_alternatives,
        }


@dataclass
class BodyReadiness:
    """Overall readiness evaluation for one or more capabilities."""

    robot_id: str
    timestamp: float
    task: str | None = None
    overall_status: str = "unknown"  # ready | caution | not_ready | emergency | unknown
    capabilities: dict[str, ReadinessItem] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
            "task": self.task,
            "overall_status": self.overall_status,
            "capabilities": {
                name: item.to_dict() for name, item in self.capabilities.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyReadiness:
        capabilities = {
            name: ReadinessItem(
                capability=item.get("capability", name),
                status=item.get("status", "unknown"),
                reasons=item.get("reasons", []),
                failed_requirements=[
                    FailedRequirement(**req)
                    for req in item.get("failed_requirements", [])
                ],
                allowed_alternatives=item.get("allowed_alternatives", []),
            )
            for name, item in data.get("capabilities", {}).items()
        }
        return cls(
            robot_id=data.get("robot_id", "unknown"),
            timestamp=data.get("timestamp", 0.0),
            task=data.get("task"),
            overall_status=data.get("overall_status", "unknown"),
            capabilities=capabilities,
        )


@dataclass
class BodySense:
    """Semantic body sense for agents and humans.

    This is the primary output of the Sense module.  It is intentionally
    low-frequency and explanation-oriented, not a raw telemetry dump.
    """

    robot_id: str
    timestamp: float

    overall_status: str = "unknown"  # ready | caution | not_ready | emergency | unknown
    risk_summary: BodyRiskSummary = field(default_factory=BodyRiskSummary)
    readiness: BodyReadiness = field(default_factory=lambda: BodyReadiness("unknown", 0.0))

    main_reasons: list[str] = field(default_factory=list)
    blocked_capabilities: list[str] = field(default_factory=list)
    degraded_capabilities: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)

    natural_language_summary: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    source_state_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "risk_summary": self.risk_summary.to_dict(),
            "readiness": self.readiness.to_dict(),
            "main_reasons": self.main_reasons,
            "blocked_capabilities": self.blocked_capabilities,
            "degraded_capabilities": self.degraded_capabilities,
            "recommended_actions": self.recommended_actions,
            "natural_language_summary": self.natural_language_summary,
            "evidence": self.evidence,
            "source_state_id": self.source_state_id,
        }


@dataclass
class BodyEvent:
    """Discrete body event detected by estimators."""

    event_id: str
    robot_id: str
    timestamp: float

    type: str
    severity: str = "info"  # info | low | medium | high | critical
    source: str = "sense"

    affected_parts: list[str] = field(default_factory=list)
    measurement: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    consequence: dict[str, Any] = field(default_factory=dict)
    recommended_actions: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyEvent:
        defaults = {
            "event_id": "",
            "robot_id": "unknown",
            "timestamp": 0.0,
            "type": "unknown",
            "severity": "info",
            "source": "sense",
            "affected_parts": [],
            "measurement": {},
            "thresholds": {},
            "consequence": {},
            "recommended_actions": [],
            "evidence": {},
        }
        return cls(**{k: data.get(k, default) for k, default in defaults.items()})


def _risk_rank(risk: str) -> int:
    """Return numeric rank for a risk level string."""
    return {
        "unknown": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }.get(risk, 0)


def max_risk(*risks: str) -> str:
    """Return the highest risk level among the given values."""
    ranking = [(r, _risk_rank(r)) for r in risks if r]
    if not ranking:
        return "unknown"
    return max(ranking, key=lambda x: x[1])[0]
