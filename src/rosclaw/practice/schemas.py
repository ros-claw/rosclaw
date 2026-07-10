"""Practice data schemas for the Physical Data Flywheel Runtime.

This module defines the canonical event envelope and payload models used by
`rosclaw-practice`. All JSONL/MCAP/SeekDB events share the same envelope so
that downstream tools can query, replay, and export practices without
source-specific parsing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from rosclaw.practice.ids import generate_event_id

SCHEMA_VERSION = "practice.event.v1"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _now_ns() -> int:
    return int(datetime.now(UTC).timestamp() * 1_000_000_000)


class PracticeEventEnvelope(BaseModel):
    """Unified envelope for every event captured by rosclaw-practice.

    Large or binary payloads should be written to files and referenced via
    ``payload_ref``; only small serializable metadata belongs in ``payload``.

    The optional ``body_id`` and ``episode_id`` fields let a single practice
    session track multiple bodies and/or episodes (e.g. RH56 left/right hand
    or repeated skill trials). ``policy_id`` links candidate/promotion events
    to a specific policy version.
    """

    schema_version: str = SCHEMA_VERSION

    practice_id: str
    session_id: str | None = None
    episode_id: str | None = None
    robot_id: str
    body_id: str | None = None

    source: Literal[
        "dds",
        "ros2",
        "camera",
        "agent",
        "provider",
        "sandbox",
        "runtime",
        "human",
        "system",
    ]
    event_type: str

    timestamp_ns: int = Field(default_factory=_now_ns)
    timestamp_utc: str = Field(default_factory=_utc_now_iso)
    source_timestamp_ns: int | None = None
    sequence_id: int | None = None

    trace_id: str | None = None
    parent_event_id: str | None = None
    event_id: str = Field(default_factory=generate_event_id)

    frame_id: str | None = None
    task_id: str | None = None
    skill_id: str | None = None
    action_id: str | None = None
    policy_id: str | None = None

    payload: dict[str, Any] = Field(default_factory=dict)
    payload_ref: dict[str, str] = Field(default_factory=dict)

    quality: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Payload models for individual sources (P0 mock + forward-compatible)
# ---------------------------------------------------------------------------


class JointStatePayload(BaseModel):
    joint_names: list[str]
    position: list[float]
    velocity: list[float] | None = None
    effort: list[float] | None = None
    temperature: list[float] | None = None
    control_mode: list[str] | None = None


class IMUPayload(BaseModel):
    orientation_xyzw: list[float] | None = None
    angular_velocity_xyz: list[float]
    linear_acceleration_xyz: list[float]
    covariance: dict[str, list[float]] | None = None


class FootContactPayload(BaseModel):
    left_contact: bool
    right_contact: bool
    left_force_n: float | None = None
    right_force_n: float | None = None
    contact_confidence: float | None = None


class OdometryPayload(BaseModel):
    frame_id: str
    child_frame_id: str
    position_xyz: list[float]
    orientation_xyzw: list[float]
    linear_velocity_xyz: list[float] | None = None
    angular_velocity_xyz: list[float] | None = None


class RGBDFramePayload(BaseModel):
    camera_id: str
    width: int
    height: int
    rgb_encoding: str
    depth_encoding: str
    rgb_ref: str
    depth_ref: str | None = None
    camera_info_ref: str | None = None
    intrinsics: list[float] | None = None
    extrinsics_ref: str | None = None
    dropped_frame_count: int | None = None


class AgentPlanPayload(BaseModel):
    task: str
    plan_id: str
    planner: str
    plan_steps: list[dict[str, Any]]
    constraints: list[str] = Field(default_factory=list)
    referenced_memory_ids: list[str] = Field(default_factory=list)
    referenced_taskcards: list[str] = Field(default_factory=list)
    decision_summary: str | None = None


class ToolCallPayload(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result_summary: dict[str, Any] | None = None
    status: Literal["started", "success", "failed"]
    latency_ms: float | None = None
    error: str | None = None


class ProviderOutputPayload(BaseModel):
    provider_id: str
    provider_type: str
    model: str | None = None
    route_reason: str | None = None
    input_summary: dict[str, Any] | None = None
    output_summary: dict[str, Any] | None = None
    latency_ms: float
    token_usage: dict[str, int] | None = None
    confidence: float | None = None
    status: Literal["success", "failed", "fallback"]


class SandboxDecisionPayload(BaseModel):
    decision_id: str
    action_id: str
    requested_action: dict[str, Any]
    decision: Literal["ALLOW", "MODIFY", "BLOCK"]
    modified_action: dict[str, Any] | None = None
    risk_score: float
    rules_triggered: list[str]
    simulation_ref: str | None = None
    reason: str
    policy_version: str
    latency_ms: float


class ExecutedActionPayload(BaseModel):
    action_id: str
    action_type: str
    command: dict[str, Any]
    controller: str
    start_time_ns: int
    end_time_ns: int | None = None
    status: Literal["started", "completed", "failed", "blocked"]
    pre_state_ref: str | None = None
    post_state_ref: str | None = None
    reward: float | None = None
    error_code: str | None = None
    error_message: str | None = None


class FailureLabelPayload(BaseModel):
    failure_id: str
    failure_type: str
    severity: Literal["low", "medium", "high", "critical"]
    source: Literal["auto", "human", "sandbox", "runtime", "critic"]
    related_action_id: str | None = None
    related_event_ids: list[str] = Field(default_factory=list)
    description: str
    evidence_refs: list[str] = Field(default_factory=list)
    suggested_fix: dict[str, Any] | None = None


class HumanFeedbackPayload(BaseModel):
    feedback_id: str
    feedback_type: Literal[
        "thumbs_up",
        "thumbs_down",
        "correction",
        "label",
        "takeover",
        "success_confirm",
        "safety_concern",
    ]
    rating: int | None = None
    text: str | None = None
    related_event_ids: list[str] = Field(default_factory=list)
    related_action_id: str | None = None
    operator_id: str | None = None


# ---------------------------------------------------------------------------
# Body-agnostic physical feedback / contact / failure / intervention payloads
# ---------------------------------------------------------------------------


class PhysicalFeedbackPayload(BaseModel):
    """Payload for ``physical_feedback_event``.

    Mirrors the generic ``PhysicalFeedbackFrame`` but is expressed as a Pydantic
    model so it can be validated and serialized inside a Practice event envelope.
    """

    frame_id: str
    body_id: str
    timestamp: float

    target: dict[str, float | None] = Field(default_factory=dict)
    actual: dict[str, float | None] = Field(default_factory=dict)
    position_error: dict[str, float | None] = Field(default_factory=dict)

    force_raw: dict[str, float | None] = Field(default_factory=dict)
    force_baseline: dict[str, float | None] = Field(default_factory=dict)
    force_net: dict[str, float | None] = Field(default_factory=dict)
    force_delta: dict[str, float | None] = Field(default_factory=dict)
    force_derivative: dict[str, float | None] = Field(default_factory=dict)

    current: dict[str, float | None] = Field(default_factory=dict)
    status: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] = Field(default_factory=dict)
    temperature: dict[str, float | None] = Field(default_factory=dict)

    primary_event: str = "unknown"
    secondary_tags: list[str] = Field(default_factory=list)
    serial_timeout: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContactEventPayload(BaseModel):
    """Payload for ``contact_event``.

    Represents a single inferred contact event produced by a body-specific
    detector and mapped to the generic ``ContactEvent`` semantics.
    """

    contact_id: str
    event_type: str
    confidence: float = 1.0
    dofs: list[str] = Field(default_factory=list)
    timestamp: float
    force_net: dict[str, float | None] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureEventPayload(BaseModel):
    """Payload for ``failure_event``.

    A structured failure observation that can be distilled into SeekDB
    ``failures`` and linked to ``how_intervention`` records.
    """

    failure_id: str
    failure_type: str
    severity: Literal["low", "medium", "high", "critical"]
    source: Literal["auto", "human", "sandbox", "runtime", "critic"]
    description: str
    related_action_id: str | None = None
    related_event_ids: list[str] = Field(default_factory=list)
    related_contact_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    suggested_fix: dict[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class HowInterventionPayload(BaseModel):
    """Payload for ``how_intervention_event``.

    Captures the compensating action taken in response to a failure, e.g.
    increasing force setpoint, backing off, or switching pose.
    """

    intervention_id: str
    failure_id: str
    episode_id: str | None = None
    description: str
    action_taken: dict[str, Any]
    outcome: Literal["pending", "resolved", "partial", "failed"] = "pending"
    before_state_ref: str | None = None
    after_state_ref: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidatePolicyPayload(BaseModel):
    """Payload for ``candidate_policy_event``.

    Records a candidate policy discovered during practice (e.g. an OK-contact
    pose found by asymmetric search) along with the evidence that supports it.
    """

    candidate_id: str
    policy_id: str | None = None
    skill_id: str | None = None
    policy_type: str
    policy_params: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    promoted: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromotionResultPayload(BaseModel):
    """Payload for ``promotion_result_event``.

    Records the outcome of a promotion gate evaluation. ``passed`` indicates
    whether the candidate policy can be promoted to the active policy set.
    """

    promotion_id: str
    candidate_id: str
    policy_id: str
    passed: bool
    gate_name: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    promoted_policy_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeSummaryPayload(BaseModel):
    """Payload for ``episode_summary_event``.

    Aggregates an episode into a compact summary that can be written to the
    catalog and distilled into SeekDB without re-parsing the full JSONL.
    """

    episode_id: str
    session_id: str | None = None
    body_id: str | None = None
    skill_id: str | None = None
    policy_id: str | None = None
    outcome: Literal["success", "failure", "partial", "unknown"] = "unknown"
    success: bool | None = None
    failure_labels: list[str] = Field(default_factory=list)
    event_count: int = 0
    contact_event_distribution: dict[str, int] = Field(default_factory=dict)
    primary_event_distribution: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: dict[str, str] = Field(default_factory=dict)


class BodyCognitionPayload(BaseModel):
    """Payload for ``body_cognition_event``.

    Distilled body knowledge such as force-model updates, thermal limits, or
    sim2real deltas. Can be written both as an artifact and as a Practice event.
    """

    body_id: str
    cognition_id: str
    cognition_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    source_event_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Sim2RealDeltaPayload(BaseModel):
    """Payload for ``sim2real_delta_event``.

    Captures a measured delta between simulation expectation and real-body
    observation for one or more DOFs.
    """

    delta_id: str
    body_id: str
    dofs: list[str] = Field(default_factory=list)
    sim_value: dict[str, float | None] = Field(default_factory=dict)
    real_value: dict[str, float | None] = Field(default_factory=dict)
    delta: dict[str, float | None] = Field(default_factory=dict)
    unit: str = ""
    source_event_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HardwareTransportErrorPayload(BaseModel):
    """Payload for ``hardware_transport_error``.

    Records a low-level transport failure (USB/serial/CH340 drop, EIO,
    re-enumeration, lock conflict, etc.) so the Practice flywheel can
    distinguish hardware-link failures from policy failures.
    """

    transport: str
    port: str | None = None
    health_state: str = "UNKNOWN"
    errno: int | None = None
    errno_name: str | None = None
    description: str
    action: str = "stop_motion_and_require_manual_ack"
    dmesg_excerpt: str | None = None
    related_action_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
