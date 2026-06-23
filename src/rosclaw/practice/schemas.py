"""Practice data schemas for the Physical Data Flywheel Runtime.

This module defines the canonical event envelope and payload models used by
`rosclaw-practice`. All JSONL/MCAP/SeekDB events share the same envelope so
that downstream tools can query, replay, and export practices without
source-specific parsing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

SCHEMA_VERSION = "practice.event.v1"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _now_ns() -> int:
    return int(datetime.now(UTC).timestamp() * 1_000_000_000)


class PracticeEventEnvelope(BaseModel):
    """Unified envelope for every event captured by rosclaw-practice.

    Large or binary payloads should be written to files and referenced via
    ``payload_ref``; only small serializable metadata belongs in ``payload``.
    """

    schema_version: str = SCHEMA_VERSION

    practice_id: str
    session_id: str | None = None
    robot_id: str

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
    event_id: str = Field(default_factory=lambda: str(uuid4()))

    frame_id: str | None = None
    task_id: str | None = None
    skill_id: str | None = None
    action_id: str | None = None

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
