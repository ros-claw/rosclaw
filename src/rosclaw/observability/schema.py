"""Canonical ROSClaw Trace v1 schema."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class SpanKind(StrEnum):
    """Physical-AI operation categories used by the timeline and span tree."""

    MISSION = "MISSION"
    AGENT = "AGENT"
    PERCEPTION = "PERCEPTION"
    CONTEXT = "CONTEXT"
    PROMPT = "PROMPT"
    LLM = "LLM"
    VLM = "VLM"
    WORLD_MODEL = "WORLD_MODEL"
    PLANNER = "PLANNER"
    TOOL = "TOOL"
    MCP = "MCP"
    SKILL = "SKILL"
    SANDBOX = "SANDBOX"
    GUARDRAIL = "GUARDRAIL"
    ROBOT_ACTION = "ROBOT_ACTION"
    ROBOT_STATE = "ROBOT_STATE"
    CRITIC = "CRITIC"
    MEMORY = "MEMORY"
    RECOVERY = "RECOVERY"
    DATA_PIPELINE = "DATA_PIPELINE"


class SpanStatus(StrEnum):
    """Terminal and in-flight span states."""

    RUNNING = "RUNNING"
    OK = "OK"
    ERROR = "ERROR"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class CaptureMode(StrEnum):
    """How much input/output detail a trace may retain."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    RESEARCH = "research"


@dataclass
class ObservabilityConfig:
    """Runtime-independent trace capture and persistence configuration."""

    enabled: bool = True
    capture_mode: CaptureMode | str = CaptureMode.STANDARD
    home: str | Path | None = None
    filename: str = "live.jsonl"
    queue_size: int = 4096
    rotate_mb: float = 64.0
    max_text_chars: int = 16_384
    max_collection_items: int = 256

    def __post_init__(self) -> None:
        self.capture_mode = CaptureMode(self.capture_mode)
        if self.queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if self.rotate_mb < 0:
            raise ValueError("rotate_mb cannot be negative")


@dataclass
class DecisionSummary:
    """Auditable decision evidence; intentionally not private chain-of-thought."""

    goal: str
    observations: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    decision: Any | None = None
    reason_summary: str = ""
    confidence: float | None = None
    evidence_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceRecord:
    """One completed (or live-start) operation in a ROSClaw trace tree."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    span_kind: str
    source: str
    operation: str
    started_at: float
    status: str
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:16]}")
    schema_version: str = "rosclaw.trace.v1"
    record_type: str = "span"
    ended_at: float | None = None
    duration_ms: float | None = None
    severity: str = "INFO"
    mission_id: str | None = None
    episode_id: str | None = None
    robot_id: str | None = None
    session_id: str | None = None
    input: Any | None = None
    output: Any | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    privacy_class: str = "redacted"
    payload_hash: str | None = None
    error: dict[str, Any] | None = None
    emitted_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)
