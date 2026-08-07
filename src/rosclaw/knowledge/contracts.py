"""Dependency-light wire contracts shared with rosclaw-know/how v2.

The authoritative schemas live in ``rosclaw-know``.  These deliberately small
copies let Core validate service traffic without importing either optional
package or their retrieval dependencies.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictWireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, validate_assignment=True, str_strip_whitespace=True
    )
    SCHEMA_VERSION: ClassVar[str | None] = None

    @classmethod
    def validate_wire_json(cls, payload: str | bytes):
        return cls.model_validate_json(payload)

    def to_wire_json(self) -> str:
        return self.model_dump_json(exclude_none=True)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return value


class ResearchConstraintsV2(StrictWireModel):
    robot_model: str | None = None
    simulator: str | None = None
    ros_distro: str | None = None
    language: list[str] = Field(default_factory=list)
    date_after: str | None = None
    software_versions: dict[str, str] = Field(default_factory=dict)


class ResearchRequestV2(StrictWireModel):
    SCHEMA_VERSION: ClassVar[str] = "rosclaw.know.research_request.v2"

    schema_version: Literal["rosclaw.know.research_request.v2"] = SCHEMA_VERSION
    request_id: str = Field(min_length=1, max_length=200)
    topic: str = Field(min_length=1, max_length=2000)
    goal: str = Field(min_length=1, max_length=4000)
    perspectives: list[str] = Field(default_factory=list, max_length=20)
    source_types: list[
        Literal[
            "repository",
            "paper",
            "official_documentation",
            "issue",
            "pull_request",
            "release",
            "web",
        ]
    ] = Field(default_factory=list, max_length=20)
    constraints: ResearchConstraintsV2 = Field(default_factory=ResearchConstraintsV2)
    depth: Literal["shallow", "standard", "deep"] = "standard"
    max_sources: int = Field(default=20, ge=1, le=200)
    token_budget: int = Field(default=50_000, ge=1_000, le=2_000_000)


class EvidenceRefV2(StrictWireModel):
    SCHEMA_VERSION: ClassVar[str] = "rosclaw.know.evidence_ref.v2"

    schema_version: Literal["rosclaw.know.evidence_ref.v2"] = SCHEMA_VERSION
    evidence_id: str = Field(min_length=1, max_length=240)
    source_id: str = Field(min_length=1, max_length=200)
    snapshot_id: str = Field(min_length=1, max_length=240)
    document_id: str = Field(min_length=1, max_length=240)
    path: str = Field(min_length=1, max_length=4000)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    section: str | None = None
    url: str = Field(min_length=1, max_length=4000)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    excerpt: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _line_range(self):
        if self.end_line is not None and self.start_line is None:
            raise ValueError("end_line requires start_line")
        if self.start_line and self.end_line and self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class ReferenceContextV2(StrictWireModel):
    task: str | None = None
    robot: str | None = None
    simulator: str | None = None
    ros_distro: str | None = None
    software_versions: dict[str, str] = Field(default_factory=dict)
    current_stage: str | None = None
    current_failure: str | None = None


class ReferencePackItemV2(StrictWireModel):
    rank: int = Field(ge=1)
    project_id: str | None = None
    knowledge_unit_ids: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    why_relevant: str = Field(min_length=1)
    relevance_dimensions: list[str] = Field(default_factory=list)
    mechanism: str = Field(min_length=1)
    what_to_borrow: list[str] = Field(default_factory=list)
    exact_files: list[str] = Field(default_factory=list)
    exact_sections: list[str] = Field(default_factory=list)
    incompatibilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    adaptation_needed: list[str] = Field(default_factory=list)
    source_version: str = Field(min_length=1)
    evidence_refs: list[EvidenceRefV2] = Field(min_length=1)
    score: float | None = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class ReferenceComparisonV2(StrictWireModel):
    shared_principles: list[str] = Field(default_factory=list)
    conflicting_assumptions: list[str] = Field(default_factory=list)
    route_tradeoffs: list[str] = Field(default_factory=list)
    preferred_references: list[str] = Field(default_factory=list)


class ReferencePackV2(StrictWireModel):
    SCHEMA_VERSION: ClassVar[str] = "rosclaw.know.reference_pack.v2"

    schema_version: Literal["rosclaw.know.reference_pack.v2"] = SCHEMA_VERSION
    reference_pack_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    context: ReferenceContextV2
    generated_at: datetime
    index_version: str = Field(min_length=1)
    items: list[ReferencePackItemV2] = Field(default_factory=list)
    comparison: ReferenceComparisonV2 = Field(default_factory=ReferenceComparisonV2)
    recommended_reading_order: list[str] = Field(default_factory=list)
    suggested_next_checks: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    token_budget: int = Field(ge=1)
    truncated: bool = False
    continuation_cursor: str | None = None
    warnings: list[str] = Field(default_factory=list)
    cached: bool = False
    stale: bool = False
    cache_age_seconds: int = Field(default=0, ge=0)

    _generated_aware = field_validator("generated_at")(_aware)

    @model_validator(mode="after")
    def _cursor_when_truncated(self):
        if self.truncated and not self.continuation_cursor:
            raise ValueError("truncated packs require a continuation_cursor")
        if self.stale and not self.cached:
            raise ValueError("stale Reference Packs must also be marked cached")
        if (self.cached or self.stale) and not self.warnings:
            raise ValueError("cached Reference Packs must explain degradation in warnings")
        return self


class AdviceRecommendationV2(StrictWireModel):
    action_type: Literal["inspect", "compare", "configure", "implement", "verify", "abstain"]
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    knowledge_unit_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceRefV2] = Field(default_factory=list)
    safety_class: Literal["advisory"] = "advisory"


class HowAdviceBundleV2(StrictWireModel):
    SCHEMA_VERSION: ClassVar[str] = "rosclaw.how.advice.v2"

    schema_version: Literal["rosclaw.how.advice.v2"] = SCHEMA_VERSION
    advice_id: str = Field(min_length=1)
    mode: Literal["discover", "consult", "diagnose", "catalyze"]
    context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_pack_id: str | None = None
    reference_pack_cached: bool = False
    reference_pack_stale: bool = False
    reference_pack_age_seconds: int = Field(default=0, ge=0)
    summary: str = Field(min_length=1)
    diagnosis: str | None = None
    recommendations: list[AdviceRecommendationV2] = Field(default_factory=list)
    compatibility_warnings: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: str | None = None
    created_at: datetime

    _created_aware = field_validator("created_at")(_aware)

    @model_validator(mode="after")
    def _abstention_explained(self):
        if self.abstained and not self.abstention_reason:
            raise ValueError("abstained advice requires abstention_reason")
        if self.reference_pack_stale and not self.reference_pack_cached:
            raise ValueError("stale advice must identify a cached Reference Pack")
        return self


class KnowledgeUsageFeedbackV1(StrictWireModel):
    SCHEMA_VERSION: ClassVar[str] = "rosclaw.knowledge_usage_feedback.v1"

    schema_version: Literal["rosclaw.knowledge_usage_feedback.v1"] = SCHEMA_VERSION
    feedback_id: str = Field(min_length=1)
    reference_pack_id: str = Field(min_length=1)
    advice_id: str | None = None
    knowledge_unit_id: str = Field(min_length=1)
    presented: bool = True
    opened: bool = False
    used_by_agent: bool = False
    verdict: Literal["useful", "irrelevant", "stale", "incompatible", "misleading", "unknown"]
    reason: str | None = None
    context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_ref: str | None = None
    practice_ref: str | None = None
    origin: Literal["user", "agent", "verifier"]
    created_at: datetime

    _created_aware = field_validator("created_at")(_aware)


_SECRET_RE = re.compile(r"(?i)(api[_-]?key|token|password|authorization)\s*[:=]\s*[^\s,;]+")


class BodyContextV2(StrictWireModel):
    robot_model: str | None = None
    robot_type: str | None = None
    sensors: list[str] = Field(default_factory=list)
    actuators: list[str] = Field(default_factory=list)
    safety_limits: list[str] = Field(default_factory=list)


class SoftwareContextV2(StrictWireModel):
    ros_distro: str | None = None
    simulator: str | None = None
    versions: dict[str, str] = Field(default_factory=dict)
    hardware_architecture: str | None = None


class RuntimeContextV2(StrictWireModel):
    task: str
    current_stage: str | None = None
    current_failure: str | None = None
    error_log: str | None = Field(default=None, max_length=20_000)
    verifier_signals: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("error_log")
    @classmethod
    def _redact_secrets(cls, value: str | None) -> str | None:
        return _SECRET_RE.sub(r"\1=[REDACTED]", value) if value else value


class MemoryEvidenceV2(StrictWireModel):
    evidence_domain: Literal["memory"] = "memory"
    memory_id: str
    summary: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)
    receipt_ref: str | None = None
    practice_ref: str | None = None
    created_at: datetime

    _memory_created_aware = field_validator("created_at")(_aware)


class HowContextV2(StrictWireModel):
    body: BodyContextV2 = Field(default_factory=BodyContextV2)
    software: SoftwareContextV2 = Field(default_factory=SoftwareContextV2)
    runtime: RuntimeContextV2
    memory_evidence: list[MemoryEvidenceV2] = Field(default_factory=list, max_length=20)

    def context_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class HowAdviceRequestV2(StrictWireModel):
    SCHEMA_VERSION: ClassVar[str] = "rosclaw.how.advice_request.v2"

    schema_version: Literal["rosclaw.how.advice_request.v2"] = SCHEMA_VERSION
    request_id: str
    mode: Literal["discover", "consult", "diagnose", "catalyze"]
    query: str = Field(min_length=1, max_length=20_000)
    context: HowContextV2
    top_k: int = Field(default=8, ge=1, le=100)
    token_budget: int = Field(default=8_000, ge=1, le=500_000)


PUBLIC_CONTRACTS = (
    ResearchRequestV2,
    EvidenceRefV2,
    ReferencePackV2,
    HowAdviceBundleV2,
    KnowledgeUsageFeedbackV1,
    HowAdviceRequestV2,
)
