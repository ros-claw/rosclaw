"""Trusted-source adapters for the ContextCompiler (PR-NA-020).

Every source returns *facts with provenance*, never model output. Sources are
small protocols so tests can substitute fakes; production adapters wrap
``rosclaw.body``, ``rosclaw.self_model``, daemon public status and the
provider/skill registries.

Evidence classes (总纲 §5.4 step 7, ordered strongest first):
``measured > verified_receipt > curated > inferred > unverified``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol


class EvidenceClass(StrEnum):
    MEASURED = "measured"
    VERIFIED_RECEIPT = "verified_receipt"
    CURATED = "curated"
    INFERRED = "inferred"
    UNVERIFIED = "unverified"


EVIDENCE_RANK: dict[EvidenceClass, int] = {
    EvidenceClass.MEASURED: 0,
    EvidenceClass.VERIFIED_RECEIPT: 1,
    EvidenceClass.CURATED: 2,
    EvidenceClass.INFERRED: 3,
    EvidenceClass.UNVERIFIED: 4,
}


@dataclass(frozen=True)
class BodyFacts:
    body_id: str
    effective_body_hash: str
    summary: str
    calibrated: bool = True
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelfFacts:
    self_snapshot_hash: str
    sequence: int
    observed_at: datetime
    health: str = "OK"
    summary: str = ""


@dataclass(frozen=True)
class CapabilityInfo:
    name: str
    kind: str = "tool"  # tool | skill | provider
    summary: str = ""
    schema_ref: str | None = None
    permission: str = "granted"  # granted | operator_only | denied | unknown
    priority: int = 0  # deterministic context-selection priority, never a permission bypass


@dataclass(frozen=True)
class MemoryItem:
    ref: str
    summary: str
    evidence_class: EvidenceClass = EvidenceClass.UNVERIFIED
    body_scope: str | None = None


@dataclass(frozen=True)
class OrgFacts:
    workers_summary: str = ""
    team_id: str | None = None
    team_epoch: int = 0
    world_revision: int = 0
    world_fresh_at: datetime | None = None


@dataclass(frozen=True)
class ConsentFacts:
    policy_hash: str
    mission_grant_public_hash: str | None = None
    public_scope_summary: str = ""
    allowed_risk_tiers: tuple[str, ...] = ("LOW",)


@dataclass(frozen=True)
class ConversationMessage:
    role: str  # user | assistant | worker | artifact
    content: str
    source: str = "user"
    ref: str | None = None


class BodySource(Protocol):
    def get_body(self, body_id: str) -> BodyFacts | None: ...


class SelfSource(Protocol):
    def get_self(self, body_id: str) -> SelfFacts | None: ...


class CapabilitySource(Protocol):
    def list_capabilities(self, query: str, limit: int) -> list[CapabilityInfo]: ...


class MemorySource(Protocol):
    def retrieve(self, query: str, limit: int) -> list[MemoryItem]: ...


class OrganizationSource(Protocol):
    def get_org(self) -> OrgFacts: ...


class ConsentSource(Protocol):
    def get_consent(self, mission_id: str) -> ConsentFacts | None: ...


@dataclass
class SourceBundle:
    """All sources the compiler reads. ``constitution_text`` is the versioned
    system policy (L0); everything else is a protocol adapter."""

    constitution_text: str
    body: BodySource
    self_source: SelfSource
    capabilities: CapabilitySource
    memory: MemorySource
    organization: OrganizationSource
    consent: ConsentSource
    runtime_status_summary: str = ""
    extra: dict = field(default_factory=dict)
