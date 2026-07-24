"""Result types for the unified runtime retrieval facade (数据库优化v4 §3.2/§3.5).

Every retrieval that serves CLI, Runtime, KNOW, HOW, or AUTO goes through
:class:`rosclaw.memory.v2.runtime_retrieval.facade.MemoryRetrievalFacade` and
returns a :class:`RetrievalResponse`.  The response always discloses which
physical index answered (or which fallback ran, and why) so a BM25 fallback
can never masquerade as a Qwen3 hybrid result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rosclaw.memory.v2.models import MemoryItem


class RetrievalPurpose(StrEnum):
    """Why the caller is retrieving (数据库优化v4 §3.2).

    The purpose selects the safety policy of the retrieval itself — most
    importantly whether a starved same-body query may retry across bodies:

    - HUMAN_SEARCH: cross-body allowed, annotated as a reference.
    - MEMORY_CONTEXT: cross-body allowed but demoted.
    - KNOW_REASONING: cross-body forbidden by default.
    - HOW_INTERVENTION: cross-body forbidden; reranker required on the
      high-risk path (its absence is disclosed, never hidden).
    - AUTO_HYPOTHESIS: cross-body allowed as exploration, annotated.
    """

    HUMAN_SEARCH = "human_search"
    MEMORY_CONTEXT = "memory_context"
    KNOW_REASONING = "know_reasoning"
    HOW_INTERVENTION = "how_intervention"
    AUTO_HYPOTHESIS = "auto_hypothesis"


# Cross-body handling per purpose (v4 §3.2 table).
CROSS_BODY_ANNOTATE = "annotate"
CROSS_BODY_DEMOTE = "demote"
CROSS_BODY_FORBID = "forbid"


@dataclass(frozen=True)
class PurposePolicy:
    """Retrieval-time safety policy derived from the purpose."""

    purpose: RetrievalPurpose
    cross_body: str
    reranker_required: bool

    @property
    def cross_body_allowed(self) -> bool:
        return self.cross_body != CROSS_BODY_FORBID


def policy_for(purpose: RetrievalPurpose) -> PurposePolicy:
    if purpose is RetrievalPurpose.HOW_INTERVENTION:
        return PurposePolicy(purpose, CROSS_BODY_FORBID, reranker_required=True)
    if purpose is RetrievalPurpose.KNOW_REASONING:
        return PurposePolicy(purpose, CROSS_BODY_FORBID, reranker_required=False)
    if purpose is RetrievalPurpose.MEMORY_CONTEXT:
        return PurposePolicy(purpose, CROSS_BODY_DEMOTE, reranker_required=False)
    # HUMAN_SEARCH / AUTO_HYPOTHESIS
    return PurposePolicy(purpose, CROSS_BODY_ANNOTATE, reranker_required=False)


@dataclass
class RetrievalCandidate:
    """One ranked memory with entity-level match disclosure (v4 §3.5).

    ``vector_rank`` / ``bm25_rank`` are ``None`` when the serving engine only
    exposes the fused RRF order (SeekDB native hybrid search) — the fused
    rank is always present and ``score_semantics`` states exactly what the
    ranks mean.
    """

    memory_id: str
    memory_type: str

    vector_rank: int | None
    bm25_rank: int | None
    fusion_rank: int
    rerank_score: float | None

    exact_entity_match: bool
    body_match: bool | None
    joint_match: bool | None
    failure_type_match: bool | None

    physical_collection: str | None
    embedding_profile_id: str | None
    score_semantics: str

    cross_body_reference: bool = False
    item: MemoryItem | None = None

    def to_dict(self) -> dict[str, Any]:
        item = self.item
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "title": item.title if item else None,
            "document": item.document if item else None,
            "body_id": item.body_id if item else None,
            "joint_name": item.joint_name if item else None,
            "failure_type": item.failure_type if item else None,
            "outcome": item.outcome if item else None,
            "evidence_count": len(item.evidence_refs) if item else 0,
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
            "fusion_rank": self.fusion_rank,
            "rerank_score": self.rerank_score,
            "exact_entity_match": self.exact_entity_match,
            "body_match": self.body_match,
            "joint_match": self.joint_match,
            "failure_type_match": self.failure_type_match,
            "cross_body_reference": self.cross_body_reference,
            "physical_collection": self.physical_collection,
            "embedding_profile_id": self.embedding_profile_id,
            "score_semantics": self.score_semantics,
        }


@dataclass
class RetrievalAttempt:
    """One step of the fallback chain, for explainability (v4 §3.7)."""

    mode: str
    ok: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "ok": self.ok, "reason": self.reason}


@dataclass
class RetrievalResponse:
    """Facade output: candidates plus the full truth block (v4 §3.7)."""

    retrieval_mode: str
    logical_collection: str
    physical_collection: str | None
    embedding_profile_id: str | None
    purpose: RetrievalPurpose
    fallback: bool
    fallback_reason: str | None
    candidates: list[RetrievalCandidate] = field(default_factory=list)
    attempts: list[RetrievalAttempt] = field(default_factory=list)
    reranker_applied: bool = False
    reranker_required: bool = False

    @property
    def score_semantics(self) -> str:
        """Response-level score semantics (never borrowed from another backend)."""
        if self.candidates:
            return self.candidates[0].score_semantics
        return "no candidates returned"

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_mode": self.retrieval_mode,
            "logical_collection": self.logical_collection,
            "physical_collection": self.physical_collection,
            "embedding_profile_id": self.embedding_profile_id,
            "purpose": self.purpose.value,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "reranker_applied": self.reranker_applied,
            "reranker_required": self.reranker_required,
            "score_semantics": self.score_semantics,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "results": [candidate.to_dict() for candidate in self.candidates],
        }
