"""Memory 2.0 unified retrieval API (§6.2, §6.3, §6.8).

Pipeline::

    Query Normalization
        ↓
    Metadata Pre-filter
        ↓
    Lexical Candidate Retrieval  +  Vector Candidate Retrieval
        ↓
    Score Normalization
        ↓
    Fusion (weighted, explainable)
        ↓
    Safety/Validity Filter
        ↓
    Top-K + Explanation

Score semantics are explicit: every :class:`RetrievalResult` carries the
per-source scores and the fusion weights that produced ``fusion_score`` —
"RRF 分数不再冒充 Cosine Similarity".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from rosclaw.memory.v2.models import MemoryItem
from rosclaw.memory.v2.repository import MemoryRepository
from rosclaw.memory.v2.tokenizer import token_set

logger = logging.getLogger("rosclaw.memory.v2.retrieval")

# Initial fusion weights (§6.3) — must be tuned by the memory benchmark.
DEFAULT_FUSION_WEIGHTS = {
    "vector": 0.35,
    "lexical": 0.20,
    "metadata": 0.15,
    "recency": 0.10,
    "evidence": 0.10,
    "confidence": 0.10,
}


@dataclass
class MemoryQuery:
    """Structured memory query (§6.2)."""

    text: str = ""
    memory_types: list[str] = field(default_factory=list)
    robot_id: str | None = None
    body_id: str | None = None
    task_id: str | None = None
    skill_id: str | None = None
    outcome: str | None = None
    time_range: tuple[float, float] | None = None
    minimum_confidence: float = 0.0
    limit: int = 10


@dataclass
class RetrievalResult:
    """One ranked memory with fully explainable scores (§6.2)."""

    memory_id: str
    memory: MemoryItem

    vector_score: float | None
    lexical_score: float | None
    metadata_score: float | None
    temporal_score: float | None
    evidence_score: float | None
    fusion_score: float

    rank: int
    score_type: str
    explanation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory.memory_type,
            "title": self.memory.title,
            "vector_score": self.vector_score,
            "lexical_score": self.lexical_score,
            "metadata_score": self.metadata_score,
            "temporal_score": self.temporal_score,
            "evidence_score": self.evidence_score,
            "fusion_score": self.fusion_score,
            "rank": self.rank,
            "score_type": self.score_type,
            "explanation": self.explanation,
        }


class MemoryRetriever:
    """Hybrid retrieval over :class:`MemoryRepository` with optional vector search."""

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        vector_store: Any | None = None,
        embedder: Any | None = None,
        fusion_weights: dict[str, float] | None = None,
        recency_half_life_days: float = 30.0,
    ):
        self._repo = repository
        self._vector_store = vector_store
        self._embedder = embedder
        self._weights = dict(fusion_weights or DEFAULT_FUSION_WEIGHTS)
        self._recency_half_life_s = recency_half_life_days * 86400.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: MemoryQuery) -> list[RetrievalResult]:
        """Run the full retrieval pipeline and return ranked, explained results."""
        # 1. Query normalization (bilingual tokenization).
        query_tokens = token_set(query.text)

        # 2. Metadata pre-filter (hard constraints, incl. cross-robot isolation).
        candidates = self._metadata_prefilter(query)
        if not candidates:
            return []

        # 3. Candidate scoring per source.
        lexical = self._lexical_scores(candidates, query_tokens)
        vector = self._vector_scores(candidates, query.text)

        # 4-5. Normalize + fuse.
        now = time.time()
        scored: list[tuple[MemoryItem, dict[str, Any]]] = []
        for item in candidates:
            parts = self._score_parts(item, query, query_tokens, lexical, vector, now)
            fusion = sum(self._weights[key] * parts[key] for key in self._weights)
            scored.append((item, {"parts": parts, "fusion": fusion}))

        # 6. Safety/validity filter.
        scored = [(item, meta) for item, meta in scored if self._passes_validity(item, meta)]

        # 7. Rank + explain.
        scored.sort(key=lambda pair: pair[1]["fusion"], reverse=True)
        results: list[RetrievalResult] = []
        for rank, (item, meta) in enumerate(scored[: query.limit], start=1):
            parts = meta["parts"]
            score_type = self._score_type(parts)
            results.append(
                RetrievalResult(
                    memory_id=item.memory_id,
                    memory=item,
                    vector_score=parts.get("vector"),
                    lexical_score=parts.get("lexical"),
                    metadata_score=parts.get("metadata"),
                    temporal_score=parts.get("recency"),
                    evidence_score=parts.get("evidence"),
                    fusion_score=meta["fusion"],
                    rank=rank,
                    score_type=score_type,
                    explanation=self._explain(item, query, parts, meta["fusion"], rank),
                )
            )
        return results

    def explain(self, memory_id: str, query: MemoryQuery) -> dict[str, Any]:
        """Explain why *memory_id* ranks where it does for *query*."""
        results = self.retrieve(MemoryQuery(**{**vars(query), "limit": 500}))
        for result in results:
            if result.memory_id == memory_id:
                return result.to_dict()
        return {"memory_id": memory_id, "found": False, "reason": "not in result set"}

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _metadata_prefilter(self, query: MemoryQuery) -> list[MemoryItem]:
        filters: dict[str, Any] = {}
        if query.robot_id:
            filters["robot_id"] = query.robot_id  # hard cross-robot isolation
        if query.body_id:
            filters["body_id"] = query.body_id
        if query.task_id:
            filters["task_id"] = query.task_id
        if query.skill_id:
            filters["skill_id"] = query.skill_id
        if query.outcome:
            filters["outcome"] = query.outcome
        candidates = self._repo.query(filters, limit=1000)
        if query.memory_types:
            allowed = set(query.memory_types)
            candidates = [item for item in candidates if item.memory_type in allowed]
        if query.time_range:
            start, end = query.time_range
            candidates = [item for item in candidates if start <= item.event_time <= end]
        if query.minimum_confidence > 0:
            candidates = [
                item for item in candidates if item.confidence >= query.minimum_confidence
            ]
        return candidates

    def _lexical_scores(
        self, candidates: list[MemoryItem], query_tokens: set[str]
    ) -> dict[str, float]:
        """Token-overlap score in [0, 1] with a title-match boost."""
        scores: dict[str, float] = {}
        if not query_tokens:
            return {item.memory_id: 0.0 for item in candidates}
        for item in candidates:
            doc_tokens = token_set(f"{item.title} {item.document}")
            if not doc_tokens:
                scores[item.memory_id] = 0.0
                continue
            overlap = len(query_tokens & doc_tokens) / len(query_tokens)
            title_tokens = token_set(item.title)
            title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
            scores[item.memory_id] = min(1.0, overlap + 0.5 * title_overlap)
        return scores

    def _vector_scores(
        self, candidates: list[MemoryItem], query_text: str
    ) -> dict[str, float] | None:
        if self._vector_store is None or self._embedder is None or not query_text.strip():
            return None
        try:
            embedding = self._embedder.encode(query_text)
            hits = self._vector_store.search(
                "memory_items", embedding, limit=max(50, len(candidates))
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector search failed; falling back to lexical only: %s", exc)
            return None
        candidate_ids = {item.memory_id for item in candidates}
        scores: dict[str, float] = {}
        for hit in hits:
            record_id = hit.get("record_id") or hit.get("id")
            if record_id in candidate_ids:
                scores[record_id] = float(hit.get("score", 0.0))
        return scores

    def _score_parts(
        self,
        item: MemoryItem,
        query: MemoryQuery,
        query_tokens: set[str],
        lexical: dict[str, float],
        vector: dict[str, float] | None,
        now: float,
    ) -> dict[str, float]:
        parts: dict[str, float] = {}
        parts["vector"] = (vector or {}).get(item.memory_id, 0.0)
        parts["lexical"] = lexical.get(item.memory_id, 0.0)

        metadata = 0.0
        if query.task_id and item.task_id == query.task_id:
            metadata += 0.5
        if query.body_id and item.body_id == query.body_id:
            metadata += 0.5
        if not (query.task_id or query.body_id):
            metadata = 0.5  # neutral when no context given
        parts["metadata"] = metadata

        age_s = max(now - item.event_time, 0.0)
        parts["recency"] = (
            0.5 ** (age_s / self._recency_half_life_s) if self._recency_half_life_s > 0 else 1.0
        )

        parts["evidence"] = min(1.0, len(item.evidence_refs) / 3.0)
        parts["confidence"] = min(1.0, (item.confidence + item.importance) / 2.0)
        return parts

    def _passes_validity(self, item: MemoryItem, meta: dict[str, Any]) -> bool:
        """Safety/validity filter (§6.8): expired and zero-relevance items drop."""
        if item.expires_at and item.expires_at <= time.time():
            meta["filtered"] = "expired"
            return False
        if meta["fusion"] <= 0.0:
            meta["filtered"] = "zero fusion score"
            return False
        return True

    def _score_type(self, parts: dict[str, float]) -> str:
        used = [name for name in ("vector", "lexical") if parts.get(name, 0.0) > 0.0]
        return "+".join(used) if used else "metadata"

    def _explain(
        self,
        item: MemoryItem,
        query: MemoryQuery,
        parts: dict[str, float],
        fusion: float,
        rank: int,
    ) -> dict[str, Any]:
        return {
            "fusion_weights": dict(self._weights),
            "score_parts": parts,
            "fusion_score": fusion,
            "rank": rank,
            "query_tokens": sorted(token_set(query.text))[:20],
            "matched_filters": {
                key: value
                for key, value in (
                    ("robot_id", query.robot_id),
                    ("body_id", query.body_id),
                    ("task_id", query.task_id),
                    ("skill_id", query.skill_id),
                    ("outcome", query.outcome),
                )
                if value is not None
            },
            "evidence_count": len(item.evidence_refs),
            "age_days": round((time.time() - item.event_time) / 86400.0, 1),
            "safety_notice": "safety" in item.tags or bool(item.metadata.get("safety")),
        }


# ---------------------------------------------------------------------------
# Safety retrieval policy (§6.8)
# ---------------------------------------------------------------------------

# Priority tiers for knowledge entering KNOW/HOW decision paths.
TIER_HARD_SAFETY = 0  # S4/S3 hard safety rules
TIER_BODY_LIMIT = 1  # deterministic body limits
TIER_HOW_VERIFIED = 2  # verified HOW rules
TIER_KNOW_CURATED = 3  # curated KNOW patterns
TIER_MEMORY = 4  # similar memory experience
TIER_LLM = 5  # LLM suggestions


@dataclass
class PolicyDecision:
    """Result of passing one retrieval result through the safety policy."""

    allowed: bool
    tier: int
    reason: str


class SafetyRetrievalPolicy:
    """Gate retrieval results before they reach KNOW/HOW (§6.8).

    Chain: Hard Safety Rule → Validity → Evidence → Applicability → Memory.

    Memory never overrides hard safety rules: results only enter the
    ``TIER_MEMORY`` tier; the caller (HOW/KNOW) is responsible for ordering
    them below verified rules and body limits.
    """

    def __init__(self, *, min_evidence: int = 1):
        self._min_evidence = min_evidence

    def evaluate(self, result: RetrievalResult, query: MemoryQuery) -> PolicyDecision:
        item = result.memory
        # Validity: expired/quarantined/superseded memories never pass.
        if item.status != "active":
            return PolicyDecision(False, TIER_MEMORY, f"status={item.status}")
        if item.expires_at and item.expires_at <= time.time():
            return PolicyDecision(False, TIER_MEMORY, "expired")
        # Evidence: HOW-impacting types need evidence.
        if (
            item.memory_type in {"intervention", "failure"}
            and len(item.evidence_refs) < self._min_evidence
        ):
            return PolicyDecision(False, TIER_MEMORY, "insufficient evidence")
        # Applicability: body-specific memory must match the caller's body.
        if query.body_id and item.body_id and item.body_id != query.body_id:
            return PolicyDecision(False, TIER_MEMORY, f"body mismatch: {item.body_id}")
        return PolicyDecision(True, TIER_MEMORY, "eligible memory candidate")

    def filter(self, results: list[RetrievalResult], query: MemoryQuery) -> list[RetrievalResult]:
        """Drop results that fail the policy; never re-orders hard rules."""
        return [r for r in results if self.evaluate(r, query).allowed]
