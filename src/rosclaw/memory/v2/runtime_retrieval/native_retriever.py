"""Versioned native retriever — queries the ACTIVE physical collection (v4 §3.5).

Pipeline::

    Query Normalization
    → Exact Entity Extraction
    → Hard Metadata Filter
    → Qwen Query Embedding (or BM25-only when provider is None)
    → ACTIVE Collection Hybrid
    → Exact Entity Validation
    → Optional Reranker
    → Candidates

Cross-body behavior is purpose-driven (v4 §3.2/§7.4): when the query names
exactly one hand and the body-constrained query starves, ``HUMAN_SEARCH`` /
``AUTO_HYPOTHESIS`` may retry unfiltered with the cross-body rows annotated,
``MEMORY_CONTEXT`` demotes them, and ``KNOW_REASONING`` / ``HOW_INTERVENTION``
never retry — an automatic intervention path must not borrow another body's
experience.
"""

from __future__ import annotations

import logging
from typing import Any

from rosclaw.embedding.protocol import EmbeddingProvider
from rosclaw.memory.v2.document import extract_exact_terms
from rosclaw.memory.v2.models import MemoryItem
from rosclaw.memory.v2.retrieval import MemoryQuery
from rosclaw.storage.versioned_collections import _exact_row_multiplier

from .active_resolver import ActiveIndexDescriptor
from .result import (
    CROSS_BODY_ANNOTATE,
    CROSS_BODY_FORBID,
    PurposePolicy,
    RetrievalCandidate,
)

logger = logging.getLogger("rosclaw.memory.v2.runtime_retrieval.native_retriever")

SCORE_SEMANTICS_HYBRID = (
    "native RRF over BM25 + manual-embedding KNN legs (rank_constant=60), "
    "exact-entity multipliers applied post-fusion; ranks, not similarities"
)
SCORE_SEMANTICS_BM25 = (
    "native BM25 + metadata filter only (embedding provider unavailable); ranks, not similarities"
)


def _hard_filters(query: MemoryQuery) -> dict[str, Any]:
    filters: dict[str, Any] = {"status": "active"}
    for key in (
        "tenant_id",
        "project_id",
        "site_id",
        "robot_id",
        "body_id",
        "task_id",
        "skill_id",
        "outcome",
    ):
        value = getattr(query, key, None)
        if value:
            filters[key] = value
    return filters


def _entity_matches(row: dict[str, Any], exact: dict[str, list[str]]) -> dict[str, bool | None]:
    """Per-entity match disclosure: True/False, or None when the query did not
    name that entity or the row is honestly unattributed (never treated as a
    match — v4 §4.4 'unknown is not wildcard')."""
    named_values: list[bool | None] = []
    joints = exact.get("joints") or []
    joint_match: bool | None = None
    if len(joints) == 1:
        joint = row.get("joint_name")
        joint_match = None if joint is None else joint == joints[0]
        named_values.append(joint_match)
    failure_types = exact.get("failure_types") or []
    failure_match: bool | None = None
    if len(failure_types) == 1:
        ftype = row.get("failure_type")
        failure_match = None if ftype is None else ftype == failure_types[0]
        named_values.append(failure_match)
    hands = exact.get("hands") or []
    body_match: bool | None = None
    wanted_body: str | None = None
    if len(hands) == 1:
        wanted_body = f"rh56_{hands[0]}_01"
        body = row.get("body_id")
        body_match = None if body is None else body == wanted_body
        named_values.append(body_match)
    elif exact.get("bodies"):
        wanted_body = exact["bodies"][0]
        body = row.get("body_id")
        body_match = None if body is None else body == wanted_body
        named_values.append(body_match)
    # Exact-entity match requires EVERY entity the query named to be matched;
    # an unattributed row (None) is not a match — unknown is not wildcard.
    return {
        "joint_match": joint_match,
        "failure_type_match": failure_match,
        "body_match": body_match,
        "wanted_body": wanted_body,
        "exact_entity_match": bool(named_values) and all(v is True for v in named_values),
    }


class VersionedNativeRetriever:
    """Hybrid retrieval against one ACTIVE physical collection."""

    def __init__(self, store: Any, *, reranker: Any | None = None) -> None:
        self._store = store
        self._reranker = reranker

    def retrieve(
        self,
        query: MemoryQuery,
        descriptor: ActiveIndexDescriptor,
        provider: EmbeddingProvider | None,
        *,
        policy: PurposePolicy,
    ) -> tuple[list[RetrievalCandidate], dict[str, Any]]:
        """Returns ``(candidates, notes)``; notes disclose mode/cross-body/reranker."""
        text = query.text or ""
        exact = extract_exact_terms(text)
        filters = _hard_filters(query)
        limit = max(1, query.limit)
        window = max(20, limit * 4)
        hands = exact.get("hands") or []

        notes: dict[str, Any] = {
            "vector": provider is not None,
            "cross_body_retry": "not_needed",
            "reranker_applied": False,
        }

        constrained = False
        if len(hands) == 1 and "body_id" not in filters:
            filters["body_id"] = f"rh56_{hands[0]}_01"
            constrained = True

        rows = self._search(descriptor.physical_collection, text, filters, limit, window, provider)

        if constrained and not rows:
            if not policy.cross_body_allowed:
                notes["cross_body_retry"] = "disabled_for_purpose"
            else:
                unfiltered = dict(filters)
                unfiltered.pop("body_id", None)
                rows = self._search(
                    descriptor.physical_collection, text, unfiltered, limit, window, provider
                )
                notes["cross_body_retry"] = policy.cross_body  # annotate | demote

        candidates = self._candidates(rows, exact, descriptor, provider, policy)
        if self._reranker is not None and candidates and text.strip():
            try:
                documents = [c.item.document if c.item else "" for c in candidates]
                scores = self._reranker.score(text, documents)
                for candidate, score in zip(candidates, scores, strict=False):
                    candidate.rerank_score = float(score)
                candidates.sort(
                    key=lambda c: (c.rerank_score is not None, c.rerank_score, -c.fusion_rank),
                    reverse=True,
                )
                for rank, candidate in enumerate(candidates, start=1):
                    candidate.fusion_rank = rank
                notes["reranker_applied"] = True
            except Exception as exc:  # noqa: BLE001
                # The reranker is an optional re-ordering layer; a failure
                # must not drop the already-retrieved candidates.
                logger.warning("reranker failed; keeping fused order: %s", exc)
                notes["reranker_error"] = str(exc)
        return candidates, notes

    # ------------------------------------------------------------------

    def _search(
        self,
        collection: str,
        text: str,
        filters: dict[str, Any],
        limit: int,
        window: int,
        provider: EmbeddingProvider | None,
    ) -> list[dict[str, Any]]:
        if provider is None:
            return self._store.fulltext_search(
                collection, text, filters=filters or None, limit=max(limit, window)
            )
        # EmbeddingUnavailableError propagates: the facade converts it into
        # the declared BM25-only fallback mode.
        vector = provider.encode_queries([text])[0]
        return self._store.hybrid_search(
            collection,
            text,
            filters=filters or None,
            limit=max(limit, window),
            candidate_window=window,
            query_embedding=vector,
        )

    def _candidates(
        self,
        rows: list[dict[str, Any]],
        exact: dict[str, list[str]],
        descriptor: ActiveIndexDescriptor,
        provider: EmbeddingProvider | None,
        policy: PurposePolicy,
    ) -> list[RetrievalCandidate]:
        semantics = SCORE_SEMANTICS_HYBRID if provider is not None else SCORE_SEMANTICS_BM25
        if exact:
            rows = sorted(
                enumerate(rows),
                key=lambda pair: (_exact_row_multiplier(pair[1], exact), -pair[0]),
                reverse=True,
            )
            ordered = [row for _, row in rows]
        else:
            ordered = list(rows)
        candidates: list[RetrievalCandidate] = []
        for rank, row in enumerate(ordered, start=1):
            matches = _entity_matches(row, exact)
            cross_body = (
                matches["wanted_body"] is not None
                and row.get("body_id") is not None
                and row.get("body_id") != matches["wanted_body"]
            )
            if cross_body and policy.cross_body is CROSS_BODY_FORBID:
                continue
            candidates.append(
                RetrievalCandidate(
                    memory_id=str(row.get("id") or ""),
                    memory_type=str(row.get("memory_type") or ""),
                    vector_rank=None,
                    bm25_rank=None,
                    fusion_rank=rank,
                    rerank_score=None,
                    exact_entity_match=bool(matches["exact_entity_match"]),
                    body_match=matches["body_match"],
                    joint_match=matches["joint_match"],
                    failure_type_match=matches["failure_type_match"],
                    physical_collection=descriptor.physical_collection,
                    embedding_profile_id=descriptor.embedding_profile_id,
                    score_semantics=semantics,
                    cross_body_reference=(
                        bool(cross_body) and policy.cross_body is CROSS_BODY_ANNOTATE
                    ),
                    item=MemoryItem.from_record(row),
                )
            )
        return candidates
