"""MemoryRetrievalFacade — the single query entry for runtime memory (v4 §3.2).

CLI, Runtime, KNOW, HOW, and AUTO all retrieve through this facade.  It
resolves the canonical ACTIVE index, pins the embedding provider to the
ACTIVE descriptor, runs the versioned native retriever, and degrades through
the declared fallback chain — always disclosing mode and reason.

Modules must not construct ``SQLiteVectorStore`` / ``SeekDBNativeStore.similar``
/ ``VersionedCollectionManager.shadow_query`` for serving queries; those are
build-time or legacy-compat paths only.
"""

from __future__ import annotations

import logging
from typing import Any

from rosclaw.embedding.errors import EmbeddingUnavailableError
from rosclaw.memory.v2.repository import MemoryRepository
from rosclaw.memory.v2.retrieval import MemoryQuery, MemoryRetriever

from .active_resolver import ActiveCollectionResolver, ActiveIndexUnavailableError
from .fallback import (
    MODE_ABSTAIN,
    MODE_ACTIVE_BM25,
    MODE_SQLITE_LEXICAL,
    REASON_ACTIVE_UNAVAILABLE,
    REASON_NATIVE_STORE_UNAVAILABLE,
    REASON_NO_MEMORY_STORE,
    REASON_PROVIDER_UNAVAILABLE,
    hybrid_mode_for,
)
from .native_retriever import VersionedNativeRetriever
from .provider_resolver import EmbeddingProviderResolver, ProviderUnavailableError
from .result import (
    RetrievalAttempt,
    RetrievalCandidate,
    RetrievalPurpose,
    RetrievalResponse,
    policy_for,
)

logger = logging.getLogger("rosclaw.memory.v2.runtime_retrieval.facade")

DEFAULT_LOGICAL_NAME = "memory_items"


class MemoryRetrievalFacade:
    """Unified retrieval over the canonical ACTIVE index with declared fallback."""

    def __init__(
        self,
        *,
        native_store: Any | None = None,
        sqlite_store: Any | None = None,
        provider_resolver: EmbeddingProviderResolver | None = None,
        reranker: Any | None = None,
        logical_name: str = DEFAULT_LOGICAL_NAME,
    ) -> None:
        self._native_store = native_store
        self._sqlite_store = sqlite_store
        self._provider_resolver = provider_resolver or EmbeddingProviderResolver()
        self._reranker = reranker
        self._logical_name = logical_name

    @property
    def logical_name(self) -> str:
        return self._logical_name

    def retrieve(
        self,
        query: MemoryQuery,
        *,
        purpose: RetrievalPurpose = RetrievalPurpose.HUMAN_SEARCH,
    ) -> RetrievalResponse:
        policy = policy_for(purpose)
        attempts: list[RetrievalAttempt] = []

        # 1) ACTIVE physical collection on the native SeekDB engine.
        if self._native_store is not None:
            response = self._try_native(query, policy=policy, attempts=attempts)
            if response is not None:
                return response
        else:
            attempts.append(
                RetrievalAttempt(
                    mode="active_hybrid",
                    ok=False,
                    reason=REASON_NATIVE_STORE_UNAVAILABLE,
                )
            )

        # 2) SQLite Memory V2 lexical (source-of-truth store, no vectors).
        # The fallback reason names why the PRIMARY path could not serve —
        # the first failed attempt — while `attempts` keeps the full chain.
        if self._sqlite_store is not None:
            try:
                candidates = self._sqlite_candidates(query)
                reason = next(
                    (a.reason for a in attempts if a.reason),
                    REASON_ACTIVE_UNAVAILABLE,
                )
                return RetrievalResponse(
                    retrieval_mode=MODE_SQLITE_LEXICAL,
                    logical_collection=self._logical_name,
                    physical_collection=None,
                    embedding_profile_id=None,
                    purpose=purpose,
                    fallback=True,
                    fallback_reason=reason,
                    candidates=candidates,
                    attempts=attempts,
                    reranker_required=policy.reranker_required,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("sqlite lexical fallback failed: %s", exc)
                attempts.append(
                    RetrievalAttempt(mode=MODE_SQLITE_LEXICAL, ok=False, reason=str(exc))
                )
        else:
            attempts.append(
                RetrievalAttempt(
                    mode=MODE_SQLITE_LEXICAL,
                    ok=False,
                    reason="sqlite store not configured",
                )
            )

        # 3) Nothing can serve — abstain honestly (v4 §3.7).  The reason is
        # the root cause (first failed attempt), not the last link tried.
        reason = next(
            (a.reason for a in attempts if a.reason),
            REASON_NO_MEMORY_STORE,
        )
        return RetrievalResponse(
            retrieval_mode=MODE_ABSTAIN,
            logical_collection=self._logical_name,
            physical_collection=None,
            embedding_profile_id=None,
            purpose=purpose,
            fallback=True,
            fallback_reason=reason,
            candidates=[],
            attempts=attempts,
            reranker_required=policy.reranker_required,
        )

    # ------------------------------------------------------------------

    def _try_native(
        self,
        query: MemoryQuery,
        *,
        policy: Any,
        attempts: list[RetrievalAttempt],
    ) -> RetrievalResponse | None:
        purpose = policy.purpose
        try:
            descriptor = ActiveCollectionResolver(self._native_store).resolve(self._logical_name)
        except ActiveIndexUnavailableError as exc:
            attempts.append(RetrievalAttempt(mode="active_hybrid", ok=False, reason=exc.reason))
            return None
        except Exception as exc:  # noqa: BLE001
            attempts.append(
                RetrievalAttempt(
                    mode="active_hybrid",
                    ok=False,
                    reason=f"{REASON_NATIVE_STORE_UNAVAILABLE}:{exc}",
                )
            )
            return None

        retriever = VersionedNativeRetriever(self._native_store, reranker=self._reranker)
        mode = hybrid_mode_for(descriptor.embedding_profile_id)

        provider = None
        provider_reason: str | None = None
        try:
            provider = self._provider_resolver.resolve(descriptor)
        except ProviderUnavailableError as exc:
            provider_reason = f"{REASON_PROVIDER_UNAVAILABLE}:{exc.reason}"
            attempts.append(RetrievalAttempt(mode=mode, ok=False, reason=provider_reason))

        if provider is not None:
            try:
                candidates, notes = retriever.retrieve(query, descriptor, provider, policy=policy)
                attempts.append(RetrievalAttempt(mode=mode, ok=True))
                return RetrievalResponse(
                    retrieval_mode=mode,
                    logical_collection=self._logical_name,
                    physical_collection=descriptor.physical_collection,
                    embedding_profile_id=descriptor.embedding_profile_id,
                    purpose=purpose,
                    fallback=False,
                    fallback_reason=None,
                    candidates=candidates,
                    attempts=attempts,
                    reranker_applied=bool(notes.get("reranker_applied")),
                    reranker_required=policy.reranker_required,
                )
            except EmbeddingUnavailableError as exc:
                provider_reason = f"{REASON_PROVIDER_UNAVAILABLE}:{exc}"
                attempts.append(RetrievalAttempt(mode=mode, ok=False, reason=provider_reason))
            except Exception as exc:  # noqa: BLE001
                attempts.append(
                    RetrievalAttempt(
                        mode=mode,
                        ok=False,
                        reason=f"{REASON_NATIVE_STORE_UNAVAILABLE}:{exc}",
                    )
                )
                return None

        # Declared degradation: BM25 + metadata on the ACTIVE collection,
        # never another model's vectors (v4 §3.4).
        try:
            candidates, notes = retriever.retrieve(query, descriptor, None, policy=policy)
        except Exception as exc:  # noqa: BLE001
            attempts.append(RetrievalAttempt(mode=MODE_ACTIVE_BM25, ok=False, reason=str(exc)))
            return None
        return RetrievalResponse(
            retrieval_mode=MODE_ACTIVE_BM25,
            logical_collection=self._logical_name,
            physical_collection=descriptor.physical_collection,
            embedding_profile_id=descriptor.embedding_profile_id,
            purpose=purpose,
            fallback=True,
            fallback_reason=provider_reason or REASON_PROVIDER_UNAVAILABLE,
            candidates=candidates,
            attempts=attempts,
            reranker_applied=bool(notes.get("reranker_applied")),
            reranker_required=policy.reranker_required,
        )

    def _sqlite_candidates(self, query: MemoryQuery) -> list[RetrievalCandidate]:
        repo = MemoryRepository(self._sqlite_store)
        retriever = MemoryRetriever(repo)  # lexical + metadata only, no vector
        results = retriever.retrieve(query)
        # Exact-entity disclosure computed from the query, not assumed:
        # the lexical path validates entities the same way the native one
        # does (unknown stays unknown).
        from rosclaw.memory.v2.document import extract_exact_terms

        exact = extract_exact_terms(query.text or "")
        candidates: list[RetrievalCandidate] = []
        for result in results:
            item = result.memory
            body_match = joint_match = failure_match = None
            exact_entity = False
            if exact:
                record = item.to_record()
                from rosclaw.memory.v2.runtime_retrieval.native_retriever import _entity_matches

                matches = _entity_matches(record, exact)
                body_match = matches["body_match"]
                joint_match = matches["joint_match"]
                failure_match = matches["failure_type_match"]
                exact_entity = bool(matches["exact_entity_match"])
            candidates.append(
                RetrievalCandidate(
                    memory_id=item.memory_id,
                    memory_type=item.memory_type,
                    vector_rank=None,
                    bm25_rank=None,
                    fusion_rank=result.rank,
                    rerank_score=None,
                    exact_entity_match=exact_entity,
                    body_match=body_match,
                    joint_match=joint_match,
                    failure_type_match=failure_match,
                    physical_collection=None,
                    embedding_profile_id=None,
                    score_semantics=(
                        "sqlite lexical+metadata fusion (source-of-truth store, "
                        "no embedding); fusion scores, not similarities"
                    ),
                    item=item,
                )
            )
        return candidates


def build_retrieval_facade(
    *,
    native_store: Any | None = None,
    sqlite_store: Any | None = None,
    provider_resolver: EmbeddingProviderResolver | None = None,
    reranker: Any | None = None,
    logical_name: str = DEFAULT_LOGICAL_NAME,
) -> MemoryRetrievalFacade:
    """Construct the facade.  Either store may be None; the facade degrades
    through the declared chain instead of failing at construction time."""
    return MemoryRetrievalFacade(
        native_store=native_store,
        sqlite_store=sqlite_store,
        provider_resolver=provider_resolver,
        reranker=reranker,
        logical_name=logical_name,
    )
