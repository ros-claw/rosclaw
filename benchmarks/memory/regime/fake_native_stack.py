"""Benchmark-native fake stack (PR-BENCH-4).

The benchmark must measure the FULL pinned path — ACTIVE collection +
pinned embedding provider + regime gate + choreography — not the degraded
sqlite-lexical fallback (which honestly disables APPLY by design, v4 §7.3).

This is a disclosed, deterministic double: hash-based fake embeddings, an
in-memory native-store shape, and an ACTIVE registry pointer — no real
SeekDB engine, no real model, and the report says so.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from rosclaw.embedding.protocol import EmbeddingProfile
from rosclaw.memory.v2.runtime_retrieval import EmbeddingProviderResolver

BENCH_PROFILE = EmbeddingProfile(
    profile_id="bench_fake_8d_v1",
    model_id="bench/fake-model",
    model_revision="rev1",
    dimension=8,
    normalize=True,
    distance="cosine",
    query_instruction="bench",
    document_instruction=None,
    max_tokens=32,
    provider_type="fake",
)

PHYSICAL = "memory_items__bench_fake_8d_v1__ik__g1"


class BenchFakeProvider:
    """Deterministic hash vectors (no model load)."""

    def __init__(self, profile: EmbeddingProfile = BENCH_PROFILE) -> None:
        self._profile = profile

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[: self._profile.dimension]]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec("doc:" + t) for t in texts]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vec("q:" + t) for t in texts]

    def health(self) -> dict:
        return {"ok": True, "profile_id": self._profile.profile_id}


def bench_provider_resolver() -> EmbeddingProviderResolver:
    return EmbeddingProviderResolver(
        provider_factory=lambda profile_id, **kwargs: BenchFakeProvider(),
        profiles={BENCH_PROFILE.profile_id: BENCH_PROFILE},
    )


def _registry_rows(physical: str, record_count: int) -> list[dict]:
    from rosclaw.storage.versioned_collections import VersionedCollectionManager

    activated = time.time()
    return [
        {
            "id": "projection_build_bench",
            "row_kind": "build",
            "build_id": "bench1",
            "logical_name": "memory_items",
            "physical_collection": physical,
            "embedding_profile_id": BENCH_PROFILE.profile_id,
            "model_id": BENCH_PROFILE.model_id,
            "model_revision": BENCH_PROFILE.model_revision,
            "dimension": BENCH_PROFILE.dimension,
            "normalize": True,
            "distance": "cosine",
            "analyzer": "ik",
            "record_count": record_count,
            "status": "ACTIVE",
            "created_at": activated - 10,
            "activated_at": activated,
        },
        {
            "id": VersionedCollectionManager._pointer_id("memory_items"),
            "row_kind": "active_pointer",
            "logical_name": "memory_items",
            "active_physical_collection": physical,
            "active_activated_at": activated,
            "previous_physical_collection": None,
            "previous_activated_at": None,
            "updated_at": activated,
        },
    ]


class BenchNativeStore:
    """Native-store double: registry + one ACTIVE physical collection.

    BM25/KNN legs are simple substring scoring over documents — disclosed,
    deterministic, and identical across runs (no hash-seeded iteration).
    """

    def __init__(self, rows: list[dict[str, Any]], physical: str = PHYSICAL) -> None:
        self._physical = physical
        self.collections = {physical: [dict(row) for row in rows]}
        self.registry_rows = _registry_rows(physical, len(rows))
        self._client = self

    # -- registry surface (projection_registry queries) --
    def query(self, table: str, filters: dict | None = None, limit: int = 100) -> list[dict]:
        if table == "projection_registry":
            pool = self.registry_rows
        else:
            pool = self.collections.get(table, [])

        def match(row: dict) -> bool:
            return all(row.get(key) == value for key, value in (filters or {}).items())

        return [row for row in pool if match(row)][:limit]

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def embedding_info(self, table: str) -> dict:
        return {"dimension": BENCH_PROFILE.dimension}

    # -- search surface --
    @staticmethod
    def _bigrams(text: str) -> set[str]:
        compact = "".join(ch for ch in text.lower() if not ch.isspace())
        if len(compact) < 2:
            return {compact} if compact else set()
        return {compact[i : i + 2] for i in range(len(compact) - 1)}

    def _score(self, row: dict, query_text: str) -> float:
        document = (str(row.get("document") or "") + " " + str(row.get("title") or "")).lower()
        query = query_text.lower()
        # Word tokens for latin/whitespace-split text, character bigrams for
        # CJK (no spaces) — both matter in the bilingual corpus.
        words = [t for t in query.split() if len(t) >= 2]
        word_hits = sum(1 for t in words if t in document)
        word_score = word_hits / len(words) if words else 0.0
        q_bigrams = self._bigrams(query)
        if q_bigrams:
            d_bigrams = self._bigrams(document)
            bigram_score = len(q_bigrams & d_bigrams) / len(q_bigrams)
        else:
            bigram_score = 0.0
        return max(word_score, bigram_score)

    def _search(
        self,
        table: str,
        query_text: str,
        filters: dict | None,
        limit: int,
        query_embedding: list[float] | None = None,
    ) -> list[dict]:
        def match(row: dict) -> bool:
            return all(row.get(key) == value for key, value in (filters or {}).items())

        scored = [
            (self._score(row, query_text), row)
            for row in self.collections.get(table, [])
            if match(row)
        ]
        # Deterministic order: score desc, then id asc (never hash order).
        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("id") or "")))
        # Honest emptiness: zero lexical overlap means no results (the
        # purpose policy's starve handling depends on it).
        return [dict(row) for score, row in scored if score > 0][:limit]

    def hybrid_search(
        self,
        table: str,
        query_text: str,
        filters: dict | None = None,
        limit: int = 5,
        candidate_window: int | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[dict]:
        return self._search(table, query_text, filters, limit, query_embedding)

    def fulltext_search(
        self,
        table: str,
        query_text: str,
        filters: dict | None = None,
        limit: int = 5,
    ) -> list[dict]:
        return self._search(table, query_text, filters, limit, None)
