"""Embedding provider protocol + profile (数据库优化v3 §7.1).

Query and document encoding paths are SEPARATE by contract (§17.6):
Qwen3-style models take a task instruction on the query side only.
An :class:`EmbeddingProfile` pins the model identity — model_id,
revision, dimension, normalization, distance — so a collection can
prove which model produced its vectors, and a query vector can be
refused when it does not match the collection's profile.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingProfile:
    profile_id: str
    model_id: str
    model_revision: str
    dimension: int
    normalize: bool
    distance: str
    query_instruction: str | None
    document_instruction: str | None
    max_tokens: int
    provider_type: str

    def cache_namespace(self, *, kind: str, text_hash: str) -> str:
        """Cache key material: model + revision + dimension + kind +
        instruction + text (§7.4 — query/document caches never mix)."""
        material = "|".join(
            [
                self.model_id,
                self.model_revision,
                str(self.dimension),
                kind,
                hashlib.sha256((self.query_instruction or "").encode()).hexdigest()
                if kind == "query"
                else hashlib.sha256((self.document_instruction or "").encode()).hexdigest(),
                text_hash,
            ]
        )
        return hashlib.sha256(material.encode()).hexdigest()


class EmbeddingProvider(Protocol):
    @property
    def profile(self) -> EmbeddingProfile: ...

    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...

    def encode_queries(self, texts: list[str]) -> list[list[float]]: ...

    def health(self) -> dict: ...
