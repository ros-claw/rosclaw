"""Embedding provider registry (数据库优化v3 §7.1).

Lazy, one provider instance per profile_id.  Offline-first: the local
snapshot cache is used as-is; HF_HUB_OFFLINE is set for the lifetime of
this process unless the caller explicitly opted into an endpoint —
models never phone home inside the robot runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

from .cache import CachedEmbeddingProvider, EmbeddingCache
from .local_sentence_transformer import LocalSentenceTransformerProvider
from .profile import PROFILES
from .protocol import EmbeddingProfile, EmbeddingProvider

_DEFAULT_CACHE = Path.home() / ".rosclaw" / "embedding_cache.sqlite"


def get_provider(
    profile_id: str,
    *,
    cache_path: str | Path | None = _DEFAULT_CACHE,
    device: str | None = None,
) -> EmbeddingProvider:
    if profile_id not in PROFILES:
        raise KeyError(f"unknown embedding profile {profile_id!r}; known: {sorted(PROFILES)}")
    profile: EmbeddingProfile = PROFILES[profile_id]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if profile.provider_type == "local_sentence_transformer":
        provider: EmbeddingProvider = LocalSentenceTransformerProvider(profile, device=device)
    else:  # pragma: no cover - defensive
        raise KeyError(f"no provider factory for type {profile.provider_type!r}")
    if cache_path is not None:
        provider = CachedEmbeddingProvider(provider, EmbeddingCache(cache_path))
    return provider


def get_profile(profile_id: str) -> EmbeddingProfile:
    return PROFILES[profile_id]
