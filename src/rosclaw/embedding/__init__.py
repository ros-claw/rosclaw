"""Embedding provider package (数据库优化v3 §7)."""

from .cache import CachedEmbeddingProvider, EmbeddingCache
from .errors import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingUnavailableError,
    ModelRevisionChangedError,
)
from .profile import PROFILES, QWEN3_06B_1024
from .protocol import EmbeddingProfile, EmbeddingProvider
from .registry import get_profile, get_provider

__all__ = [
    "PROFILES",
    "QWEN3_06B_1024",
    "CachedEmbeddingProvider",
    "EmbeddingCache",
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmbeddingProfile",
    "EmbeddingProvider",
    "EmbeddingUnavailableError",
    "ModelRevisionChangedError",
    "get_profile",
    "get_provider",
]
