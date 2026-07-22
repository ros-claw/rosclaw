"""Embedding provider errors (数据库优化v3 §7/§11)."""


class EmbeddingError(RuntimeError):
    """Base class for embedding provider failures."""


class EmbeddingUnavailableError(EmbeddingError):
    """The provider (service or local model) is unreachable.

    Callers MUST degrade (BM25 + metadata filter), never block the
    robot, and NEVER substitute a different model's vectors into a
    collection built by another model (§11).
    """


class EmbeddingDimensionMismatchError(EmbeddingError):
    """The provider returned a vector that does not match its profile."""


class ModelRevisionChangedError(EmbeddingError):
    """The resolved model revision differs from the pinned profile."""
