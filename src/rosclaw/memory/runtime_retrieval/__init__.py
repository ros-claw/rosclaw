"""Unified runtime retrieval facade (数据库优化v4 §3, PR-MEM-5).

The canonical query path for CLI, Runtime, KNOW, HOW, and AUTO: resolve the
ACTIVE physical collection, pin the embedding provider to the ACTIVE
descriptor, retrieve with purpose-aware safety policy, and degrade through
the declared fallback chain with explicit reasons.
"""

from .active_resolver import (
    ActiveCollectionResolver,
    ActiveIndexDescriptor,
    ActiveIndexUnavailableError,
)
from .facade import DEFAULT_LOGICAL_NAME, MemoryRetrievalFacade, build_retrieval_facade
from .fallback import (
    MODE_ABSTAIN,
    MODE_ACTIVE_BM25,
    MODE_SQLITE_LEXICAL,
    hybrid_mode_for,
)
from .health import RetrievalHealthProbe
from .provider_resolver import EmbeddingProviderResolver, ProviderUnavailableError
from .result import (
    PurposePolicy,
    RetrievalAttempt,
    RetrievalCandidate,
    RetrievalPurpose,
    RetrievalResponse,
    policy_for,
)

__all__ = [
    "DEFAULT_LOGICAL_NAME",
    "MODE_ABSTAIN",
    "MODE_ACTIVE_BM25",
    "MODE_SQLITE_LEXICAL",
    "ActiveCollectionResolver",
    "ActiveIndexDescriptor",
    "ActiveIndexUnavailableError",
    "EmbeddingProviderResolver",
    "MemoryRetrievalFacade",
    "ProviderUnavailableError",
    "PurposePolicy",
    "RetrievalAttempt",
    "RetrievalCandidate",
    "RetrievalHealthProbe",
    "RetrievalPurpose",
    "RetrievalResponse",
    "build_retrieval_facade",
    "hybrid_mode_for",
    "policy_for",
]
