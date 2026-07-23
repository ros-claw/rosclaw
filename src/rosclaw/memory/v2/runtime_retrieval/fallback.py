"""Declared fallback chain for runtime retrieval (数据库优化v4 §3.7).

Chain::

    ACTIVE + Qwen Provider 正常      → <family>_hybrid          (e.g. qwen3_hybrid)
    ACTIVE 存在，Provider 不可用      → active_bm25_metadata
    SeekDB / ACTIVE 不可用           → sqlite_memory_v2_lexical
    以上均不可用                      → abstain

Every response names its mode and, when degraded, the machine-readable
reason.  Silence is forbidden: a caller must never have to guess which index
answered.
"""

from __future__ import annotations

# Retrieval modes.
MODE_ACTIVE_BM25 = "active_bm25_metadata"
MODE_SQLITE_LEXICAL = "sqlite_memory_v2_lexical"
MODE_ABSTAIN = "abstain"


def hybrid_mode_for(profile_id: str) -> str:
    """e.g. ``qwen3_06b_768_v1`` → ``qwen3_hybrid`` (v4 §3.7 example output)."""
    family = profile_id.split("_", 1)[0] if profile_id else "active"
    return f"{family}_hybrid"


# Fallback reason codes (composed as "<code>" or "<code>:<detail>").
REASON_PROVIDER_UNAVAILABLE = "embedding_provider_unavailable"
REASON_ACTIVE_UNAVAILABLE = "active_index_unavailable"
REASON_NATIVE_STORE_UNAVAILABLE = "seekdb_native_store_unavailable"
REASON_SQLITE_UNAVAILABLE = "sqlite_store_unavailable"
REASON_NO_MEMORY_STORE = "no_memory_store_available"
