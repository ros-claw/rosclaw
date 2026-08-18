"""Typed runtime configuration (PR-DF-18 / phase-II §15).

Canonical in-process config models.  Pre-DF-18 shapes (flat
``seekdb_*`` fields, dict ``storage``/``darwin`` sections) are accepted
only at the RuntimeConfig constructor boundary and folded into these
models once by ``rosclaw.config.compat``.
"""

from rosclaw.config.compat import (
    normalize_darwin_config,
    normalize_evolution_config,
    normalize_knowledge_config,
    normalize_storage_config,
)
from rosclaw.config.loader import load_typed_configs
from rosclaw.config.models import (
    ArtifactStoreConfig,
    DarwinConfig,
    EvolutionConfig,
    KnowledgeConfig,
    OutboxConfig,
    RetrievalStoreConfig,
    StorageConfig,
    StructuredStoreConfig,
)

__all__ = [
    "ArtifactStoreConfig",
    "DarwinConfig",
    "EvolutionConfig",
    "KnowledgeConfig",
    "OutboxConfig",
    "RetrievalStoreConfig",
    "StorageConfig",
    "StructuredStoreConfig",
    "load_typed_configs",
    "normalize_darwin_config",
    "normalize_evolution_config",
    "normalize_knowledge_config",
    "normalize_storage_config",
]
