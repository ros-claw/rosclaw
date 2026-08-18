"""DEPRECATED package (PR-DF-09 / ADR-0010): legacy Knowledge runtime.

Canonical package: ``rosclaw.knowledge``.  This package stays as the
compatibility shim for at least one full minor release; import legacy
symbols via ``rosclaw.knowledge.legacy`` so the eventual physical move is
invisible to consumers.

Compatibility facade for legacy ROSClaw Knowledge integrations.

Authoritative v2 implementation lives in ``rosclaw-know``. This package is a
rollback-compatible runtime adapter and must not grow new retrieval algorithms.

Provides structured knowledge for Agent Runtime:
- Robot capabilities from e-URDF semantic tags
- Symptom matching for failure recovery
- Cross-domain engineering analogies
- v1.5: Task Packs (pre-flight knowledge for agents)
- v1.5: EventBus-triggered catalog updates from sandbox/runtime episodes

Architecture:

    KnowledgeInterface (resident, query side)
        - Loads knowledge_graph from SeekDB at startup
        - Answers queries via local keyword/regex matching
        - Zero LLM calls in hot path

    KnowledgeBatchEngine (resident, batch side, v1.5)
        - Triggered by EventBus events (sandbox + runtime)
        - Wraps rosclaw_know.sim_ingest direct path (Sprint 12)
        - Inert when rosclaw-know is not installed

    AssetsLoader (resident, asset publication side, v1.5)
        - Reloads bridge_index.json on assets_refreshed events
        - Invalidates task_pack_adapter cache

    task_pack_adapter (function, hot-path, v1.5)
        - task_pack_for(task_id) → dict with FailureMode + FixPattern context
        - Returns empty pack when rosclaw-know is not installed
"""

from rosclaw.core.lifecycle import LifecycleMixin, LifecycleState

from .assets_loader import AssetsLoader
from .batch_engine import KnowledgeBatchEngine
from .embodiment_card import EmbodimentCard
from .interface import KnowledgeInterface
from .task_card import TaskCard
from .task_pack_adapter import task_pack_for
from .verifier_card import VerifierCard

# Canonical-in-spirit alias for the legacy entry point (ADR-0010 §4):
# two "official-looking" Knowledge runtimes must not coexist unnamed.
LegacyKnowledgeRuntime = KnowledgeInterface

__all__ = [
    "AssetsLoader",
    "EmbodimentCard",
    "KnowledgeBatchEngine",
    "KnowledgeInterface",
    "LegacyKnowledgeRuntime",
    "LifecycleMixin",
    "LifecycleState",
    "TaskCard",
    "VerifierCard",
    "task_pack_for",
]
