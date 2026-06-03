"""ROSClaw Knowledge Module — query + batch sides.

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
from .interface import KnowledgeInterface
from .task_pack_adapter import task_pack_for

__all__ = [
    "AssetsLoader",
    "KnowledgeBatchEngine",
    "KnowledgeInterface",
    "LifecycleMixin",
    "LifecycleState",
    "task_pack_for",
]
