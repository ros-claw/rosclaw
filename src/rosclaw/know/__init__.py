"""ROSClaw Knowledge Graph Module (v1.0).

Provides structured knowledge for Agent Runtime:
- Robot capabilities from e-URDF semantic tags
- Symptom matching for failure recovery
- Cross-domain engineering analogies

Architecture:
    KnowledgeInterface (resident, query side)
        - Loads knowledge_graph from SeekDB at startup
        - Answers queries via local keyword/regex matching (v1.0)
        - Zero LLM calls in hot path

    KnowledgeBatchEngine (transient, batch side)
        - Triggered by EventBus events
        - Wraps part/rosclaw-know/ pipeline (future)
"""

from rosclaw.core.lifecycle import LifecycleMixin, LifecycleState
from .interface import KnowledgeInterface

__all__ = ["KnowledgeInterface", "LifecycleMixin", "LifecycleState"]
