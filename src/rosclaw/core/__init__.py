"""
ROSClaw Core - OS Kernel Layer

The core module provides the foundational runtime infrastructure:
- EventBus: Publish/subscribe message bus for all module communication
- Runtime: Central orchestrator managing all grounding engines
- Lifecycle: Module initialization, startup, and shutdown coordination
"""

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin, LifecycleState
from rosclaw.core.runtime import Runtime, RuntimeConfig

__all__ = [
    "EventBus",
    "Event",
    "EventPriority",
    "Runtime",
    "RuntimeConfig",
    "LifecycleState",
    "LifecycleMixin",
]
