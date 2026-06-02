"""Swarm Runtime Manager - Collaboration Grounding

Manages multi-robot coordination through the EventBus.
All communication goes through EventBus — no direct module calls.
"""

import logging
from typing import Optional

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin

logger = logging.getLogger("rosclaw.swarm.manager")


class SwarmRuntimeManager(LifecycleMixin):
    """
    Manages coordination between multiple robot agents.

    Provides:
    - Agent registry and discovery via EventBus
    - Task allocation and planning
    - Swarm-level event coordination
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        super().__init__()
        self.event_bus = event_bus
        self._agents: dict[str, dict] = {}
        self._tasks: dict[str, dict] = {}

    def _do_initialize(self) -> None:
        """Initialize swarm manager."""
        if self.event_bus is not None:
            self.event_bus.subscribe("swarm.register", self._on_register_request)
            self.event_bus.subscribe("swarm.allocate", self._on_allocate_request)
            self.event_bus.subscribe("swarm.status", self._on_status_request)
        logger.info("Swarm manager initialized")

    def _do_stop(self) -> None:
        if self.event_bus is not None:
            self.event_bus.unsubscribe("swarm.register", self._on_register_request)
            self.event_bus.unsubscribe("swarm.allocate", self._on_allocate_request)
            self.event_bus.unsubscribe("swarm.status", self._on_status_request)

    def _on_register_request(self, event: Event) -> None:
        """Handle agent registration via EventBus."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        agent_id = payload.get("agent_id")
        capabilities = payload.get("capabilities", [])
        if agent_id:
            self.register_agent(agent_id, capabilities)

    def _on_allocate_request(self, event: Event) -> None:
        """Handle task allocation via EventBus."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        task = payload.get("task", {})
        agent_id = self.allocate_task(task)
        if self.event_bus is not None:
            self.event_bus.publish(Event(
                topic="swarm.allocate_result",
                payload={"task": task, "agent_id": agent_id},
                source="swarm",
                priority=EventPriority.HIGH,
            ))

    def _on_status_request(self, event: Event) -> None:
        """Handle status query via EventBus."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        agent_id = payload.get("agent_id")
        status = self.get_agent_status(agent_id) if agent_id else {"agents": list(self._agents.keys())}
        if self.event_bus is not None:
            self.event_bus.publish(Event(
                topic="swarm.status_result",
                payload={"agent_id": agent_id, "status": status},
                source="swarm",
                priority=EventPriority.NORMAL,
            ))

    def register_agent(self, agent_id: str, capabilities: list[str]) -> None:
        """Register a robot agent with the swarm."""
        self._agents[agent_id] = {
            "id": agent_id,
            "capabilities": capabilities,
            "status": "idle",
        }
        logger.info("Agent registered: %s (%s)", agent_id, capabilities)
        if self.event_bus is not None:
            self.event_bus.publish(Event(
                topic="swarm.agent_registered",
                payload={"agent_id": agent_id, "capabilities": capabilities},
                source="swarm",
                priority=EventPriority.NORMAL,
            ))

    def allocate_task(self, task: dict) -> Optional[str]:
        """Allocate a task to an available agent."""
        required = task.get("required_capabilities", [])
        for agent_id, agent in self._agents.items():
            if agent["status"] == "idle" and all(c in agent["capabilities"] for c in required):
                agent["status"] = "busy"
                task_id = task.get("id", f"task_{len(self._tasks)}")
                self._tasks[task_id] = {
                    "id": task_id,
                    "agent": agent_id,
                    "task": task,
                    "status": "allocated",
                }
                return agent_id
        return None

    def get_agent_status(self, agent_id: str) -> Optional[dict]:
        """Get status of a registered agent."""
        return self._agents.get(agent_id)

    @property
    def agent_count(self) -> int:
        """Number of registered agents."""
        return len(self._agents)
