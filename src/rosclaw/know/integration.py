"""KnowIntegration — Runtime-facing integration layer for KNOW queries.

Provides a clean API for Runtime to query knowledge before making
provider routing decisions, and records knowledge usage for Practice
tracking.
"""

from __future__ import annotations

from typing import Any, Optional

from rosclaw.know.interface import KnowledgeInterface
from rosclaw.core.event_bus import EventBus, Event, EventPriority


class KnowIntegration:
    """Integration wrapper that bridges Runtime → KNOW → Practice/Memory.

    Usage:
        ki = KnowIntegration(robot_id="ur5e", event_bus=bus)
        result = ki.query_before_decision("ur5e", "pick_and_place")
        # result contains capability match, safety limits, alternatives
    """

    def __init__(self, robot_id: str, event_bus: Optional[EventBus] = None,
                 seekdb_client: Any = None):
        self.robot_id = robot_id
        self.event_bus = event_bus
        self._know = KnowledgeInterface(
            robot_id=robot_id,
            event_bus=event_bus,
            seekdb_client=seekdb_client,
        )
        self._know._do_initialize()

    def query_before_decision(self, robot_id: str, task: str) -> dict[str, Any]:
        """Query KNOW before Runtime makes a provider routing decision.

        Returns capability match, safety limits, simulation profile,
        alternative robots, and known risks.
        """
        # Decompose task first
        decomposition = self._know.task_decomposition_hint(task)

        # Check if robot can perform
        can_perf = self._know.can_perform_task(robot_id, task)

        # Get capability match for provider selection
        skill_name = task.lower().replace(" ", "_")
        cap_match = self._know.query_for_provider_selection(skill_name, robot_id)

        # Publish event for Practice / Memory tracking
        if self.event_bus is not None:
            self.event_bus.publish(Event(
                topic="rosclaw.knowledge.pre_check",
                payload={
                    "robot_id": robot_id,
                    "task": task,
                    "decomposition": decomposition,
                    "can_perform": can_perf,
                    "capability_match": cap_match,
                },
                source="know_integration",
                priority=EventPriority.NORMAL,
            ))

        return {
            "robot_id": robot_id,
            "task": task,
            "decomposition": decomposition,
            "can_perform": can_perf,
            "capability_match": cap_match,
            "timestamp": __import__("time").time(),
        }

    def record_usage(self, context: dict[str, Any]) -> None:
        """Record knowledge usage after task execution."""
        self._know.record_knowledge_usage(context)
