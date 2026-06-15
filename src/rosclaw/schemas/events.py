"""Unified EventEnvelope — bridges core EventBus Event and auto EventEnvelope."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosclaw.core.event_bus import Event


@dataclass
class EventEnvelope:
    """Canonical event envelope for all ROSClaw inter-module communication.

    Designed to be compatible with both:
    - core.event_bus.Event (topic-based, priority, trace_id)
    - auto.events.schemas.EventEnvelope (event_type, run_id, payload)

    When publishing through core EventBus, use ``to_core_event()``.
    When receiving from auto subscribers, use ``from_auto_event()``.
    """

    event_id: str = ""
    event_type: str = ""  # canonical topic name, e.g. "rosclaw.auto.proposal.created"
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    trace_id: str = ""
    run_id: str = ""
    task_id: str = ""
    robot_id: str = ""
    skill_id: str = ""
    source: str = "rosclaw"
    priority: str = "NORMAL"  # CRITICAL | HIGH | NORMAL | LOW | BACKGROUND
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "robot_id": self.robot_id,
            "skill_id": self.skill_id,
            "source": self.source,
            "priority": self.priority,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EventEnvelope:
        return cls(
            event_id=d.get("event_id", ""),
            event_type=d.get("event_type", ""),
            timestamp=d.get("timestamp", ""),
            trace_id=d.get("trace_id", ""),
            run_id=d.get("run_id", ""),
            task_id=d.get("task_id", ""),
            robot_id=d.get("robot_id", ""),
            skill_id=d.get("skill_id", ""),
            source=d.get("source", "rosclaw"),
            priority=d.get("priority", "NORMAL"),
            payload=dict(d.get("payload", {})),
            metadata=dict(d.get("metadata", {})),
        )

    def to_core_event(self) -> Event:
        """Convert to core.event_bus.Event for publishing via EventBus."""
        from rosclaw.core.event_bus import Event, EventPriority

        priority_map = {
            "CRITICAL": EventPriority.CRITICAL,
            "HIGH": EventPriority.HIGH,
            "NORMAL": EventPriority.NORMAL,
            "LOW": EventPriority.LOW,
            "BACKGROUND": EventPriority.BACKGROUND,
        }
        return Event(
            topic=self.event_type,
            payload=self.to_dict(),
            source=self.source,
            priority=priority_map.get(self.priority, EventPriority.NORMAL),
            trace_id=self.trace_id,
            metadata=self.metadata,
        )

    @classmethod
    def from_core_event(cls, event: Event) -> EventEnvelope:
        """Build from a core.event_bus.Event received from EventBus."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        return cls.from_dict(payload)
