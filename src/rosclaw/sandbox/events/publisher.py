"""RuntimePublisher — publishes sandbox events to the Runtime EventBus."""

from __future__ import annotations

from typing import Any, Optional

from rosclaw.core.event_bus import EventBus, Event, EventPriority


class RuntimePublisher:
    """Publishes sandbox lifecycle and episode events to the v1.0 EventBus."""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self._event_bus = event_bus

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish an event to the EventBus."""
        if self._event_bus is None:
            return
        self._event_bus.publish(Event(
            topic=topic,
            payload=payload,
            source="sandbox",
            priority=EventPriority.NORMAL,
        ))
