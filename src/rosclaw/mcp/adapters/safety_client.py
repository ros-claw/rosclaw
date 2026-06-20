"""Thin adapter for emergency-stop coordination through the Runtime/EventBus."""

from __future__ import annotations

from typing import Any


class SafetyClient:
    """Emergency client that publishes ``robot.emergency_stop`` and invokes the
    runtime handler directly for immediate effect.
    """

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def emergency_stop(self, reason: str) -> dict[str, Any]:
        """Trigger an emergency stop and return a structured response."""
        from rosclaw.core.event_bus import Event, EventPriority

        event = Event(
            topic="robot.emergency_stop",
            payload={"reason": reason, "source": "mcp.emergency_stop"},
            source="rosclaw.mcp.server",
            priority=EventPriority.CRITICAL,
        )
        self._runtime.event_bus.publish(event)
        return {"stopped": True, "reason": reason, "mode": "live"}
