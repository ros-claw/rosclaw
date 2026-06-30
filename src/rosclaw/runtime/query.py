"""RuntimeQueryAPI — read-only query interface over the RuntimeBus.

Dashboard, MCP servers, and CLI commands use this API to ask the runtime for
live state instead of scanning files. Queries can target exact event types or
glob patterns such as ``camera.*``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.event import RuntimeEvent


class RuntimeQueryAPI:
    """Read-only query API over RuntimeBus history and replay logs."""

    def __init__(self, runtime_bus: RuntimeBus) -> None:
        self.bus = runtime_bus

    def latest(self, event_type: str) -> RuntimeEvent | None:
        """Return the most recent event matching ``event_type`` (supports globs).

        Uses in-memory history so the result reflects live runtime state without
        scanning JSONL logs.
        """
        events = self.bus.get_history(event_type=event_type, limit=1)
        return events[0] if events else None

    def latest_n(self, event_type: str, n: int = 10) -> list[RuntimeEvent]:
        """Return the ``n`` most recent events matching ``event_type``.

        Results are ordered newest-first, mirroring dashboard and monitoring
        conventions.
        """
        events = self.bus.get_history(event_type=event_type, limit=n)
        return list(reversed(events))

    def since(self, event_type: str, timestamp: datetime) -> list[RuntimeEvent]:
        """Return events of ``event_type`` published after ``timestamp``."""
        events = self.bus.replay(event_type=event_type, start=timestamp, limit=1000)
        return list(reversed(events))

    def trace(
        self, trace_id: str, event_type: str | None = None
    ) -> list[RuntimeEvent]:
        """Return all events belonging to ``trace_id`` (optionally filtered)."""
        events = self.bus.replay(
            trace_id=trace_id, event_type=event_type, limit=10000
        )
        return list(reversed(events))

    def latest_payload(self, event_type: str) -> dict[str, Any] | None:
        """Convenience helper returning only the payload of the latest event."""
        ev = self.latest(event_type)
        return ev.payload if ev else None
