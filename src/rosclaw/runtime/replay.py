"""Runtime replay engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.event import RuntimeEvent


class RuntimeReplay:
    """Replay engine for runtime events, episodes, skills, and providers."""

    def __init__(self, runtime_bus: RuntimeBus) -> None:
        self.bus = runtime_bus

    def replay_event(self, event_id: str) -> RuntimeEvent | None:
        """Replay a single event by id."""
        events = self.bus.replay(trace_id=event_id, limit=1)
        if events:
            return events[0]
        # Fall back to id matching.
        events = self.bus.replay(limit=10000)
        for ev in events:
            if ev.id == event_id:
                return ev
        return None

    def replay_episode(self, episode_id: str, limit: int = 10000) -> list[RuntimeEvent]:
        """Replay all events belonging to an episode/trace."""
        return self.bus.replay(trace_id=episode_id, limit=limit)

    def replay_skill(
        self, skill_id: str, episode_id: str | None = None, limit: int = 1000
    ) -> list[RuntimeEvent]:
        """Replay skill.invoke and skill.complete events."""
        results: list[RuntimeEvent] = []
        events = self.bus.replay(trace_id=episode_id, limit=limit) if episode_id else self.bus.replay(limit=limit)
        for ev in events:
            if ev.type in ("skill.invoke", "skill.complete") and ev.payload.get("skill_id") == skill_id:
                results.append(ev)
        return results

    def replay_provider(
        self, request_id: str | None = None, episode_id: str | None = None, limit: int = 1000
    ) -> list[RuntimeEvent]:
        """Replay provider.request and provider.result events."""
        results: list[RuntimeEvent] = []
        events = self.bus.replay(trace_id=episode_id, limit=limit) if episode_id else self.bus.replay(limit=limit)
        for ev in events:
            if ev.type in ("provider.request", "provider.result"):
                if request_id is None or ev.payload.get("request_id") == request_id:
                    results.append(ev)
        return results

    def replay_time_range(
        self, start: datetime, end: datetime, event_type: str | None = None, limit: int = 1000
    ) -> list[RuntimeEvent]:
        """Replay events in a time range."""
        return self.bus.replay(event_type=event_type, start=start, end=end, limit=limit)
