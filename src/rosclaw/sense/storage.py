"""In-memory and JSONL storage for sense history."""

from __future__ import annotations

import json
import logging
import os
from collections import deque

from rosclaw.sense.schemas import BodyEvent, BodySense, BodyState

logger = logging.getLogger("rosclaw.sense.storage")


class SenseStorage:
    """Ring-buffer storage for the latest BodyState, BodySense, and BodyEvents.

    Optionally appends BodySense snapshots to a JSONL file for later replay
    or analysis.
    """

    def __init__(self, max_event_history: int = 1000, jsonl_path: str | None = None):
        self.max_event_history = max_event_history
        self.jsonl_path = jsonl_path
        self._latest_state: BodyState | None = None
        self._latest_sense: BodySense | None = None
        self._events: deque[BodyEvent] = deque(maxlen=max_event_history)

    @property
    def latest_state(self) -> BodyState | None:
        return self._latest_state

    @property
    def latest_sense(self) -> BodySense | None:
        return self._latest_sense

    def update_state(self, state: BodyState) -> None:
        """Store the latest raw BodyState."""
        self._latest_state = state

    def update_sense(self, sense: BodySense) -> None:
        """Store the latest BodySense and optionally append to JSONL."""
        self._latest_sense = sense
        if self.jsonl_path:
            self._append_jsonl(sense)

    def add_event(self, event: BodyEvent) -> None:
        """Store a BodyEvent in the ring buffer."""
        self._events.append(event)

    def get_events(
        self,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[BodyEvent]:
        """Return recent events, optionally filtered by type."""
        events = list(self._events)
        if event_type:
            events = [e for e in events if e.type == event_type]
        if limit is not None:
            events = events[-limit:]
        return events

    def _append_jsonl(self, sense: BodySense) -> None:
        if self.jsonl_path is None:
            return
        try:
            os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sense.to_dict(), default=str) + "\n")
        except OSError as e:
            logger.warning("Failed to append sense to %s: %s", self.jsonl_path, e)
