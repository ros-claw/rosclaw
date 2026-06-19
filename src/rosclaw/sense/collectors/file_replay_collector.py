"""File replay collector for BodyState sequences stored as JSONL."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from rosclaw.sense.collectors.base import BodyStateCollector
from rosclaw.sense.schemas import BodyState

logger = logging.getLogger("rosclaw.sense.collectors.file_replay")


class FileReplayCollector(BodyStateCollector):
    """Replay BodyState snapshots from a JSONL file.

    This is a stub for Phase 1.  It supports reading the latest snapshot from
    a JSONL file and will be extended with timestamp-based playback, speed
    control, and single-step ticks in a follow-up.
    """

    name = "file_replay"

    def __init__(self, replay_path: str | None = None, robot_id: str = "unknown"):
        self.replay_path = replay_path
        self.robot_id = robot_id
        self._frames: list[dict[str, Any]] = []
        self._index = 0

    def _load(self) -> None:
        if self.replay_path is None or self._frames:
            return
        try:
            with open(self.replay_path, encoding="utf-8") as f:
                self._frames = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            logger.warning("Replay file not found: %s", self.replay_path)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse replay file %s: %s", self.replay_path, e)

    def collect(self) -> BodyState:
        """Return the current replay frame or a fallback unknown state."""
        self._load()
        if not self._frames:
            return BodyState(
                robot_id=self.robot_id,
                timestamp=time.time(),
                source="file_replay:empty",
            )
        frame = self._frames[self._index % len(self._frames)]
        state = BodyState.from_dict(frame)
        # Update timestamp to now for realistic replay
        state.timestamp = time.time()
        return state

    def step(self) -> None:
        """Advance to the next frame."""
        self._load()
        if self._frames:
            self._index = (self._index + 1) % len(self._frames)
