"""Mock runtime adapter for testing the practice runtime without real robots."""

from __future__ import annotations

from collections.abc import Iterable
from time import monotonic_ns
from typing import Any

from rosclaw.practice.adapters.base import SourceAdapter, SourceHealth
from rosclaw.practice.schemas import ExecutedActionPayload, PracticeEventEnvelope


class MockRuntimeAdapter(SourceAdapter):
    """Generates synthetic runtime/action events."""

    source_name = "runtime"

    def __init__(self, robot_id: str):
        self._robot_id = robot_id
        self._practice_id: str | None = None
        self._step = 0
        self._running = False
        self._action_id = ""

    def start(self, session: Any) -> None:
        self._practice_id = getattr(session, "practice_id", None)
        self._running = True
        self._step = 0
        self._action_id = f"action_{self._practice_id}"

    def stop(self) -> None:
        self._running = False

    def health(self) -> SourceHealth:
        return SourceHealth(source=self.source_name, healthy=True)

    def poll(self) -> Iterable[PracticeEventEnvelope]:
        if not self._running or self._practice_id is None:
            return

        self._step += 1
        if self._step == 1:
            yield PracticeEventEnvelope(
                practice_id=self._practice_id,
                robot_id=self._robot_id,
                source="runtime",
                event_type="runtime.action_proposed",
                payload={
                    "action_id": self._action_id,
                    "action_type": "mock_move",
                    "command": {"target": [0.5, 0.0, 0.2]},
                },
            )
        elif self._step == 2:
            start_ns = monotonic_ns()
            action = ExecutedActionPayload(
                action_id=self._action_id,
                action_type="mock_move",
                command={"target": [0.5, 0.0, 0.2]},
                controller="mock",
                start_time_ns=start_ns,
                status="completed",
                reward=0.8,
            )
            yield PracticeEventEnvelope(
                practice_id=self._practice_id,
                robot_id=self._robot_id,
                source="runtime",
                event_type="runtime.action_executed",
                payload=action.model_dump(),
            )
        elif self._step == 3:
            yield PracticeEventEnvelope(
                practice_id=self._practice_id,
                robot_id=self._robot_id,
                source="runtime",
                event_type="runtime.reward_observed",
                payload={"action_id": self._action_id, "reward": 0.8},
            )

    def on_event(self, callback: Any) -> None:
        pass
