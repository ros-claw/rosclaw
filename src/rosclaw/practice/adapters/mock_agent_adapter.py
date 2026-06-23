"""Mock agent adapter for testing the practice runtime without a real agent."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rosclaw.practice.adapters.base import SourceAdapter, SourceHealth
from rosclaw.practice.schemas import (
    AgentPlanPayload,
    PracticeEventEnvelope,
    ToolCallPayload,
)


class MockAgentAdapter(SourceAdapter):
    """Generates synthetic agent trace events."""

    source_name = "agent"

    def __init__(self, robot_id: str, task: str = "mock task"):
        self._robot_id = robot_id
        self._task = task
        self._practice_id: str | None = None
        self._step = 0
        self._running = False

    def start(self, session: Any) -> None:
        self._practice_id = getattr(session, "practice_id", None)
        self._running = True
        self._step = 0

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
                source="agent",
                event_type="agent.task_received",
                payload={"task": self._task},
            )
        elif self._step == 2:
            plan = AgentPlanPayload(
                task=self._task,
                plan_id=f"plan_{self._practice_id}",
                planner="mock_agent",
                plan_steps=[
                    {"step": 1, "name": "perceive"},
                    {"step": 2, "name": "plan"},
                    {"step": 3, "name": "act"},
                ],
                decision_summary="Mock plan generated for testing.",
            )
            yield PracticeEventEnvelope(
                practice_id=self._practice_id,
                robot_id=self._robot_id,
                source="agent",
                event_type="agent.plan_created",
                payload=plan.model_dump(),
            )
        elif self._step == 3:
            tool = ToolCallPayload(
                tool_call_id=f"tc_{self._practice_id}",
                tool_name="mock_grasp",
                arguments={"object": "cup"},
                status="started",
            )
            yield PracticeEventEnvelope(
                practice_id=self._practice_id,
                robot_id=self._robot_id,
                source="agent",
                event_type="agent.tool_call_started",
                payload=tool.model_dump(),
            )
        elif self._step == 4:
            tool = ToolCallPayload(
                tool_call_id=f"tc_{self._practice_id}",
                tool_name="mock_grasp",
                arguments={"object": "cup"},
                result_summary={"success": True},
                status="success",
                latency_ms=12.0,
            )
            yield PracticeEventEnvelope(
                practice_id=self._practice_id,
                robot_id=self._robot_id,
                source="agent",
                event_type="agent.tool_call_finished",
                payload=tool.model_dump(),
            )

    def on_event(self, callback: Any) -> None:
        pass
