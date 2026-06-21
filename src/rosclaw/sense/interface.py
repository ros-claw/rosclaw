"""High-level SenseInterface used by CLI, MCP, and Runtime integration."""

from __future__ import annotations

from typing import Any

from rosclaw.core.event_bus import EventBus
from rosclaw.sense.collectors import MockCollector
from rosclaw.sense.config import SenseConfig
from rosclaw.sense.runtime import SenseRuntime
from rosclaw.sense.schemas import BodyReadiness, BodySense, BodyState


class SenseInterface:
    """Convenience wrapper around SenseRuntime for one-off queries.

    This is useful for CLI commands and ad-hoc diagnostics where a long-lived
    background loop is not required.
    """

    def __init__(
        self,
        robot_id: str = "rosclaw_default",
        collector: str = "mock",
        scenario: str = "normal",
        event_bus: EventBus | None = None,
    ):
        self.robot_id = robot_id
        self.event_bus = event_bus or EventBus()
        self._runtime = SenseRuntime(
            config=SenseConfig(
                robot_id=robot_id,
                collector=collector,
                update_hz=0.0,  # no background loop
            ),
            event_bus=self.event_bus,
        )
        # If using mock collector, override scenario through the collector directly.
        if collector == "mock" and isinstance(self._runtime._collector, MockCollector):
            self._runtime._collector.scenario = scenario

    def initialize(self) -> None:
        self._runtime.initialize()

    def stop(self) -> None:
        self._runtime.stop()

    def get_body_sense(self) -> BodySense:
        """Return a fresh BodySense snapshot."""
        return self._runtime.tick()

    def tick(self) -> BodySense:
        """Alias for :meth:`get_body_sense` used by sense-aware adapters."""
        return self._runtime.tick()

    def get_body_state(self) -> BodyState:
        """Return the latest raw BodyState."""
        return self._runtime.get_latest_state() or self._runtime._collector.collect()

    def get_latest_state(self) -> BodyState | None:
        """Alias for :meth:`get_body_state`."""
        return self.get_body_state()

    def get_latest_sense(self) -> BodySense | None:
        """Return the latest cached BodySense without forcing a tick."""
        return self._runtime.get_latest_sense()

    def get_readiness(
        self,
        task: str | None = None,
        requirements: dict[str, Any] | None = None,
    ) -> BodyReadiness:
        """Evaluate readiness for a task or all capabilities."""
        return self._runtime.get_readiness(task=task, requirements=requirements)

    def explain_block(self, task: str) -> str:
        """Explain why a task is blocked or degraded."""
        return self._runtime.explain_block(task)
