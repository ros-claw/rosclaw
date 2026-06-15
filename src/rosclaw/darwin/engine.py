"""Darwin Engine — multi-seed evaluation and regression detection."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from rosclaw.core.event_bus import Event, EventPriority

from .events import DarwinBenchmarkCompletedEvent
from .runner import BenchmarkRunner
from .scenarios import StressScenario, get_scenario

logger = logging.getLogger("rosclaw.darwin.engine")


class DarwinEngine:
    """Evaluation pressure engine for ROSClaw skills.

    Runs multi-seed benchmarks, detects regressions, and publishes
    DarwinBenchmarkCompletedEvent for Auto to consume.
    """

    def __init__(
        self,
        event_bus: Any | None = None,
        seekdb_client: Any | None = None,
        default_seeds: int = 10,
        default_episodes: int = 50,
    ):
        self._bus = event_bus
        self._seekdb = seekdb_client
        self.default_seeds = default_seeds
        self.default_episodes = default_episodes
        self._history: list[DarwinBenchmarkCompletedEvent] = []

    def run_benchmark(
        self,
        task_id: str,
        skill_id: str,
        candidate_skill_id: str | None = None,
        scenario_id: str | None = None,
        seeds: int | None = None,
        episodes: int | None = None,
    ) -> DarwinBenchmarkCompletedEvent:
        """Run a benchmark and return the completed event."""
        benchmark_id = f"darwin_{uuid.uuid4().hex[:8]}"
        seeds_list = list(range(seeds or self.default_seeds))
        eps = episodes or self.default_episodes

        scenario: StressScenario | None = None
        if scenario_id:
            scenario = get_scenario(scenario_id)

        runner = BenchmarkRunner(episodes=eps, seeds=seeds_list)

        # Run baseline
        baseline_metrics = runner.run(skill_id, task_id, scenario)

        # Run candidate if provided
        candidate_metrics = baseline_metrics
        if candidate_skill_id:
            candidate_metrics = runner.run(candidate_skill_id, task_id, scenario)

        regression = candidate_metrics.is_regression(baseline_metrics)

        event = DarwinBenchmarkCompletedEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            benchmark_id=benchmark_id,
            task_id=task_id,
            skill_id=skill_id,
            candidate_skill_id=candidate_skill_id or "",
            metrics=candidate_metrics.to_dict(),
            baseline_metrics=baseline_metrics.to_dict(),
            regression_detected=regression,
            seeds=len(seeds_list),
            episodes=eps,
            source="rosclaw-darwin",
        )

        self._history.append(event)

        # Write to SeekDB
        if self._seekdb is not None:
            try:
                self._seekdb.insert(
                    "darwin_benchmarks",
                    {
                        "id": benchmark_id,
                        "task_id": task_id,
                        "skill_id": skill_id,
                        "candidate_skill_id": candidate_skill_id or "",
                        "metrics": event.metrics,
                        "baseline_metrics": event.baseline_metrics,
                        "regression_detected": regression,
                        "seeds": len(seeds_list),
                        "episodes": eps,
                    },
                )
            except Exception as exc:
                logger.warning("SeekDB write failed: %s", exc)

        # Publish event
        if self._bus is not None:
            try:
                core_event = Event(
                    topic="rosclaw.darwin.benchmark.completed",
                    payload=event.to_dict(),
                    source="rosclaw-darwin",
                    priority=EventPriority.NORMAL,
                    trace_id=event.trace_id,
                )
                self._bus.publish(core_event)
            except Exception as exc:
                logger.warning("EventBus publish failed: %s", exc)

        logger.info(
            "Benchmark %s: baseline=%.2f candidate=%.2f regression=%s",
            benchmark_id,
            baseline_metrics.success_rate,
            candidate_metrics.success_rate,
            regression,
        )
        return event

    def list_history(self, task_id: str | None = None) -> list[DarwinBenchmarkCompletedEvent]:
        if task_id is None:
            return list(self._history)
        return [e for e in self._history if e.task_id == task_id]
