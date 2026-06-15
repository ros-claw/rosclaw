"""Darwin event schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DarwinBenchmarkCompletedEvent:
    """Emitted when a Darwin benchmark run completes."""

    event_id: str = ""
    benchmark_id: str = ""
    task_id: str = ""
    skill_id: str = ""
    candidate_skill_id: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    regression_detected: bool = False
    seeds: int = 0
    episodes: int = 0
    trace_id: str = ""
    source: str = "rosclaw-darwin"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "benchmark_id": self.benchmark_id,
            "task_id": self.task_id,
            "skill_id": self.skill_id,
            "candidate_skill_id": self.candidate_skill_id,
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "regression_detected": self.regression_detected,
            "seeds": self.seeds,
            "episodes": self.episodes,
            "trace_id": self.trace_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DarwinBenchmarkCompletedEvent:
        return cls(
            event_id=d.get("event_id", ""),
            benchmark_id=d.get("benchmark_id", ""),
            task_id=d.get("task_id", ""),
            skill_id=d.get("skill_id", ""),
            candidate_skill_id=d.get("candidate_skill_id", ""),
            metrics=dict(d.get("metrics", {})),
            baseline_metrics=dict(d.get("baseline_metrics", {})),
            regression_detected=d.get("regression_detected", False),
            seeds=d.get("seeds", 0),
            episodes=d.get("episodes", 0),
            trace_id=d.get("trace_id", ""),
            source=d.get("source", "rosclaw-darwin"),
        )
