"""Darwin metric aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics across seeds."""

    success_rate: float = 0.0
    success_rate_std: float = 0.0
    collision_rate: float = 0.0
    collision_rate_std: float = 0.0
    completion_time_mean: float = 0.0
    completion_time_std: float = 0.0
    safety_violation_count: int = 0
    seeds: int = 0
    episodes: int = 0
    raw: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_rate": self.success_rate,
            "success_rate_std": self.success_rate_std,
            "collision_rate": self.collision_rate,
            "collision_rate_std": self.collision_rate_std,
            "completion_time_mean": self.completion_time_mean,
            "completion_time_std": self.completion_time_std,
            "safety_violation_count": self.safety_violation_count,
            "seeds": self.seeds,
            "episodes": self.episodes,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkMetrics:
        return cls(
            success_rate=d.get("success_rate", 0.0),
            success_rate_std=d.get("success_rate_std", 0.0),
            collision_rate=d.get("collision_rate", 0.0),
            collision_rate_std=d.get("collision_rate_std", 0.0),
            completion_time_mean=d.get("completion_time_mean", 0.0),
            completion_time_std=d.get("completion_time_std", 0.0),
            safety_violation_count=d.get("safety_violation_count", 0),
            seeds=d.get("seeds", 0),
            episodes=d.get("episodes", 0),
            raw=list(d.get("raw", [])),
        )

    @property
    def regression_threshold(self) -> float:
        return 0.05

    def is_regression(self, baseline: BenchmarkMetrics) -> bool:
        """Detect regression vs baseline."""
        if self.success_rate < baseline.success_rate - self.regression_threshold:
            return True
        if self.collision_rate > baseline.collision_rate + self.regression_threshold:
            return True
        return self.safety_violation_count > baseline.safety_violation_count
