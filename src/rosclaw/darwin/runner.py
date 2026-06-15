"""Darwin benchmark runner."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .metrics import BenchmarkMetrics

logger = logging.getLogger("rosclaw.darwin.runner")


@dataclass
class SeedResult:
    """Result from a single seed."""

    seed: int = 0
    success_count: int = 0
    collision_count: int = 0
    total_episodes: int = 0
    completion_times: list[float] = field(default_factory=list)
    safety_violations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "success_count": self.success_count,
            "collision_count": self.collision_count,
            "total_episodes": self.total_episodes,
            "completion_times": self.completion_times,
            "safety_violations": self.safety_violations,
        }


class BenchmarkRunner:
    """Runs multi-seed benchmarks and aggregates metrics."""

    def __init__(self, episodes: int = 50, seeds: list[int] | None = None):
        self.episodes = episodes
        self.seeds = seeds or list(range(10))

    def run(self, skill_id: str, task_id: str, scenario: Any | None = None) -> BenchmarkMetrics:
        """Run benchmark and return aggregated metrics.

        In production this would invoke sandbox / real robot.
        Here we provide a deterministic simulation fallback.
        """
        import random

        seed_results: list[SeedResult] = []
        for seed in self.seeds:
            rng = random.Random(seed)
            # Deterministic simulation: skill_id hash influences base rate
            base_success = 0.35 + (hash(skill_id) % 100) / 300.0
            if scenario and hasattr(scenario, "difficulty"):
                if scenario.difficulty == "hard":
                    base_success -= 0.1
                elif scenario.difficulty == "extreme":
                    base_success -= 0.2
            success_count = sum(
                1 for _ in range(self.episodes) if rng.random() < base_success
            )
            collision_count = sum(
                1 for _ in range(self.episodes) if rng.random() < 0.05
            )
            completion_times = [rng.uniform(2.0, 8.0) for _ in range(success_count)]
            sr = SeedResult(
                seed=seed,
                success_count=success_count,
                collision_count=collision_count,
                total_episodes=self.episodes,
                completion_times=completion_times,
                safety_violations=rng.randint(0, 2),
            )
            seed_results.append(sr)

        # Aggregate across seeds
        sum(s.success_count for s in seed_results)
        sum(s.collision_count for s in seed_results)
        total_episodes = sum(s.total_episodes for s in seed_results)
        all_times = [t for s in seed_results for t in s.completion_times]
        total_safety = sum(s.safety_violations for s in seed_results)

        success_rates = [s.success_count / s.total_episodes for s in seed_results]
        collision_rates = [s.collision_count / s.total_episodes for s in seed_results]

        import statistics

        return BenchmarkMetrics(
            success_rate=statistics.mean(success_rates) if success_rates else 0.0,
            success_rate_std=statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0,
            collision_rate=statistics.mean(collision_rates) if collision_rates else 0.0,
            collision_rate_std=statistics.stdev(collision_rates) if len(collision_rates) > 1 else 0.0,
            completion_time_mean=statistics.mean(all_times) if all_times else 0.0,
            completion_time_std=statistics.stdev(all_times) if len(all_times) > 1 else 0.0,
            safety_violation_count=total_safety,
            seeds=len(self.seeds),
            episodes=total_episodes,
            raw=[s.to_dict() for s in seed_results],
        )
