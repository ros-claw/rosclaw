"""Lightweight metrics aggregation for proposal/shadow rollouts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutMetrics:
    """Collect timing and counters for a rollout."""

    step_latencies_ms: list[float] = field(default_factory=list)
    inference_latencies_ms: list[float] = field(default_factory=list)
    mapping_latencies_ms: list[float] = field(default_factory=list)
    sandbox_latencies_ms: list[float] = field(default_factory=list)
    observation_validation_failures: int = 0
    mapping_blocks: int = 0
    sandbox_blocks: int = 0
    nan_inf_blocks: int = 0
    deadline_misses: int = 0
    hardware_actions_executed: int = 0

    def record_step(self, latency_ms: float) -> None:
        self.step_latencies_ms.append(latency_ms)

    def record_inference(self, latency_ms: float) -> None:
        self.inference_latencies_ms.append(latency_ms)

    def record_mapping(self, latency_ms: float) -> None:
        self.mapping_latencies_ms.append(latency_ms)

    def record_sandbox(self, latency_ms: float) -> None:
        self.sandbox_latencies_ms.append(latency_ms)

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"count": 0, "min_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0, "p95_ms": 0.0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        p95 = sorted_vals[int(n * 0.95)] if n >= 2 else sorted_vals[0]
        return {
            "count": n,
            "min_ms": round(sorted_vals[0], 3),
            "max_ms": round(sorted_vals[-1], 3),
            "mean_ms": round(sum(sorted_vals) / n, 3),
            "p95_ms": round(p95, 3),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_latency_ms": self._summary(self.step_latencies_ms),
            "inference_latency_ms": self._summary(self.inference_latencies_ms),
            "mapping_latency_ms": self._summary(self.mapping_latencies_ms),
            "sandbox_latency_ms": self._summary(self.sandbox_latencies_ms),
            "observation_validation_failures": self.observation_validation_failures,
            "mapping_blocks": self.mapping_blocks,
            "sandbox_blocks": self.sandbox_blocks,
            "nan_inf_blocks": self.nan_inf_blocks,
            "deadline_misses": self.deadline_misses,
            "hardware_actions_executed": self.hardware_actions_executed,
        }


class StepTimer:
    """Context manager that records elapsed wall time in milliseconds."""

    def __init__(self) -> None:
        self.start_ns: int = 0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "StepTimer":
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter_ns() - self.start_ns) / 1e6
