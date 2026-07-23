"""Continuous safety and temporal robustness monitors."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SafetyPredicateMonitor:
    """Compute rho for the six safety margins defined by SimForge 2.0."""

    required_clearance_m: float = 0.02
    velocity_limit: float = 1.0
    force_limit: float = 100.0
    deadline_sec: float = 5.0
    stop_distance_limit_m: float = 0.25
    max_observations: int = 100_000
    _observations: list[dict[str, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        limits = (
            self.required_clearance_m,
            self.velocity_limit,
            self.force_limit,
            self.deadline_sec,
            self.stop_distance_limit_m,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in limits
        ):
            raise ValueError("safety monitor limits must be finite and positive")
        if (
            isinstance(self.max_observations, bool)
            or not isinstance(self.max_observations, int)
            or not 1 <= self.max_observations <= 1_000_000
        ):
            raise ValueError("max_observations must be in [1, 1000000]")

    def observe(self, metrics: dict[str, Any]) -> float:
        if len(self._observations) >= self.max_observations:
            raise RuntimeError("safety monitor observation budget exhausted")
        values = {
            "clearance_margin": _lower_margin(
                metrics, "minimum_clearance_m", self.required_clearance_m
            ),
            "joint_margin": _lower_margin(metrics, "joint_limit_margin_rad", 0.0),
            "velocity_margin": _upper_margin(metrics, "peak_velocity", self.velocity_limit),
            "force_margin": _upper_margin(metrics, "peak_force_n", self.force_limit),
            "deadline_margin": _upper_margin(metrics, "elapsed_sec", self.deadline_sec),
            "stop_distance_margin": _upper_margin(
                metrics, "actual_stop_distance_m", self.stop_distance_limit_m
            ),
        }
        rho = min(values.values())
        values["rho"] = rho
        self._observations.append(values)
        return rho

    @property
    def robustness(self) -> float:
        if not self._observations:
            return -math.inf
        return min(observation["rho"] for observation in self._observations)

    @property
    def margins(self) -> dict[str, float]:
        if not self._observations:
            return {}
        names = self._observations[0].keys()
        return {name: min(item[name] for item in self._observations) for name in names}


@dataclass
class TemporalPredicateMonitor:
    """Bounded G(always) and F(eventually) monitor with quantitative margins."""

    horizon_sec: float
    max_observations: int = 100_000
    _always: list[tuple[float, float]] = field(default_factory=list, init=False)
    _eventually: list[tuple[float, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.horizon_sec, bool)
            or not isinstance(self.horizon_sec, (int, float))
            or not math.isfinite(float(self.horizon_sec))
            or self.horizon_sec <= 0
        ):
            raise ValueError("temporal horizon must be finite and positive")
        if (
            isinstance(self.max_observations, bool)
            or not isinstance(self.max_observations, int)
            or not 1 <= self.max_observations <= 1_000_000
        ):
            raise ValueError("max_observations must be in [1, 1000000]")

    def observe_always(self, *, timestamp_sec: float, margin: float) -> None:
        self._append(self._always, timestamp_sec, margin)

    def observe_eventually(self, *, timestamp_sec: float, margin: float) -> None:
        self._append(self._eventually, timestamp_sec, margin)

    @property
    def always_robustness(self) -> float:
        values = [margin for timestamp, margin in self._always if timestamp <= self.horizon_sec]
        return min(values) if values else -math.inf

    @property
    def eventually_robustness(self) -> float:
        values = [margin for timestamp, margin in self._eventually if timestamp <= self.horizon_sec]
        return max(values) if values else -math.inf

    @property
    def satisfied(self) -> bool:
        return self.always_robustness >= 0 and self.eventually_robustness >= 0

    def _append(self, target: list[tuple[float, float]], timestamp: float, margin: float) -> None:
        if len(self._always) + len(self._eventually) >= self.max_observations:
            raise RuntimeError("temporal monitor observation budget exhausted")
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        normalized = float(margin)
        if not math.isfinite(normalized):
            normalized = -math.inf
        target.append((timestamp, normalized))


class RobustnessAggregator:
    @staticmethod
    def minimum(values: list[float] | tuple[float, ...]) -> float:
        finite = _finite_values(values)
        return min(finite) if finite else -math.inf

    @staticmethod
    def quantile(values: list[float] | tuple[float, ...], probability: float) -> float:
        if not 0 <= probability <= 1:
            raise ValueError("quantile probability must be in [0, 1]")
        ordered = sorted(_finite_values(values))
        if not ordered:
            return -math.inf
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    @classmethod
    def lower_tail_cvar(
        cls, values: list[float] | tuple[float, ...], probability: float = 0.05
    ) -> float:
        if not 0 < probability <= 1:
            raise ValueError("CVaR probability must be in (0, 1]")
        ordered = sorted(_finite_values(values))
        if not ordered:
            return -math.inf
        count = max(1, math.ceil(len(ordered) * probability))
        return sum(ordered[:count]) / count


def _finite(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def _lower_margin(metrics: dict[str, Any], name: str, lower_bound: float) -> float:
    value = _finite(metrics, name)
    return value - lower_bound if value is not None else -math.inf


def _upper_margin(metrics: dict[str, Any], name: str, upper_bound: float) -> float:
    value = _finite(metrics, name)
    return upper_bound - value if value is not None else -math.inf


def _finite_values(values: list[float] | tuple[float, ...]) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        normalized = float(value)
        if math.isfinite(normalized):
            result.append(normalized)
    return result


__all__ = [
    "RobustnessAggregator",
    "SafetyPredicateMonitor",
    "TemporalPredicateMonitor",
]
