"""Rolling-window regime features (数据库优化v4 §4.4).

Windows:

* short — last ``short_window_rounds`` samples (5–10 rounds)
* medium — last ``medium_window_rounds`` samples (~30 rounds)
* long — the whole session

A feature that has no data is ``None`` and listed in ``missing_features`` —
missing temperature is never treated as "temperature matches" (v4 §4.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TelemetrySample:
    """One round/action-level observation used for regime features."""

    timestamp: float
    temperature_c: float | None = None
    position_error: float | None = None
    time_to_reach_ms: float | None = None
    invalid: bool = False
    failure: bool = False
    comm_error: bool = False
    action_count: int = 0
    gesture_interval_sec: float | None = None
    evidence_ref: str | None = None


@dataclass
class WindowStats:
    """Aggregated features over one rolling window."""

    sample_count: int = 0
    temperature_last: float | None = None
    temperature_slope_c_per_min: float | None = None
    position_error_mean: float | None = None
    position_error_p95: float | None = None
    time_to_reach_mean_ms: float | None = None
    time_to_reach_p95_ms: float | None = None
    invalid_rate: float | None = None
    failure_rate: float | None = None
    consecutive_failures: int = 0
    comm_error_rate: float | None = None
    action_frequency_hz: float | None = None
    gesture_interval_mean_sec: float | None = None
    gesture_interval_cv: float | None = None
    window_elapsed_sec: float = 0.0
    missing: list[str] = field(default_factory=list)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[rank]


def _slope_c_per_min(points: list[tuple[float, float]]) -> float | None:
    """Least-squares slope over (timestamp, temperature) points, °C/min."""
    if len(points) < 2:
        return None
    n = len(points)
    t0 = points[0][0]
    xs = [(t - t0) / 60.0 for t, _ in points]
    ys = [v for _, v in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denom


def compute_window(samples: list[TelemetrySample]) -> WindowStats:
    """Aggregate one window; missing data is reported, never defaulted."""
    stats = WindowStats(sample_count=len(samples))
    if not samples:
        stats.missing = ["all"]
        return stats

    temps = [(s.timestamp, s.temperature_c) for s in samples if s.temperature_c is not None]
    if temps:
        stats.temperature_last = temps[-1][1]
        stats.temperature_slope_c_per_min = _slope_c_per_min(temps)
    else:
        stats.missing.append("temperature_c")
        stats.missing.append("temperature_slope_c_per_min")

    pos_errors = [s.position_error for s in samples if s.position_error is not None]
    if pos_errors:
        stats.position_error_mean = sum(pos_errors) / len(pos_errors)
        stats.position_error_p95 = percentile(pos_errors, 0.95)
    else:
        stats.missing.append("position_error")

    ttrs = [s.time_to_reach_ms for s in samples if s.time_to_reach_ms is not None]
    if ttrs:
        stats.time_to_reach_mean_ms = sum(ttrs) / len(ttrs)
        stats.time_to_reach_p95_ms = percentile(ttrs, 0.95)
    else:
        stats.missing.append("time_to_reach_ms")

    stats.invalid_rate = sum(1 for s in samples if s.invalid) / len(samples)
    stats.failure_rate = sum(1 for s in samples if s.failure) / len(samples)
    consecutive = 0
    for sample in reversed(samples):
        if sample.failure or sample.invalid:
            consecutive += 1
        else:
            break
    stats.consecutive_failures = consecutive

    comm_total = sum(1 for s in samples if s.comm_error)
    stats.comm_error_rate = comm_total / len(samples)

    elapsed = max(samples[-1].timestamp - samples[0].timestamp, 0.0)
    stats.window_elapsed_sec = elapsed
    total_actions = sum(s.action_count for s in samples)
    if elapsed > 0 and total_actions > 0:
        stats.action_frequency_hz = total_actions / elapsed

    intervals = [s.gesture_interval_sec for s in samples if s.gesture_interval_sec is not None]
    if len(intervals) >= 2:
        mean = sum(intervals) / len(intervals)
        stats.gesture_interval_mean_sec = mean
        if mean > 0:
            variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
            stats.gesture_interval_cv = math.sqrt(variance) / mean
    else:
        stats.missing.append("gesture_interval_sec")
    return stats


def compute_windows(
    samples: list[TelemetrySample],
    *,
    short: int,
    medium: int,
) -> dict[str, WindowStats]:
    """short / medium / long(session) window stats (v4 §4.4)."""
    return {
        "short": compute_window(samples[-short:] if short > 0 else samples),
        "medium": compute_window(samples[-medium:] if medium > 0 else samples),
        "long": compute_window(samples),
    }


# ---------------------------------------------------------------------------
# EWMA / CUSUM (v4 §4.5 — deterministic change detection primitives)
# ---------------------------------------------------------------------------


@dataclass
class Ewma:
    """Exponentially weighted moving average with deviation tracking."""

    alpha: float
    mean: float | None = None
    variance: float = 0.0

    def update(self, value: float) -> float:
        if self.mean is None:
            self.mean = value
            self.variance = 0.0
            return self.mean
        delta = value - self.mean
        self.mean = self.mean + self.alpha * delta
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * delta * delta)
        return self.mean

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


@dataclass
class Cusum:
    """One-sided CUSUM for upward drift detection."""

    drift: float
    threshold: float
    accumulator: float = 0.0

    def update(self, value: float, reference: float) -> bool:
        """Feed a new value; returns True when the threshold is crossed."""
        self.accumulator = max(0.0, self.accumulator + (value - reference - self.drift))
        return self.accumulator > self.threshold

    def reset(self) -> None:
        self.accumulator = 0.0
