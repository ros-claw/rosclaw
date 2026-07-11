"""Resampling primitives for Gate B.1 source-stream synchronization.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It implements the per-feature resampling strategies used to align
multi-rate source streams onto a canonical fixed-FPS timeline.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any

from rosclaw.integrations.lerobot.clock_mapping import ClockMappingResult
from rosclaw.integrations.lerobot.source_stream_schema import SourceStream
from rosclaw.integrations.lerobot.sync_config import SyncPolicy
from rosclaw.integrations.lerobot.sync_provenance import (
    PROVENANCE_AGGREGATED,
    PROVENANCE_EXACT,
    PROVENANCE_HELD,
    PROVENANCE_INTERPOLATED,
    PROVENANCE_NEAREST,
    PROVENANCE_UNKNOWN,
    VALIDITY_INVALID,
    VALIDITY_VALID,
)


_NS_TO_S = 1e-9
_EPS = 1e-9


@dataclass
class ResampledFeature:
    """Resampled values for one feature across all canonical frames."""

    key: str
    values: list[Any] = field(default_factory=list)
    valid: list[int] = field(default_factory=list)
    provenance: list[int] = field(default_factory=list)
    source_skew_ms: list[float | None] = field(default_factory=list)
    source_count: list[int | None] = field(default_factory=list)
    peak_values: list[Any] | None = None


def _samples_with_target_sec(
    stream: SourceStream,
    mapping_result: ClockMappingResult,
) -> list[tuple[float, Any, bool]]:
    """Return (target_sec, value_or_image_path, valid) for each sample."""
    mapping = None
    if stream.samples:
        clock = stream.samples[0].clock_domain
        mapping = mapping_result.mappings.get(clock)
    out: list[tuple[float, Any, bool]] = []
    for sample in stream.samples:
        ts_ns = sample.source_timestamp_ns
        if mapping is not None:
            ts_ns = mapping.apply(ts_ns)
        target_sec = ts_ns * _NS_TO_S
        if sample.image_path is not None:
            value = sample.image_path
        else:
            value = sample.value
        out.append((target_sec, value, sample.valid))
    return out


def _empty_resampled(key: str, num_frames: int) -> ResampledFeature:
    return ResampledFeature(
        key=key,
        values=[None] * num_frames,
        valid=[VALIDITY_INVALID] * num_frames,
        provenance=[PROVENANCE_UNKNOWN] * num_frames,
        source_skew_ms=[None] * num_frames,
        source_count=[None] * num_frames,
    )


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + alpha * (b - a)


def _lerp_value(
    v0: Any,
    v1: Any,
    alpha: float,
) -> Any:
    """Element-wise linear interpolation for scalars or equal-length lists."""
    if isinstance(v0, list) and isinstance(v1, list):
        if len(v0) != len(v1):
            return None
        return [_lerp(float(a), float(b), alpha) for a, b in zip(v0, v1)]
    return _lerp(float(v0), float(v1), alpha)


def _vectorize_binary(
    values: list[Any],
    op: Any,
    initializer: Any | None = None,
) -> Any:
    """Reduce a list of vector/scalar values with ``op``."""
    if not values:
        return None
    first = values[0]
    if isinstance(first, list):
        dim = len(first)
        result = list(first) if initializer is None else [initializer] * dim
        for value in values[1:]:
            result = [op(r, v) for r, v in zip(result, value)]
        return result
    result = first if initializer is None else initializer
    for value in values[1:]:
        result = op(result, value)
    return result


def resample_linear(
    samples: list[tuple[float, Any, bool]],
    timestamps_sec: list[float],
    max_gap_ms: float | None,
) -> ResampledFeature:
    """Linear interpolation resampling."""
    key = "linear"
    num_frames = len(timestamps_sec)
    result = _empty_resampled(key, num_frames)
    if not samples:
        return result

    times = [s[0] for s in samples]
    first_t, last_t = times[0], times[-1]
    max_gap = (max_gap_ms / 1000.0) if max_gap_ms is not None else float("inf")

    for i, t in enumerate(timestamps_sec):
        if t + _EPS < first_t or t - _EPS > last_t:
            continue

        idx = bisect.bisect_right(times, t + _EPS) - 1
        if idx < 0:
            idx = 0

        t0 = times[idx]
        v0 = samples[idx][1]
        valid0 = samples[idx][2]

        if abs(t - t0) <= _EPS:
            if valid0:
                result.values[i] = v0
                result.valid[i] = VALIDITY_VALID
                result.provenance[i] = PROVENANCE_EXACT
                result.source_skew_ms[i] = 0.0
                result.source_count[i] = 1
            continue

        if idx + 1 >= len(samples):
            continue
        t1 = times[idx + 1]
        v1 = samples[idx + 1][1]
        valid1 = samples[idx + 1][2]
        if not (valid0 and valid1):
            continue
        if t1 - t0 > max_gap + _EPS:
            continue
        if t + _EPS < t0 or t - _EPS > t1:
            continue

        alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
        value = _lerp_value(v0, v1, alpha)
        if value is None:
            continue
        result.values[i] = value
        result.valid[i] = VALIDITY_VALID
        result.provenance[i] = PROVENANCE_INTERPOLATED
        result.source_skew_ms[i] = 0.0
        result.source_count[i] = 2

    return result


def resample_previous(
    samples: list[tuple[float, Any, bool]],
    timestamps_sec: list[float],
    max_age_ms: float | None,
) -> ResampledFeature:
    """Hold-last resampling."""
    key = "previous"
    num_frames = len(timestamps_sec)
    result = _empty_resampled(key, num_frames)
    if not samples:
        return result

    times = [s[0] for s in samples]
    first_t, last_t = times[0], times[-1]
    max_age = (max_age_ms / 1000.0) if max_age_ms is not None else float("inf")

    for i, t in enumerate(timestamps_sec):
        if t + _EPS < first_t or t - _EPS > last_t:
            continue

        idx = bisect.bisect_right(times, t + _EPS) - 1
        if idx < 0:
            continue

        t0 = times[idx]
        age = t - t0
        if age > max_age + _EPS:
            continue

        sample = samples[idx]
        if not sample[2]:
            continue

        result.values[i] = sample[1]
        result.valid[i] = VALIDITY_VALID
        result.provenance[i] = PROVENANCE_EXACT if age <= _EPS else PROVENANCE_HELD
        result.source_skew_ms[i] = age * 1000.0
        result.source_count[i] = 1

    return result


def resample_nearest(
    samples: list[tuple[float, Any, bool]],
    timestamps_sec: list[float],
    max_skew_ms: float | None,
) -> ResampledFeature:
    """Nearest-neighbor resampling with skew limit."""
    key = "nearest"
    num_frames = len(timestamps_sec)
    result = _empty_resampled(key, num_frames)
    if not samples:
        return result

    times = [s[0] for s in samples]
    first_t, last_t = times[0], times[-1]
    max_skew = (max_skew_ms / 1000.0) if max_skew_ms is not None else float("inf")

    for i, t in enumerate(timestamps_sec):
        if t + _EPS < first_t or t - _EPS > last_t:
            continue

        idx = bisect.bisect_left(times, t)
        candidates: list[int] = []
        if 0 <= idx < len(times):
            candidates.append(idx)
        if idx - 1 >= 0:
            candidates.append(idx - 1)
        if not candidates:
            continue

        best_idx = min(candidates, key=lambda k: abs(times[k] - t))
        skew = abs(times[best_idx] - t)
        if skew > max_skew + _EPS:
            continue

        sample = samples[best_idx]
        if not sample[2]:
            continue

        result.values[i] = sample[1]
        result.valid[i] = VALIDITY_VALID
        result.provenance[i] = PROVENANCE_EXACT if skew <= _EPS else PROVENANCE_NEAREST
        result.source_skew_ms[i] = skew * 1000.0
        result.source_count[i] = 1

    return result


def resample_interval_mean(
    samples: list[tuple[float, Any, bool]],
    timestamps_sec: list[float],
    fps: float,
    emit_peak_abs: bool = False,
) -> ResampledFeature:
    """Interval mean resampling with optional per-axis peak absolute value."""
    key = "interval_mean"
    num_frames = len(timestamps_sec)
    result = _empty_resampled(key, num_frames)
    result.peak_values = [None] * num_frames if emit_peak_abs else None
    if not samples:
        return result

    half_window = 0.5 / fps

    for i, t in enumerate(timestamps_sec):
        low = t - half_window
        high = t + half_window
        window = [
            s for s in samples
            if s[2] and low - _EPS <= s[0] < high - _EPS
        ]
        if not window:
            continue

        values = [s[1] for s in window]
        mean_value = _vectorize_binary(values, lambda a, b: a + b)
        if mean_value is None:
            continue
        n = len(values)
        if isinstance(mean_value, list):
            mean_value = [v / n for v in mean_value]
            if emit_peak_abs:
                peak = [max(abs(float(v)) for v in dim_values) for dim_values in zip(*values)]
                result.peak_values[i] = peak
        else:
            mean_value = mean_value / n
            if emit_peak_abs:
                result.peak_values[i] = max(abs(float(v)) for v in values)

        result.values[i] = mean_value
        result.valid[i] = VALIDITY_VALID
        result.provenance[i] = PROVENANCE_AGGREGATED
        result.source_skew_ms[i] = None
        result.source_count[i] = n

    return result


def resample_interval_any(
    samples: list[tuple[float, Any, bool]],
    timestamps_sec: list[float],
    fps: float,
) -> ResampledFeature:
    """Interval any-true resampling for boolean/discrete features.

    Output per-axis:
      1 if any true in window
      0 if any false and no true in window
      missing if no samples in window
    """
    key = "interval_any"
    num_frames = len(timestamps_sec)
    result = _empty_resampled(key, num_frames)
    if not samples:
        return result

    half_window = 0.5 / fps

    for i, t in enumerate(timestamps_sec):
        low = t - half_window
        high = t + half_window
        window = [
            s for s in samples
            if s[2] and low - _EPS <= s[0] < high - _EPS
        ]
        if not window:
            continue

        bool_values = [s[1] for s in window]
        # Reduce element-wise OR then AND-NOT-OR to detect explicit false.
        if isinstance(bool_values[0], list):
            dim = len(bool_values[0])
            any_true = [any(bool(v[d]) for v in bool_values) for d in range(dim)]
            any_false = [any(not bool(v[d]) for v in bool_values) for d in range(dim)]
            out = [1 if any_true[d] else (0 if any_false[d] else -1) for d in range(dim)]
        else:
            any_true = any(bool(v) for v in bool_values)
            any_false = any(not bool(v) for v in bool_values)
            out = 1 if any_true else (0 if any_false else -1)

        result.values[i] = out
        result.valid[i] = VALIDITY_VALID
        result.provenance[i] = PROVENANCE_AGGREGATED
        result.source_skew_ms[i] = None
        result.source_count[i] = len(window)

    return result


def resample_stream(
    stream: SourceStream,
    timestamps_sec: list[float],
    policy: SyncPolicy,
    mapping_result: ClockMappingResult,
) -> ResampledFeature:
    """Dispatch to the correct resampling strategy for ``stream``."""
    samples = _samples_with_target_sec(stream, mapping_result)

    if policy.method == "linear":
        return resample_linear(samples, timestamps_sec, policy.max_gap_ms)
    if policy.method == "previous":
        return resample_previous(samples, timestamps_sec, policy.max_age_ms)
    if policy.method == "nearest":
        return resample_nearest(samples, timestamps_sec, policy.max_skew_ms)
    if policy.method == "interval_mean":
        result = resample_interval_mean(
            samples, timestamps_sec, 1.0 / (timestamps_sec[1] - timestamps_sec[0]) if len(timestamps_sec) > 1 else 10.0,
            emit_peak_abs=policy.emit_peak_abs or policy.emit_peak,
        )
        return result
    if policy.method == "interval_any":
        return resample_interval_any(
            samples, timestamps_sec, 1.0 / (timestamps_sec[1] - timestamps_sec[0]) if len(timestamps_sec) > 1 else 10.0
        )

    # Fallback to previous.
    return resample_previous(samples, timestamps_sec, policy.max_age_ms)


__all__ = [
    "ResampledFeature",
    "resample_interval_any",
    "resample_interval_mean",
    "resample_linear",
    "resample_nearest",
    "resample_previous",
    "resample_stream",
]
