"""Event-stream resampling for Gate B.1 source-stream synchronization.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It turns discrete event streams (sandbox decisions, failure flags,
intervention flags) into per-canonical-frame values.
"""

from __future__ import annotations

import bisect
from typing import Any

from rosclaw.integrations.lerobot.clock_mapping import ClockMappingResult
from rosclaw.integrations.lerobot.resample import _empty_resampled
from rosclaw.integrations.lerobot.source_stream_schema import SourceStream
from rosclaw.integrations.lerobot.sync_config import SyncPolicy
from rosclaw.integrations.lerobot.sync_provenance import (
    PROVENANCE_AGGREGATED,
    PROVENANCE_EXACT,
    PROVENANCE_HELD,
    VALIDITY_VALID,
)

_EPS = 1e-9
_NS_TO_S = 1e-9


def _samples_with_target_sec(
    stream: SourceStream,
    mapping_result: ClockMappingResult,
) -> list[tuple[float, dict[str, Any], bool]]:
    """Return (target_sec, event_dict, valid) for each event sample."""
    mapping = None
    if stream.samples:
        clock = stream.samples[0].clock_domain
        mapping = mapping_result.mappings.get(clock)
    out: list[tuple[float, dict[str, Any], bool]] = []
    for sample in stream.samples:
        ts_ns = sample.source_timestamp_ns
        if mapping is not None:
            ts_ns = mapping.apply(ts_ns)
        target_sec = ts_ns * _NS_TO_S
        out.append((target_sec, sample.event or {}, sample.valid))
    return out


def resample_event_active_interval_any(
    stream: SourceStream,
    timestamps_sec: list[float],
    mapping_result: ClockMappingResult,
    fps: float,
) -> Any:
    """Resample an event stream whose ``event.active`` boolean is aggregated per window."""
    samples = _samples_with_target_sec(stream, mapping_result)
    key = stream.key
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

        any_true = any(bool(s[1].get("active")) for s in window)
        any_false = any(not bool(s[1].get("active")) for s in window)
        out = 1 if any_true else (0 if any_false else -1)

        result.values[i] = out
        result.valid[i] = VALIDITY_VALID
        result.provenance[i] = PROVENANCE_AGGREGATED
        result.source_count[i] = len(window)

    return result


def resample_sandbox_decision_previous(
    stream: SourceStream,
    timestamps_sec: list[float],
    mapping_result: ClockMappingResult,
    max_age_ms: float | None = None,
) -> Any:
    """Hold-last resampling for sandbox decision strings."""
    samples = _samples_with_target_sec(stream, mapping_result)
    key = stream.key
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
        decision = sample[1].get("decision")
        if decision is None:
            continue

        result.values[i] = str(decision)
        result.valid[i] = VALIDITY_VALID
        result.provenance[i] = PROVENANCE_EXACT if age <= _EPS else PROVENANCE_HELD
        result.source_skew_ms[i] = age * 1000.0
        result.source_count[i] = 1

    return result


def resample_event_stream(
    stream: SourceStream,
    timestamps_sec: list[float],
    policy: SyncPolicy,
    mapping_result: ClockMappingResult,
    fps: float,
) -> Any:
    """Dispatch event resampling based on stream key and policy."""
    if stream.key == "rosclaw.sandbox":
        return resample_sandbox_decision_previous(
            stream, timestamps_sec, mapping_result, max_age_ms=policy.max_age_ms
        )
    return resample_event_active_interval_any(stream, timestamps_sec, mapping_result, fps)


__all__ = [
    "resample_event_active_interval_any",
    "resample_event_stream",
    "resample_sandbox_decision_previous",
]
