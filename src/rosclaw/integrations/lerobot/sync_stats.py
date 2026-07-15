"""Per-feature synchronization statistics for Gate B.1.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It turns a synchronization result into a structured statistics table
suitable for ``sync_stats.parquet``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.canonical_timeline import CanonicalTimeline
from rosclaw.integrations.lerobot.clock_mapping import ClockMappingResult
from rosclaw.integrations.lerobot.resample import ResampledFeature
from rosclaw.integrations.lerobot.source_stream_schema import SourceStream
from rosclaw.integrations.lerobot.sync_config import SyncConfig
from rosclaw.integrations.lerobot.sync_provenance import (
    PROVENANCE_AGGREGATED,
    PROVENANCE_EXACT,
    PROVENANCE_HELD,
    PROVENANCE_INTERPOLATED,
    PROVENANCE_NEAREST,
    PROVENANCE_UNKNOWN,
    VALIDITY_VALID,
)

_NS_TO_S = 1e-9


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _is_active(value: Any) -> bool:
    if isinstance(value, list):
        return any(v == 1 for v in value)
    return value == 1


@dataclass
class SyncFeatureStats:
    """Synchronization statistics for one feature."""

    episode_index: int
    feature_key: str
    source_clock_domain: str
    target_clock_domain: str
    clock_mapping_model: str
    sync_method: str

    source_samples: int
    target_frames: int
    valid_frames: int
    missing_frames: int
    dropped_frames: int

    exact_frames: int
    nearest_frames: int
    interpolated_frames: int
    held_frames: int
    aggregated_frames: int
    unknown_frames: int

    coverage_ratio: float | None
    source_rate_hz: float | None
    effective_output_rate_hz: float | None

    mean_skew_ms: float | None
    p50_skew_ms: float | None
    p95_skew_ms: float | None
    max_skew_ms: float | None

    mean_hold_age_ms: float | None
    p95_hold_age_ms: float | None
    max_hold_age_ms: float | None

    mean_samples_per_window: float | None
    min_samples_per_window: int | None
    max_samples_per_window: int | None

    # Event-only fields
    event_count: int | None = None
    active_output_frames: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "episode_index": self.episode_index,
            "feature_key": self.feature_key,
            "source_clock_domain": self.source_clock_domain,
            "target_clock_domain": self.target_clock_domain,
            "clock_mapping_model": self.clock_mapping_model,
            "sync_method": self.sync_method,
            "source_samples": self.source_samples,
            "target_frames": self.target_frames,
            "valid_frames": self.valid_frames,
            "missing_frames": self.missing_frames,
            "dropped_frames": self.dropped_frames,
            "exact_frames": self.exact_frames,
            "nearest_frames": self.nearest_frames,
            "interpolated_frames": self.interpolated_frames,
            "held_frames": self.held_frames,
            "aggregated_frames": self.aggregated_frames,
            "unknown_frames": self.unknown_frames,
            "coverage_ratio": self.coverage_ratio,
            "source_rate_hz": self.source_rate_hz,
            "effective_output_rate_hz": self.effective_output_rate_hz,
            "mean_skew_ms": self.mean_skew_ms,
            "p50_skew_ms": self.p50_skew_ms,
            "p95_skew_ms": self.p95_skew_ms,
            "max_skew_ms": self.max_skew_ms,
            "mean_hold_age_ms": self.mean_hold_age_ms,
            "p95_hold_age_ms": self.p95_hold_age_ms,
            "max_hold_age_ms": self.max_hold_age_ms,
            "mean_samples_per_window": self.mean_samples_per_window,
            "min_samples_per_window": self.min_samples_per_window,
            "max_samples_per_window": self.max_samples_per_window,
        }
        if self.event_count is not None:
            out["event_count"] = self.event_count
        if self.active_output_frames is not None:
            out["active_output_frames"] = self.active_output_frames
        return out


def _clock_domain(stream: SourceStream) -> str:
    if stream.samples:
        return stream.samples[0].clock_domain
    return "unknown"


def _mapping_model(mapping_result: ClockMappingResult, clock_domain: str) -> str:
    mapping = mapping_result.mappings.get(clock_domain)
    if mapping is None:
        return "identity"
    return mapping.model


def _stream_duration(stream: SourceStream, mapping_result: ClockMappingResult) -> float | None:
    if not stream.samples:
        return None
    mapping = mapping_result.mappings.get(stream.samples[0].clock_domain)
    first_ns = stream.samples[0].source_timestamp_ns
    last_ns = stream.samples[-1].source_timestamp_ns
    if mapping is not None:
        first_ns = mapping.apply(first_ns)
        last_ns = mapping.apply(last_ns)
    duration = (last_ns - first_ns) * _NS_TO_S
    return duration if duration > 0 else None


def compute_sync_feature_stats(
    feature: ResampledFeature,
    stream: SourceStream,
    sync_config: SyncConfig,
    mapping_result: ClockMappingResult,
    timeline: CanonicalTimeline,
    episode_index: int = 0,
    dropped_frames: int = 0,
) -> SyncFeatureStats:
    """Compute synchronization statistics for one feature."""
    policy = sync_config.policy_for(feature.key)
    clock_domain = _clock_domain(stream)
    mapping_model = _mapping_model(mapping_result, clock_domain)

    target_frames = len(feature.values)
    valid_frames = sum(1 for v in feature.valid if v == VALIDITY_VALID)
    missing_frames = target_frames - valid_frames

    provenance_counts: dict[int, int] = {
        PROVENANCE_UNKNOWN: 0,
        PROVENANCE_EXACT: 0,
        PROVENANCE_NEAREST: 0,
        PROVENANCE_INTERPOLATED: 0,
        PROVENANCE_HELD: 0,
        PROVENANCE_AGGREGATED: 0,
    }
    for p in feature.provenance:
        provenance_counts[p] = provenance_counts.get(p, 0) + 1

    skews = [s for s in feature.source_skew_ms if s is not None]
    hold_ages = [
        s
        for s, p in zip(feature.source_skew_ms, feature.provenance, strict=False)
        if s is not None and p == PROVENANCE_HELD
    ]
    samples_per_window = [
        c for c in feature.source_count if c is not None
    ]

    duration = timeline.end_sec - timeline.start_sec
    source_duration = _stream_duration(stream, mapping_result)

    coverage_ratio = valid_frames / target_frames if target_frames else None
    source_rate_hz = (
        len(stream.samples) / source_duration if source_duration else None
    )
    effective_output_rate_hz = (
        valid_frames / duration if duration > 0 else None
    )

    event_count: int | None = None
    active_output_frames: int | None = None
    if stream.stream_type == "event" or policy.method == "interval_any":
        event_count = len(stream.samples)
        active_output_frames = sum(
            1 for v in feature.values if _is_active(v)
        )

    return SyncFeatureStats(
        episode_index=episode_index,
        feature_key=feature.key,
        source_clock_domain=clock_domain,
        target_clock_domain=mapping_result.target_clock,
        clock_mapping_model=mapping_model,
        sync_method=policy.method,
        source_samples=len(stream.samples),
        target_frames=target_frames,
        valid_frames=valid_frames,
        missing_frames=missing_frames,
        dropped_frames=dropped_frames,
        exact_frames=provenance_counts[PROVENANCE_EXACT],
        nearest_frames=provenance_counts[PROVENANCE_NEAREST],
        interpolated_frames=provenance_counts[PROVENANCE_INTERPOLATED],
        held_frames=provenance_counts[PROVENANCE_HELD],
        aggregated_frames=provenance_counts[PROVENANCE_AGGREGATED],
        unknown_frames=provenance_counts[PROVENANCE_UNKNOWN],
        coverage_ratio=coverage_ratio,
        source_rate_hz=source_rate_hz,
        effective_output_rate_hz=effective_output_rate_hz,
        mean_skew_ms=_mean(skews),
        p50_skew_ms=_percentile(skews, 0.5),
        p95_skew_ms=_percentile(skews, 0.95),
        max_skew_ms=max(skews) if skews else None,
        mean_hold_age_ms=_mean(hold_ages),
        p95_hold_age_ms=_percentile(hold_ages, 0.95),
        max_hold_age_ms=max(hold_ages) if hold_ages else None,
        mean_samples_per_window=_mean([float(c) for c in samples_per_window]),
        min_samples_per_window=min(samples_per_window) if samples_per_window else None,
        max_samples_per_window=max(samples_per_window) if samples_per_window else None,
        event_count=event_count,
        active_output_frames=active_output_frames,
    )


def compute_sync_stats(
    features: dict[str, ResampledFeature],
    streams: dict[str, SourceStream],
    sync_config: SyncConfig,
    mapping_result: ClockMappingResult,
    timeline: CanonicalTimeline,
    episode_index: int = 0,
    dropped_frames: int = 0,
) -> list[SyncFeatureStats]:
    """Compute statistics for every synchronized feature."""
    stats: list[SyncFeatureStats] = []
    for key, feature in features.items():
        stream = streams.get(key)
        if stream is None:
            continue
        stats.append(
            compute_sync_feature_stats(
                feature,
                stream,
                sync_config,
                mapping_result,
                timeline,
                episode_index=episode_index,
                dropped_frames=dropped_frames,
            )
        )
    return stats


def write_sync_stats_parquet(
    stats: list[SyncFeatureStats],
    output_path: Path | str,
) -> Path:
    """Write statistics to a parquet file (JSONL fallback if pandas is unavailable)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [s.to_dict() for s in stats]
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False)
    except Exception:
        # Fallback: JSONL
        fallback = output_path.with_suffix(".jsonl")
        with fallback.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")
        return fallback
    return output_path


__all__ = [
    "SyncFeatureStats",
    "compute_sync_feature_stats",
    "compute_sync_stats",
    "write_sync_stats_parquet",
]
