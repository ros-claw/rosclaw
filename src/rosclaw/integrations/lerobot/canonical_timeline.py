"""Canonical timeline construction for Gate B.1 source-stream synchronization.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It turns a set of mapped source-clock timestamps into a deterministic
fixed-FPS frame grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rosclaw.integrations.lerobot.clock_mapping import (
    ClockMappingResult,
    normalize_bundle_timestamps,
)
from rosclaw.integrations.lerobot.practice_normalizer import NormalizationError
from rosclaw.integrations.lerobot.source_stream_schema import SourceStreamBundle

_NS_TO_S = 1e-9
_EPSILON = 1e-9


@dataclass
class CanonicalTimeline:
    """A fixed-FPS frame grid aligned to episode_time_sec."""

    start_sec: float
    end_sec: float
    fps: float
    timestamps_sec: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "fps": self.fps,
            "timestamps_sec": self.timestamps_sec,
            "num_frames": len(self.timestamps_sec),
        }

    @property
    def num_frames(self) -> int:
        return len(self.timestamps_sec)


def _stream_target_bounds(
    bundle: SourceStreamBundle,
    mapping_result: ClockMappingResult,
) -> dict[str, tuple[float, float]]:
    """Return (start_sec, end_sec) for each stream in target clock domain."""
    normalized = normalize_bundle_timestamps(bundle, mapping_result)
    bounds: dict[str, tuple[float, float]] = {}
    for key, pairs in normalized.items():
        valid_pairs = [p for p in pairs if p[1] is not None]
        if not valid_pairs:
            continue
        start_ns = min(p[1] for p in valid_pairs)
        end_ns = max(p[1] for p in valid_pairs)
        bounds[key] = (start_ns * _NS_TO_S, end_ns * _NS_TO_S)
    return bounds


def build_canonical_timeline(
    bundle: SourceStreamBundle,
    mapping_result: ClockMappingResult,
    fps: float,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> CanonicalTimeline:
    """Build a fixed-FPS timeline from mapped source streams.

    If ``start_sec``/``end_sec`` are not provided, the timeline is derived from
    the intersection of the required streams (``observation.state`` and
    ``action``).  Optional streams never extend the episode bounds.
    """
    if fps <= 0:
        raise NormalizationError(
            "invalid_fps",
            f"FPS must be positive, got {fps}.",
        )

    bounds = _stream_target_bounds(bundle, mapping_result)

    if start_sec is None or end_sec is None:
        required_keys = ["observation.state", "action"]
        required_bounds = [bounds[k] for k in required_keys if k in bounds]
        if len(required_bounds) < len(required_keys):
            missing = [k for k in required_keys if k not in bounds]
            raise NormalizationError(
                "required_stream_missing_for_timeline",
                f"Cannot build canonical timeline without required streams: {missing}.",
            )
        inferred_start = max(b[0] for b in required_bounds)
        inferred_end = min(b[1] for b in required_bounds)
        if start_sec is None:
            start_sec = inferred_start
        if end_sec is None:
            end_sec = inferred_end

    if end_sec < start_sec - _EPSILON:
        raise NormalizationError(
            "timeline_negative_duration",
            f"Episode end {end_sec} is before start {start_sec}.",
        )

    # Use epsilon to avoid creating an extra frame when end is just past a grid point.
    raw_count = (end_sec - start_sec) * fps
    num_frames = int(raw_count + _EPSILON) + 1
    if num_frames <= 0:
        raise NormalizationError(
            "timeline_zero_frames",
            f"Canonical timeline has zero frames for range [{start_sec}, {end_sec}] at {fps} FPS.",
        )

    timestamps_sec = [start_sec + k / fps for k in range(num_frames)]
    # Clamp the last timestamp to not exceed end_sec beyond epsilon.
    if timestamps_sec[-1] > end_sec + _EPSILON:
        timestamps_sec[-1] = end_sec

    return CanonicalTimeline(
        start_sec=start_sec,
        end_sec=end_sec,
        fps=fps,
        timestamps_sec=timestamps_sec,
    )


__all__ = [
    "CanonicalTimeline",
    "build_canonical_timeline",
]
