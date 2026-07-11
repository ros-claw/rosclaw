"""High-level synchronization orchestrator for Gate B.1 source streams.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It wires clock mapping, canonical timeline construction, per-feature
resampling, and missingness policy into a single deterministic pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rosclaw.integrations.lerobot.canonical_timeline import (
    CanonicalTimeline,
    build_canonical_timeline,
)
from rosclaw.integrations.lerobot.clock_mapping import (
    ClockMappingResult,
    build_clock_mappings,
)
from rosclaw.integrations.lerobot.resample import ResampledFeature, resample_stream
from rosclaw.integrations.lerobot.source_stream_schema import SourceStreamBundle
from rosclaw.integrations.lerobot.sync_config import SyncConfig
from rosclaw.integrations.lerobot.sync_missingness import (
    MissingnessResult,
    apply_missing_policy,
)


@dataclass
class SynchronizationResult:
    """All outputs of the Gate B.1 synchronization pipeline."""

    timeline: CanonicalTimeline
    features: dict[str, ResampledFeature]
    dropped_frame_indices: list[int]
    kept_frame_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeline": self.timeline.to_dict(),
            "features": {k: self._feature_to_dict(v) for k, v in self.features.items()},
            "dropped_frame_indices": self.dropped_frame_indices,
            "kept_frame_indices": self.kept_frame_indices,
        }

    @staticmethod
    def _feature_to_dict(feature: ResampledFeature) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": feature.key,
            "values": feature.values,
            "valid": feature.valid,
            "provenance": feature.provenance,
            "source_skew_ms": feature.source_skew_ms,
            "source_count": feature.source_count,
        }
        if feature.peak_values is not None:
            out["peak_values"] = feature.peak_values
        return out


def synchronize_bundle(
    bundle: SourceStreamBundle,
    sync_config: SyncConfig,
    missing_policy: str = "nan",
    *,
    mapping_result: ClockMappingResult | None = None,
    timeline: CanonicalTimeline | None = None,
) -> SynchronizationResult:
    """Run the full Gate B.1 synchronization pipeline.

    Parameters
    ----------
    bundle:
        Raw source streams with clock metadata.
    sync_config:
        Target FPS and per-feature resampling policies.
    missing_policy:
        One of ``error``, ``drop-frame``, ``fill-last``, ``nan``.
    mapping_result:
        Optional pre-built clock mappings.  If ``None``, mappings are built
        from ``bundle``.
    timeline:
        Optional pre-built canonical timeline.  If ``None``, the timeline is
        inferred from the required-stream intersection.

    Returns
    -------
    SynchronizationResult with the canonical timeline, synchronized features,
    and frame keep/drop bookkeeping.
    """
    if mapping_result is None:
        mapping_result = build_clock_mappings(bundle)
    if timeline is None:
        timeline = build_canonical_timeline(
            bundle, mapping_result, sync_config.target_fps
        )

    features: dict[str, ResampledFeature] = {}
    for key, stream in bundle.streams.items():
        policy = sync_config.policy_for(key)
        features[key] = resample_stream(
            stream, timeline.timestamps_sec, policy, mapping_result
        )

    missingness: MissingnessResult = apply_missing_policy(
        missing_policy, features, sync_config, timeline.timestamps_sec
    )

    return SynchronizationResult(
        timeline=timeline,
        features=missingness.features,
        dropped_frame_indices=missingness.dropped_frame_indices,
        kept_frame_indices=missingness.kept_frame_indices,
    )


__all__ = [
    "SynchronizationResult",
    "synchronize_bundle",
]
