"""Missingness policy application for Gate B.1 synchronized features.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It implements the four global missing-data policies and guarantees
atomic drop-frame behavior across all modalities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rosclaw.integrations.lerobot.practice_normalizer import NormalizationError
from rosclaw.integrations.lerobot.resample import ResampledFeature
from rosclaw.integrations.lerobot.sync_config import SyncConfig
from rosclaw.integrations.lerobot.sync_provenance import (
    PROVENANCE_FILLED_NAN,
    PROVENANCE_HELD,
    PROVENANCE_UNKNOWN,
    VALIDITY_INVALID,
    VALIDITY_VALID,
)


_MISSING_POLICIES = {"error", "drop-frame", "fill-last", "nan"}


@dataclass
class MissingnessResult:
    """Result of applying a missingness policy."""

    features: dict[str, ResampledFeature]
    dropped_frame_indices: list[int]
    kept_frame_indices: list[int]
    errors: list[dict[str, Any]] = field(default_factory=list)


def _is_missing(value: Any) -> bool:
    return value is None


def _nan_sentinel_for(value: Any, template: Any | None = None) -> Any:
    """Return a missing sentinel that matches the value shape.

    If ``value`` is ``None`` a non-``None`` ``template`` from the same feature is
    used to infer the correct shape and dtype.
    """
    shape_source = value if value is not None else template
    if shape_source is None:
        return None

    if isinstance(shape_source, list):
        if not shape_source:
            return []
        first = shape_source[0]
        if isinstance(first, bool):
            return [-1] * len(shape_source)
        # Float / int arrays use NaN for floats and -1 for ints.
        if isinstance(first, (int, float)):
            return [float("nan") if isinstance(first, float) else -1 for _ in shape_source]
        return [None] * len(shape_source)
    if isinstance(shape_source, bool):
        return -1
    if isinstance(shape_source, str):
        return "UNKNOWN"
    if isinstance(shape_source, int):
        return -1
    if isinstance(shape_source, float):
        return float("nan")
    return None


def _find_missing_frames(features: dict[str, ResampledFeature]) -> set[int]:
    num_frames = _frame_count(features)
    missing: set[int] = set()
    for feature in features.values():
        for i in range(num_frames):
            if feature.valid[i] != VALIDITY_VALID:
                missing.add(i)
    return missing


def _frame_count(features: dict[str, ResampledFeature]) -> int:
    for feature in features.values():
        return len(feature.values)
    return 0


def _build_error_details(
    features: dict[str, ResampledFeature],
    timestamps_sec: list[float] | None,
) -> list[dict[str, Any]]:
    num_frames = _frame_count(features)
    errors: list[dict[str, Any]] = []
    for feature in features.values():
        for i in range(num_frames):
            if feature.valid[i] != VALIDITY_VALID:
                errors.append(
                    {
                        "feature_key": feature.key,
                        "target_frame_index": i,
                        "target_timestamp": timestamps_sec[i] if timestamps_sec else None,
                        "reason": "required_feature_missing",
                    }
                )
    return errors


def apply_error_policy(
    features: dict[str, ResampledFeature],
    timestamps_sec: list[float] | None = None,
) -> MissingnessResult:
    """Fail immediately if any feature is missing for any frame."""
    missing = _find_missing_frames(features)
    if missing:
        errors = _build_error_details(features, timestamps_sec)
        first = errors[0]
        raise NormalizationError(
            "sync_required_feature_missing",
            f"Required feature '{first['feature_key']}' missing at frame {first['target_frame_index']}.",
            details=str(errors),
        )
    num_frames = _frame_count(features)
    return MissingnessResult(
        features=features,
        dropped_frame_indices=[],
        kept_frame_indices=list(range(num_frames)),
    )


def apply_drop_frame_policy(
    features: dict[str, ResampledFeature],
) -> MissingnessResult:
    """Atomically drop every frame that has at least one missing feature."""
    num_frames = _frame_count(features)
    missing = _find_missing_frames(features)
    kept = [i for i in range(num_frames) if i not in missing]

    new_features: dict[str, ResampledFeature] = {}
    for key, feature in features.items():
        new_features[key] = ResampledFeature(
            key=feature.key,
            values=[feature.values[i] for i in kept],
            valid=[feature.valid[i] for i in kept],
            provenance=[feature.provenance[i] for i in kept],
            source_skew_ms=[feature.source_skew_ms[i] for i in kept],
            source_count=[feature.source_count[i] for i in kept],
            peak_values=[feature.peak_values[i] for i in kept] if feature.peak_values is not None else None,
        )

    return MissingnessResult(
        features=new_features,
        dropped_frame_indices=sorted(missing),
        kept_frame_indices=kept,
    )


def _fill_last_for_feature(
    feature: ResampledFeature,
    timestamps_sec: list[float],
    max_age_ms: float | None,
) -> ResampledFeature:
    """Fill missing frames by holding the previous valid value."""
    num_frames = len(feature.values)
    if num_frames == 0:
        return feature

    max_age = (max_age_ms / 1000.0) if max_age_ms is not None else float("inf")
    values = list(feature.values)
    valid = list(feature.valid)
    provenance = list(feature.provenance)
    skew = list(feature.source_skew_ms)
    count = list(feature.source_count)

    last_valid_idx: int | None = None
    for i in range(num_frames):
        if valid[i] == VALIDITY_VALID:
            last_valid_idx = i
            continue
        if last_valid_idx is None:
            continue
        age = timestamps_sec[i] - timestamps_sec[last_valid_idx]
        if age > max_age:
            continue
        values[i] = values[last_valid_idx]
        valid[i] = VALIDITY_VALID
        provenance[i] = PROVENANCE_HELD
        skew[i] = age * 1000.0
        count[i] = 0

    return ResampledFeature(
        key=feature.key,
        values=values,
        valid=valid,
        provenance=provenance,
        source_skew_ms=skew,
        source_count=count,
        peak_values=feature.peak_values,
    )


def apply_fill_last_policy(
    features: dict[str, ResampledFeature],
    sync_config: SyncConfig,
    timestamps_sec: list[float],
) -> MissingnessResult:
    """Hold-last policy: fill allowed features from previous valid frame."""
    new_features: dict[str, ResampledFeature] = {}
    for key, feature in features.items():
        policy = sync_config.policy_for(key)
        if policy.method in {"previous", "interval_any"}:
            new_features[key] = _fill_last_for_feature(
                feature, timestamps_sec, policy.max_age_ms
            )
        else:
            new_features[key] = feature

    # Any remaining missing frames are left invalid; caller may still need nan fill.
    return MissingnessResult(
        features=new_features,
        dropped_frame_indices=[],
        kept_frame_indices=list(range(_frame_count(new_features))),
    )


def apply_nan_policy(
    features: dict[str, ResampledFeature],
) -> MissingnessResult:
    """Fill missing frames with type-appropriate unknown sentinels."""
    new_features: dict[str, ResampledFeature] = {}
    for key, feature in features.items():
        # Find a representative non-None value to infer shape for missing frames.
        template = next((v for v in feature.values if v is not None), None)
        values = list(feature.values)
        provenance = list(feature.provenance)
        for i in range(len(values)):
            if feature.valid[i] != VALIDITY_VALID:
                values[i] = _nan_sentinel_for(values[i], template=template)
                provenance[i] = PROVENANCE_FILLED_NAN
        new_features[key] = ResampledFeature(
            key=feature.key,
            values=values,
            valid=list(feature.valid),
            provenance=provenance,
            source_skew_ms=list(feature.source_skew_ms),
            source_count=list(feature.source_count),
            peak_values=feature.peak_values,
        )

    return MissingnessResult(
        features=new_features,
        dropped_frame_indices=[],
        kept_frame_indices=list(range(_frame_count(new_features))),
    )


def apply_missing_policy(
    policy: str,
    features: dict[str, ResampledFeature],
    sync_config: SyncConfig,
    timestamps_sec: list[float],
) -> MissingnessResult:
    """Apply the global missing-data policy to synchronized features."""
    if policy not in _MISSING_POLICIES:
        raise NormalizationError(
            "invalid_missing_policy",
            f"Unknown missing policy '{policy}'. Expected one of {_MISSING_POLICIES}.",
        )

    if policy == "error":
        return apply_error_policy(features, timestamps_sec)
    if policy == "drop-frame":
        return apply_drop_frame_policy(features)
    if policy == "fill-last":
        return apply_fill_last_policy(features, sync_config, timestamps_sec)
    return apply_nan_policy(features)


__all__ = [
    "MissingnessResult",
    "apply_drop_frame_policy",
    "apply_error_policy",
    "apply_fill_last_policy",
    "apply_missing_policy",
    "apply_nan_policy",
]
