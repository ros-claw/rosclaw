"""Infer LeRobotDataset feature schema from a NormalizedPracticeEpisode.

This module lives in the ROSClaw core Python and must not import torch,
lerobot, or PIL at module import time.  Pillow is only required when image
dimensions need to be read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.practice_normalizer import NormalizedPracticeEpisode


class FeatureInferenceError(Exception):
    """Raised when feature inference fails due to inconsistent data."""

    def __init__(self, code: str, message: str, details: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


# Mirror of the worker feature spec for dry-run and reporting.
ROSCLAW_FEATURE_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "safety": {
        "rosclaw.sandbox.decision": {"dtype": "int8", "shape": [1], "names": None},
        "rosclaw.sandbox.modified": {"dtype": "int8", "shape": [1], "names": None},
        "rosclaw.sandbox.risk_score": {"dtype": "float32", "shape": [1], "names": None},
    },
    "failure": {
        "rosclaw.failure.active": {"dtype": "int8", "shape": [1], "names": None},
        "rosclaw.failure.code": {"dtype": "int16", "shape": [1], "names": None},
    },
    "intervention": {
        "rosclaw.intervention.active": {"dtype": "int8", "shape": [1], "names": None},
        "rosclaw.intervention.source": {"dtype": "int8", "shape": [1], "names": None},
    },
    "action": {
        "rosclaw.action.source": {"dtype": "int8", "shape": [1], "names": None},
        "rosclaw.action.was_clamped": {"dtype": "int8", "shape": [1], "names": None},
    },
    "outcome": {
        "rosclaw.done": {"dtype": "int8", "shape": [1], "names": None},
        "rosclaw.success": {"dtype": "int8", "shape": [1], "names": None},
    },
    "physical_telemetry": {
        "observation.motor_current": {"dtype": "float32", "shape": [1], "names": ["motor"]},
        "observation.joint_temperature": {"dtype": "float32", "shape": [1], "names": ["joint"]},
        "observation.force_torque": {"dtype": "float32", "shape": [1], "names": ["axis"]},
        "observation.contact": {"dtype": "int8", "shape": [1], "names": ["contact"]},
        "observation.joint_velocity": {"dtype": "float32", "shape": [1], "names": ["joint"]},
        "observation.joint_effort": {"dtype": "float32", "shape": [1], "names": ["joint"]},
    },
}

def _infer_telemetry_features(
    episode: NormalizedPracticeEpisode,
) -> dict[str, dict[str, Any]]:
    """Infer physical telemetry feature shapes from the first non-empty frame."""
    features: dict[str, dict[str, Any]] = {}
    telemetry_fields = [
        ("observation.motor_current", "motor_current"),
        ("observation.joint_temperature", "joint_temperature"),
        ("observation.force_torque", "force_torque"),
        ("observation.contact", "contact"),
        ("observation.joint_velocity", "joint_velocity"),
        ("observation.joint_effort", "joint_effort"),
    ]
    for feature_key, attr_name in telemetry_fields:
        dim: int | None = None
        for frame in episode.frames:
            values = getattr(frame, attr_name)
            if values:
                dim = len(values)
                break
        if dim is not None and dim > 0:
            spec = dict(ROSCLAW_FEATURE_SPECS["physical_telemetry"][feature_key])
            spec["shape"] = [dim]
            features[feature_key] = spec
    return features


def _read_image_shape(image_path: Path, camera_name: str) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise FeatureInferenceError(
            "image_reader_unavailable",
            "Pillow is required to infer image feature dimensions.",
            f"Could not import PIL while reading camera '{camera_name}' at {image_path}.",
        ) from exc

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return (img.height, img.width)
    except Exception as exc:  # noqa: BLE001
        raise FeatureInferenceError(
            "image_file_not_found",
            f"Could not read image '{camera_name}' at {image_path}: {exc}",
        ) from exc


def infer_features(
    episode: NormalizedPracticeEpisode,
    feature_groups: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a LeRobot feature dict for state, action, images, and ROSClaw groups.

    Args:
        episode: Normalized practice episode.
        feature_groups: Optional ROSClaw feature groups to include (e.g.
            ``["safety", "action"]``).  Defaults to ``["outcome"]`` for v2
            episodes when not provided, preserving P2 minimal behavior otherwise.

    Raises:
        FeatureInferenceError: if dimensions are inconsistent.
    """
    if not episode.frames:
        raise FeatureInferenceError(
            "normalized_episode_invalid",
            "Cannot infer features from an empty episode.",
        )

    feature_groups = list(feature_groups or [])

    state_dim: int | None = None
    action_dim: int | None = None
    image_sizes: dict[str, tuple[int, int]] = {}

    for frame in episode.frames:
        if state_dim is None:
            state_dim = len(frame.observation_state)
        elif len(frame.observation_state) != state_dim:
            raise FeatureInferenceError(
                "state_dim_mismatch",
                f"State dimension mismatch: expected {state_dim}, got {len(frame.observation_state)} "
                f"at frame {frame.frame_index}.",
            )

        if action_dim is None:
            action_dim = len(frame.action)
        elif len(frame.action) != action_dim:
            raise FeatureInferenceError(
                "action_dim_mismatch",
                f"Action dimension mismatch: expected {action_dim}, got {len(frame.action)} "
                f"at frame {frame.frame_index}.",
            )

        base_dir = Path(episode.metadata.get("source_dir", "")) if episode.metadata else Path()
        for camera_name, image_rel in frame.observation_images.items():
            image_path = base_dir / image_rel
            if not image_path.exists():
                raise FeatureInferenceError(
                    "image_file_not_found",
                    f"Image file not found for camera '{camera_name}': {image_path}",
                )
            size = _read_image_shape(image_path, camera_name)

            prev_size = image_sizes.get(camera_name)
            if prev_size is None:
                image_sizes[camera_name] = size
            elif size != prev_size:
                raise FeatureInferenceError(
                    "image_shape_mismatch",
                    f"Image shape mismatch for camera '{camera_name}': expected {prev_size}, got {size} "
                    f"at frame {frame.frame_index}.",
                )

    features: dict[str, dict[str, Any]] = {}
    if state_dim is not None and state_dim > 0:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [state_dim],
            "names": None,
        }
    if action_dim is not None and action_dim > 0:
        features["action"] = {
            "dtype": "float32",
            "shape": [action_dim],
            "names": None,
        }
    for camera_name, (h, w) in sorted(image_sizes.items()):
        features[f"observation.images.{camera_name}"] = {
            "dtype": "image",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
        }

    for group in feature_groups:
        if group == "physical_telemetry":
            features.update(_infer_telemetry_features(episode))
            continue
        for key, spec in ROSCLAW_FEATURE_SPECS.get(group, {}).items():
            features[key] = dict(spec)

    return features


def feature_summary(features: dict[str, dict[str, Any]]) -> dict[str, list[int]]:
    """Return a compact summary for reports and CLI output."""
    summary: dict[str, list[int]] = {}
    for key, value in features.items():
        summary[key] = list(value.get("shape", []))
    return summary


__all__ = [
    "FeatureInferenceError",
    "ROSCLAW_FEATURE_SPECS",
    "feature_summary",
    "infer_features",
]
