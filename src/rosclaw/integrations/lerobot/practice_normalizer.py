"""Normalize a ROSClaw Practice episode into a LeRobot-ready intermediate format.

This module lives in the ROSClaw core Python and must not import torch,
lerobot, or PIL at module import time.  It validates frame consistency, image
sizes, and dimensions before the LeRobot runtime worker ever sees the data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NORMALIZED_SCHEMA_VERSION = "rosclaw.practice.normalized.v2"
LEGACY_NORMALIZED_SCHEMA_VERSION = "rosclaw.practice.normalized.v1"


class NormalizationError(Exception):
    """Raised when an episode cannot be normalized."""

    def __init__(self, code: str, message: str, details: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _read_image_size(image_path: Path, camera_name: str) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise NormalizationError(
            "image_reader_unavailable",
            "Pillow is required to validate practice episode images.",
            f"Could not import PIL while reading camera '{camera_name}' at {image_path}.",
        ) from exc

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return (img.width, img.height)
    except Exception as exc:  # noqa: BLE001
        raise NormalizationError(
            "image_file_not_found",
            f"Could not read image '{camera_name}' at {image_path}: {exc}",
        ) from exc


@dataclass
class NormalizedRobot:
    robot_id: str = "unknown"
    body_profile: str | None = None
    body_yaml_path: str | None = None
    body_hash: str | None = None
    eurdf_repo: str | None = None
    eurdf_revision: str | None = None
    provider_type: str | None = None
    provider_name: str | None = None
    policy_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"robot_id": self.robot_id}
        for key in (
            "body_profile",
            "body_yaml_path",
            "body_hash",
            "eurdf_repo",
            "eurdf_revision",
            "provider_type",
            "provider_name",
            "policy_path",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedRobot:
        return cls(
            robot_id=data.get("robot_id") or "unknown",
            body_profile=data.get("body_profile"),
            body_yaml_path=data.get("body_yaml_path"),
            body_hash=data.get("body_hash"),
            eurdf_repo=data.get("eurdf_repo"),
            eurdf_revision=data.get("eurdf_revision"),
            provider_type=data.get("provider_type"),
            provider_name=data.get("provider_name"),
            policy_path=data.get("policy_path"),
        )


@dataclass
class NormalizedTask:
    text: str = ""
    task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"text": self.text}
        if self.task_id is not None:
            out["task_id"] = self.task_id
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> NormalizedTask:
        if isinstance(data, str):
            return cls(text=data)
        return cls(text=data.get("text", ""), task_id=data.get("task_id"))


@dataclass
class NormalizedSafety:
    """Sandbox/safety decision for a single frame."""

    decision: str = "UNKNOWN"
    modified: bool | None = None
    risk_score: float | None = None
    reason_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"decision": self.decision}
        if self.modified is not None:
            out["modified"] = self.modified
        if self.risk_score is not None:
            out["risk_score"] = self.risk_score
        if self.reason_code is not None:
            out["reason_code"] = self.reason_code
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> NormalizedSafety:
        if not data:
            return cls()
        modified = data.get("modified")
        modified = None if modified is None else bool(modified)
        return cls(
            decision=data.get("decision") or "UNKNOWN",
            modified=modified,
            risk_score=data.get("risk_score"),
            reason_code=data.get("reason_code"),
        )


@dataclass
class NormalizedFailure:
    """Failure event for a single frame."""

    active: bool | None = None
    code: str | None = None
    severity: int = 0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"severity": self.severity}
        if self.active is not None:
            out["active"] = self.active
        if self.code is not None:
            out["code"] = self.code
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> NormalizedFailure:
        if not data:
            return cls()
        active = data.get("active")
        active = None if active is None else bool(active)
        return cls(
            active=active,
            code=data.get("code"),
            severity=int(data.get("severity", 0)),
        )


@dataclass
class NormalizedIntervention:
    """Human/operator intervention for a single frame."""

    active: bool | None = None
    source: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.active is not None:
            out["active"] = self.active
        if self.source is not None:
            out["source"] = self.source
        if self.confidence is not None:
            out["confidence"] = self.confidence
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> NormalizedIntervention:
        if not data:
            return cls()
        active = data.get("active")
        active = None if active is None else bool(active)
        return cls(
            active=active,
            source=data.get("source"),
            confidence=data.get("confidence"),
        )


@dataclass
class NormalizedActionContext:
    """Action provenance for a single frame."""

    source: str = "UNKNOWN"
    was_clamped: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source": self.source}
        if self.was_clamped is not None:
            out["was_clamped"] = self.was_clamped
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> NormalizedActionContext:
        if not data:
            return cls()
        was_clamped = data.get("was_clamped")
        was_clamped = None if was_clamped is None else bool(was_clamped)
        return cls(
            source=data.get("source") or "UNKNOWN",
            was_clamped=was_clamped,
        )


@dataclass
class NormalizedFrame:
    frame_index: int = 0
    timestamp: float = 0.0
    observation_state: list[float] = field(default_factory=list)
    observation_images: dict[str, str] = field(default_factory=dict)
    action: list[float] = field(default_factory=list)
    done: bool | None = None
    success: bool | None = None
    safety: NormalizedSafety | None = None
    failure: NormalizedFailure | None = None
    intervention: NormalizedIntervention | None = None
    action_context: NormalizedActionContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Time sync (Gate B)
    source_timestamp_ns: int | None = None
    clock_domain: str | None = None
    episode_time_sec: float | None = None
    # Physical telemetry (Gate B)
    motor_current: list[float] | None = None
    joint_temperature: list[float] | None = None
    force_torque: list[float] | None = None
    contact: list[bool] | None = None
    joint_velocity: list[float] | None = None
    joint_effort: list[float] | None = None

    def __post_init__(self):
        if self.safety is None:
            self.safety = NormalizedSafety()
        if self.failure is None:
            self.failure = NormalizedFailure()
        if self.intervention is None:
            self.intervention = NormalizedIntervention()
        if self.action_context is None:
            self.action_context = NormalizedActionContext()

    def _observation_dict(self) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "state": self.observation_state,
            "images": self.observation_images,
        }
        if self.motor_current is not None:
            obs["motor_current"] = self.motor_current
        if self.joint_temperature is not None:
            obs["joint_temperature"] = self.joint_temperature
        if self.force_torque is not None:
            obs["force_torque"] = self.force_torque
        if self.contact is not None:
            obs["contact"] = self.contact
        if self.joint_velocity is not None:
            obs["joint_velocity"] = self.joint_velocity
        if self.joint_effort is not None:
            obs["joint_effort"] = self.joint_effort
        return obs

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "observation": self._observation_dict(),
            "action": self.action,
            "metadata": self.metadata,
        }
        if self.done is not None:
            out["done"] = self.done
        if self.success is not None:
            out["success"] = self.success
        if self.source_timestamp_ns is not None:
            out["source_timestamp_ns"] = self.source_timestamp_ns
        if self.clock_domain is not None:
            out["clock_domain"] = self.clock_domain
        if self.episode_time_sec is not None:
            out["episode_time_sec"] = self.episode_time_sec
        if self.safety:
            safety_dict = self.safety.to_dict()
            if safety_dict.get("decision") != "UNKNOWN" or safety_dict.get("modified") is not None:
                out["safety"] = safety_dict
        if self.failure:
            failure_dict = self.failure.to_dict()
            if failure_dict.get("active") is not None:
                out["failure"] = failure_dict
        if self.intervention:
            intervention_dict = self.intervention.to_dict()
            if intervention_dict.get("active") is not None:
                out["intervention"] = intervention_dict
        if self.action_context:
            action_dict = self.action_context.to_dict()
            if action_dict.get("source") != "UNKNOWN" or action_dict.get("was_clamped") is not None:
                out["action_context"] = action_dict
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedFrame:
        obs = data.get("observation", {})

        def _optional_bool(value: Any) -> bool | None:
            if value is None:
                return None
            return bool(value)

        def _float_list(value: Any) -> list[float] | None:
            if value is None:
                return None
            return [float(v) for v in value]

        def _bool_list(value: Any) -> list[bool] | None:
            if value is None:
                return None
            return [bool(v) for v in value]

        return cls(
            frame_index=int(data.get("frame_index", 0)),
            timestamp=float(data.get("timestamp", 0.0)),
            observation_state=list(obs.get("state", [])),
            observation_images=dict(obs.get("images", {})),
            action=list(data.get("action", [])),
            done=_optional_bool(data.get("done")),
            success=_optional_bool(data.get("success")),
            source_timestamp_ns=data.get("source_timestamp_ns"),
            clock_domain=data.get("clock_domain"),
            episode_time_sec=data.get("episode_time_sec"),
            safety=NormalizedSafety.from_dict(data.get("safety")),
            failure=NormalizedFailure.from_dict(data.get("failure")),
            intervention=NormalizedIntervention.from_dict(data.get("intervention")),
            action_context=NormalizedActionContext.from_dict(data.get("action_context")),
            metadata=dict(data.get("metadata", {})),
            motor_current=_float_list(obs.get("motor_current")),
            joint_temperature=_float_list(obs.get("joint_temperature")),
            force_torque=_float_list(obs.get("force_torque")),
            contact=_bool_list(obs.get("contact")),
            joint_velocity=_float_list(obs.get("joint_velocity")),
            joint_effort=_float_list(obs.get("joint_effort")),
        )


@dataclass
class NormalizedPracticeEpisode:
    """Intermediate representation of a ROSClaw Practice episode."""

    schema_version: str = NORMALIZED_SCHEMA_VERSION
    episode_id: str = ""
    robot: NormalizedRobot = field(default_factory=NormalizedRobot)
    task: NormalizedTask = field(default_factory=NormalizedTask)
    fps: float = 10.0
    frames: list[NormalizedFrame] = field(default_factory=list)
    environment: str | None = None
    operator: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "robot": self.robot.to_dict(),
            "task": self.task.to_dict(),
            "fps": self.fps,
            "frames": [f.to_dict() for f in self.frames],
            "metadata": self.metadata,
        }
        if self.environment is not None:
            out["environment"] = self.environment
        if self.operator is not None:
            out["operator"] = self.operator
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedPracticeEpisode:
        return cls(
            schema_version=data.get("schema_version", LEGACY_NORMALIZED_SCHEMA_VERSION),
            episode_id=data.get("episode_id", ""),
            robot=NormalizedRobot.from_dict(data.get("robot", {})),
            task=NormalizedTask.from_dict(data.get("task", {})),
            fps=float(data.get("fps", 10.0)),
            frames=[NormalizedFrame.from_dict(f) for f in data.get("frames", [])],
            environment=data.get("environment"),
            operator=data.get("operator"),
            metadata=dict(data.get("metadata", {})),
        )


def normalize_practice_episode(
    episode_path: Path | str,
    *,
    task: str | None = None,
    robot_id: str | None = None,
    body_profile: str | None = None,
    fps: float | None = None,
) -> NormalizedPracticeEpisode:
    """Load and validate a Practice episode directory or single JSON file.

    Supported inputs:
      - ``<episode_dir>/episode.json`` plus relative image paths.
      - A single ``episode.json`` file with absolute or relative image paths.

    v1 episodes are accepted and migrated to v2 with unknown/inactive defaults
    for safety, failure, intervention, and action context.
    """
    episode_path = Path(episode_path)

    if episode_path.is_dir():
        episode_file = episode_path / "episode.json"
        base_dir = episode_path
    else:
        episode_file = episode_path
        base_dir = episode_path.parent

    if not episode_file.exists():
        raise NormalizationError(
            "practice_episode_not_found",
            f"Episode file not found: {episode_file}",
        )

    try:
        raw = json.loads(episode_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NormalizationError(
            "practice_episode_invalid",
            f"Invalid JSON in episode file: {episode_file}",
            str(exc),
        ) from exc

    if not isinstance(raw, dict):
        raise NormalizationError(
            "practice_episode_invalid",
            "Episode file must contain a JSON object.",
        )

    episode = NormalizedPracticeEpisode.from_dict(raw)
    if not episode.frames:
        raise NormalizationError(
            "practice_episode_invalid",
            "Episode contains no frames.",
        )

    # Apply CLI overrides.
    if task:
        episode.task.text = task
    if robot_id:
        episode.robot.robot_id = robot_id
    if body_profile:
        episode.robot.body_profile = body_profile
    if fps is not None:
        episode.fps = fps

    if not episode.task.text:
        episode.task.text = "rosclaw practice"

    if not episode.episode_id:
        episode.episode_id = base_dir.name

    episode.schema_version = NORMALIZED_SCHEMA_VERSION
    episode.metadata["source_dir"] = str(base_dir.resolve())

    _validate_frames(episode, base_dir)
    return episode


def _validate_telemetry_dimensions(frame: NormalizedFrame, dims: dict[str, int]) -> None:
    """Ensure telemetry arrays have consistent dimensions across frames."""
    telemetry_fields = {
        "motor_current": frame.motor_current,
        "joint_temperature": frame.joint_temperature,
        "force_torque": frame.force_torque,
        "contact": frame.contact,
        "joint_velocity": frame.joint_velocity,
        "joint_effort": frame.joint_effort,
    }
    for name, values in telemetry_fields.items():
        if values is None:
            continue
        length = len(values)
        expected = dims.get(name)
        if expected is None:
            dims[name] = length
        elif length != expected:
            raise NormalizationError(
                f"{name}_dim_mismatch",
                f"{name} dimension mismatch: expected {expected}, got {length} "
                f"at frame {frame.frame_index}",
            )


def _validate_frames(episode: NormalizedPracticeEpisode, base_dir: Path) -> None:
    """Validate frame indexing, timestamps, dimensions, and image files."""
    frames = episode.frames
    expected_indices = list(range(len(frames)))
    actual_indices = [f.frame_index for f in frames]
    if actual_indices != expected_indices:
        raise NormalizationError(
            "frame_index_not_contiguous",
            f"Frame indices are not contiguous 0..N-1: {actual_indices}",
        )

    timestamps = [f.timestamp for f in frames]
    if any(timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)):
        raise NormalizationError(
            "timestamp_not_monotonic",
            "Frame timestamps must be strictly increasing.",
        )

    state_dim: int | None = None
    action_dim: int | None = None
    image_size: tuple[int, int] | None = None
    telemetry_dims: dict[str, int] = {}

    for frame in frames:
        if state_dim is None:
            state_dim = len(frame.observation_state)
        elif len(frame.observation_state) != state_dim:
            raise NormalizationError(
                "state_dim_mismatch",
                f"State dimension mismatch: expected {state_dim}, got {len(frame.observation_state)} "
                f"at frame {frame.frame_index}",
            )

        if action_dim is None:
            action_dim = len(frame.action)
        elif len(frame.action) != action_dim:
            raise NormalizationError(
                "action_dim_mismatch",
                f"Action dimension mismatch: expected {action_dim}, got {len(frame.action)} "
                f"at frame {frame.frame_index}",
            )

        _validate_telemetry_dimensions(frame, telemetry_dims)

        for camera_name, image_rel in frame.observation_images.items():
            image_path = base_dir / image_rel
            if not image_path.exists():
                raise NormalizationError(
                    "image_file_not_found",
                    f"Image file not found for camera '{camera_name}': {image_path}",
                )
            size = _read_image_size(image_path, camera_name)

            if image_size is None:
                image_size = size
            elif size != image_size:
                raise NormalizationError(
                    "image_shape_mismatch",
                    f"Image shape mismatch for camera '{camera_name}': expected {image_size}, got {size} "
                    f"at frame {frame.frame_index}",
                )


def write_normalized_episode(
    episode: NormalizedPracticeEpisode,
    output_path: Path | str,
) -> Path:
    """Serialize a normalized episode to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(episode.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


__all__ = [
    "LEGACY_NORMALIZED_SCHEMA_VERSION",
    "NORMALIZED_SCHEMA_VERSION",
    "NormalizationError",
    "NormalizedActionContext",
    "NormalizedFailure",
    "NormalizedFrame",
    "NormalizedIntervention",
    "NormalizedPracticeEpisode",
    "NormalizedRobot",
    "NormalizedSafety",
    "NormalizedTask",
    "normalize_practice_episode",
    "write_normalized_episode",
]
