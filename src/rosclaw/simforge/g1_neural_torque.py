"""Fail-closed recurrent G1 joint-torque policy for MuJoCo only.

This module deliberately has no ROS, DDS, vendor SDK, Registry, or hardware
dependency.  A candidate consumes raw proprioception plus task context and
emits all 29 joint torques, but the output remains behind an independent
amplitude/rate/power/joint-limit safety envelope.  Any malformed, incompatible,
or over-projecting proposal falls back to the qualified parent controller.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    G1_HARD_TORQUE_LIMITS,
)

_SCHEMA = "rosclaw.simforge.g1_neural_torque_artifact.v1"
_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_SHA256 = __import__("re").compile(r"^sha256:[0-9a-f]{64}$")

G1_NEURAL_TORQUE_OBSERVATIONS = (
    *(f"joint_position/{name}" for name in G1_DDS_JOINT_NAMES),
    *(f"joint_velocity/{name}" for name in G1_DDS_JOINT_NAMES),
    "projected_gravity/x",
    "projected_gravity/y",
    "projected_gravity/z",
    "ball_relative/x",
    "ball_relative/y",
    "ball_relative/z",
    "ball_velocity/x",
    "ball_velocity/y",
    "ball_velocity/z",
    "target/y",
    "target/z",
    "phase/sin",
    "phase/cos",
    "contact/left",
    "contact/right",
    *(f"previous_torque_ratio/{name}" for name in G1_DDS_JOINT_NAMES),
)
G1_NEURAL_TORQUE_ACTIONS = tuple(f"joint_torque/{name}" for name in G1_DDS_JOINT_NAMES)


@dataclass(frozen=True)
class G1TorqueSafetyConfig:
    """Immutable outer envelope around a neural torque proposal."""

    torque_guard_scale: float = 0.70
    max_delta_ratio_per_step: float = 0.08
    joint_limit_margin_rad: float = 0.035
    maximum_mechanical_power_w: float = 2500.0
    maximum_projection_ratio: float = 0.35
    maximum_parent_deviation_ratio: float = 0.08
    maximum_observation_z: float = 6.0
    minimum_upright_gravity_z: float = -0.75
    minimum_pelvis_height_m: float = 0.62
    recovery_cooldown_steps: int = 0
    warmup_steps: int = 50
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if not 0.05 <= self.torque_guard_scale <= 0.85:
            raise ValueError("neural torque guard scale must be in [0.05, 0.85]")
        if not 0.001 <= self.max_delta_ratio_per_step <= 0.25:
            raise ValueError("neural torque delta ratio must be in [0.001, 0.25]")
        if not 0.005 <= self.joint_limit_margin_rad <= 0.20:
            raise ValueError("joint limit margin must be in [0.005, 0.20] rad")
        if not 100.0 <= self.maximum_mechanical_power_w <= 5000.0:
            raise ValueError("mechanical power ceiling must be in [100, 5000] W")
        if not 0.0 <= self.maximum_projection_ratio <= 1.0:
            raise ValueError("maximum projection ratio must be in [0, 1]")
        if not 0.0 <= self.maximum_parent_deviation_ratio <= 0.25:
            raise ValueError("maximum parent torque deviation ratio must be in [0, 0.25]")
        if not 2.0 <= self.maximum_observation_z <= 20.0:
            raise ValueError("maximum neural torque observation z must be in [2, 20]")
        if not -0.999 <= self.minimum_upright_gravity_z <= -0.50:
            raise ValueError("minimum upright gravity z must be in [-0.999, -0.50]")
        if not 0.40 <= self.minimum_pelvis_height_m <= 0.95:
            raise ValueError("minimum pelvis height must be in [0.40, 0.95] m")
        if not 0 <= self.recovery_cooldown_steps <= 5000:
            raise ValueError("recovery cooldown must be in [0, 5000] steps")
        if not 0 <= self.warmup_steps <= 5000:
            raise ValueError("neural torque warmup steps must be in [0, 5000]")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("neural torque candidates are restricted to SIM_ONLY")


@dataclass(frozen=True)
class G1TorqueControlFrame:
    """One simulator state used to construct the end-to-end policy input."""

    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_lower_limits: np.ndarray
    joint_upper_limits: np.ndarray
    torso_quaternion_wxyz: np.ndarray
    pelvis_position: np.ndarray
    ball_position: np.ndarray
    ball_velocity: np.ndarray
    target_y_m: float
    target_z_m: float
    policy_phase: float
    left_contact: bool
    right_contact: bool


@dataclass(frozen=True)
class G1TorqueProjection:
    torque: np.ndarray
    used_parent: bool
    reason: str | None
    projected_joint_count: int
    maximum_limit_ratio: float
    mechanical_power_w: float


@dataclass(frozen=True)
class G1TorquePolicyReceipt:
    artifact_hash: str
    body_hash: str
    parent_policy_hash: str
    inference_count: int
    learned_output_count: int
    fallback_count: int
    nonfinite_fallback_count: int
    projection_fallback_count: int
    out_of_distribution_fallback_count: int
    warmup_fallback_count: int
    state_guard_fallback_count: int
    projected_joint_count: int
    maximum_limit_ratio: float
    peak_mechanical_power_w: float
    direct_torque_output: bool
    activation_ceiling: str
    hardware_authorized: bool = False
    dds_opened: bool = False
    schema_version: str = "rosclaw.simforge.g1_neural_torque_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class G1TorquePolicy(Protocol):
    """Minimal callback surface accepted by the MuJoCo backend."""

    def reset(self) -> None: ...

    def command(
        self,
        frame: G1TorqueControlFrame,
        parent_torque: np.ndarray,
    ) -> np.ndarray: ...

    def note_applied(self, torque: np.ndarray) -> None: ...


@dataclass(frozen=True)
class G1NeuralTorqueArtifact:
    """Verified, inference-only recurrent actor."""

    artifact_hash: str
    body_hash: str
    parent_policy_hash: str
    dataset_hash: str
    observation_names: tuple[str, ...]
    action_names: tuple[str, ...]
    action_limits: tuple[float, ...]
    hidden_dim: int
    observation_clip: float
    update_index: int
    safety: G1TorqueSafetyConfig
    tensors: Mapping[str, np.ndarray]
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))


@dataclass(frozen=True)
class G1TeacherTorqueEpisode:
    observations: np.ndarray
    actions: np.ndarray
    parent_actions: np.ndarray

    def __post_init__(self) -> None:
        observations = np.asarray(self.observations, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        parent = np.asarray(self.parent_actions, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != len(G1_NEURAL_TORQUE_OBSERVATIONS):
            raise ValueError("teacher observations have the wrong shape")
        if actions.shape != (len(observations), len(G1_DDS_JOINT_NAMES)):
            raise ValueError("teacher actions have the wrong shape")
        if parent.shape != actions.shape:
            raise ValueError("teacher parent actions must align with actions")
        if not all(np.all(np.isfinite(value)) for value in (observations, actions, parent)):
            raise ValueError("teacher torque episode must contain only finite values")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "parent_actions", parent)


class G1TeacherTorqueCollector:
    """Pass-through policy that records the 500 Hz parent-torque teacher."""

    def __init__(self) -> None:
        self._observations: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._parents: list[np.ndarray] = []
        self._previous: np.ndarray = np.zeros(len(G1_DDS_JOINT_NAMES), dtype=np.float64)
        self._pending_observation: np.ndarray | None = None
        self._pending_parent: np.ndarray | None = None

    def reset(self) -> None:
        self._observations.clear()
        self._actions.clear()
        self._parents.clear()
        self._previous.fill(0.0)
        self._pending_observation = None
        self._pending_parent = None

    def command(
        self,
        frame: G1TorqueControlFrame,
        parent_torque: np.ndarray,
    ) -> np.ndarray:
        if self._pending_observation is not None:
            raise RuntimeError("teacher collector did not receive note_applied")
        self._pending_observation = build_g1_neural_torque_observation(frame, self._previous)
        self._pending_parent = _vector(parent_torque, label="parent torque")
        return self._pending_parent.copy()

    def note_applied(self, torque: np.ndarray) -> None:
        if self._pending_observation is None or self._pending_parent is None:
            raise RuntimeError("teacher collector received applied torque without a command")
        applied = _vector(torque, label="applied teacher torque")
        self._observations.append(self._pending_observation)
        self._actions.append(applied)
        self._parents.append(self._pending_parent)
        self._previous = applied.copy()
        self._pending_observation = None
        self._pending_parent = None

    def episode(self) -> G1TeacherTorqueEpisode:
        if self._pending_observation is not None or not self._observations:
            raise ValueError("teacher torque episode is empty or incomplete")
        return G1TeacherTorqueEpisode(
            observations=np.asarray(self._observations, dtype=np.float32),
            actions=np.asarray(self._actions, dtype=np.float32),
            parent_actions=np.asarray(self._parents, dtype=np.float32),
        )


class G1TorqueSafetyProjector:
    """Independent state-aware envelope for direct neural torques."""

    def __init__(self, config: G1TorqueSafetyConfig) -> None:
        self.config = config
        self._hard = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
        self._guarded = self._hard * config.torque_guard_scale
        self._parent_guarded = self._hard * 0.85
        self._delta = self._hard * config.max_delta_ratio_per_step

    def project(
        self,
        proposed: np.ndarray,
        *,
        parent: np.ndarray,
        previous: np.ndarray,
        frame: G1TorqueControlFrame,
    ) -> G1TorqueProjection:
        parent_value = self._safe_parent(parent)
        previous_value = self._safe_previous(previous, parent_value)
        try:
            candidate = _vector(proposed, label="neural torque proposal")
        except ValueError as exc:
            return self._fallback(parent_value, frame, f"invalid_proposal:{exc}")

        clipped = np.clip(candidate, -self._guarded, self._guarded)
        parent_delta = self._hard * self.config.maximum_parent_deviation_ratio
        clipped = np.clip(clipped, parent_value - parent_delta, parent_value + parent_delta)
        clipped = np.clip(clipped, previous_value - self._delta, previous_value + self._delta)
        lower = _vector(frame.joint_lower_limits, label="joint lower limits")
        upper = _vector(frame.joint_upper_limits, label="joint upper limits")
        position = _vector(frame.joint_position, label="joint position")
        if np.any(lower >= upper):
            return self._fallback(parent_value, frame, "invalid_joint_limits")
        outward_low = (position <= lower + self.config.joint_limit_margin_rad) & (clipped < 0.0)
        outward_high = (position >= upper - self.config.joint_limit_margin_rad) & (clipped > 0.0)
        if np.any(outward_low | outward_high):
            return self._fallback(parent_value, frame, "joint_limit_guard")
        clipped = self._limit_power(clipped, frame.joint_velocity)
        changed = np.abs(clipped - candidate) > 1e-8
        projection_ratio = float(np.count_nonzero(changed)) / len(changed)
        if projection_ratio > self.config.maximum_projection_ratio:
            return self._fallback(parent_value, frame, "projection_ratio_exceeded")
        return self._result(
            clipped,
            frame=frame,
            used_parent=False,
            reason=None,
            projected_joint_count=int(np.count_nonzero(changed)),
        )

    def fallback(
        self,
        *,
        parent: np.ndarray,
        frame: G1TorqueControlFrame,
        reason: str,
    ) -> G1TorqueProjection:
        """Return the independently bounded parent with an auditable reason."""

        if not reason.strip():
            raise ValueError("neural torque fallback reason must not be empty")
        return self._fallback(self._safe_parent(parent), frame, reason)

    def _safe_parent(self, parent: np.ndarray) -> np.ndarray:
        try:
            value = _vector(parent, label="parent torque")
        except ValueError:
            return np.zeros_like(self._hard)
        return np.clip(value, -self._parent_guarded, self._parent_guarded)

    def _safe_previous(self, previous: np.ndarray, parent: np.ndarray) -> np.ndarray:
        try:
            return np.clip(
                _vector(previous, label="previous torque"), -self._guarded, self._guarded
            )
        except ValueError:
            return parent.copy()

    def _fallback(
        self,
        parent: np.ndarray,
        frame: G1TorqueControlFrame,
        reason: str,
    ) -> G1TorqueProjection:
        # The qualified parent has already passed the backend's immutable
        # torque guard.  Preserve it exactly; candidate-only limits must not
        # turn a fail-closed fallback into a different controller.
        safe = np.clip(parent, -self._parent_guarded, self._parent_guarded)
        return self._result(
            safe,
            frame=frame,
            used_parent=True,
            reason=reason,
            projected_joint_count=0,
        )

    def _limit_power(self, torque: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        speed = _vector(velocity, label="joint velocity")
        power = float(np.sum(np.abs(torque * speed)))
        if power <= self.config.maximum_mechanical_power_w or power <= 1e-12:
            return torque
        return torque * (self.config.maximum_mechanical_power_w / power)

    def _result(
        self,
        torque: np.ndarray,
        *,
        frame: G1TorqueControlFrame,
        used_parent: bool,
        reason: str | None,
        projected_joint_count: int,
    ) -> G1TorqueProjection:
        ratio = float(np.max(np.abs(torque) / self._hard))
        power = float(
            np.sum(np.abs(torque * _vector(frame.joint_velocity, label="joint velocity")))
        )
        return G1TorqueProjection(
            torque=np.asarray(torque, dtype=np.float64),
            used_parent=used_parent,
            reason=reason,
            projected_joint_count=projected_joint_count,
            maximum_limit_ratio=ratio,
            mechanical_power_w=power,
        )


class G1NeuralTorquePolicy:
    """Allocation-light NumPy GRU actor with parent fallback and receipts."""

    def __init__(
        self,
        artifact: G1NeuralTorqueArtifact,
        *,
        expected_body_hash: str,
        expected_parent_policy_hash: str,
    ) -> None:
        if artifact.body_hash != expected_body_hash:
            raise ValueError("neural torque artifact body hash mismatch")
        if artifact.parent_policy_hash != expected_parent_policy_hash:
            raise ValueError("neural torque artifact parent policy hash mismatch")
        self.artifact = artifact
        self.projector = G1TorqueSafetyProjector(artifact.safety)
        self._hidden: np.ndarray = np.zeros(artifact.hidden_dim, dtype=np.float32)
        self._previous: np.ndarray = np.zeros(len(G1_DDS_JOINT_NAMES), dtype=np.float64)
        self._pending: G1TorqueProjection | None = None
        self._observations: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._parents: list[np.ndarray] = []
        self._fallback_reasons: list[str] = []
        self._projected_joint_count = 0
        self._maximum_limit_ratio = 0.0
        self._peak_power = 0.0
        self._recovery_cooldown = 0

    def reset(self) -> None:
        self._hidden.fill(0.0)
        self._previous.fill(0.0)
        self._pending = None
        self._observations.clear()
        self._actions.clear()
        self._parents.clear()
        self._fallback_reasons.clear()
        self._projected_joint_count = 0
        self._maximum_limit_ratio = 0.0
        self._peak_power = 0.0
        self._recovery_cooldown = 0

    def command(
        self,
        frame: G1TorqueControlFrame,
        parent_torque: np.ndarray,
    ) -> np.ndarray:
        if self._pending is not None:
            raise RuntimeError("neural torque policy did not receive note_applied")
        parent = np.asarray(parent_torque, dtype=np.float64)
        try:
            raw = build_g1_neural_torque_observation(frame, self._previous)
            normalized = self._normalize(raw)
            if len(self._observations) < self.artifact.safety.warmup_steps:
                projection = self.projector.fallback(
                    parent=parent,
                    frame=frame,
                    reason="warmup_parent",
                )
                self._advance_hidden(normalized)
            elif (
                float(frame.pelvis_position[2])
                < self.artifact.safety.minimum_pelvis_height_m
                or float(raw[60]) > self.artifact.safety.minimum_upright_gravity_z
            ):
                self._recovery_cooldown = self.artifact.safety.recovery_cooldown_steps
                projection = self.projector.fallback(
                    parent=parent,
                    frame=frame,
                    reason="state_recovery_parent",
                )
                self._advance_hidden(normalized)
            elif self._recovery_cooldown > 0:
                self._recovery_cooldown -= 1
                projection = self.projector.fallback(
                    parent=parent,
                    frame=frame,
                    reason="recovery_cooldown_parent",
                )
                self._advance_hidden(normalized)
            elif float(np.max(np.abs(normalized))) > self.artifact.safety.maximum_observation_z:
                projection = self.projector.fallback(
                    parent=parent,
                    frame=frame,
                    reason="observation_out_of_distribution",
                )
                self._hidden.fill(0.0)
            else:
                proposed = self._infer_normalized(normalized)
                projection = self.projector.project(
                    proposed,
                    parent=parent,
                    previous=self._previous,
                    frame=frame,
                )
        except (ValueError, FloatingPointError) as exc:
            projection = self.projector.project(
                np.full(len(G1_DDS_JOINT_NAMES), np.nan),
                parent=parent,
                previous=self._previous,
                frame=frame,
            )
            projection = G1TorqueProjection(
                torque=projection.torque,
                used_parent=True,
                reason=f"observation_or_inference:{type(exc).__name__}",
                projected_joint_count=projection.projected_joint_count,
                maximum_limit_ratio=projection.maximum_limit_ratio,
                mechanical_power_w=projection.mechanical_power_w,
            )
            raw = np.zeros(len(self.artifact.observation_names), dtype=np.float32)
        self._pending = projection
        self._observations.append(raw)
        self._parents.append(np.asarray(parent, dtype=np.float32))
        if projection.reason is not None:
            self._fallback_reasons.append(projection.reason)
        self._projected_joint_count += projection.projected_joint_count
        self._maximum_limit_ratio = max(
            self._maximum_limit_ratio,
            projection.maximum_limit_ratio,
        )
        self._peak_power = max(self._peak_power, projection.mechanical_power_w)
        return projection.torque.copy()

    def note_applied(self, torque: np.ndarray) -> None:
        if self._pending is None:
            raise RuntimeError("neural torque policy received applied torque without command")
        applied = _vector(torque, label="applied neural torque")
        self._actions.append(np.asarray(applied, dtype=np.float32))
        self._previous = applied.copy()
        self._pending = None

    def build_receipt(self) -> G1TorquePolicyReceipt:
        if self._pending is not None or not self._actions:
            raise ValueError("cannot build an incomplete neural torque receipt")
        fallback_count = len(self._fallback_reasons)
        return G1TorquePolicyReceipt(
            artifact_hash=self.artifact.artifact_hash,
            body_hash=self.artifact.body_hash,
            parent_policy_hash=self.artifact.parent_policy_hash,
            inference_count=len(self._actions),
            learned_output_count=len(self._actions) - fallback_count,
            fallback_count=fallback_count,
            nonfinite_fallback_count=sum(
                "invalid_proposal" in reason or "observation_or_inference" in reason
                for reason in self._fallback_reasons
            ),
            projection_fallback_count=sum(
                reason == "projection_ratio_exceeded" for reason in self._fallback_reasons
            ),
            out_of_distribution_fallback_count=sum(
                reason == "observation_out_of_distribution" for reason in self._fallback_reasons
            ),
            warmup_fallback_count=sum(
                reason == "warmup_parent" for reason in self._fallback_reasons
            ),
            state_guard_fallback_count=sum(
                reason
                in {
                    "state_recovery_parent",
                    "recovery_cooldown_parent",
                    "joint_limit_guard",
                }
                for reason in self._fallback_reasons
            ),
            projected_joint_count=self._projected_joint_count,
            maximum_limit_ratio=self._maximum_limit_ratio,
            peak_mechanical_power_w=self._peak_power,
            direct_torque_output=True,
            activation_ceiling=self.artifact.safety.activation_ceiling,
        )

    def episode(self) -> G1TeacherTorqueEpisode:
        if self._pending is not None or not self._actions:
            raise ValueError("neural torque episode is empty or incomplete")
        return G1TeacherTorqueEpisode(
            observations=np.asarray(self._observations, dtype=np.float32),
            actions=np.asarray(self._actions, dtype=np.float32),
            parent_actions=np.asarray(self._parents, dtype=np.float32),
        )

    def _normalize(self, raw_observation: np.ndarray) -> np.ndarray:
        tensors = self.artifact.tensors
        return np.clip(
            (raw_observation - tensors["observation_mean"]) / tensors["observation_std"],
            -self.artifact.observation_clip,
            self.artifact.observation_clip,
        ).astype(np.float32)

    def _advance_hidden(self, normalized: np.ndarray) -> None:
        tensors = self.artifact.tensors
        weight_ih = tensors["actor.gru.weight_ih_l0"]
        weight_hh = tensors["actor.gru.weight_hh_l0"]
        bias_ih = tensors["actor.gru.bias_ih_l0"]
        bias_hh = tensors["actor.gru.bias_hh_l0"]
        input_gates = weight_ih @ normalized + bias_ih
        hidden_gates = weight_hh @ self._hidden + bias_hh
        reset_i, update_i, new_i = np.split(input_gates, 3)
        reset_h, update_h, new_h = np.split(hidden_gates, 3)
        reset = _sigmoid(reset_i + reset_h)
        update = _sigmoid(update_i + update_h)
        new = np.tanh(new_i + reset * new_h)
        self._hidden = ((1.0 - update) * new + update * self._hidden).astype(np.float32)

    def _infer_normalized(self, normalized: np.ndarray) -> np.ndarray:
        self._advance_hidden(normalized)
        tensors = self.artifact.tensors
        output = tensors["actor.head.weight"] @ self._hidden + tensors["actor.head.bias"]
        mean = output[: len(G1_DDS_JOINT_NAMES)]
        proposed = np.tanh(mean) * tensors["action_limits"]
        if not np.all(np.isfinite(proposed)):
            raise FloatingPointError("neural torque actor produced non-finite output")
        return proposed.astype(np.float64)


def build_g1_neural_torque_observation(
    frame: G1TorqueControlFrame,
    previous_torque: np.ndarray,
) -> np.ndarray:
    """Build the frozen raw observation order shared by training and runtime."""

    position = _vector(frame.joint_position, label="joint position")
    velocity = _vector(frame.joint_velocity, label="joint velocity")
    previous = _vector(previous_torque, label="previous torque")
    quaternion = np.asarray(frame.torso_quaternion_wxyz, dtype=np.float64)
    pelvis = np.asarray(frame.pelvis_position, dtype=np.float64)
    ball = np.asarray(frame.ball_position, dtype=np.float64)
    ball_velocity = np.asarray(frame.ball_velocity, dtype=np.float64)
    if quaternion.shape != (4,) or pelvis.shape != (3,) or ball.shape != (3,):
        raise ValueError("neural torque body pose fields have the wrong shape")
    if ball_velocity.shape != (3,):
        raise ValueError("neural torque ball velocity has the wrong shape")
    scalars = (frame.target_y_m, frame.target_z_m, frame.policy_phase)
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("neural torque task context must be finite")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("neural torque torso quaternion is invalid")
    quaternion = quaternion / norm
    gravity = _inverse_rotate(quaternion, np.asarray((0.0, 0.0, -1.0)))
    phase = float(np.clip(frame.policy_phase, 0.0, 1.0)) * (2.0 * math.pi)
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    value = np.concatenate(
        (
            position,
            velocity,
            gravity,
            ball - pelvis,
            ball_velocity,
            np.asarray((frame.target_y_m, frame.target_z_m)),
            np.asarray((math.sin(phase), math.cos(phase))),
            np.asarray((float(frame.left_contact), float(frame.right_contact))),
            previous / limits,
        )
    ).astype(np.float32)
    if value.shape != (len(G1_NEURAL_TORQUE_OBSERVATIONS),) or not np.all(np.isfinite(value)):
        raise ValueError("neural torque observation is malformed or non-finite")
    return value


def load_g1_neural_torque_artifact(
    path: Path,
    *,
    expected_hash: str | None = None,
    expected_body_hash: str | None = None,
) -> G1NeuralTorqueArtifact:
    """Load a bounded safe-tensor actor artifact without pickle execution."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"neural torque artifact is missing: {resolved}")
    size = resolved.stat().st_size
    if size <= 0 or size > _MAX_ARTIFACT_BYTES:
        raise ValueError("neural torque artifact size is outside the loader limit")
    payload = resolved.read_bytes()
    artifact_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    if expected_hash is not None and artifact_hash != expected_hash:
        raise ValueError("neural torque artifact hash mismatch")
    metadata_end = _json_object_end(payload)
    try:
        metadata = json.loads(payload[:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("neural torque artifact metadata is invalid") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported neural torque artifact schema")
    body_hash = _hash(metadata, "body_hash")
    parent_hash = _hash(metadata, "parent_policy_hash")
    dataset_hash = _hash(metadata, "dataset_hash")
    if expected_body_hash is not None and body_hash != expected_body_hash:
        raise ValueError("neural torque artifact body hash mismatch")
    observations = _names(metadata, "observation_names")
    actions = _names(metadata, "action_names")
    if observations != G1_NEURAL_TORQUE_OBSERVATIONS:
        raise ValueError("neural torque observation contract mismatch")
    if actions != G1_NEURAL_TORQUE_ACTIONS:
        raise ValueError("neural torque action contract mismatch")
    limits = _float_tuple(metadata, "action_limits", len(actions))
    hard = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    if np.any(np.asarray(limits) <= 0.0) or np.any(np.asarray(limits) > hard * 0.85 + 1e-8):
        raise ValueError("neural torque action limits exceed the immutable ceiling")
    hidden_dim = _positive_int(metadata, "hidden_dim", maximum=1024)
    update_index = _nonnegative_int(metadata, "update_index")
    observation_clip = float(metadata.get("observation_clip", math.nan))
    if not math.isfinite(observation_clip) or not 1.0 <= observation_clip <= 20.0:
        raise ValueError("neural torque observation clip is invalid")
    safety_raw = metadata.get("safety")
    if not isinstance(safety_raw, dict):
        raise ValueError("neural torque artifact lacks its safety envelope")
    try:
        safety = G1TorqueSafetyConfig(**safety_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("neural torque artifact safety envelope is invalid") from exc
    if not np.allclose(
        limits,
        hard * safety.torque_guard_scale,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("neural torque action limits do not match the safety envelope")
    tensors = _parse_tensors(payload, metadata_end)
    _verify_tensors(tensors, hidden_dim=hidden_dim, action_limits=limits)
    return G1NeuralTorqueArtifact(
        artifact_hash=artifact_hash,
        body_hash=body_hash,
        parent_policy_hash=parent_hash,
        dataset_hash=dataset_hash,
        observation_names=observations,
        action_names=actions,
        action_limits=limits,
        hidden_dim=hidden_dim,
        observation_clip=observation_clip,
        update_index=update_index,
        safety=safety,
        tensors=tensors,
    )


def serialize_g1_neural_torque_artifact(
    *,
    body_hash: str,
    parent_policy_hash: str,
    dataset_hash: str,
    hidden_dim: int,
    observation_clip: float,
    update_index: int,
    safety: G1TorqueSafetyConfig,
    tensors: Mapping[str, np.ndarray],
) -> bytes:
    """Create deterministic metadata+float32 tensor bytes for content addressing."""

    for label, value in (
        ("body_hash", body_hash),
        ("parent_policy_hash", parent_policy_hash),
        ("dataset_hash", dataset_hash),
    ):
        if not _SHA256.fullmatch(value):
            raise ValueError(f"{label} must be a sha256 content hash")
    limits = tuple(float(value) * safety.torque_guard_scale for value in G1_HARD_TORQUE_LIMITS)
    metadata = {
        "schema_version": _SCHEMA,
        "body_hash": body_hash,
        "parent_policy_hash": parent_policy_hash,
        "dataset_hash": dataset_hash,
        "observation_names": list(G1_NEURAL_TORQUE_OBSERVATIONS),
        "action_names": list(G1_NEURAL_TORQUE_ACTIONS),
        "action_limits": list(limits),
        "action_semantics": "DIRECT_JOINT_TORQUE",
        "hidden_dim": hidden_dim,
        "observation_clip": observation_clip,
        "update_index": update_index,
        "safety": asdict(safety),
        "activation_ceiling": "SIM_ONLY",
    }
    chunks = [json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()]
    for name, original in sorted(tensors.items()):
        value = np.ascontiguousarray(original, dtype=np.float32)
        if not name or not np.all(np.isfinite(value)):
            raise ValueError("neural torque artifact tensors must be named and finite")
        name_bytes = name.encode()
        dtype_bytes = b"float32"
        chunks.extend(
            (
                struct.pack("!I", len(name_bytes)),
                name_bytes,
                struct.pack("!I", len(dtype_bytes)),
                dtype_bytes,
                struct.pack("!I", value.ndim),
                struct.pack("!" + "Q" * value.ndim, *value.shape),
                value.tobytes(),
            )
        )
    payload = b"".join(chunks)
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise ValueError("neural torque artifact exceeds the loader size limit")
    return payload


def _verify_tensors(
    tensors: Mapping[str, np.ndarray],
    *,
    hidden_dim: int,
    action_limits: tuple[float, ...],
) -> None:
    observation_dim = len(G1_NEURAL_TORQUE_OBSERVATIONS)
    action_dim = len(G1_DDS_JOINT_NAMES)
    expected = {
        "action_limits": (action_dim,),
        "observation_mean": (observation_dim,),
        "observation_std": (observation_dim,),
        "actor.gru.weight_ih_l0": (3 * hidden_dim, observation_dim),
        "actor.gru.weight_hh_l0": (3 * hidden_dim, hidden_dim),
        "actor.gru.bias_ih_l0": (3 * hidden_dim,),
        "actor.gru.bias_hh_l0": (3 * hidden_dim,),
        "actor.head.weight": (2 * action_dim, hidden_dim),
        "actor.head.bias": (2 * action_dim,),
    }
    if set(tensors) != set(expected):
        raise ValueError("neural torque artifact has an unexpected tensor set")
    for name, shape in expected.items():
        if tensors[name].shape != shape:
            raise ValueError(f"neural torque tensor {name} has the wrong shape")
    if np.any(tensors["observation_std"] <= 1e-6):
        raise ValueError("neural torque observation standard deviations must be positive")
    if not np.allclose(tensors["action_limits"], action_limits, rtol=0.0, atol=1e-5):
        raise ValueError("neural torque tensor limits do not match metadata")


def _parse_tensors(payload: bytes, offset: int) -> Mapping[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    cursor = offset
    while cursor < len(payload):
        name_size, cursor = _read_u32(payload, cursor, "tensor name length")
        name_raw, cursor = _read(payload, cursor, name_size, "tensor name")
        dtype_size, cursor = _read_u32(payload, cursor, "tensor dtype length")
        dtype_raw, cursor = _read(payload, cursor, dtype_size, "tensor dtype")
        rank, cursor = _read_u32(payload, cursor, "tensor rank")
        if rank > 4:
            raise ValueError("neural torque tensor rank exceeds four")
        shape_raw, cursor = _read(payload, cursor, 8 * rank, "tensor shape")
        shape = tuple(struct.unpack("!" + "Q" * rank, shape_raw)) if rank else ()
        try:
            name = name_raw.decode("utf-8")
            dtype = dtype_raw.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("neural torque tensor descriptor is invalid") from exc
        if not name or name in result or dtype != "float32":
            raise ValueError("neural torque tensor name or dtype is invalid")
        if any(dimension <= 0 or dimension > 2_000_000 for dimension in shape):
            raise ValueError("neural torque tensor shape is invalid")
        count = math.prod(shape)
        raw, cursor = _read(payload, cursor, count * 4, f"tensor {name}")
        value = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        if not np.all(np.isfinite(value)):
            raise ValueError(f"neural torque tensor {name} is non-finite")
        value.flags.writeable = False
        result[name] = value
    if not result or cursor != len(payload):
        raise ValueError("neural torque tensor payload is empty or truncated")
    return MappingProxyType(result)


def _json_object_end(payload: bytes) -> int:
    if not payload or payload[0] != ord("{"):
        raise ValueError("neural torque artifact must begin with JSON metadata")
    depth = 0
    in_string = False
    escaped = False
    for index, byte in enumerate(payload):
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue
        if byte == ord('"'):
            in_string = True
        elif byte == ord("{"):
            depth += 1
        elif byte == ord("}"):
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError("neural torque artifact metadata is incomplete")


def _read_u32(payload: bytes, cursor: int, label: str) -> tuple[int, int]:
    raw, cursor = _read(payload, cursor, 4, label)
    return struct.unpack("!I", raw)[0], cursor


def _read(payload: bytes, cursor: int, count: int, label: str) -> tuple[bytes, int]:
    if count < 0 or cursor < 0 or cursor + count > len(payload):
        raise ValueError(f"neural torque artifact is truncated at {label}")
    return payload[cursor : cursor + count], cursor + count


def _names(metadata: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"neural torque {key} must be a non-empty list")
    names = tuple(item for item in value if isinstance(item, str) and item.strip())
    if len(names) != len(value) or len(set(names)) != len(names):
        raise ValueError(f"neural torque {key} must contain unique strings")
    return names


def _float_tuple(metadata: Mapping[str, Any], key: str, count: int) -> tuple[float, ...]:
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"neural torque {key} has the wrong length")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"neural torque {key} must contain finite numbers")
    return result


def _hash(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"neural torque {key} must be a sha256 content hash")
    return value


def _positive_int(metadata: Mapping[str, Any], key: str, *, maximum: int) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise ValueError(f"neural torque {key} must be in [1, {maximum}]")
    return value


def _nonnegative_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"neural torque {key} must be a non-negative integer")
    return value


def _vector(value: np.ndarray, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (len(G1_DDS_JOINT_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 29-vector")
    return result.copy()


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _inverse_rotate(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = map(float, quaternion_wxyz)
    rotation = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return rotation.T @ vector


__all__ = [
    "G1_NEURAL_TORQUE_ACTIONS",
    "G1_NEURAL_TORQUE_OBSERVATIONS",
    "G1NeuralTorqueArtifact",
    "G1NeuralTorquePolicy",
    "G1TeacherTorqueCollector",
    "G1TeacherTorqueEpisode",
    "G1TorqueControlFrame",
    "G1TorquePolicy",
    "G1TorquePolicyReceipt",
    "G1TorqueProjection",
    "G1TorqueSafetyConfig",
    "G1TorqueSafetyProjector",
    "build_g1_neural_torque_observation",
    "load_g1_neural_torque_artifact",
    "serialize_g1_neural_torque_artifact",
]
