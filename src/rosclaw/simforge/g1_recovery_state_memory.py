"""SIM-only short-horizon recovery-state memory for G1 post-kick control.

The policy observes a brief proprioceptive window after contact and landing,
then consults a conservative evidence neighborhood.  A primitive is selected
only when enough nearby DEVELOPMENT exemplars agree on the primitive and all
matching exemplars retain a positive measured advantage.  Negative exemplars
are first-class abstention votes.  The actor cannot emit torque, joint targets,
ROS/DDS commands, or hardware authorization.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.simforge.g1_contextual_recovery import G1ContextualRecoveryPrimitive
from rosclaw.simforge.g1_muscle_memory import G1_MUSCLE_MEMORY_OBSERVATIONS
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024

G1_RECOVERY_STATE_FEATURES = (
    "pelvis_velocity_x_m_s",
    "pelvis_velocity_y_m_s",
    "pelvis_velocity_z_m_s",
    "torso_roll_rad",
    "torso_pitch_rad",
    "torso_angular_velocity_x_rad_s",
    "torso_angular_velocity_y_rad_s",
    "torso_angular_velocity_z_rad_s",
    "com_y_relative_m",
    "contact_impulse_ns",
    "ball_relative_position_x_m",
    "ball_relative_position_y_m",
    "ball_relative_position_z_m",
    "ball_relative_velocity_x_m_s",
    "ball_relative_velocity_y_m_s",
    "ball_relative_velocity_z_m_s",
)

G1_RECOVERY_STATE_OBSERVATIONS = G1_MUSCLE_MEMORY_OBSERVATIONS + (
    "ball_relative_position_x_m",
    "ball_relative_position_y_m",
    "ball_relative_position_z_m",
    "ball_relative_velocity_x_m_s",
    "ball_relative_velocity_y_m_s",
    "ball_relative_velocity_z_m_s",
)


@dataclass(frozen=True)
class G1RecoveryStateArtifact:
    """Content-addressed temporal evidence-neighborhood policy."""

    body_hash: str
    motion_hash: str
    baseline_recovery_config_hash: str
    fallback_recovery_config_hash: str
    training_dataset_hash: str
    observation_mean: tuple[float, ...]
    observation_scale: tuple[float, ...]
    descriptor_feature_names: tuple[str, ...]
    descriptor_prototypes: tuple[tuple[float, ...], ...]
    prototype_primitive_indices: tuple[int, ...]
    prototype_composite_advantages: tuple[float, ...]
    prototype_component_minimums: tuple[float, ...]
    primitives: tuple[G1ContextualRecoveryPrimitive, ...]
    selection_window_frames: int
    neighbor_count: int
    maximum_neighbor_distance: float
    minimum_primitive_consensus: float
    minimum_advantage_lower_bound: float
    minimum_component_lower_bound: float
    maximum_feature_z: float
    training_episode_count: int
    training_seed: int
    negative_veto_distance_ratio: float | None = None
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.g1_goalforge.recovery_state_artifact.v2"

    def __post_init__(self) -> None:
        for label, value in (
            ("body_hash", self.body_hash),
            ("motion_hash", self.motion_hash),
            ("baseline_recovery_config_hash", self.baseline_recovery_config_hash),
            ("fallback_recovery_config_hash", self.fallback_recovery_config_hash),
            ("training_dataset_hash", self.training_dataset_hash),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256 content hash")
        supported_schemas = {
            "rosclaw.g1_goalforge.recovery_state_artifact.v2",
            "rosclaw.g1_goalforge.recovery_state_artifact.v3",
        }
        if self.schema_version not in supported_schemas:
            raise ValueError("unsupported recovery-state artifact schema")
        if self.schema_version.endswith(".v2"):
            if self.negative_veto_distance_ratio is not None:
                raise ValueError("v2 recovery-state artifacts cannot enable negative veto")
        elif (
            isinstance(self.negative_veto_distance_ratio, bool)
            or not isinstance(self.negative_veto_distance_ratio, (int, float))
            or not math.isfinite(self.negative_veto_distance_ratio)
            or not 1.0 <= self.negative_veto_distance_ratio <= 4.0
        ):
            raise ValueError("v3 recovery-state negative veto ratio must be in [1, 4]")
        if self.descriptor_feature_names != G1_RECOVERY_STATE_FEATURES:
            raise ValueError("recovery-state descriptor feature contract mismatch")
        observation_count = len(G1_RECOVERY_STATE_OBSERVATIONS)
        if (
            len(self.observation_mean) != observation_count
            or len(self.observation_scale) != observation_count
        ):
            raise ValueError("recovery-state normalization shape is invalid")
        descriptor_width = 2 * len(self.descriptor_feature_names)
        prototype_count = len(self.descriptor_prototypes)
        if not 3 <= prototype_count <= 64:
            raise ValueError("recovery-state artifact requires 3 to 64 prototypes")
        if any(len(row) != descriptor_width for row in self.descriptor_prototypes):
            raise ValueError("recovery-state descriptor prototype shape is invalid")
        aligned = (
            len(self.prototype_primitive_indices),
            len(self.prototype_composite_advantages),
            len(self.prototype_component_minimums),
        )
        if any(count != prototype_count for count in aligned):
            raise ValueError("recovery-state prototype evidence is misaligned")
        if not 1 <= len(self.primitives) <= 16:
            raise ValueError("recovery-state primitive library is invalid")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < -1
            or index >= len(self.primitives)
            for index in self.prototype_primitive_indices
        ):
            raise ValueError("recovery-state prototype route is invalid")
        arrays = (
            np.asarray(self.observation_mean, dtype=np.float64),
            np.asarray(self.observation_scale, dtype=np.float64),
            np.asarray(self.descriptor_prototypes, dtype=np.float64),
            np.asarray(self.prototype_composite_advantages, dtype=np.float64),
            np.asarray(self.prototype_component_minimums, dtype=np.float64),
        )
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("recovery-state artifact contains non-finite values")
        if np.any(arrays[1] <= 0.0):
            raise ValueError("recovery-state observation scales must be positive")
        integer_fields = (
            self.selection_window_frames,
            self.neighbor_count,
            self.training_episode_count,
            self.training_seed,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise ValueError("recovery-state count fields must be integers")
        if not 3 <= self.selection_window_frames <= 12:
            raise ValueError("recovery-state window must contain 3 to 12 frames")
        if not 2 <= self.neighbor_count <= 5 or self.neighbor_count > prototype_count:
            raise ValueError("recovery-state neighbor count is invalid")
        if self.training_episode_count <= 0 or self.training_seed < 0:
            raise ValueError("recovery-state training evidence is invalid")
        thresholds = (
            self.maximum_neighbor_distance,
            self.minimum_primitive_consensus,
            self.minimum_advantage_lower_bound,
            self.minimum_component_lower_bound,
            self.maximum_feature_z,
        )
        if any(isinstance(value, bool) for value in thresholds) or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in thresholds
        ):
            raise ValueError("recovery-state thresholds must be finite numbers")
        if not 0.05 <= self.maximum_neighbor_distance <= 2.0:
            raise ValueError("recovery-state distance must be in [0.05, 2]")
        if not 0.50 <= self.minimum_primitive_consensus <= 1.0:
            raise ValueError("recovery-state consensus must be in [0.50, 1]")
        if not 0.0 <= self.minimum_advantage_lower_bound <= 0.50:
            raise ValueError("recovery-state advantage lower bound is invalid")
        if not -0.10 <= self.minimum_component_lower_bound <= 0.0:
            raise ValueError("recovery-state component lower bound is invalid")
        if not 2.0 <= self.maximum_feature_z <= 12.0:
            raise ValueError("recovery-state feature envelope must be in [2, 12]")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("recovery-state memory must remain SIM_ONLY")

    @property
    def artifact_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.schema_version.endswith(".v2"):
            # Preserve the byte-level v2 contract and its content hashes.
            value.pop("negative_veto_distance_ratio")
        value["observation_names"] = list(G1_RECOVERY_STATE_OBSERVATIONS)
        value["observation_mean"] = list(self.observation_mean)
        value["observation_scale"] = list(self.observation_scale)
        value["descriptor_feature_names"] = list(self.descriptor_feature_names)
        value["descriptor_prototypes"] = [list(row) for row in self.descriptor_prototypes]
        value["prototype_primitive_indices"] = list(self.prototype_primitive_indices)
        value["prototype_composite_advantages"] = list(self.prototype_composite_advantages)
        value["prototype_component_minimums"] = list(self.prototype_component_minimums)
        value["primitives"] = [primitive.to_dict() for primitive in self.primitives]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> G1RecoveryStateArtifact:
        expected = {
            "schema_version",
            "body_hash",
            "motion_hash",
            "baseline_recovery_config_hash",
            "fallback_recovery_config_hash",
            "training_dataset_hash",
            "observation_names",
            "observation_mean",
            "observation_scale",
            "descriptor_feature_names",
            "descriptor_prototypes",
            "prototype_primitive_indices",
            "prototype_composite_advantages",
            "prototype_component_minimums",
            "primitives",
            "selection_window_frames",
            "neighbor_count",
            "maximum_neighbor_distance",
            "minimum_primitive_consensus",
            "minimum_advantage_lower_bound",
            "minimum_component_lower_bound",
            "maximum_feature_z",
            "training_episode_count",
            "training_seed",
            "activation_ceiling",
        }
        schema = value.get("schema_version")
        if schema == "rosclaw.g1_goalforge.recovery_state_artifact.v3":
            expected.add("negative_veto_distance_ratio")
        if set(value) != expected:
            raise ValueError("recovery-state artifact fields are invalid")
        if tuple(value["observation_names"]) != G1_RECOVERY_STATE_OBSERVATIONS:
            raise ValueError("recovery-state observation contract mismatch")
        features = value["descriptor_feature_names"]
        prototypes = value["descriptor_prototypes"]
        primitives = value["primitives"]
        if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
            raise ValueError("recovery-state feature names must be strings")
        if not isinstance(prototypes, list) or not all(isinstance(row, list) for row in prototypes):
            raise ValueError("recovery-state prototypes must be numeric arrays")
        if not isinstance(primitives, list) or not all(
            isinstance(item, Mapping) for item in primitives
        ):
            raise ValueError("recovery-state primitives must be objects")
        return cls(
            schema_version=str(value["schema_version"]),
            body_hash=str(value["body_hash"]),
            motion_hash=str(value["motion_hash"]),
            baseline_recovery_config_hash=str(value["baseline_recovery_config_hash"]),
            fallback_recovery_config_hash=str(value["fallback_recovery_config_hash"]),
            training_dataset_hash=str(value["training_dataset_hash"]),
            observation_mean=_float_tuple(value["observation_mean"]),
            observation_scale=_float_tuple(value["observation_scale"]),
            descriptor_feature_names=tuple(features),
            descriptor_prototypes=tuple(_float_tuple(row) for row in prototypes),
            prototype_primitive_indices=_int_tuple(value["prototype_primitive_indices"]),
            prototype_composite_advantages=_float_tuple(value["prototype_composite_advantages"]),
            prototype_component_minimums=_float_tuple(value["prototype_component_minimums"]),
            primitives=tuple(G1ContextualRecoveryPrimitive.from_dict(item) for item in primitives),
            selection_window_frames=_strict_int(value["selection_window_frames"]),
            neighbor_count=_strict_int(value["neighbor_count"]),
            maximum_neighbor_distance=_strict_float(value["maximum_neighbor_distance"]),
            minimum_primitive_consensus=_strict_float(value["minimum_primitive_consensus"]),
            minimum_advantage_lower_bound=_strict_float(value["minimum_advantage_lower_bound"]),
            minimum_component_lower_bound=_strict_float(value["minimum_component_lower_bound"]),
            maximum_feature_z=_strict_float(value["maximum_feature_z"]),
            training_episode_count=_strict_int(value["training_episode_count"]),
            training_seed=_strict_int(value["training_seed"]),
            negative_veto_distance_ratio=(
                _strict_float(value["negative_veto_distance_ratio"])
                if schema == "rosclaw.g1_goalforge.recovery_state_artifact.v3"
                else None
            ),
            activation_ceiling=str(value["activation_ceiling"]),
        )


@dataclass(frozen=True)
class G1RecoveryStateSelection:
    ready: bool
    primitive_index: int | None
    nearest_distance: float | None
    neighbor_count: int
    primitive_consensus: float
    advantage_lower_bound: float | None
    component_lower_bound: float | None
    out_of_distribution: bool
    fallback_reason: str | None
    nearest_negative_distance: float | None = None


@dataclass(frozen=True)
class G1RecoveryStateReceipt:
    artifact_hash: str
    selected_primitive_index: int | None
    nearest_distance: float | None
    neighbor_count: int
    primitive_consensus: float
    advantage_lower_bound: float | None
    component_lower_bound: float | None
    fallback_reason: str | None
    pending_count: int
    selection_count: int
    fallback_count: int
    descriptor_hash: str | None
    nearest_negative_distance: float | None = None
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.g1_goalforge.recovery_state_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class G1RecoveryStatePolicy:
    """Conservative temporal k-nearest-evidence primitive router."""

    def __init__(self, artifact: G1RecoveryStateArtifact) -> None:
        self.artifact = artifact
        self._mean = np.asarray(artifact.observation_mean, dtype=np.float64)
        self._scale = np.asarray(artifact.observation_scale, dtype=np.float64)
        self._prototypes = np.asarray(artifact.descriptor_prototypes, dtype=np.float64)
        self._prototype_routes = np.asarray(
            artifact.prototype_primitive_indices,
            dtype=np.int64,
        )
        self._advantages = np.asarray(
            artifact.prototype_composite_advantages,
            dtype=np.float64,
        )
        self._component_minimums = np.asarray(
            artifact.prototype_component_minimums,
            dtype=np.float64,
        )
        self._feature_indices = np.asarray(
            [G1_RECOVERY_STATE_OBSERVATIONS.index(name) for name in G1_RECOVERY_STATE_FEATURES],
            dtype=np.int64,
        )
        self.reset()

    def reset(self) -> None:
        self._window: deque[np.ndarray] = deque(maxlen=self.artifact.selection_window_frames)
        self._selection: G1RecoveryStateSelection | None = None
        self._descriptor: np.ndarray | None = None
        self._pending_count = 0
        self._selection_count = 0
        self._fallback_count = 0

    def require_compatible(
        self,
        *,
        body_hash: str,
        motion_hash: str,
        baseline_recovery_config_hash: str,
        fallback_recovery_config_hash: str,
    ) -> None:
        expected = self.artifact
        if body_hash != expected.body_hash:
            raise ValueError("recovery-state Body hash mismatch")
        if motion_hash != expected.motion_hash:
            raise ValueError("recovery-state motion hash mismatch")
        if baseline_recovery_config_hash != expected.baseline_recovery_config_hash:
            raise ValueError("recovery-state baseline config hash mismatch")
        if fallback_recovery_config_hash != expected.fallback_recovery_config_hash:
            raise ValueError("recovery-state fallback config hash mismatch")

    def select(self, observation: Mapping[str, float]) -> G1RecoveryStateSelection:
        if self._selection is not None:
            return self._selection
        missing = set(G1_RECOVERY_STATE_OBSERVATIONS).difference(observation)
        if missing:
            return self._fallback("missing_observation")
        try:
            ordered = np.asarray(
                [float(observation[name]) for name in G1_RECOVERY_STATE_OBSERVATIONS],
                dtype=np.float64,
            )
        except (TypeError, ValueError, OverflowError):
            return self._fallback("invalid_observation")
        if not np.all(np.isfinite(ordered)):
            return self._fallback("nonfinite_observation")
        normalized = (ordered - self._mean) / self._scale
        if float(np.max(np.abs(normalized))) > self.artifact.maximum_feature_z:
            return self._fallback("feature_envelope_exceeded")
        self._window.append(normalized[self._feature_indices].copy())
        if len(self._window) < self.artifact.selection_window_frames:
            self._pending_count += 1
            return G1RecoveryStateSelection(
                ready=False,
                primitive_index=None,
                nearest_distance=None,
                neighbor_count=0,
                primitive_consensus=0.0,
                advantage_lower_bound=None,
                component_lower_bound=None,
                out_of_distribution=False,
                fallback_reason=None,
            )
        window = np.asarray(self._window, dtype=np.float64)
        descriptor = np.concatenate((np.mean(window, axis=0), window[-1] - window[0]))
        self._descriptor = descriptor
        distances = np.linalg.norm(self._prototypes - descriptor, axis=1) / math.sqrt(
            descriptor.size
        )
        order = np.argsort(distances, kind="stable")
        neighbors = order[: self.artifact.neighbor_count]
        nearest = float(distances[neighbors[0]])
        negative_distances = distances[self._prototype_routes < 0]
        nearest_negative = (
            float(np.min(negative_distances)) if len(negative_distances) else None
        )
        covered = distances[neighbors] <= self.artifact.maximum_neighbor_distance
        if not bool(np.all(covered)):
            return self._fallback(
                "insufficient_covered_neighbors",
                nearest_distance=nearest,
                neighbor_count=int(np.count_nonzero(covered)),
            )
        routes = self._prototype_routes[neighbors]
        if int(routes[0]) < 0:
            return self._fallback(
                "nearest_exemplar_abstains",
                nearest_distance=nearest,
                neighbor_count=len(neighbors),
                consensus=float(np.count_nonzero(routes < 0) / len(neighbors)),
                nearest_negative_distance=nearest_negative,
            )
        route = int(routes[0])
        agreeing = routes == route
        consensus = float(np.count_nonzero(agreeing) / len(neighbors))
        if consensus + 1e-12 < self.artifact.minimum_primitive_consensus:
            return self._fallback(
                "primitive_consensus_below_gate",
                nearest_distance=nearest,
                neighbor_count=len(neighbors),
                consensus=consensus,
                nearest_negative_distance=nearest_negative,
            )
        veto_ratio = self.artifact.negative_veto_distance_ratio
        if (
            veto_ratio is not None
            and nearest_negative is not None
            and nearest_negative <= self.artifact.maximum_neighbor_distance
            and nearest_negative <= nearest * veto_ratio
        ):
            return self._fallback(
                "negative_evidence_veto",
                nearest_distance=nearest,
                neighbor_count=len(neighbors),
                consensus=consensus,
                nearest_negative_distance=nearest_negative,
            )
        agreeing_neighbors = neighbors[agreeing]
        advantage_lower = float(np.min(self._advantages[agreeing_neighbors]))
        component_lower = float(np.min(self._component_minimums[agreeing_neighbors]))
        if advantage_lower + 1e-12 < self.artifact.minimum_advantage_lower_bound:
            return self._fallback(
                "advantage_lower_bound_below_gate",
                nearest_distance=nearest,
                neighbor_count=len(neighbors),
                consensus=consensus,
                advantage_lower=advantage_lower,
                component_lower=component_lower,
                nearest_negative_distance=nearest_negative,
            )
        if component_lower + 1e-12 < self.artifact.minimum_component_lower_bound:
            return self._fallback(
                "component_lower_bound_below_gate",
                nearest_distance=nearest,
                neighbor_count=len(neighbors),
                consensus=consensus,
                advantage_lower=advantage_lower,
                component_lower=component_lower,
                nearest_negative_distance=nearest_negative,
            )
        self._selection = G1RecoveryStateSelection(
            ready=True,
            primitive_index=route,
            nearest_distance=nearest,
            neighbor_count=len(neighbors),
            primitive_consensus=consensus,
            advantage_lower_bound=advantage_lower,
            component_lower_bound=component_lower,
            out_of_distribution=False,
            fallback_reason=None,
            nearest_negative_distance=nearest_negative,
        )
        self._selection_count += 1
        return self._selection

    def _fallback(
        self,
        reason: str,
        *,
        nearest_distance: float | None = None,
        neighbor_count: int = 0,
        consensus: float = 0.0,
        advantage_lower: float | None = None,
        component_lower: float | None = None,
        nearest_negative_distance: float | None = None,
    ) -> G1RecoveryStateSelection:
        self._selection = G1RecoveryStateSelection(
            ready=True,
            primitive_index=None,
            nearest_distance=nearest_distance,
            neighbor_count=neighbor_count,
            primitive_consensus=consensus,
            advantage_lower_bound=advantage_lower,
            component_lower_bound=component_lower,
            out_of_distribution=True,
            fallback_reason=reason,
            nearest_negative_distance=nearest_negative_distance,
        )
        self._fallback_count += 1
        return self._selection

    def build_receipt(self) -> G1RecoveryStateReceipt:
        selection = self._selection
        descriptor_hash = None
        if self._descriptor is not None:
            descriptor_hash = (
                "sha256:"
                + hashlib.sha256(np.ascontiguousarray(self._descriptor).tobytes()).hexdigest()
            )
        return G1RecoveryStateReceipt(
            artifact_hash=self.artifact.artifact_hash,
            selected_primitive_index=(selection.primitive_index if selection else None),
            nearest_distance=(selection.nearest_distance if selection else None),
            neighbor_count=(selection.neighbor_count if selection else 0),
            primitive_consensus=(selection.primitive_consensus if selection else 0.0),
            advantage_lower_bound=(selection.advantage_lower_bound if selection else None),
            component_lower_bound=(selection.component_lower_bound if selection else None),
            fallback_reason=(selection.fallback_reason if selection else None),
            pending_count=self._pending_count,
            selection_count=self._selection_count,
            fallback_count=self._fallback_count,
            descriptor_hash=descriptor_hash,
            nearest_negative_distance=(
                selection.nearest_negative_distance if selection else None
            ),
        )


def load_g1_recovery_state_artifact(path: Path) -> G1RecoveryStateArtifact:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("recovery-state artifact is missing or exceeds 2 MiB")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("recovery-state artifact is not readable JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("recovery-state artifact root must be an object")
    return G1RecoveryStateArtifact.from_dict(value)


def _strict_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("recovery-state integer field is invalid")
    return value


def _strict_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("recovery-state numeric field is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("recovery-state numeric field must be finite")
    return result


def _float_tuple(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError("recovery-state numeric vector must be an array")
    return tuple(_strict_float(item) for item in value)


def _int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError("recovery-state integer vector must be an array")
    return tuple(_strict_int(item) for item in value)


__all__ = [
    "G1_RECOVERY_STATE_FEATURES",
    "G1_RECOVERY_STATE_OBSERVATIONS",
    "G1RecoveryStateArtifact",
    "G1RecoveryStatePolicy",
    "G1RecoveryStateReceipt",
    "G1RecoveryStateSelection",
    "load_g1_recovery_state_artifact",
]
