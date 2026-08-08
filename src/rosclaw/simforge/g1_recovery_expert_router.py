"""Fail-closed proprioceptive routing for learned G1 recovery experts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from rosclaw.simforge.g1_contact_recovery_torque_policy import (
    G1RecoveryContextSnapshot,
)
from rosclaw.simforge.g1_neural_torque import (
    G1NeuralTorquePolicy,
    G1TorqueControlFrame,
    G1TorquePolicyReceipt,
    G1TorqueProjection,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json

G1_RECOVERY_CONTEXT_FEATURES = (
    "policy_phase",
    "pelvis_height_m",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    "base_linear_velocity_x_mps",
    "base_linear_velocity_y_mps",
    "base_linear_velocity_z_mps",
    "base_angular_velocity_x_rps",
    "base_angular_velocity_y_rps",
    "base_angular_velocity_z_rps",
    "ball_speed_mps",
    "ball_direction_x",
    "ball_direction_y",
    "ball_direction_z",
    "left_contact",
    "right_contact",
)


@dataclass(frozen=True)
class G1RecoveryExpertPrototype:
    context_hash: str
    normalized_context: tuple[float, ...]
    expert_id: str | None
    measured_naturalness_gain: float
    task_preserved: bool
    schema_version: str = "rosclaw.simforge.g1_recovery_expert_prototype.v1"

    def __post_init__(self) -> None:
        if len(self.normalized_context) != len(G1_RECOVERY_CONTEXT_FEATURES):
            raise ValueError("recovery expert prototype feature count is invalid")
        if not all(math.isfinite(value) for value in self.normalized_context):
            raise ValueError("recovery expert prototype must be finite")
        if not math.isfinite(self.measured_naturalness_gain):
            raise ValueError("recovery expert prototype gain must be finite")
        if self.expert_id is not None and not self.expert_id:
            raise ValueError("recovery expert id must be non-empty")


@dataclass(frozen=True)
class G1RecoveryExpertRoute:
    context_hash: str
    expert_id: str | None
    normalized_distance: float
    expected_naturalness_gain: float
    eligible: bool
    fallback_reason: str | None
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.simforge.g1_recovery_expert_route.v1"


@dataclass(frozen=True)
class G1RecoveryExpertRouterArtifact:
    body_hash: str
    parent_policy_hash: str
    source_evidence_hash: str
    normalization_center: tuple[float, ...]
    normalization_scale: tuple[float, ...]
    prototypes: tuple[G1RecoveryExpertPrototype, ...]
    expert_artifact_hashes: Mapping[str, str]
    maximum_normalized_distance: float = 0.50
    minimum_expected_gain: float = 0.05
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.simforge.g1_recovery_expert_router.v1"

    def __post_init__(self) -> None:
        count = len(G1_RECOVERY_CONTEXT_FEATURES)
        if len(self.normalization_center) != count or len(self.normalization_scale) != count:
            raise ValueError("recovery expert router normalization shape is invalid")
        numeric = (*self.normalization_center, *self.normalization_scale)
        if not all(math.isfinite(value) for value in numeric) or any(
            value <= 0.0 for value in self.normalization_scale
        ):
            raise ValueError("recovery expert router normalization is invalid")
        if not self.prototypes:
            raise ValueError("recovery expert router requires prototypes")
        hashes = dict(self.expert_artifact_hashes)
        if not hashes or any(not key or not value.startswith("sha256:") for key, value in hashes.items()):
            raise ValueError("recovery expert router artifact map is invalid")
        unknown = {
            item.expert_id
            for item in self.prototypes
            if item.expert_id is not None and item.expert_id not in hashes
        }
        if unknown:
            raise ValueError("recovery expert prototype references an unknown expert")
        if not 0.0 < self.maximum_normalized_distance <= 20.0:
            raise ValueError("recovery expert router distance must be in (0, 20]")
        if not 0.0 < self.minimum_expected_gain <= 1.0:
            raise ValueError("recovery expert router gain gate must be in (0, 1]")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("recovery expert router is restricted to SIM_ONLY")
        object.__setattr__(self, "expert_artifact_hashes", MappingProxyType(hashes))

    @property
    def artifact_hash(self) -> str:
        return hash_json(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "parent_policy_hash": self.parent_policy_hash,
            "source_evidence_hash": self.source_evidence_hash,
            "feature_names": list(G1_RECOVERY_CONTEXT_FEATURES),
            "normalization_center": list(self.normalization_center),
            "normalization_scale": list(self.normalization_scale),
            "prototypes": [asdict(item) for item in self.prototypes],
            "expert_artifact_hashes": dict(self.expert_artifact_hashes),
            "maximum_normalized_distance": self.maximum_normalized_distance,
            "minimum_expected_gain": self.minimum_expected_gain,
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": False,
        }
        if include_hash:
            value["artifact_hash"] = self.artifact_hash
        return value

    def route(self, context: G1RecoveryContextSnapshot) -> G1RecoveryExpertRoute:
        normalized = (
            _context_vector(context) - np.asarray(self.normalization_center, dtype=np.float64)
        ) / np.asarray(self.normalization_scale, dtype=np.float64)
        distances = np.asarray(
            [
                np.linalg.norm(normalized - np.asarray(item.normalized_context, dtype=np.float64))
                / math.sqrt(len(normalized))
                for item in self.prototypes
            ],
            dtype=np.float64,
        )
        index = int(np.argmin(distances))
        prototype = self.prototypes[index]
        distance = float(distances[index])
        reason: str | None = None
        if distance > self.maximum_normalized_distance:
            reason = "outside_sealed_context_envelope"
        elif prototype.expert_id is None:
            reason = "prototype_requires_parent_fallback"
        elif not prototype.task_preserved:
            reason = "prototype_task_regression"
        elif prototype.measured_naturalness_gain < self.minimum_expected_gain:
            reason = "prototype_gain_below_gate"
        return G1RecoveryExpertRoute(
            context_hash=context.context_hash,
            expert_id=prototype.expert_id if reason is None else None,
            normalized_distance=distance,
            expected_naturalness_gain=prototype.measured_naturalness_gain,
            eligible=reason is None,
            fallback_reason=reason,
        )


@dataclass(frozen=True)
class G1RoutedRecoveryTorqueReceipt(G1TorquePolicyReceipt):
    router_artifact_hash: str = ""
    expert_artifact_hashes: tuple[str, ...] = ()
    selected_expert_id: str | None = None
    selected_expert_artifact_hash: str | None = None
    route: G1RecoveryExpertRoute | None = None
    routing_context: G1RecoveryContextSnapshot | None = None
    routing_context_hash: str | None = None
    routing_delay_steps: int = 0
    route_fallback_count: int = 0
    schema_version: str = "rosclaw.simforge.g1_routed_recovery_torque_receipt.v1"


@dataclass(frozen=True)
class _RoutedArtifactIdentity:
    artifact_hash: str
    body_hash: str
    parent_policy_hash: str


class G1RoutedRecoveryTorquePolicy:
    """Run all experts in shadow and activate only the sealed routed expert."""

    def __init__(
        self,
        router: G1RecoveryExpertRouterArtifact,
        experts: Mapping[str, G1NeuralTorquePolicy],
        *,
        minimum_ball_speed_mps: float = 0.50,
        routing_delay_steps: int = 50,
    ) -> None:
        values = dict(experts)
        if set(values) != set(router.expert_artifact_hashes):
            raise ValueError("recovery expert policy set does not match router")
        for expert_id, policy in values.items():
            expected = router.expert_artifact_hashes[expert_id]
            if policy.artifact.artifact_hash != expected:
                raise ValueError("recovery expert artifact hash does not match router")
            if policy.artifact.body_hash != router.body_hash:
                raise ValueError("recovery expert body does not match router")
            if policy.artifact.parent_policy_hash != router.parent_policy_hash:
                raise ValueError("recovery expert parent does not match router")
        if not math.isfinite(minimum_ball_speed_mps) or minimum_ball_speed_mps <= 0.0:
            raise ValueError("recovery expert contact threshold must be positive")
        if not 1 <= routing_delay_steps <= 250:
            raise ValueError("recovery expert routing delay must be in [1, 250] steps")
        self.router = router
        self.experts = MappingProxyType(values)
        self.minimum_ball_speed_mps = minimum_ball_speed_mps
        self.routing_delay_steps = routing_delay_steps
        self.artifact = _RoutedArtifactIdentity(
            artifact_hash=hash_json(
                {
                    "router_artifact_hash": router.artifact_hash,
                    "expert_artifact_hashes": dict(router.expert_artifact_hashes),
                    "routing_delay_steps": routing_delay_steps,
                }
            ),
            body_hash=router.body_hash,
            parent_policy_hash=router.parent_policy_hash,
        )
        self.reset()

    def reset(self) -> None:
        for policy in self.experts.values():
            policy.reset()
        self._pending = False
        self._route: G1RecoveryExpertRoute | None = None
        self._routing_context: G1RecoveryContextSnapshot | None = None
        self._selected_expert_id: str | None = None
        self._pending_projection: G1TorqueProjection | None = None
        self._inference_count = 0
        self._route_fallback_count = 0
        self._contact_steps = 0

    @property
    def pending_projection(self) -> G1TorqueProjection:
        if self._pending_projection is None:
            raise RuntimeError("routed recovery policy has no pending projection")
        return self._pending_projection

    def command(
        self,
        frame: G1TorqueControlFrame,
        parent_torque: np.ndarray,
        *,
        allow_exploration: bool = True,
    ) -> np.ndarray:
        del allow_exploration
        if self._pending:
            raise RuntimeError("routed recovery policy did not receive note_applied")
        proposals = {
            expert_id: policy.command(frame, parent_torque, allow_exploration=False)
            for expert_id, policy in self.experts.items()
        }
        ball_speed = float(np.linalg.norm(np.asarray(frame.ball_velocity, dtype=np.float64)))
        contact_ready = bool(
            frame.ball_contact_observed
            and math.isfinite(ball_speed)
            and ball_speed >= self.minimum_ball_speed_mps
        )
        if contact_ready:
            self._contact_steps += 1
        if self._route is None and self._contact_steps >= self.routing_delay_steps:
            try:
                context = G1RecoveryContextSnapshot.from_frame(
                    frame,
                    ball_speed_mps=ball_speed,
                )
                self._routing_context = context
                self._route = self.router.route(context)
                self._selected_expert_id = self._route.expert_id
            except ValueError:
                self._route = G1RecoveryExpertRoute(
                    context_hash="",
                    expert_id=None,
                    normalized_distance=math.inf,
                    expected_naturalness_gain=0.0,
                    eligible=False,
                    fallback_reason="invalid_contact_context",
                )
        selected = self._selected_expert_id
        if selected is None:
            parent = np.asarray(parent_torque, dtype=np.float64).copy()
            self._pending_projection = G1TorqueProjection(
                torque=parent,
                used_parent=True,
                reason="expert_router_parent",
                projected_joint_count=0,
                maximum_limit_ratio=0.0,
                mechanical_power_w=0.0,
            )
            command = parent
            self._route_fallback_count += 1
        else:
            self._pending_projection = self.experts[selected].pending_projection
            command = proposals[selected]
        self._pending = True
        self._inference_count += 1
        return np.asarray(command, dtype=np.float64).copy()

    def note_applied(self, torque: np.ndarray) -> None:
        if not self._pending:
            raise RuntimeError("routed recovery policy has no pending command")
        for policy in self.experts.values():
            policy.note_applied(torque)
        self._pending = False
        self._pending_projection = None

    def build_receipt(self) -> G1RoutedRecoveryTorqueReceipt:
        if self._pending or not self._inference_count:
            raise ValueError("cannot build an incomplete routed recovery receipt")
        receipts = {name: policy.build_receipt() for name, policy in self.experts.items()}
        selected = receipts.get(self._selected_expert_id or "")
        fallback_count = (
            self._inference_count
            if selected is None
            else min(self._inference_count, selected.fallback_count + self._route_fallback_count)
        )
        return G1RoutedRecoveryTorqueReceipt(
            artifact_hash=self.artifact.artifact_hash,
            body_hash=self.artifact.body_hash,
            parent_policy_hash=self.artifact.parent_policy_hash,
            inference_count=self._inference_count,
            learned_output_count=self._inference_count - fallback_count,
            fallback_count=fallback_count,
            nonfinite_fallback_count=selected.nonfinite_fallback_count if selected else 0,
            projection_fallback_count=selected.projection_fallback_count if selected else 0,
            out_of_distribution_fallback_count=(
                selected.out_of_distribution_fallback_count if selected else 0
            ),
            warmup_fallback_count=selected.warmup_fallback_count if selected else 0,
            state_guard_fallback_count=selected.state_guard_fallback_count if selected else 0,
            projected_joint_count=selected.projected_joint_count if selected else 0,
            maximum_limit_ratio=selected.maximum_limit_ratio if selected else 0.0,
            peak_mechanical_power_w=selected.peak_mechanical_power_w if selected else 0.0,
            direct_torque_output=True,
            activation_ceiling="SIM_ONLY",
            router_artifact_hash=self.router.artifact_hash,
            expert_artifact_hashes=tuple(sorted(self.router.expert_artifact_hashes.values())),
            selected_expert_id=self._selected_expert_id,
            selected_expert_artifact_hash=(
                self.router.expert_artifact_hashes[self._selected_expert_id]
                if self._selected_expert_id is not None
                else None
            ),
            route=self._route,
            routing_context=self._routing_context,
            routing_context_hash=(
                self._routing_context.context_hash
                if self._routing_context is not None
                else None
            ),
            routing_delay_steps=self.routing_delay_steps,
            route_fallback_count=self._route_fallback_count,
        )


def build_g1_recovery_expert_router(
    *,
    body_hash: str,
    parent_policy_hash: str,
    source_evidence_hash: str,
    contexts: tuple[G1RecoveryContextSnapshot, ...],
    selected_expert_ids: tuple[str | None, ...],
    measured_gains: tuple[float, ...],
    task_preserved: tuple[bool, ...],
    expert_artifact_hashes: Mapping[str, str],
    maximum_normalized_distance: float = 0.50,
    minimum_expected_gain: float = 0.05,
) -> G1RecoveryExpertRouterArtifact:
    if not (
        contexts
        and len(contexts)
        == len(selected_expert_ids)
        == len(measured_gains)
        == len(task_preserved)
    ):
        raise ValueError("recovery expert router training rows are misaligned")
    matrix = np.stack([_context_vector(item) for item in contexts])
    center = np.median(matrix, axis=0)
    scale = np.maximum(np.std(matrix, axis=0), _feature_scale_floor())
    normalized = (matrix - center) / scale
    prototypes = tuple(
        G1RecoveryExpertPrototype(
            context_hash=context.context_hash,
            normalized_context=tuple(map(float, row)),
            expert_id=expert_id,
            measured_naturalness_gain=float(gain),
            task_preserved=bool(preserved),
        )
        for context, row, expert_id, gain, preserved in zip(
            contexts,
            normalized,
            selected_expert_ids,
            measured_gains,
            task_preserved,
            strict=True,
        )
    )
    return G1RecoveryExpertRouterArtifact(
        body_hash=body_hash,
        parent_policy_hash=parent_policy_hash,
        source_evidence_hash=source_evidence_hash,
        normalization_center=tuple(map(float, center)),
        normalization_scale=tuple(map(float, scale)),
        prototypes=prototypes,
        expert_artifact_hashes=expert_artifact_hashes,
        maximum_normalized_distance=maximum_normalized_distance,
        minimum_expected_gain=minimum_expected_gain,
    )


def write_g1_recovery_expert_router(
    path: Path,
    artifact: G1RecoveryExpertRouterArtifact,
) -> None:
    path.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_g1_recovery_expert_router(path: Path) -> G1RecoveryExpertRouterArtifact:
    source = path.expanduser().resolve()
    if not source.is_file() or source.stat().st_size > 1024 * 1024:
        raise ValueError("recovery expert router artifact is missing or oversized")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("recovery expert router artifact must be an object")
    expected = {
        "schema_version",
        "artifact_hash",
        "body_hash",
        "parent_policy_hash",
        "source_evidence_hash",
        "feature_names",
        "normalization_center",
        "normalization_scale",
        "prototypes",
        "expert_artifact_hashes",
        "maximum_normalized_distance",
        "minimum_expected_gain",
        "activation_ceiling",
        "hardware_authorized",
    }
    if set(value) != expected:
        raise ValueError("recovery expert router artifact fields are invalid")
    if tuple(value["feature_names"]) != G1_RECOVERY_CONTEXT_FEATURES:
        raise ValueError("recovery expert router feature contract is invalid")
    if value["hardware_authorized"] is not False:
        raise ValueError("recovery expert router cannot authorize hardware")
    prototypes = value["prototypes"]
    expert_hashes = value["expert_artifact_hashes"]
    if not isinstance(prototypes, list) or not isinstance(expert_hashes, dict):
        raise ValueError("recovery expert router collections are invalid")
    artifact = G1RecoveryExpertRouterArtifact(
        schema_version=str(value["schema_version"]),
        body_hash=str(value["body_hash"]),
        parent_policy_hash=str(value["parent_policy_hash"]),
        source_evidence_hash=str(value["source_evidence_hash"]),
        normalization_center=tuple(map(float, value["normalization_center"])),
        normalization_scale=tuple(map(float, value["normalization_scale"])),
        prototypes=tuple(G1RecoveryExpertPrototype(**item) for item in prototypes),
        expert_artifact_hashes={str(key): str(item) for key, item in expert_hashes.items()},
        maximum_normalized_distance=float(value["maximum_normalized_distance"]),
        minimum_expected_gain=float(value["minimum_expected_gain"]),
        activation_ceiling=str(value["activation_ceiling"]),
    )
    if value["artifact_hash"] != artifact.artifact_hash:
        raise ValueError("recovery expert router artifact hash mismatch")
    return artifact


def _context_vector(context: G1RecoveryContextSnapshot) -> np.ndarray:
    values = asdict(context)
    return np.asarray([float(values[name]) for name in G1_RECOVERY_CONTEXT_FEATURES], dtype=np.float64)


def _feature_scale_floor() -> np.ndarray:
    return np.asarray(
        (0.01, 0.01, 0.02, 0.02, 0.01, 0.03, 0.03, 0.03, 0.10, 0.10, 0.10,
         0.20, 0.02, 0.02, 0.02, 0.50, 0.50),
        dtype=np.float64,
    )


__all__ = [
    "G1_RECOVERY_CONTEXT_FEATURES",
    "G1RecoveryExpertPrototype",
    "G1RecoveryExpertRoute",
    "G1RecoveryExpertRouterArtifact",
    "G1RoutedRecoveryTorquePolicy",
    "G1RoutedRecoveryTorqueReceipt",
    "build_g1_recovery_expert_router",
    "load_g1_recovery_expert_router",
    "write_g1_recovery_expert_router",
]
