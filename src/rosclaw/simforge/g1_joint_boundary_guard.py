"""Velocity-aware, SIM-only joint-boundary projection for G1 policies.

This is a safety overlay, not another actuator writer.  It receives the torque
already proposed by the qualified parent controller and changes only commands
that would continue driving a sealed set of joints through their measured
limits.  The selected joints are part of the evidence-bound configuration:
using a small, audited set avoids the instability observed when an aggressive
whole-body guard fights the kick policy at every joint.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.g1_neural_torque import (
    G1TorqueControlFrame,
    G1TorquePolicyReceipt,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES


@dataclass(frozen=True)
class G1JointBoundaryGuardConfig:
    """Evidence-bound envelope for a preventive torque projection."""

    protected_joint_names: tuple[str, ...] = (
        "left_knee_joint",
        "left_ankle_roll_joint",
        "waist_pitch_joint",
    )
    margin_rad: float = 0.025
    prediction_horizon_sec: float = 0.05
    boundary_kp: float = 60.0
    boundary_kd: float = 5.0
    minimum_policy_phase: float = 0.0
    maximum_correction_nm: float = 40.0
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.simforge.g1_joint_boundary_guard_config.v1"

    def __post_init__(self) -> None:
        names = tuple(self.protected_joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("joint-boundary guard joints must be non-empty and unique")
        if any(name not in G1_DDS_JOINT_NAMES for name in names):
            raise ValueError("joint-boundary guard references an unknown G1 joint")
        values = (
            self.margin_rad,
            self.prediction_horizon_sec,
            self.boundary_kp,
            self.boundary_kd,
            self.minimum_policy_phase,
            self.maximum_correction_nm,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("joint-boundary guard config must be finite")
        if not 0.005 <= self.margin_rad <= 0.10:
            raise ValueError("joint-boundary guard margin must be in [0.005, 0.10] rad")
        if not 0.01 <= self.prediction_horizon_sec <= 0.20:
            raise ValueError("joint-boundary guard horizon must be in [0.01, 0.20] sec")
        if not 20.0 <= self.boundary_kp <= 200.0:
            raise ValueError("joint-boundary guard kp must be in [20, 200]")
        if not 1.0 <= self.boundary_kd <= 20.0:
            raise ValueError("joint-boundary guard kd must be in [1, 20]")
        if not 0.0 <= self.minimum_policy_phase <= 0.95:
            raise ValueError("joint-boundary guard phase must be in [0, 0.95]")
        if not 1.0 <= self.maximum_correction_nm <= 80.0:
            raise ValueError("joint-boundary guard correction must be in [1, 80] Nm")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("joint-boundary guard is restricted to SIM_ONLY")
        object.__setattr__(self, "protected_joint_names", names)

    @property
    def config_hash(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class G1JointBoundaryGuardReceipt(G1TorquePolicyReceipt):
    config_hash: str = ""
    projection_count: int = 0
    maximum_correction_nm: float = 0.0
    maximum_predicted_boundary_excess_rad: float = 0.0
    protected_joint_names: tuple[str, ...] = ()
    direct_torque_actor: bool = False
    safety_projection_only: bool = True
    schema_version: str = "rosclaw.simforge.g1_joint_boundary_guard_receipt.v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class G1JointBoundaryGuardPolicy:
    """Project only outward parent torque near audited joint boundaries."""

    safety_projection_only = True
    activation_ceiling = "SIM_ONLY"

    def __init__(
        self,
        *,
        body_hash: str,
        parent_policy_hash: str,
        config: G1JointBoundaryGuardConfig | None = None,
    ) -> None:
        if not body_hash.startswith("sha256:") or not parent_policy_hash.startswith("sha256:"):
            raise ValueError("joint-boundary guard requires body and parent hashes")
        self.body_hash = body_hash
        self.parent_policy_hash = parent_policy_hash
        self.config = config or G1JointBoundaryGuardConfig()
        self.artifact_hash = canonical_hash(
            {
                "body_hash": body_hash,
                "parent_policy_hash": parent_policy_hash,
                "config_hash": self.config.config_hash,
            }
        )
        indices = {name: index for index, name in enumerate(G1_DDS_JOINT_NAMES)}
        self._protected = np.asarray(
            [indices[name] for name in self.config.protected_joint_names],
            dtype=np.int64,
        )
        self.reset()

    def reset(self) -> None:
        self._pending = False
        self._inference_count = 0
        self._projection_count = 0
        self._projected_joint_count = 0
        self._maximum_correction_nm = 0.0
        self._maximum_predicted_excess_rad = 0.0

    def command(self, frame: G1TorqueControlFrame, parent_torque: np.ndarray) -> np.ndarray:
        if self._pending:
            raise RuntimeError("joint-boundary guard did not receive note_applied")
        parent = np.asarray(parent_torque, dtype=np.float64)
        arrays = (
            parent,
            np.asarray(frame.joint_position, dtype=np.float64),
            np.asarray(frame.joint_velocity, dtype=np.float64),
            np.asarray(frame.joint_lower_limits, dtype=np.float64),
            np.asarray(frame.joint_upper_limits, dtype=np.float64),
        )
        if any(value.shape != (29,) or not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("joint-boundary guard requires finite 29-DoF inputs")
        projected = parent.copy()
        active: np.ndarray = np.zeros(29, dtype=np.bool_)
        excess: np.ndarray = np.zeros(29, dtype=np.float64)
        if frame.policy_phase >= self.config.minimum_policy_phase:
            projected, active, excess = project_g1_joint_boundary_torque(
                joint_position=arrays[1],
                joint_velocity=arrays[2],
                commanded_torque=parent,
                joint_lower_limits=arrays[3],
                joint_upper_limits=arrays[4],
                protected_joint_indices=self._protected,
                config=self.config,
            )
        correction = np.clip(
            projected - parent,
            -self.config.maximum_correction_nm,
            self.config.maximum_correction_nm,
        )
        projected = parent + correction
        changed = np.abs(correction) > 1e-12
        self._inference_count += 1
        self._projection_count += int(np.any(changed))
        self._projected_joint_count += int(np.count_nonzero(changed))
        self._maximum_correction_nm = max(
            self._maximum_correction_nm,
            float(np.max(np.abs(correction))),
        )
        if np.any(active):
            self._maximum_predicted_excess_rad = max(
                self._maximum_predicted_excess_rad,
                float(np.max(excess[active])),
            )
        self._pending = True
        return projected

    def note_applied(self, torque: np.ndarray) -> None:
        if not self._pending:
            raise RuntimeError("joint-boundary guard has no pending command")
        value = np.asarray(torque, dtype=np.float64)
        if value.shape != (29,) or not np.all(np.isfinite(value)):
            raise ValueError("joint-boundary guard applied torque must be finite 29-DoF")
        self._pending = False

    def build_receipt(self) -> G1JointBoundaryGuardReceipt:
        if self._pending or self._inference_count <= 0:
            raise ValueError("cannot build an incomplete joint-boundary guard receipt")
        return G1JointBoundaryGuardReceipt(
            artifact_hash=self.artifact_hash,
            config_hash=self.config.config_hash,
            body_hash=self.body_hash,
            parent_policy_hash=self.parent_policy_hash,
            inference_count=self._inference_count,
            learned_output_count=0,
            fallback_count=0,
            nonfinite_fallback_count=0,
            projection_fallback_count=0,
            out_of_distribution_fallback_count=0,
            warmup_fallback_count=0,
            state_guard_fallback_count=0,
            projection_count=self._projection_count,
            projected_joint_count=self._projected_joint_count,
            maximum_limit_ratio=0.0,
            peak_mechanical_power_w=0.0,
            direct_torque_output=False,
            activation_ceiling="SIM_ONLY",
            maximum_correction_nm=self._maximum_correction_nm,
            maximum_predicted_boundary_excess_rad=(
                self._maximum_predicted_excess_rad
            ),
            protected_joint_names=self.config.protected_joint_names,
        )


def project_g1_joint_boundary_torque(
    *,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    commanded_torque: np.ndarray,
    joint_lower_limits: np.ndarray,
    joint_upper_limits: np.ndarray,
    protected_joint_indices: np.ndarray,
    config: G1JointBoundaryGuardConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return projected torque, active mask, and predicted boundary excess."""

    arrays = (
        np.asarray(joint_position, dtype=np.float64),
        np.asarray(joint_velocity, dtype=np.float64),
        np.asarray(commanded_torque, dtype=np.float64),
        np.asarray(joint_lower_limits, dtype=np.float64),
        np.asarray(joint_upper_limits, dtype=np.float64),
    )
    indices = np.asarray(protected_joint_indices)
    if any(value.shape != (29,) or not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("joint-boundary projection requires finite 29-DoF vectors")
    if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in {"i", "u"}:
        raise ValueError("joint-boundary projection requires integer protected indices")
    if np.any(indices < 0) or np.any(indices >= 29) or len(np.unique(indices)) != len(indices):
        raise ValueError("joint-boundary protected indices are invalid")
    position, velocity, torque, lower_limit, upper_limit = arrays
    lower = lower_limit + config.margin_rad
    upper = upper_limit - config.margin_rad
    if np.any(lower[indices] >= upper[indices]):
        raise ValueError("joint-boundary margin collapses a protected joint range")
    predicted = position + config.prediction_horizon_sec * velocity
    protected: np.ndarray = np.zeros(29, dtype=np.bool_)
    protected[indices] = True
    lower_threat = protected & (predicted < lower)
    upper_threat = protected & (predicted > upper)
    lower_brake = config.boundary_kp * (lower - position) - config.boundary_kd * velocity
    upper_brake = config.boundary_kp * (upper - position) - config.boundary_kd * velocity
    projected = torque.copy()
    projected[lower_threat] = np.maximum(projected[lower_threat], lower_brake[lower_threat])
    projected[upper_threat] = np.minimum(projected[upper_threat], upper_brake[upper_threat])
    active = lower_threat | upper_threat
    excess = np.maximum(lower - predicted, predicted - upper)
    excess[~active] = 0.0
    return projected, active, excess


__all__ = [
    "G1JointBoundaryGuardConfig",
    "G1JointBoundaryGuardPolicy",
    "G1JointBoundaryGuardReceipt",
    "project_g1_joint_boundary_torque",
]
