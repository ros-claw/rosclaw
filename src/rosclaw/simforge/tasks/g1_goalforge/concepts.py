"""Immutable public contracts for the G1 GoalForge task."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

GOALFORGE_TASK_ID = "g1_penalty_kick"

# Unitree HG LowCmd/LowState order for the 29-DoF G1.  This is deliberately a
# hard contract: a policy with a different order is not eligible to execute.
G1_DDS_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_HARD_TORQUE_LIMITS = (
    88.0,
    139.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    139.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    50.0,
    50.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
)


class GoalForgeStatus(StrEnum):
    SUCCESS = "SUCCESS"
    TARGET_MISS_LEFT = "TARGET_MISS_LEFT"
    TARGET_MISS_RIGHT = "TARGET_MISS_RIGHT"
    TARGET_MISS_HIGH = "TARGET_MISS_HIGH"
    SHOT_TOO_WEAK = "SHOT_TOO_WEAK"
    SHOT_TOO_STRONG = "SHOT_TOO_STRONG"
    BALL_NOT_CONTACTED = "BALL_NOT_CONTACTED"
    EARLY_BALL_CONTACT = "EARLY_BALL_CONTACT"
    LATE_BALL_CONTACT = "LATE_BALL_CONTACT"
    WRONG_FOOT_CONTACT = "WRONG_FOOT_CONTACT"
    SUPPORT_FOOT_SLIP = "SUPPORT_FOOT_SLIP"
    COM_OUTSIDE_SUPPORT = "COM_OUTSIDE_SUPPORT"
    POST_KICK_FALL = "POST_KICK_FALL"
    RECOVERY_STEP_FAILED = "RECOVERY_STEP_FAILED"
    TORSO_OVERSHOOT = "TORSO_OVERSHOOT"
    JOINT_LIMIT_EXCEEDED = "JOINT_LIMIT_EXCEEDED"
    TORQUE_LIMIT_EXCEEDED = "TORQUE_LIMIT_EXCEEDED"
    ACTUATOR_SATURATION = "ACTUATOR_SATURATION"
    BODY_CALIBRATION_MISMATCH = "BODY_CALIBRATION_MISMATCH"
    POLICY_BODY_INCOMPATIBLE = "POLICY_BODY_INCOMPATIBLE"
    BALL_OBSERVATION_STALE = "BALL_OBSERVATION_STALE"
    AGENT_LOST = "AGENT_LOST"
    POLICY_WORKER_CRASH = "POLICY_WORKER_CRASH"
    DDS_LOST = "DDS_LOST"
    STATE_FEEDBACK_STALE = "STATE_FEEDBACK_STALE"
    EXECUTOR_UNAVAILABLE = "EXECUTOR_UNAVAILABLE"
    BALL_OUT_OF_REACH = "BALL_OUT_OF_REACH"
    ROBOT_NOT_STABLE = "ROBOT_NOT_STABLE"
    TASK_PHYSICALLY_IMPOSSIBLE = "TASK_PHYSICALLY_IMPOSSIBLE"
    NON_FINITE_STATE = "NON_FINITE_STATE"


@dataclass(frozen=True)
class ShotParameters:
    """Bounded, interpretable adapter around a fixed whole-body kick prior."""

    stance_offset_x: float = 0.0
    stance_offset_y: float = 0.0
    pelvis_yaw_offset: float = 0.0
    kick_foot: str = "right"
    com_shift_y: float = 0.0
    swing_amplitude: float = 1.0
    swing_speed_scale: float = 1.0
    foot_yaw_offset: float = 0.0
    contact_phase_offset: float = 0.0
    kick_trigger_delay: float = 0.0
    recovery_step_length: float = 0.04
    recovery_step_yaw: float = 0.0
    policy_type: str = "fixed_prior"
    dataset_snapshot_hash: str | None = None

    def __post_init__(self) -> None:
        bounds = {
            "stance_offset_x": (self.stance_offset_x, -0.12, 0.12),
            "stance_offset_y": (self.stance_offset_y, -0.12, 0.12),
            "pelvis_yaw_offset": (self.pelvis_yaw_offset, -0.20, 0.20),
            "com_shift_y": (self.com_shift_y, -0.08, 0.08),
            "swing_amplitude": (self.swing_amplitude, 0.75, 1.15),
            "swing_speed_scale": (self.swing_speed_scale, 0.80, 1.15),
            "foot_yaw_offset": (self.foot_yaw_offset, -0.12, 0.12),
            "contact_phase_offset": (self.contact_phase_offset, -0.10, 0.10),
            "kick_trigger_delay": (self.kick_trigger_delay, 0.0, 0.20),
            "recovery_step_length": (self.recovery_step_length, 0.0, 0.15),
            "recovery_step_yaw": (self.recovery_step_yaw, -0.15, 0.15),
        }
        for name, (value, minimum, maximum) in bounds.items():
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        if self.kick_foot not in {"left", "right"}:
            raise ValueError("kick_foot must be left or right")
        if self.policy_type not in {
            "fixed_prior",
            "parameter",
            "trajectory",
            "skill_graph",
            "learned_adapter",
        }:
            raise ValueError("unsupported GoalForge policy_type")
        if self.policy_type == "learned_adapter" and (
            not self.dataset_snapshot_hash or not self.dataset_snapshot_hash.startswith("sha256:")
        ):
            raise ValueError("learned adapters require a dataset snapshot hash")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def policy_hash(self) -> str:
        return hash_json(self.to_dict())


@dataclass(frozen=True)
class GoalForgeResult:
    status: GoalForgeStatus
    success: bool
    physics_executed: bool
    contact_observed: bool
    kick_foot_contacted: bool
    goal_crossed: bool
    target_zone_hit: bool
    target_error_m: float
    ball_speed_mps: float
    ball_contact_time_sec: float | None
    contact_impulse_ns: float
    support_foot_slip_m: float
    com_margin_min_m: float
    torso_roll_peak_rad: float
    torso_pitch_peak_rad: float
    peak_torque_scale: float
    joint_limit_violation: bool
    torque_limit_violation: bool
    actuator_saturation: bool
    post_kick_fall: bool
    post_kick_stability_time_sec: float
    final_pelvis_height_m: float
    physics_steps: int
    finite_state: bool
    robustness: float

    def summary_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return {
            key: (item if not isinstance(item, float) or math.isfinite(item) else None)
            for key, item in value.items()
        }


@dataclass(frozen=True)
class SimulationReceiptV4:
    episode_id: str
    body_hash: str
    policy_hash: str
    kick_prior_hash: str
    scenario_commitment: str
    seed_commitment: str
    request_hash: str
    trajectory_hash: str
    result_hash: str
    backend: str
    backend_commit: str
    physics_steps: int
    independently_verified: bool
    strict_replay: bool
    evidence_domain: str = "SHADOW"
    schema_version: str = "rosclaw.g1_goalforge.simulation_receipt.v1"

    def __post_init__(self) -> None:
        digests = (
            self.body_hash,
            self.policy_hash,
            self.kick_prior_hash,
            self.scenario_commitment,
            self.seed_commitment,
            self.request_hash,
            self.trajectory_hash,
            self.result_hash,
        )
        if any(not item.startswith("sha256:") for item in digests):
            raise ValueError("GoalForge receipt hashes must be sha256 digests")
        if self.physics_steps <= 0:
            raise ValueError("GoalForge receipt requires physical simulation steps")
        if self.evidence_domain != "SHADOW":
            raise ValueError("GoalForge Phase 4 only permits SHADOW evidence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def receipt_hash(self) -> str:
        return hash_json(self.to_dict())


def hash_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def hash_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "G1_DDS_JOINT_NAMES",
    "G1_HARD_TORQUE_LIMITS",
    "GOALFORGE_TASK_ID",
    "GoalForgeResult",
    "GoalForgeStatus",
    "ShotParameters",
    "SimulationReceiptV4",
    "hash_bytes",
    "hash_json",
]
