"""Continuous learned run-up to precision free kick in one MuJoCo world."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import platform
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.approach_strike_residual import (
    G1ApproachStrikeResidualConfig,
    G1ApproachStrikeResidualController,
)
from rosclaw.growth.ballistic_contact_impulse_actor import (
    G1BallisticContactImpulseActor,
    g1_ballistic_contact_impulse_context_hash,
    g1_ballistic_contact_impulse_effect,
)
from rosclaw.growth.ballistic_contact_residual import (
    G1BallisticContactResidualConfig,
    blend_g1_ballistic_contact_target,
)
from rosclaw.growth.ballistic_contact_torque_residual import (
    G1BallisticContactTorqueResidualConfig,
    g1_ballistic_contact_torque_residual,
)
from rosclaw.growth.episodic_contact_memory import (
    G1EpisodicContactMemory,
    g1_episodic_contact_context_hash,
    g1_episodic_contact_effect,
    g1_episodic_contact_observation,
)
from rosclaw.growth.football_motion_prior import (
    G1FootballMotionPrior,
    blend_g1_football_motion_prior_target,
    blend_g1_football_motion_prior_velocity,
)
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    _adapt_target,
    _contact_observation,
    _fill_state,
    _load_robonaldo,
    _ModelIds,
    _policy_repeat_count,
    _roll_pitch,
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw.simforge.g1_cerebellar_recovery import (
    shared_post_impact_recovery_config,
)
from rosclaw.simforge.g1_joint_boundary_guard import (
    G1JointBoundaryGuardConfig,
    project_g1_joint_boundary_torque,
)
from rosclaw.simforge.g1_learned_runup import (
    G1LearnedGaitQualification,
    G1LearnedRunupConfig,
    G1LearnedRunupController,
    qualify_g1_learned_gait,
)
from rosclaw.simforge.g1_loft_teacher import (
    G1LoftTeacherConfig,
    g1_loft_teacher_effect,
)
from rosclaw.simforge.g1_sonic_runup import (
    G1SonicQualification,
    G1SonicRunupConfig,
    G1SonicRunupController,
    qualify_g1_sonic,
)
from rosclaw.simforge.g1_stadium_scene import (
    G1TrainingGoalSpec,
    apply_g1_compliant_goal_net_force,
    build_g1_stadium_model,
    g1_goal_net_contact_plane_x,
    g1_stadium_scene_hash,
)
from rosclaw.simforge.g1_torque_authority import (
    project_g1_additive_torque_authority,
    project_g1_torque_authority,
)
from rosclaw.simforge.g1_transition_bridge import (
    G1TransitionBridgeConfig,
    G1VelocityMatchedTransitionBridge,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_HARD_TORQUE_LIMITS,
    ShotParameters,
    hash_bytes,
    hash_json,
)

if TYPE_CHECKING:
    from rosclaw.growth.ballistic_skill_memory import G1BallisticSkillMemory
    from rosclaw.growth.football_outcome_model import G1FootballOutcomeModel
    from rosclaw.growth.proprioceptive_expert_router import (
        G1ProprioceptiveExpertRouter,
    )

_MOTION_REL = Path("policy/robonaldo/model/freekick_motion.npz")


class G1FootballEventPhase(IntEnum):
    """Stable phase ids shared by collection, triage and learning jobs."""

    APPROACH = 0
    ALIGN_BRAKE = 1
    PLANT_BRIDGE = 2
    LOAD = 3
    SWING = 4
    CONTACT = 5
    FOLLOW_THROUGH = 6
    RECOVERY = 7
    READY = 8


@dataclass(frozen=True)
class G1FreeKickFlowConfig:
    """Velocity-matched controller handoff and compliant-net contract."""

    kick_phase_start_frame: int = 150
    contextual_phase_yaw_threshold_rad: float = 0.0
    contextual_high_yaw_kick_phase_start_frame: int = 190
    bridge_duration_sec: float = 0.60
    bridge_entry_velocity_scale: float = 0.0
    bridge_exit_velocity_scale: float = 0.0
    bridge_boundary_velocity_limit_rad_s: float = 2.0
    history_prime_frames: int = 5
    aim_bias_y_m: float = 0.30
    aim_bias_z_m: float = 0.0
    shot_reference_plane_x_m: float = 0.0
    shot_reference_target_y_m: float | None = None
    shot_reference_target_z_m: float | None = None
    net_capture_depth_m: float = 0.80
    net_stiffness_n_m: float = 42.0
    net_damping_n_s_m: float = 8.5
    shot_pelvis_yaw_offset_rad: float = 0.10
    shot_foot_yaw_offset_rad: float = 0.01
    shot_foot_pitch_offset_rad: float = 0.0
    shot_loft_synergy_rad: float = 0.0
    shot_loft_teacher_target_vz_mps: float = 0.0
    shot_loft_teacher_gain_n_per_mps: float = 24.0
    shot_loft_teacher_max_force_n: float = 60.0
    shot_loft_teacher_target_vx_mps: float = 0.0
    shot_loft_teacher_forward_gain_n_per_mps: float = 20.0
    shot_loft_teacher_max_forward_force_n: float = 80.0
    shot_loft_teacher_target_vy_mps: float = 0.0
    shot_loft_teacher_lateral_gain_n_per_mps: float = 20.0
    shot_loft_teacher_max_lateral_force_n: float = 80.0
    shot_loft_teacher_start_policy_frame: int = 230
    shot_loft_teacher_end_policy_frame: int = 335
    shot_loft_teacher_foot_pitch_bonus_rad: float = 0.0
    shot_loft_teacher_max_foot_ball_distance_m: float = 0.0
    shot_com_shift_y_m: float = -0.065
    shot_swing_amplitude: float = 0.85
    shot_swing_speed_scale: float = 0.90
    shot_load_speed_scale: float = 1.0
    shot_contact_phase_offset: float = -0.015
    strike_gain_schedule_start_policy_frame: int = 0
    strike_gain_scales: tuple[float, ...] = (1.0,) * 29
    follow_through_gain_scales: tuple[float, ...] = (1.0,) * 29
    authority_calibration_hash: str | None = None
    contextual_phase_calibration_hash: str | None = None
    proprioceptive_router_hash: str | None = None
    football_outcome_model_hash: str | None = None
    football_motion_prior_hash: str | None = None
    football_motion_prior_blend: float = 0.0
    football_motion_prior_velocity_blend: float = 0.0
    football_motion_prior_contact_policy_frame: int = 265
    ballistic_contact_impulse_actor_hash: str | None = None
    episodic_contact_memory_hash: str | None = None
    ballistic_contact_residual_rad: tuple[float, ...] = (0.0,) * 6
    ballistic_contact_torque_residual_nm: tuple[float, ...] = (0.0,) * 6
    ballistic_contact_torque_preload_nm: tuple[float, ...] = (0.0,) * 6
    ballistic_contact_torque_phase_offset_sec: tuple[float, ...] = (0.0,) * 6
    ballistic_counterbalance_torque_residual_nm: tuple[float, ...] = (0.0,) * 6
    ballistic_contact_torque_policy_frame: int = 256
    ballistic_contact_torque_lead_duration_sec: float = 0.16
    ballistic_contact_torque_trail_duration_sec: float = 0.08
    ballistic_contact_policy_frame: int = 256
    ballistic_contact_lead_duration_sec: float = 0.16
    ballistic_contact_trail_duration_sec: float = 0.08
    football_retry_recovery_duration_sec: float = 0.0
    football_retry_follow_through_gain_scale: float = 1.0
    shared_cerebellar_recovery_enabled: bool = False
    shot_recovery_step_length_m: float = 0.11
    shot_recovery_step_yaw_rad: float = -0.05
    post_contact_damping_scale: float = 1.0
    post_contact_damping_delay_sec: float = 0.18
    post_contact_damping_ramp_sec: float = 0.45
    torque_authority_projection_ratio: float = 0.0
    torque_authority_projection_max_fraction: float = 0.01
    ballistic_skill_memory_hash: str | None = None
    ballistic_skill_id: str | None = None
    approach_provider: str = "groot_history"
    schema_version: str = "rosclaw.simforge.g1_free_kick_flow_config.v38"

    def __post_init__(self) -> None:
        if not isinstance(self.shared_cerebellar_recovery_enabled, bool):
            raise ValueError("shared cerebellar recovery flag must be boolean")
        values = (
            self.bridge_duration_sec,
            self.contextual_phase_yaw_threshold_rad,
            self.bridge_entry_velocity_scale,
            self.bridge_exit_velocity_scale,
            self.bridge_boundary_velocity_limit_rad_s,
            self.aim_bias_y_m,
            self.aim_bias_z_m,
            self.shot_reference_plane_x_m,
            self.net_capture_depth_m,
            self.net_stiffness_n_m,
            self.net_damping_n_s_m,
            self.shot_pelvis_yaw_offset_rad,
            self.shot_foot_yaw_offset_rad,
            self.shot_foot_pitch_offset_rad,
            self.shot_loft_synergy_rad,
            self.shot_loft_teacher_target_vz_mps,
            self.shot_loft_teacher_gain_n_per_mps,
            self.shot_loft_teacher_max_force_n,
            self.shot_loft_teacher_target_vx_mps,
            self.shot_loft_teacher_forward_gain_n_per_mps,
            self.shot_loft_teacher_max_forward_force_n,
            self.shot_loft_teacher_target_vy_mps,
            self.shot_loft_teacher_lateral_gain_n_per_mps,
            self.shot_loft_teacher_max_lateral_force_n,
            self.shot_loft_teacher_foot_pitch_bonus_rad,
            self.shot_loft_teacher_max_foot_ball_distance_m,
            self.shot_com_shift_y_m,
            self.shot_swing_amplitude,
            self.shot_swing_speed_scale,
            self.shot_load_speed_scale,
            self.shot_contact_phase_offset,
            self.football_retry_recovery_duration_sec,
            self.football_retry_follow_through_gain_scale,
            self.shot_recovery_step_length_m,
            self.shot_recovery_step_yaw_rad,
            self.post_contact_damping_scale,
            self.post_contact_damping_delay_sec,
            self.post_contact_damping_ramp_sec,
            self.torque_authority_projection_ratio,
            self.torque_authority_projection_max_fraction,
            self.football_motion_prior_blend,
            self.football_motion_prior_velocity_blend,
            self.ballistic_contact_lead_duration_sec,
            self.ballistic_contact_trail_duration_sec,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("free-kick flow config must be finite")
        if not 100 <= self.kick_phase_start_frame <= 240:
            raise ValueError("kick phase start must be in [100, 240]")
        if not 100 <= self.contextual_high_yaw_kick_phase_start_frame <= 240:
            raise ValueError("contextual high-yaw kick phase must be in [100, 240]")
        if self.contextual_phase_yaw_threshold_rad != 0.0 and not (
            0.05 <= self.contextual_phase_yaw_threshold_rad <= 0.35
        ):
            raise ValueError("contextual phase yaw threshold must be zero or in [0.05, 0.35] rad")
        if not 0.16 <= self.bridge_duration_sec <= 0.80:
            raise ValueError("skill bridge duration must be in [0.16, 0.80] s")
        G1TransitionBridgeConfig(
            duration_sec=self.bridge_duration_sec,
            entry_velocity_scale=self.bridge_entry_velocity_scale,
            exit_velocity_scale=self.bridge_exit_velocity_scale,
            maximum_boundary_velocity_rad_s=self.bridge_boundary_velocity_limit_rad_s,
        )
        if not 3 <= self.history_prime_frames <= 5:
            raise ValueError("skill bridge history prime must be in [3, 5] frames")
        if not -0.5 <= self.aim_bias_y_m <= 2.0:
            raise ValueError("aim bias must be in [-0.5, 2.0] m")
        if not -0.30 <= self.aim_bias_z_m <= 1.20:
            raise ValueError("vertical aim bias must be in [-0.30, 1.20] m")
        if self.shot_reference_plane_x_m != 0.0 and not (
            3.0 <= self.shot_reference_plane_x_m <= 12.0
        ):
            raise ValueError("shot reference plane must be zero or in [3, 12] m")
        if self.shot_reference_target_y_m is not None and not (
            math.isfinite(self.shot_reference_target_y_m)
            and -2.0 <= self.shot_reference_target_y_m <= 2.0
        ):
            raise ValueError("shot reference target y must be in [-2, 2] m")
        if self.shot_reference_target_z_m is not None and not (
            math.isfinite(self.shot_reference_target_z_m)
            and 0.105 <= self.shot_reference_target_z_m <= 2.40
        ):
            raise ValueError("shot reference target z must be in [0.105, 2.40] m")
        if (self.shot_reference_target_y_m is None) != (self.shot_reference_target_z_m is None):
            raise ValueError("shot reference target y and z must be provided together")
        if not 0.20 <= self.net_capture_depth_m <= 2.50:
            raise ValueError("net capture depth must be in [0.20, 2.50] m")
        if not 10.0 <= self.net_stiffness_n_m <= 250.0:
            raise ValueError("net stiffness must be in [10, 250] N/m")
        if not 2.0 <= self.net_damping_n_s_m <= 30.0:
            raise ValueError("net damping must be in [2, 30] N s/m")
        if self.approach_provider not in {"groot_history", "sonic_fullbody"}:
            raise ValueError("approach provider must be groot_history or sonic_fullbody")
        if not -0.20 <= self.shot_pelvis_yaw_offset_rad <= 0.20:
            raise ValueError("shot pelvis yaw offset must be in [-0.20, 0.20] rad")
        if not -0.10 <= self.shot_foot_yaw_offset_rad <= 0.10:
            raise ValueError("shot foot yaw offset must be in [-0.10, 0.10] rad")
        if not -0.18 <= self.shot_foot_pitch_offset_rad <= 0.18:
            raise ValueError("shot foot pitch offset must be in [-0.18, 0.18] rad")
        if not 0.0 <= self.shot_loft_synergy_rad <= 0.30:
            raise ValueError("shot loft synergy must be in [0, 0.30] rad")
        G1LoftTeacherConfig(
            target_vertical_speed_mps=self.shot_loft_teacher_target_vz_mps,
            velocity_gain_n_per_mps=self.shot_loft_teacher_gain_n_per_mps,
            maximum_vertical_force_n=self.shot_loft_teacher_max_force_n,
            target_forward_speed_mps=self.shot_loft_teacher_target_vx_mps,
            forward_velocity_gain_n_per_mps=(self.shot_loft_teacher_forward_gain_n_per_mps),
            maximum_forward_force_n=self.shot_loft_teacher_max_forward_force_n,
            target_lateral_speed_mps=self.shot_loft_teacher_target_vy_mps,
            lateral_velocity_gain_n_per_mps=(self.shot_loft_teacher_lateral_gain_n_per_mps),
            maximum_lateral_force_n=self.shot_loft_teacher_max_lateral_force_n,
            start_policy_frame=self.shot_loft_teacher_start_policy_frame,
            end_policy_frame=self.shot_loft_teacher_end_policy_frame,
            maximum_foot_ball_distance_m=(self.shot_loft_teacher_max_foot_ball_distance_m),
        )
        if not 0.0 <= self.shot_loft_teacher_foot_pitch_bonus_rad <= 0.12:
            raise ValueError("loft teacher foot pitch bonus must be in [0, 0.12] rad")
        if self.shot_loft_teacher_target_vz_mps == 0.0 and (
            self.shot_loft_teacher_foot_pitch_bonus_rad != 0.0
        ):
            raise ValueError("loft teacher foot pitch bonus requires the teacher to be enabled")
        if not -0.08 <= self.shot_com_shift_y_m <= 0.02:
            raise ValueError("shot COM shift must be in [-0.08, 0.02] m")
        if not 0.70 <= self.shot_swing_amplitude <= 1.15:
            raise ValueError("shot swing amplitude must be in [0.70, 1.15]")
        if not 0.80 <= self.shot_swing_speed_scale <= 1.50:
            raise ValueError("shot swing speed scale must be in [0.80, 1.50]")
        if not 1.0 <= self.shot_load_speed_scale <= 1.50:
            raise ValueError("shot load speed scale must be in [1.0, 1.50]")
        if not -0.08 <= self.shot_contact_phase_offset <= 0.08:
            raise ValueError("shot contact phase offset must be in [-0.08, 0.08]")
        if self.strike_gain_schedule_start_policy_frame != 0 and not (
            185 <= self.strike_gain_schedule_start_policy_frame <= 335
        ):
            raise ValueError(
                "strike gain schedule start must be zero or in [185, 335] policy frames"
            )
        if len(self.strike_gain_scales) != 29 or not all(
            math.isfinite(value) and 0.5 <= value <= 1.0 for value in self.strike_gain_scales
        ):
            raise ValueError("strike gain scales must contain 29 values in [0.5, 1.0]")
        if len(self.follow_through_gain_scales) != 29 or not all(
            math.isfinite(value) and 0.5 <= value <= 1.0
            for value in self.follow_through_gain_scales
        ):
            raise ValueError("follow-through gain scales must contain 29 values in [0.5, 1.0]")
        if (
            self.authority_calibration_hash is not None
            and not self.authority_calibration_hash.startswith("sha256:")
        ):
            raise ValueError("flow authority calibration hash must be SHA-256")
        if (
            self.contextual_phase_calibration_hash is not None
            and not self.contextual_phase_calibration_hash.startswith("sha256:")
        ):
            raise ValueError("contextual phase calibration hash must be SHA-256")
        if (
            self.contextual_phase_calibration_hash is not None
            and self.contextual_phase_yaw_threshold_rad == 0.0
        ):
            raise ValueError("contextual phase calibration requires enabled routing")
        if self.proprioceptive_router_hash is not None and not (
            self.proprioceptive_router_hash.startswith("sha256:")
        ):
            raise ValueError("proprioceptive router hash must be SHA-256")
        if self.proprioceptive_router_hash is not None and (
            self.contextual_phase_yaw_threshold_rad != 0.0
            or self.contextual_phase_calibration_hash is not None
        ):
            raise ValueError("proprioceptive and legacy contextual routers are exclusive")
        if self.football_outcome_model_hash is not None and not (
            self.football_outcome_model_hash.startswith("sha256:")
        ):
            raise ValueError("football outcome model hash must be SHA-256")
        if self.football_outcome_model_hash is not None and (
            self.proprioceptive_router_hash is not None
            or self.contextual_phase_yaw_threshold_rad != 0.0
            or self.contextual_phase_calibration_hash is not None
        ):
            raise ValueError("football outcome model and contextual routers are exclusive")
        if self.football_motion_prior_hash is not None and not (
            self.football_motion_prior_hash.startswith("sha256:")
        ):
            raise ValueError("football motion prior hash must be SHA-256")
        if not 0.0 <= self.football_motion_prior_blend <= 0.50:
            raise ValueError("football motion prior blend must be in [0, 0.50]")
        if not 0.0 <= self.football_motion_prior_velocity_blend <= 0.50:
            raise ValueError("football motion prior velocity blend must be in [0, 0.50]")
        motion_prior_enabled = bool(
            self.football_motion_prior_blend > 0.0
            or self.football_motion_prior_velocity_blend > 0.0
        )
        if (self.football_motion_prior_hash is not None) != motion_prior_enabled:
            raise ValueError("football motion prior hash and non-zero blend must be paired")
        if not 240 <= self.football_motion_prior_contact_policy_frame <= 300:
            raise ValueError("football motion prior contact frame must be in [240, 300]")
        if self.ballistic_contact_impulse_actor_hash is not None and not (
            self.ballistic_contact_impulse_actor_hash.startswith("sha256:")
        ):
            raise ValueError("ballistic contact impulse actor hash must be SHA-256")
        if (
            self.ballistic_contact_impulse_actor_hash is not None
            and G1LoftTeacherConfig(
                target_vertical_speed_mps=self.shot_loft_teacher_target_vz_mps,
                target_forward_speed_mps=self.shot_loft_teacher_target_vx_mps,
                target_lateral_speed_mps=self.shot_loft_teacher_target_vy_mps,
            ).enabled
        ):
            raise ValueError("learned contact impulse actor and SIM teacher are exclusive")
        if self.episodic_contact_memory_hash is not None and not (
            self.episodic_contact_memory_hash.startswith("sha256:")
        ):
            raise ValueError("episodic contact memory hash must be SHA-256")
        if (
            self.episodic_contact_memory_hash is not None
            and self.ballistic_contact_impulse_actor_hash is not None
        ):
            raise ValueError("episodic memory and contact impulse actor are exclusive")
        if (
            self.episodic_contact_memory_hash is not None
            and G1LoftTeacherConfig(
                target_vertical_speed_mps=self.shot_loft_teacher_target_vz_mps,
                target_forward_speed_mps=self.shot_loft_teacher_target_vx_mps,
                target_lateral_speed_mps=self.shot_loft_teacher_target_vy_mps,
            ).enabled
        ):
            raise ValueError("episodic contact memory and SIM teacher are exclusive")
        G1BallisticContactResidualConfig(
            right_leg_residual_rad=self.ballistic_contact_residual_rad,
            contact_policy_frame=self.ballistic_contact_policy_frame,
            lead_duration_sec=self.ballistic_contact_lead_duration_sec,
            trail_duration_sec=self.ballistic_contact_trail_duration_sec,
        )
        G1BallisticContactTorqueResidualConfig(
            right_leg_residual_nm=self.ballistic_contact_torque_residual_nm,
            right_leg_preload_nm=self.ballistic_contact_torque_preload_nm,
            right_leg_phase_offset_sec=(self.ballistic_contact_torque_phase_offset_sec),
            counterbalance_residual_nm=(self.ballistic_counterbalance_torque_residual_nm),
            contact_policy_frame=self.ballistic_contact_torque_policy_frame,
            lead_duration_sec=self.ballistic_contact_torque_lead_duration_sec,
            trail_duration_sec=self.ballistic_contact_torque_trail_duration_sec,
        )
        if self.football_retry_recovery_duration_sec != 0.0 and not (
            0.4 <= self.football_retry_recovery_duration_sec <= 2.0
        ):
            raise ValueError("football retry recovery must be zero or in [0.4, 2.0] s")
        if (
            self.football_retry_recovery_duration_sec > 0.0
            and self.football_outcome_model_hash is None
        ):
            raise ValueError("football retry recovery requires an outcome model")
        if not 0.7 <= self.football_retry_follow_through_gain_scale <= 1.0:
            raise ValueError("football retry follow-through gain scale must be in [0.7, 1.0]")
        if not 0.0 <= self.shot_recovery_step_length_m <= 0.15:
            raise ValueError("shot recovery step length must be in [0.0, 0.15] m")
        if not -0.15 <= self.shot_recovery_step_yaw_rad <= 0.15:
            raise ValueError("shot recovery step yaw must be in [-0.15, 0.15] rad")
        if not 1.0 <= self.post_contact_damping_scale <= 2.5:
            raise ValueError("post-contact damping scale must be in [1.0, 2.5]")
        if not 0.0 <= self.post_contact_damping_delay_sec <= 0.4:
            raise ValueError("post-contact damping delay must be in [0.0, 0.4] s")
        if not 0.1 <= self.post_contact_damping_ramp_sec <= 0.8:
            raise ValueError("post-contact damping ramp must be in [0.1, 0.8] s")
        if self.torque_authority_projection_ratio != 0.0 and not (
            0.90 <= self.torque_authority_projection_ratio <= 0.99
        ):
            raise ValueError("torque authority ratio must be zero or in [0.90, 0.99]")
        if not 0.001 <= self.torque_authority_projection_max_fraction <= 0.05:
            raise ValueError("torque authority projection fraction must be in [0.001, 0.05]")
        if (self.ballistic_skill_memory_hash is None) != (self.ballistic_skill_id is None):
            raise ValueError("ballistic skill memory hash and skill id must be paired")
        if self.ballistic_skill_memory_hash is not None and not (
            self.ballistic_skill_memory_hash.startswith("sha256:")
        ):
            raise ValueError("ballistic skill memory hash must be SHA-256")
        if self.ballistic_skill_id is not None and not self.ballistic_skill_id.startswith(
            "sonic-seed-"
        ):
            raise ValueError("ballistic skill id is invalid")
        if self.ballistic_skill_id is not None and self.approach_provider != "sonic_fullbody":
            raise ValueError("ballistic skill memory requires the SONIC provider")
        if (
            self.football_retry_follow_through_gain_scale != 1.0
            and self.football_retry_recovery_duration_sec == 0.0
        ):
            raise ValueError("retry follow-through scaling requires retry recovery")


@dataclass(frozen=True)
class G1FreeKickResult:
    finite_state: bool
    learned_runup_executed: bool
    learned_approach_strike_residual_executed: bool
    loft_teacher_executed: bool
    loft_teacher_active_frames: int
    loft_teacher_peak_torque_nm: float
    loft_teacher_peak_force_n: float
    joint_boundary_guard_active_steps: int
    joint_boundary_guard_peak_correction_nm: float
    residual_accepted_frames: int
    residual_rejected_frames: int
    residual_peak_nm: float
    residual_rms_nm: float
    residual_effect_fraction: float
    continuous_single_world: bool
    state_reset_after_start: bool
    initial_ball_distance_m: float
    shot_distance_m: float
    runup_distance_m: float
    runup_peak_speed_mps: float
    runup_min_pelvis_height_m: float
    runup_peak_tilt_rad: float
    runup_terminal_speed_mps: float
    handoff_yaw_rad: float
    handoff_roll_rad: float
    handoff_pitch_rad: float
    handoff_pelvis_x_m: float
    handoff_pelvis_y_m: float
    handoff_joint_velocity_rms_rad_s: float
    selected_kick_phase_start_frame: int
    contextual_phase_expert_executed: bool
    proprioceptive_router_executed: bool
    proprioceptive_router_fallback: bool
    proprioceptive_router_nearest_distance: float | None
    proprioceptive_router_distance_margin: float | None
    handoff_to_contact_sec: float | None
    pre_contact_motion_pause_sec: float
    handoff_min_forward_speed_mps: float
    handoff_low_forward_speed_duration_sec: float
    handoff_forward_speed_retention_ratio: float
    skill_bridge_max_joint_delta_rad: float
    skill_bridge_rms_joint_delta_rad: float
    skill_bridge_entry_velocity_rms_rad_s: float
    skill_bridge_target_exit_velocity_rms_rad_s: float
    skill_bridge_exit_velocity_error_rms_rad_s: float
    skill_bridge_peak_target_acceleration_rms_rad_s2: float
    kick_contact_observed: bool
    contact_time_sec: float | None
    kick_contact_point_xyz_m: tuple[float, float, float] | None
    kick_contact_height_relative_ball_center_m: float | None
    ball_launch_velocity_xyz_mps: tuple[float, float, float] | None
    ball_apex_height_m: float
    ball_speed_peak_mps: float
    goal_crossed: bool
    goal_crossing_xyz_m: tuple[float, float, float] | None
    goal_mouth_hit: bool
    goal_plane_target_error_m: float | None
    net_capture_xyz_m: tuple[float, float, float] | None
    net_capture_target_error_m: float | None
    final_ball_xyz_m: tuple[float, float, float]
    final_ball_yz_target_error_m: float
    ball_retained_in_goal: bool
    precision_radius_m: float
    declared_target_corner: str
    declared_corner_distance_m: float | None
    upper_corner_distance_m: float | None
    lower_corner_distance_m: float | None
    kick_min_pelvis_height_m: float
    kick_peak_tilt_rad: float
    final_pelvis_height_m: float
    final_speed_mps: float
    post_kick_fall: bool
    joint_limit_violation: bool
    torque_limit_violation: bool
    actuator_saturation: bool
    actuator_saturation_steps: int
    actuator_saturation_fraction: float
    actuator_peak_demand_ratio: float
    physics_steps: int
    post_contact_peak_pelvis_speed_mps: float = 0.0
    post_contact_backward_displacement_m: float = 0.0
    post_contact_forward_velocity_reversals: int = 0
    post_contact_settling_time_sec: float = 0.0
    post_contact_peak_joint_velocity_rms_rad_s: float = 0.0
    post_contact_final_joint_velocity_rms_rad_s: float = 0.0
    post_contact_mean_pelvis_speed_mps: float = 0.0
    post_contact_mean_joint_velocity_rms_rad_s: float = 0.0
    cerebellar_recovery_executed: bool = False
    cerebellar_recovery_active_frames: int = 0
    cerebellar_recovery_peak_blend_fraction: float = 0.0
    football_outcome_model_executed: bool = False
    football_outcome_retry_recommended: bool = False
    football_outcome_predicted_hard_safe_probability: float | None = None
    football_outcome_predicted_precision_probability: float | None = None
    football_outcome_predicted_penalized_error_m: float | None = None
    football_outcome_retry_recovery_executed: bool = False
    football_outcome_retry_recovery_duration_sec: float = 0.0
    football_outcome_retry_initial_speed_mps: float | None = None
    football_outcome_retry_final_speed_mps: float | None = None
    football_motion_prior_executed: bool = False
    football_motion_prior_active_frames: int = 0
    football_motion_prior_peak_target_delta_rad: float = 0.0
    football_motion_prior_velocity_executed: bool = False
    football_motion_prior_velocity_active_frames: int = 0
    football_motion_prior_peak_target_velocity_delta_rad_s: float = 0.0
    ballistic_contact_residual_executed: bool = False
    ballistic_contact_residual_active_frames: int = 0
    ballistic_contact_residual_peak_target_delta_rad: float = 0.0
    ballistic_contact_torque_residual_executed: bool = False
    ballistic_contact_torque_residual_active_frames: int = 0
    ballistic_contact_torque_residual_peak_nm: float = 0.0
    ballistic_contact_impulse_actor_executed: bool = False
    ballistic_contact_impulse_actor_active_frames: int = 0
    ballistic_contact_impulse_actor_peak_torque_nm: float = 0.0
    ballistic_contact_impulse_actor_peak_lateral_force_n: float = 0.0
    ballistic_contact_impulse_actor_peak_vertical_force_n: float = 0.0
    ballistic_contact_impulse_actor_target_conditioned: bool = False
    ballistic_contact_impulse_actor_out_of_envelope_frames: int = 0
    ballistic_contact_impulse_actor_peak_desired_lateral_launch_speed_mps: float = 0.0
    ballistic_contact_impulse_actor_peak_desired_vertical_launch_speed_mps: float = 0.0
    episodic_contact_memory_executed: bool = False
    episodic_contact_memory_active_frames: int = 0
    episodic_contact_memory_out_of_support_frames: int = 0
    episodic_contact_memory_peak_torque_nm: float = 0.0
    episodic_contact_memory_peak_lateral_force_n: float = 0.0
    episodic_contact_memory_peak_vertical_force_n: float = 0.0
    episodic_contact_memory_peak_desired_lateral_launch_speed_mps: float = 0.0
    episodic_contact_memory_peak_desired_vertical_launch_speed_mps: float = 0.0
    episodic_contact_memory_selected_seed_label: int | None = None
    episodic_contact_memory_peak_context_distance: float = 0.0
    torque_authority_projection_enabled: bool = False
    torque_authority_projection_steps: int = 0
    torque_authority_projection_fraction: float = 0.0
    torque_authority_projection_peak_correction_nm: float = 0.0
    torque_authority_preprojection_peak_demand_ratio: float = 0.0
    torque_authority_projection_qualified: bool = True
    contact_task_authority_projection_steps: int = 0
    contact_task_authority_scale_min: float = 1.0
    ballistic_skill_memory_executed: bool = False
    ballistic_skill_id: str | None = None
    ballistic_skill_nearest_distance: float | None = None
    ballistic_skill_distance_margin: float | None = None
    kick_contact_foot_position_xyz_m: tuple[float, float, float] | None = None
    kick_contact_foot_velocity_xyz_mps: tuple[float, float, float] | None = None
    kick_contact_normal_xyz: tuple[float, float, float] | None = None
    kick_contact_force_world_xyz_n: tuple[float, float, float] | None = None
    kick_contact_peak_force_n: float | None = None
    schema_version: str = "rosclaw.simforge.g1_free_kick_result.v27"

    @property
    def perceptual_continuity_passed(self) -> bool:
        """Require forward progress, not merely moving limbs in place."""

        return bool(
            self.handoff_to_contact_sec is not None
            and self.handoff_to_contact_sec <= 1.10
            and self.pre_contact_motion_pause_sec <= 0.10
            and self.handoff_min_forward_speed_mps >= 0.15
            and self.handoff_low_forward_speed_duration_sec <= 0.10
            and self.handoff_forward_speed_retention_ratio >= 0.45
        )

    @property
    def passed(self) -> bool:
        return bool(
            self.finite_state
            and self.learned_runup_executed
            and not self.loft_teacher_executed
            and self.continuous_single_world
            and not self.state_reset_after_start
            and self.initial_ball_distance_m >= 4.0
            and self.runup_distance_m >= 3.0
            and self.runup_peak_speed_mps >= 1.0
            and self.runup_min_pelvis_height_m >= 0.70
            and self.runup_peak_tilt_rad <= 0.30
            and 0.10 <= self.runup_terminal_speed_mps <= 0.40
            and self.handoff_to_contact_sec is not None
            and self.handoff_to_contact_sec <= 2.8
            and self.pre_contact_motion_pause_sec <= 0.25
            and self.skill_bridge_max_joint_delta_rad <= 1.25
            and self.skill_bridge_rms_joint_delta_rad <= 0.40
            and self.kick_contact_observed
            and self.ball_speed_peak_mps >= 6.0
            and self.goal_crossed
            and self.goal_mouth_hit
            and self.goal_plane_target_error_m is not None
            and self.goal_plane_target_error_m <= self.precision_radius_m
            and self.net_capture_target_error_m is not None
            and self.net_capture_target_error_m <= self.precision_radius_m + 0.05
            and self.ball_retained_in_goal
            and self.declared_corner_distance_m is not None
            and self.declared_corner_distance_m <= 0.25
            and self.kick_min_pelvis_height_m >= 0.68
            and self.kick_peak_tilt_rad <= 0.40
            and self.final_pelvis_height_m >= 0.70
            and self.final_speed_mps <= 0.20
            and self.post_contact_backward_displacement_m <= 0.20
            and self.post_contact_forward_velocity_reversals <= 12
            and self.post_contact_settling_time_sec <= 5.0
            and self.post_contact_final_joint_velocity_rms_rad_s <= 0.10
            and self.post_contact_mean_pelvis_speed_mps <= 0.12
            and self.post_contact_mean_joint_velocity_rms_rad_s <= 0.25
            and not self.post_kick_fall
            and not self.joint_limit_violation
            and not self.torque_limit_violation
            and not self.actuator_saturation
            and self.torque_authority_projection_qualified
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "perceptual_continuity_passed": self.perceptual_continuity_passed,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class G1FreeKickEvidence:
    body_hash: str
    kick_prior_hash: str
    learned_gait_qualification: G1LearnedGaitQualification
    sonic_qualification: G1SonicQualification | None
    sonic_runup_config: G1SonicRunupConfig | None
    sonic_reference_digest: str | None
    approach_strike_candidate_hash: str | None
    approach_strike_residual_config: G1ApproachStrikeResidualConfig | None
    stadium_scene_hash: str
    implementation_hash: str
    request_hash: str
    trajectory_path: str
    trajectory_hash: str
    trajectory_digest: str
    strict_replay: bool
    runup_config: G1LearnedRunupConfig
    flow_config: G1FreeKickFlowConfig
    goal_spec: G1TrainingGoalSpec
    result: G1FreeKickResult
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "DEVELOPMENT_SHOWCASE"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.simforge.g1_free_kick_evidence.v27"

    @property
    def passed(self) -> bool:
        return bool(
            self.strict_replay
            and self.learned_gait_qualification.eligible
            and (
                self.flow_config.approach_provider != "sonic_fullbody"
                or (
                    self.sonic_qualification is not None
                    and self.sonic_qualification.eligible
                    and self.sonic_runup_config is not None
                    and self.sonic_reference_digest is not None
                )
            )
            and (
                self.flow_config.ballistic_skill_memory_hash is None
                or (
                    self.result.ballistic_skill_memory_executed
                    and self.result.ballistic_skill_id == self.flow_config.ballistic_skill_id
                    and self.result.ballistic_skill_nearest_distance is not None
                    and self.result.ballistic_skill_distance_margin is not None
                )
            )
            and self.result.passed
            and self.activation_ceiling == "SIM_ONLY"
            and not self.hardware_command_sent
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "learned_gait_qualification": asdict(self.learned_gait_qualification),
            "sonic_qualification": (
                None if self.sonic_qualification is None else asdict(self.sonic_qualification)
            ),
            "sonic_runup_config": (
                None if self.sonic_runup_config is None else asdict(self.sonic_runup_config)
            ),
            "approach_strike_residual_config": (
                None
                if self.approach_strike_residual_config is None
                else asdict(self.approach_strike_residual_config)
            ),
            "runup_config": asdict(self.runup_config),
            "flow_config": asdict(self.flow_config),
            "goal_spec": asdict(self.goal_spec),
            "result": self.result.to_dict(),
            "passed": self.passed,
            "claims": {
                "learned_neural_runup_policy": True,
                "sonic_fullbody_proprioceptive_policy": (
                    self.flow_config.approach_provider == "sonic_fullbody"
                ),
                "sonic_planner_pixels_or_kinematics_used_for_scoring": False,
                "support_bound_approach_strike_iql_residual": (
                    self.approach_strike_candidate_hash is not None
                ),
                "train_only_support_bound_football_motion_prior": (
                    self.result.football_motion_prior_executed
                ),
                "bounded_ballistic_contact_residual": (
                    self.result.ballistic_contact_residual_executed
                ),
                "bounded_sim_only_ballistic_contact_torque_residual": (
                    self.result.ballistic_contact_torque_residual_executed
                ),
                "audited_torque_authority_projection": (
                    self.result.torque_authority_projection_enabled
                ),
                "direction_preserving_contact_task_authority_projection": (
                    self.result.contact_task_authority_projection_steps > 0
                ),
                "learned_proprioceptive_contact_impulse_actor": (
                    self.result.ballistic_contact_impulse_actor_executed
                ),
                "state_routed_episodic_contact_memory": (
                    self.result.episodic_contact_memory_executed
                ),
                "full_state_ballistic_skill_memory": (self.result.ballistic_skill_memory_executed),
                "sim_only_operational_space_loft_teacher": (self.result.loft_teacher_executed),
                "post_contact_right_ankle_boundary_projection": (
                    self.result.joint_boundary_guard_active_steps > 0
                ),
                "event_phase_contract": {phase.name: int(phase) for phase in G1FootballEventPhase},
                "continuous_runup_kick_recovery": True,
                "perceptual_run_to_strike_continuity": (self.result.perceptual_continuity_passed),
                "scoring_goal_decoupled_from_motion_reference": (
                    self.flow_config.shot_reference_plane_x_m > 0.0
                ),
                "post_contact_recovery_metrics_from_physics_trace": True,
                "shared_contact_gated_cerebellar_recovery": (
                    self.result.cerebellar_recovery_executed
                    and self.result.cerebellar_recovery_active_frames > 0
                ),
                "contact_dynamics_observed_from_physics": bool(
                    self.result.kick_contact_foot_position_xyz_m is not None
                    and self.result.kick_contact_foot_velocity_xyz_mps is not None
                    and self.result.kick_contact_normal_xyz is not None
                    and self.result.kick_contact_force_world_xyz_n is not None
                    and self.result.kick_contact_peak_force_n is not None
                ),
                "velocity_matched_mid_phase_handoff": (
                    self.flow_config.bridge_entry_velocity_scale > 0.0
                    or self.flow_config.bridge_exit_velocity_scale > 0.0
                ),
                "proprioceptive_contextual_phase_expert": (
                    self.result.contextual_phase_expert_executed
                ),
                "multi_feature_proprioceptive_expert_router": (
                    self.result.proprioceptive_router_executed
                ),
                "success_failure_football_outcome_memory": (
                    self.result.football_outcome_model_executed
                ),
                "football_success_requires_ball_contact": True,
                "recovery_only_is_task_success": False,
                "retry_recommendation_is_terminal_abstention": False,
                "continuous_retry_recovery_before_mandatory_shot": (
                    self.result.football_outcome_retry_recovery_executed
                ),
                "single_rigid_body_world": True,
                "state_teleport_or_reset_after_start": False,
                "native_collision_goal_frame": True,
                "native_collision_net": False,
                "compliant_net_capture_force_model": True,
                "net_capture_from_physics_trajectory": True,
                "precision_scoring_from_physics_state": True,
                "rendered_pixels_used_for_scoring": False,
                "sealed_generalization_evidence": False,
                "promotion_evidence": False,
                "real_hardware": False,
            },
        }


def run_g1_free_kick_showcase(
    *,
    asset_root: Path,
    gait_policy_root: Path,
    output_dir: Path,
    source_checkout: Path,
    runup_config: G1LearnedRunupConfig | None = None,
    flow_config: G1FreeKickFlowConfig | None = None,
    goal_spec: G1TrainingGoalSpec | None = None,
    sonic_model_root: Path | None = None,
    sonic_runup_config: G1SonicRunupConfig | None = None,
    approach_strike_candidate_path: Path | None = None,
    approach_strike_residual_config: G1ApproachStrikeResidualConfig | None = None,
    proprioceptive_expert_router: G1ProprioceptiveExpertRouter | None = None,
    football_outcome_model: G1FootballOutcomeModel | None = None,
    football_motion_prior: G1FootballMotionPrior | None = None,
    ballistic_skill_memory: G1BallisticSkillMemory | None = None,
    ballistic_contact_impulse_actor: G1BallisticContactImpulseActor | None = None,
    episodic_contact_memory: G1EpisodicContactMemory | None = None,
) -> G1FreeKickEvidence:
    """Execute and strictly replay one continuous long run-up free kick."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("free-kick evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    runup = runup_config or G1LearnedRunupConfig()
    flow = flow_config or G1FreeKickFlowConfig()
    goal = goal_spec or G1TrainingGoalSpec()
    if flow.net_capture_depth_m > goal.depth_m:
        raise ValueError("net capture depth extends beyond the visible goal net")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if football_motion_prior is None and flow.football_motion_prior_hash is not None:
        raise ValueError("flow declares a football motion prior but none was supplied")
    if football_motion_prior is not None:
        if football_motion_prior.body_hash != qualification.body_hash:
            raise ValueError("football motion prior Body hash mismatch")
        if flow.football_motion_prior_hash != football_motion_prior.prior_hash:
            raise ValueError("flow football motion prior hash mismatch")
        if (
            flow.football_motion_prior_velocity_blend > 0.0
            and not football_motion_prior.whole_body_velocity_reference_rad_s
        ):
            raise ValueError("flow velocity blend requires a velocity-aware football prior")
    if (
        ballistic_contact_impulse_actor is None
        and flow.ballistic_contact_impulse_actor_hash is not None
    ):
        raise ValueError("flow declares a contact impulse actor but none was supplied")
    if ballistic_contact_impulse_actor is not None:
        if ballistic_contact_impulse_actor.body_hash != qualification.body_hash:
            raise ValueError("contact impulse actor Body hash mismatch")
        if flow.ballistic_contact_impulse_actor_hash != ballistic_contact_impulse_actor.actor_hash:
            raise ValueError("flow contact impulse actor hash mismatch")
    if episodic_contact_memory is None and flow.episodic_contact_memory_hash is not None:
        raise ValueError("flow declares episodic contact memory but none was supplied")
    if episodic_contact_memory is not None:
        if episodic_contact_memory.body_hash != qualification.body_hash:
            raise ValueError("episodic contact memory Body hash mismatch")
        if flow.episodic_contact_memory_hash != episodic_contact_memory.memory_hash:
            raise ValueError("flow episodic contact memory hash mismatch")
    if ballistic_skill_memory is None and flow.ballistic_skill_memory_hash is not None:
        raise ValueError("flow declares a ballistic skill memory but none was supplied")
    if ballistic_skill_memory is not None:
        if ballistic_skill_memory.body_hash != qualification.body_hash:
            raise ValueError("ballistic skill memory Body hash mismatch")
        if flow.ballistic_skill_memory_hash != ballistic_skill_memory.memory_hash:
            raise ValueError("flow ballistic skill memory hash mismatch")
        if flow.ballistic_skill_id is None:
            raise ValueError("flow ballistic skill id is missing")
        ballistic_skill_memory.prototype(flow.ballistic_skill_id)
    if proprioceptive_expert_router is None and flow.proprioceptive_router_hash is not None:
        raise ValueError("flow declares a proprioceptive router but none was supplied")
    if proprioceptive_expert_router is not None:
        if proprioceptive_expert_router.body_hash != qualification.body_hash:
            raise ValueError("proprioceptive expert router Body hash mismatch")
        if flow.proprioceptive_router_hash != proprioceptive_expert_router.router_hash:
            raise ValueError("flow proprioceptive router hash mismatch")
    if football_outcome_model is None and flow.football_outcome_model_hash is not None:
        raise ValueError("flow declares a football outcome model but none was supplied")
    if football_outcome_model is not None:
        if proprioceptive_expert_router is not None:
            raise ValueError("football outcome model and expert router are exclusive")
        if flow.approach_provider != "sonic_fullbody":
            raise ValueError("football outcome model requires the SONIC provider")
        if football_outcome_model.body_hash != qualification.body_hash:
            raise ValueError("football outcome model Body hash mismatch")
        if flow.football_outcome_model_hash != football_outcome_model.model_hash:
            raise ValueError("flow football outcome model hash mismatch")
    gait_qualification = qualify_g1_learned_gait(gait_policy_root)
    gait_qualification.require_eligible()
    sonic_config: G1SonicRunupConfig | None = None
    sonic_qualification: G1SonicQualification | None = None
    if flow.approach_provider == "sonic_fullbody":
        if sonic_model_root is None:
            raise ValueError("sonic_fullbody approach requires sonic_model_root")
        sonic_config = sonic_runup_config or G1SonicRunupConfig()
        sonic_qualification = qualify_g1_sonic(sonic_model_root)
        sonic_qualification.require_eligible()
    residual_controller: G1ApproachStrikeResidualController | None = None
    residual_config: G1ApproachStrikeResidualConfig | None = None
    if approach_strike_candidate_path is not None:
        if flow.approach_provider != "sonic_fullbody":
            raise ValueError("approach-strike residual requires the SONIC full-body provider")
        residual_config = approach_strike_residual_config or G1ApproachStrikeResidualConfig()
        residual_controller = G1ApproachStrikeResidualController(
            approach_strike_candidate_path,
            residual_config,
        )
    if ballistic_contact_impulse_actor is not None:
        actor_context_hash = g1_ballistic_contact_impulse_context_hash(
            flow_config=asdict(flow),
            goal_spec=asdict(goal),
            runup_config=asdict(runup),
            sonic_runup_config=(None if sonic_config is None else asdict(sonic_config)),
            approach_strike_candidate_hash=(
                None if residual_controller is None else residual_controller.candidate_hash
            ),
            target_conditioned=(
                ballistic_contact_impulse_actor.schema_version
                == "rosclaw.growth.g1_ballistic_contact_impulse_actor.v2"
            ),
        )
        if actor_context_hash != ballistic_contact_impulse_actor.experiment_context_hash:
            raise ValueError("contact impulse actor experiment context mismatch")
    if episodic_contact_memory is not None:
        memory_context_hash = g1_episodic_contact_context_hash(
            flow_config=asdict(flow),
            goal_spec=asdict(goal),
            runup_config=asdict(runup),
            sonic_runup_config=(None if sonic_config is None else asdict(sonic_config)),
            approach_strike_candidate_hash=(
                None if residual_controller is None else residual_controller.candidate_hash
            ),
        )
        if memory_context_hash != episodic_contact_memory.experiment_context_hash:
            raise ValueError("episodic contact memory experiment context mismatch")
    implementation_hash = hash_json(
        {
            name: hash_bytes(Path(__file__).with_name(name).read_bytes())
            for name in (
                "g1_free_kick_showcase.py",
                "g1_cerebellar_recovery.py",
                "g1_joint_boundary_guard.py",
                "g1_learned_runup.py",
                "g1_loft_teacher.py",
                "g1_sonic_runup.py",
                "g1_stadium_scene.py",
                "g1_torque_authority.py",
                "g1_transition_bridge.py",
                "g1_two_player_relay.py",
            )
        }
        | {
            relative: hash_bytes(Path(__file__).parents[1].joinpath(relative).read_bytes())
            for relative in (
                "growth/approach_strike_contracts.py",
                "growth/approach_strike_residual.py",
                "growth/ballistic_contact_residual.py",
                "growth/ballistic_contact_impulse_actor.py",
                "growth/ballistic_contact_torque_residual.py",
                "growth/ballistic_skill_memory.py",
                "growth/episodic_contact_memory.py",
                "growth/learners/iql.py",
                "growth/phase_conditioned_residual.py",
                "growth/football_outcome_model.py",
                "growth/football_motion_prior.py",
                "growth/proprioceptive_expert_router.py",
                "simforge/backends/unitree_mujoco_backend.py",
                "simforge/tasks/g1_goalforge/concepts.py",
            )
        }
    )
    if (
        ballistic_skill_memory is not None
        and ballistic_skill_memory.implementation_hash != implementation_hash
    ):
        raise ValueError("ballistic skill memory implementation hash mismatch")
    if (
        ballistic_contact_impulse_actor is not None
        and ballistic_contact_impulse_actor.implementation_hash != implementation_hash
    ):
        raise ValueError("contact impulse actor implementation hash mismatch")
    if (
        episodic_contact_memory is not None
        and episodic_contact_memory.implementation_hash != implementation_hash
    ):
        raise ValueError("episodic contact memory implementation hash mismatch")
    request = {
        "schema_version": "rosclaw.simforge.g1_free_kick_request.v34",
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "learned_gait_qualification_hash": gait_qualification.qualification_hash,
        "sonic_qualification_hash": (
            None if sonic_qualification is None else sonic_qualification.qualification_hash
        ),
        "stadium_scene_hash": g1_stadium_scene_hash(asset_root, goal),
        "implementation_hash": implementation_hash,
        "runup_config": asdict(runup),
        "flow_config": asdict(flow),
        "sonic_runup_config": (None if sonic_config is None else asdict(sonic_config)),
        "approach_strike_candidate_hash": (
            None if residual_controller is None else residual_controller.candidate_hash
        ),
        "approach_strike_residual_config": (
            None if residual_config is None else asdict(residual_config)
        ),
        "football_motion_prior_hash": (
            None if football_motion_prior is None else football_motion_prior.prior_hash
        ),
        "ballistic_contact_impulse_actor_hash": (
            None
            if ballistic_contact_impulse_actor is None
            else ballistic_contact_impulse_actor.actor_hash
        ),
        "episodic_contact_memory_hash": (
            None if episodic_contact_memory is None else episodic_contact_memory.memory_hash
        ),
        "goal_spec": asdict(goal),
        "activation_ceiling": "SIM_ONLY",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    if football_outcome_model is not None:
        runtime_context_hash = _football_experiment_context_hash(
            flow=flow,
            sonic=sonic_config,
            runup=runup,
            goal=goal,
        )
        if runtime_context_hash != football_outcome_model.experiment_context_hash:
            raise ValueError("football outcome model experiment context mismatch")
    if ballistic_skill_memory is not None:
        from rosclaw.growth.ballistic_skill_memory import (
            ballistic_skill_experiment_context_hash,
        )

        if sonic_config is None:
            raise ValueError("ballistic skill memory requires a SONIC configuration")
        runtime_context_hash = ballistic_skill_experiment_context_hash(
            flow_config=asdict(flow),
            sonic_runup_config=asdict(sonic_config),
            runup_config=asdict(runup),
            goal_spec=asdict(goal),
            approach_strike_candidate_hash=(
                None if residual_controller is None else residual_controller.candidate_hash
            ),
        )
        if runtime_context_hash != ballistic_skill_memory.experiment_context_hash:
            raise ValueError("ballistic skill memory experiment context mismatch")
    request_path = root / "request.json"
    _write_json(request_path, request)
    result, trajectory = _simulate(
        asset_root=asset_root,
        gait_policy_root=gait_policy_root,
        runup=runup,
        flow=flow,
        goal=goal,
        sonic_model_root=sonic_model_root,
        sonic_config=sonic_config,
        residual_controller=residual_controller,
        proprioceptive_expert_router=proprioceptive_expert_router,
        football_outcome_model=football_outcome_model,
        football_motion_prior=football_motion_prior,
        ballistic_skill_memory=ballistic_skill_memory,
        ballistic_contact_impulse_actor=ballistic_contact_impulse_actor,
        episodic_contact_memory=episodic_contact_memory,
    )
    replay_result, replay_trajectory = _simulate(
        asset_root=asset_root,
        gait_policy_root=gait_policy_root,
        runup=runup,
        flow=flow,
        goal=goal,
        sonic_model_root=sonic_model_root,
        sonic_config=sonic_config,
        residual_controller=residual_controller,
        proprioceptive_expert_router=proprioceptive_expert_router,
        football_outcome_model=football_outcome_model,
        football_motion_prior=football_motion_prior,
        ballistic_skill_memory=ballistic_skill_memory,
        ballistic_contact_impulse_actor=ballistic_contact_impulse_actor,
        episodic_contact_memory=episodic_contact_memory,
    )
    digest = trajectory_digest(trajectory)
    strict_replay = bool(
        result.to_dict() == replay_result.to_dict()
        and digest == trajectory_digest(replay_trajectory)
    )
    trajectory_path = root / "g1-free-kick-trajectory.npz"
    np.savez_compressed(trajectory_path, **trajectory)  # type: ignore[arg-type]
    evidence = G1FreeKickEvidence(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        learned_gait_qualification=gait_qualification,
        sonic_qualification=sonic_qualification,
        sonic_runup_config=sonic_config,
        sonic_reference_digest=(
            None
            if flow.approach_provider != "sonic_fullbody"
            else str(trajectory["sonic_reference_digest"].item())
        ),
        approach_strike_candidate_hash=(
            None if residual_controller is None else residual_controller.candidate_hash
        ),
        approach_strike_residual_config=residual_config,
        stadium_scene_hash=request["stadium_scene_hash"],
        implementation_hash=implementation_hash,
        request_hash=hash_bytes(request_path.read_bytes()),
        trajectory_path=str(trajectory_path),
        trajectory_hash=_file_hash(trajectory_path),
        trajectory_digest=digest,
        strict_replay=strict_replay,
        runup_config=runup,
        flow_config=flow,
        goal_spec=goal,
        result=result,
    )
    _write_json(root / "g1-free-kick.json", evidence.to_dict())
    return evidence


def _simulate(
    *,
    asset_root: Path,
    gait_policy_root: Path,
    runup: G1LearnedRunupConfig,
    flow: G1FreeKickFlowConfig,
    goal: G1TrainingGoalSpec,
    sonic_model_root: Path | None,
    sonic_config: G1SonicRunupConfig | None,
    residual_controller: G1ApproachStrikeResidualController | None,
    proprioceptive_expert_router: G1ProprioceptiveExpertRouter | None,
    football_outcome_model: G1FootballOutcomeModel | None,
    football_motion_prior: G1FootballMotionPrior | None,
    ballistic_skill_memory: G1BallisticSkillMemory | None,
    ballistic_contact_impulse_actor: G1BallisticContactImpulseActor | None,
    episodic_contact_memory: G1EpisodicContactMemory | None,
) -> tuple[G1FreeKickResult, dict[str, np.ndarray]]:
    import mujoco

    asset = asset_root.expanduser().resolve()
    model = build_g1_stadium_model(asset, goal)
    data = mujoco.MjData(model)
    model.opt.timestep = runup.physics_dt_sec
    ids = _ModelIds.from_model(model)
    _configure_surface(model, goal)
    gait: G1LearnedRunupController | None = None
    sonic: G1SonicRunupController | None = None
    if flow.approach_provider == "sonic_fullbody":
        if sonic_model_root is None or sonic_config is None:
            raise ValueError("SONIC simulation requires a qualified model root and config")
        if not math.isclose(
            sonic_config.physics_dt_sec, runup.physics_dt_sec, abs_tol=1e-12
        ) or not math.isclose(sonic_config.policy_dt_sec, runup.control_dt_sec, abs_tol=1e-12):
            raise ValueError("SONIC and free-kick control clocks must match")
        sonic = G1SonicRunupController(sonic_model_root, sonic_config)
    else:
        gait = G1LearnedRunupController(gait_policy_root)
    data.qpos[:3] = (runup.start_x_m, runup.start_y_m, 0.793)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    if sonic is not None:
        data.qpos[7:36] = sonic.default_angles
    else:
        assert gait is not None
        data.qpos[7:22] = gait.default_lower
    data.qpos[ids.ball_qpos : ids.ball_qpos + 3] = (1.0, 0.0, goal.ball_radius_m)
    data.qpos[ids.ball_qpos + 3 : ids.ball_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    if sonic is not None:
        sonic.reset(data)

    trace: dict[str, list[Any]] = {
        "time": [],
        "joint_position": [],
        "joint_velocity": [],
        "joint_torque": [],
        "commanded_torque": [],
        "commanded_torque_peak_abs": [],
        "safety_projected_torque": [],
        "executed_torque": [],
        "torque_projection_applied": [],
        "learned_residual_torque": [],
        "learned_residual_accepted": [],
        "learned_residual_confidence": [],
        "loft_teacher_torque": [],
        "loft_teacher_force_n": [],
        "loft_teacher_forward_force_n": [],
        "loft_teacher_lateral_force_n": [],
        "loft_teacher_foot_vx_mps": [],
        "loft_teacher_foot_vy_mps": [],
        "loft_teacher_foot_vz_mps": [],
        "loft_teacher_active": [],
        "loft_teacher_pre_action_observation": [],
        "loft_teacher_pre_action_observation_valid": [],
        "ballistic_contact_impulse_actor_torque": [],
        "ballistic_contact_impulse_actor_lateral_force_n": [],
        "ballistic_contact_impulse_actor_vertical_force_n": [],
        "ballistic_contact_impulse_actor_foot_vy_mps": [],
        "ballistic_contact_impulse_actor_foot_vz_mps": [],
        "ballistic_contact_impulse_actor_active": [],
        "episodic_contact_memory_torque": [],
        "episodic_contact_memory_lateral_force_n": [],
        "episodic_contact_memory_vertical_force_n": [],
        "episodic_contact_memory_active": [],
        "episodic_contact_memory_context_supported": [],
        "episodic_contact_memory_launch_supported": [],
        "episodic_contact_memory_selected_seed_label": [],
        "episodic_contact_memory_context_distance": [],
        "joint_boundary_guard_correction": [],
        "joint_boundary_guard_active": [],
        "football_motion_prior_target_delta": [],
        "football_motion_prior_active": [],
        "ballistic_contact_target_delta": [],
        "ballistic_contact_residual_active": [],
        "ballistic_contact_torque_residual": [],
        "ballistic_contact_torque_residual_active": [],
        "cerebellar_recovery_active": [],
        "cerebellar_recovery_blend_fraction": [],
        "policy_action": [],
        "controller_target_velocity": [],
        "pelvis_pose": [],
        "pelvis_velocity": [],
        "torso_quaternion": [],
        "ball_pose": [],
        "ball_velocity": [],
        "right_foot_position": [],
        "right_foot_linear_velocity": [],
        "ball_contact_force_peak_n": [],
        "ball_contact_normal": [],
        "ball_contact_force_world": [],
        "controller_mode": [],
        "event_phase": [],
        "policy_phase": [],
        "goal_crossing": [],
    }
    hard_limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    loft_teacher_config = G1LoftTeacherConfig(
        target_vertical_speed_mps=flow.shot_loft_teacher_target_vz_mps,
        velocity_gain_n_per_mps=flow.shot_loft_teacher_gain_n_per_mps,
        maximum_vertical_force_n=flow.shot_loft_teacher_max_force_n,
        target_forward_speed_mps=flow.shot_loft_teacher_target_vx_mps,
        forward_velocity_gain_n_per_mps=(flow.shot_loft_teacher_forward_gain_n_per_mps),
        maximum_forward_force_n=flow.shot_loft_teacher_max_forward_force_n,
        target_lateral_speed_mps=flow.shot_loft_teacher_target_vy_mps,
        lateral_velocity_gain_n_per_mps=(flow.shot_loft_teacher_lateral_gain_n_per_mps),
        maximum_lateral_force_n=flow.shot_loft_teacher_max_lateral_force_n,
        start_policy_frame=flow.shot_loft_teacher_start_policy_frame,
        end_policy_frame=flow.shot_loft_teacher_end_policy_frame,
        maximum_foot_ball_distance_m=(flow.shot_loft_teacher_max_foot_ball_distance_m),
    )
    joint_boundary_guard_config = G1JointBoundaryGuardConfig(
        protected_joint_names=("right_ankle_pitch_joint",),
        margin_rad=0.02,
        prediction_horizon_sec=0.05,
        boundary_kp=80.0,
        boundary_kd=6.0,
        maximum_correction_nm=40.0,
    )
    right_ankle_pitch_index = np.asarray((10,), dtype=np.int64)
    joint_lower_limits = np.asarray(model.jnt_range[1:30, 0], dtype=np.float64)
    joint_upper_limits = np.asarray(model.jnt_range[1:30, 1], dtype=np.float64)
    boundary_guard_active_steps = 0
    boundary_guard_peak_correction = 0.0
    football_motion_prior_active_frames = 0
    football_motion_prior_peak_target_delta = 0.0
    football_motion_prior_velocity_active_frames = 0
    football_motion_prior_peak_target_velocity_delta = 0.0
    ballistic_contact_config = G1BallisticContactResidualConfig(
        right_leg_residual_rad=flow.ballistic_contact_residual_rad,
        contact_policy_frame=flow.ballistic_contact_policy_frame,
        lead_duration_sec=flow.ballistic_contact_lead_duration_sec,
        trail_duration_sec=flow.ballistic_contact_trail_duration_sec,
    )
    ballistic_contact_active_frames = 0
    ballistic_contact_peak_target_delta = 0.0
    ballistic_contact_torque_config = G1BallisticContactTorqueResidualConfig(
        right_leg_residual_nm=flow.ballistic_contact_torque_residual_nm,
        right_leg_preload_nm=flow.ballistic_contact_torque_preload_nm,
        right_leg_phase_offset_sec=flow.ballistic_contact_torque_phase_offset_sec,
        counterbalance_residual_nm=(flow.ballistic_counterbalance_torque_residual_nm),
        contact_policy_frame=flow.ballistic_contact_torque_policy_frame,
        lead_duration_sec=flow.ballistic_contact_torque_lead_duration_sec,
        trail_duration_sec=flow.ballistic_contact_torque_trail_duration_sec,
    )
    ballistic_contact_torque_active_frames = 0
    ballistic_contact_torque_peak = 0.0
    ballistic_contact_impulse_actor_active_frames = 0
    ballistic_contact_impulse_actor_peak_torque = 0.0
    ballistic_contact_impulse_actor_peak_lateral_force = 0.0
    ballistic_contact_impulse_actor_peak_vertical_force = 0.0
    ballistic_contact_impulse_actor_out_of_envelope_frames = 0
    ballistic_contact_impulse_actor_peak_desired_lateral_launch_speed = 0.0
    ballistic_contact_impulse_actor_peak_desired_vertical_launch_speed = 0.0
    episodic_contact_memory_active_frames = 0
    episodic_contact_memory_out_of_support_frames = 0
    episodic_contact_memory_peak_torque = 0.0
    episodic_contact_memory_peak_lateral_force = 0.0
    episodic_contact_memory_peak_vertical_force = 0.0
    episodic_contact_memory_peak_desired_lateral_launch_speed = 0.0
    episodic_contact_memory_peak_desired_vertical_launch_speed = 0.0
    episodic_contact_memory_selected_seed_label: int | None = None
    episodic_contact_memory_mixed_contexts = False
    episodic_contact_memory_peak_context_distance = 0.0
    finite = True
    saturation = False
    saturation_steps = 0
    peak_demand_ratio = 0.0
    torque_authority_projection_steps = 0
    torque_authority_projection_peak_correction = 0.0
    torque_authority_preprojection_peak_demand_ratio = 0.0
    contact_task_authority_projection_steps = 0
    contact_task_authority_scale_min = 1.0
    torque_violation = False
    joint_violation = False
    physics_steps = 0
    runup_min_height = float(data.qpos[2])
    runup_peak_tilt = 0.0
    runup_start_x = float(data.qpos[0])
    initial_ball_distance = float(
        np.linalg.norm(data.qpos[ids.ball_qpos : ids.ball_qpos + 2] - data.qpos[:2])
    )
    runup_peak_speed = 0.0
    substeps = int(round(runup.control_dt_sec / runup.physics_dt_sec))
    runup_frames = (
        sonic.config.execution_frames
        if sonic is not None
        else int(round(runup.total_duration_sec / runup.control_dt_sec))
    )
    last_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)

    def project_authority(raw_torque: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal peak_demand_ratio
        nonlocal saturation
        nonlocal saturation_steps
        nonlocal torque_authority_projection_steps
        nonlocal torque_authority_projection_peak_correction
        nonlocal torque_authority_preprojection_peak_demand_ratio

        raw_value = np.asarray(raw_torque, dtype=np.float64)
        preprojection_ratio = float(np.max(np.abs(raw_value) / hard_limits))
        torque_authority_preprojection_peak_demand_ratio = max(
            torque_authority_preprojection_peak_demand_ratio,
            preprojection_ratio,
        )
        projected = raw_value
        if flow.torque_authority_projection_ratio > 0.0:
            projection = project_g1_torque_authority(
                commanded_torque_nm=raw_value,
                hard_limits_nm=hard_limits,
                maximum_demand_ratio=flow.torque_authority_projection_ratio,
            )
            projected = projection.projected_torque_nm
            torque_authority_projection_steps += int(projection.active)
            torque_authority_projection_peak_correction = max(
                torque_authority_projection_peak_correction,
                float(np.max(np.abs(projection.correction_nm))),
            )
        demand_ratio = float(np.max(np.abs(projected) / hard_limits))
        peak_demand_ratio = max(peak_demand_ratio, demand_ratio)
        saturated_step = bool(demand_ratio >= 0.999)
        saturation = saturation or saturated_step
        saturation_steps += int(saturated_step)
        return np.clip(projected, -hard_limits, hard_limits)

    for frame in range(runup_frames):
        if sonic is not None:
            sonic.update(data, frame)
            mode = 5
        else:
            command, mode = _runup_command(frame * runup.control_dt_sec, runup)
        runup_event = (
            G1FootballEventPhase.ALIGN_BRAKE
            if frame >= runup_frames - int(round(0.60 / runup.control_dt_sec))
            else G1FootballEventPhase.APPROACH
        )
        if sonic is not None:
            runup_target = sonic.target.copy()
            baseline = sonic.raw_torque(data)
        else:
            assert gait is not None
            runup_target = np.zeros(29, dtype=np.float64)
            runup_target[:15] = gait.target
            baseline = np.concatenate(
                (
                    (gait.target - data.qpos[7:22]) * gait.lower_kp
                    - data.qvel[6:21] * gait.lower_kd,
                    -data.qpos[22:36] * gait.arm_kp - data.qvel[21:35] * gait.arm_kd,
                )
            )
        residual, residual_accepted, residual_confidence = _residual_for_frame(
            residual_controller,
            data=data,
            ids=ids,
            target=runup_target,
            event_phase=runup_event,
            baseline_torque=baseline,
        )
        # The neural target updates at 50 Hz; the stabilizing PD loop closes at
        # the 500 Hz physics rate. Holding torque for an entire policy interval
        # destabilizes this model even though holding the target is intended.
        for _ in range(substeps):
            if sonic is not None:
                raw = sonic.raw_torque(data) + residual
            else:
                assert gait is not None
                raw_lower = (gait.target - data.qpos[7:22]) * gait.lower_kp - data.qvel[
                    6:21
                ] * gait.lower_kd
                raw_arms = -data.qpos[22:36] * gait.arm_kp - data.qvel[21:35] * gait.arm_kd
                raw = np.concatenate((raw_lower, raw_arms)) + residual
            last_torque = project_authority(raw)
            torque_violation = torque_violation or bool(np.any(np.abs(last_torque) > hard_limits))
            data.ctrl[:] = last_torque
            mujoco.mj_step(model, data)
            physics_steps += 1
        if sonic is not None:
            sonic.observe(data)
        else:
            assert gait is not None
            gait.update(data, command)
        roll, pitch = _roll_pitch(data.xquat[ids.torso])
        runup_min_height = min(runup_min_height, float(data.qpos[2]))
        runup_peak_tilt = max(runup_peak_tilt, abs(roll), abs(pitch))
        runup_peak_speed = max(runup_peak_speed, float(np.linalg.norm(data.qvel[:2])))
        finite = finite and _finite(data)
        joint_violation = joint_violation or _joint_violation(model, data)
        _append_trace(
            trace,
            data,
            ids,
            last_torque,
            runup_target,
            np.zeros(29, dtype=np.float64),
            mode,
            0.0,
            False,
            runup_event,
            raw,
            residual,
            residual_accepted,
            residual_confidence,
        )

    from rosclaw.growth.proprioceptive_expert_router import strike_handoff_features

    handoff_features = strike_handoff_features(
        np.asarray(data.qpos[:7], dtype=np.float64),
        np.asarray(data.qvel[6:35], dtype=np.float64),
    )
    handoff_yaw = _signed_yaw(data.qpos[3:7])
    ballistic_skill_selection = None
    if ballistic_skill_memory is not None:
        from rosclaw.growth.ballistic_skill_memory import ballistic_handoff_state

        if sonic_config is None:
            raise ValueError("ballistic skill memory requires a SONIC configuration")
        full_handoff_state = ballistic_handoff_state(
            pelvis_pose_xyz_wxyz=np.asarray(data.qpos[:7], dtype=np.float64),
            joint_position=np.asarray(data.qpos[7:36], dtype=np.float64),
            pelvis_velocity_linear_angular=np.asarray(data.qvel[:6], dtype=np.float64),
            joint_velocity=np.asarray(data.qvel[6:35], dtype=np.float64),
        )
        ballistic_skill_selection = ballistic_skill_memory.select(full_handoff_state)
        if ballistic_skill_selection.abstained:
            raise ValueError(
                "ballistic skill memory rejected the measured handoff: "
                f"{ballistic_skill_selection.failure_code}; "
                f"distance={ballistic_skill_selection.nearest_distance:.6f}"
            )
        if ballistic_skill_selection.selected_skill_id != flow.ballistic_skill_id:
            raise ValueError("ballistic skill memory selected a different skill island")
        skill = ballistic_skill_memory.prototype(flow.ballistic_skill_id or "")
        if sonic_config.planner_seed != skill.planner_seed:
            raise ValueError("ballistic skill planner seed binding mismatch")
        if not np.allclose(
            np.asarray(flow.ballistic_contact_residual_rad),
            np.asarray(skill.action_rad),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("ballistic skill action binding mismatch")
        if flow.ballistic_contact_policy_frame != skill.contact_policy_frame:
            raise ValueError("ballistic skill contact-frame binding mismatch")
        if not math.isclose(
            flow.post_contact_damping_scale,
            skill.post_contact_damping_scale,
            abs_tol=1e-12,
        ):
            raise ValueError("ballistic skill damping binding mismatch")
    router_selection = (
        None
        if proprioceptive_expert_router is None
        else proprioceptive_expert_router.select(handoff_features)
    )
    outcome_decision = (
        None if football_outcome_model is None else football_outcome_model.decide(handoff_features)
    )
    initial_retry_recommended = bool(
        outcome_decision is not None and outcome_decision.retry_recommended
    )
    retry_recovery_executed = False
    retry_initial_speed: float | None = None
    retry_final_speed: float | None = None
    if (
        outcome_decision is not None
        and outcome_decision.retry_recommended
        and flow.football_retry_recovery_duration_sec > 0.0
    ):
        if sonic is None:
            raise ValueError("football retry recovery requires the SONIC provider")
        retry_recovery_executed = True
        retry_initial_speed = float(np.linalg.norm(data.qvel[:2]))
        recovery_frames = int(
            round(flow.football_retry_recovery_duration_sec / runup.control_dt_sec)
        )
        sonic.extend_stationary_recovery(recovery_frames)
        for recovery_frame in range(recovery_frames):
            sonic.update_recovery_extension(data, recovery_frame)
            retry_target = sonic.target.copy()
            baseline = sonic.raw_torque(data)
            residual, residual_accepted, residual_confidence = _residual_for_frame(
                residual_controller,
                data=data,
                ids=ids,
                target=retry_target,
                event_phase=G1FootballEventPhase.ALIGN_BRAKE,
                baseline_torque=baseline,
            )
            for _ in range(substeps):
                raw = sonic.raw_torque(data) + residual
                last_torque = project_authority(raw)
                torque_violation = torque_violation or bool(
                    np.any(np.abs(last_torque) > hard_limits)
                )
                data.ctrl[:] = last_torque
                mujoco.mj_step(model, data)
                physics_steps += 1
            sonic.observe(data)
            roll, pitch = _roll_pitch(data.xquat[ids.torso])
            runup_min_height = min(runup_min_height, float(data.qpos[2]))
            runup_peak_tilt = max(runup_peak_tilt, abs(roll), abs(pitch))
            runup_peak_speed = max(runup_peak_speed, float(np.linalg.norm(data.qvel[:2])))
            finite = finite and _finite(data)
            joint_violation = joint_violation or _joint_violation(model, data)
            _append_trace(
                trace,
                data,
                ids,
                last_torque,
                retry_target,
                np.zeros(29, dtype=np.float64),
                6,
                0.0,
                False,
                G1FootballEventPhase.ALIGN_BRAKE,
                raw,
                residual,
                residual_accepted,
                residual_confidence,
            )
        handoff_features = strike_handoff_features(
            np.asarray(data.qpos[:7], dtype=np.float64),
            np.asarray(data.qvel[6:35], dtype=np.float64),
        )
        handoff_yaw = _signed_yaw(data.qpos[3:7])
        retry_final_speed = float(np.linalg.norm(data.qvel[:2]))
        outcome_decision = football_outcome_model.decide(handoff_features)

    runup_end_x = float(data.qpos[0])
    runup_terminal_speed = float(np.linalg.norm(data.qvel[:2]))
    kick_min_height = float(data.qpos[2])
    kick_peak_tilt = 0.0
    handoff_time = float(data.time)
    pause_run_sec = 0.0
    pause_max_sec = 0.0
    handoff_min_forward_speed = math.inf
    low_forward_run_sec = 0.0
    low_forward_max_sec = 0.0
    state_type, output_type, policy_type, mujoco_to_isaac = _load_robonaldo(asset)
    state = state_type(29)
    output = output_type(29)
    _fill_state(state, model, data, ids)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = policy_type(state, output)
    # RoboNaldo's motion prior uses target distance to generate its strike
    # pose.  A farther scoring plane must not silently flatten a qualified
    # muscle-memory motion.  A non-zero reference plane preserves that strike
    # morphology while the physical ball and scoring goal remain untouched.
    # The value is request-hashed and actor-context-bound for strict replay.
    motion_reference_x = flow.shot_reference_plane_x_m or goal.plane_x_m
    motion_reference_y = (
        goal.target_y_m
        if flow.shot_reference_target_y_m is None
        else flow.shot_reference_target_y_m
    )
    motion_reference_z = (
        goal.target_z_m
        if flow.shot_reference_target_z_m is None
        else flow.shot_reference_target_z_m
    )
    policy.target_pos_w = np.asarray(
        (
            motion_reference_x,
            motion_reference_y + flow.aim_bias_y_m,
            motion_reference_z + flow.aim_bias_z_m,
        ),
        dtype=np.float32,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        policy.enter()
    if outcome_decision is not None:
        phase_start = outcome_decision.selected_phase_start_frame
        contextual_phase_expert = False
    elif router_selection is None:
        phase_start, contextual_phase_expert = _select_contextual_phase(flow, handoff_yaw)
    else:
        phase_start = router_selection.phase_start_frame
        contextual_phase_expert = phase_start != flow.kick_phase_start_frame
    policy._ref_anchor_world_origin = policy._init_to_world @ policy.motion_body_pos[
        phase_start, 9
    ].astype(np.float64)
    phase_target = np.asarray(
        policy.motion_joint_pos[phase_start][mujoco_to_isaac], dtype=np.float64
    )
    phase_velocity = np.asarray(
        policy.motion_joint_vel[phase_start][mujoco_to_isaac], dtype=np.float64
    )
    bridge_entry = data.qpos[7:36].copy()
    bridge_entry_velocity = data.qvel[6:35].copy()
    bridge_delta = phase_target - bridge_entry
    bridge_max = float(np.max(np.abs(bridge_delta)))
    bridge_rms = float(np.sqrt(np.mean(np.square(bridge_delta))))
    bridge_frames = int(round(flow.bridge_duration_sec / runup.control_dt_sec))
    transition_bridge = G1VelocityMatchedTransitionBridge(
        entry_position=bridge_entry,
        entry_velocity=bridge_entry_velocity,
        exit_position=phase_target,
        exit_velocity=phase_velocity,
        config=G1TransitionBridgeConfig(
            duration_sec=flow.bridge_duration_sec,
            entry_velocity_scale=flow.bridge_entry_velocity_scale,
            exit_velocity_scale=flow.bridge_exit_velocity_scale,
            maximum_boundary_velocity_rad_s=flow.bridge_boundary_velocity_limit_rad_s,
        ),
    )
    bridge_peak_target_acceleration_rms = 0.0
    # Gain scheduling keeps the mid-stride skill splice inside the same hard
    # torque envelope as the learned gait. The quintic bridge still reaches
    # the motion prior without an impulsive full-gain catch-up.
    bridge_kp = np.minimum(
        np.asarray(policy.kps, dtype=np.float64),
        hard_limits * 0.58 / np.maximum(np.abs(bridge_delta), 0.05),
    )
    bridge_kd = np.asarray(policy.kds, dtype=np.float64) * 0.72
    for bridge_frame in range(bridge_frames):
        bridge_sample = transition_bridge.sample((bridge_frame + 1) * runup.control_dt_sec)
        bridge_target = bridge_sample.position
        bridge_target_velocity = bridge_sample.velocity
        bridge_peak_target_acceleration_rms = max(
            bridge_peak_target_acceleration_rms,
            float(np.sqrt(np.mean(np.square(bridge_sample.acceleration)))),
        )
        baseline = (bridge_target - data.qpos[7:36]) * bridge_kp + (
            bridge_target_velocity - data.qvel[6:35]
        ) * bridge_kd
        residual, residual_accepted, residual_confidence = _residual_for_frame(
            residual_controller,
            data=data,
            ids=ids,
            target=bridge_target,
            event_phase=G1FootballEventPhase.PLANT_BRIDGE,
            baseline_torque=baseline,
        )
        for _ in range(substeps):
            raw = (
                (bridge_target - data.qpos[7:36]) * bridge_kp
                + (bridge_target_velocity - data.qvel[6:35]) * bridge_kd
                + residual
            )
            last_torque = project_authority(raw)
            torque_violation = torque_violation or bool(np.any(np.abs(last_torque) > hard_limits))
            data.ctrl[:] = last_torque
            mujoco.mj_step(model, data)
            physics_steps += 1
        _fill_state(state, model, data, ids)
        if bridge_frame >= bridge_frames - flow.history_prime_frames:
            policy.time_step = (
                policy.WARMUP_STEPS + phase_start - (bridge_frames - 1 - bridge_frame)
            )
            with contextlib.redirect_stdout(io.StringIO()):
                policy._build_obs()
        roll, pitch = _roll_pitch(data.xquat[ids.torso])
        kick_min_height = min(kick_min_height, float(data.qpos[2]))
        kick_peak_tilt = max(kick_peak_tilt, abs(roll), abs(pitch))
        finite = finite and _finite(data)
        joint_violation = joint_violation or _joint_violation(model, data)
        pause_run_sec, pause_max_sec = _update_motion_pause(
            data, runup.control_dt_sec, pause_run_sec, pause_max_sec
        )
        (
            handoff_min_forward_speed,
            low_forward_run_sec,
            low_forward_max_sec,
        ) = _update_forward_continuity(
            data,
            runup.control_dt_sec,
            handoff_min_forward_speed,
            low_forward_run_sec,
            low_forward_max_sec,
        )
        _append_trace(
            trace,
            data,
            ids,
            last_torque,
            bridge_target,
            bridge_target_velocity,
            3,
            phase_start / max(1, int(policy.motion_total_steps) - 1),
            False,
            G1FootballEventPhase.PLANT_BRIDGE,
            raw,
            residual,
            residual_accepted,
            residual_confidence,
        )
    bridge_entry_velocity_rms = float(np.sqrt(np.mean(np.square(transition_bridge.entry_velocity))))
    bridge_target_exit_velocity_rms = float(
        np.sqrt(np.mean(np.square(transition_bridge.exit_velocity)))
    )
    bridge_exit_velocity_error_rms = float(
        np.sqrt(np.mean(np.square(data.qvel[6:35] - transition_bridge.exit_velocity)))
    )
    policy.time_step = policy.WARMUP_STEPS + phase_start
    parameters = ShotParameters(
        stance_offset_y=-0.035,
        pelvis_yaw_offset=flow.shot_pelvis_yaw_offset_rad,
        com_shift_y=flow.shot_com_shift_y_m,
        swing_amplitude=flow.shot_swing_amplitude,
        swing_speed_scale=flow.shot_swing_speed_scale,
        foot_yaw_offset=flow.shot_foot_yaw_offset_rad,
        foot_pitch_offset=flow.shot_foot_pitch_offset_rad,
        loft_synergy=flow.shot_loft_synergy_rad,
        contact_phase_offset=flow.shot_contact_phase_offset,
        recovery_step_length=flow.shot_recovery_step_length_m,
        recovery_step_yaw=flow.shot_recovery_step_yaw_rad,
        policy_type="skill_graph",
    )
    cerebellar_recovery = None
    if flow.shared_cerebellar_recovery_enabled:
        # Use the exact same Body/motion-bound controller and trained config
        # as the coupled passer and shooter.  The scenario supplies only the
        # calibrated SIM regime; the controller remains contact/landing gated.
        from rosclaw.simforge.g1_two_player_relay import _base_scenario

        recovery_backend = G1MuJoCoBackend(asset_root=asset)
        cerebellar_recovery = recovery_backend.build_cerebellar_recovery_controller(
            _base_scenario(),
            shared_post_impact_recovery_config(),
        )
        cerebellar_recovery.reset()
    cerebellar_recovery_active_frames = 0
    cerebellar_recovery_peak_blend_fraction = 0.0
    # Slower swings need extra policy frames. Faster swings must not shorten
    # the physical observation tail: the ball still needs time to cross the
    # goal plane and the body still needs a full recovery window.
    slowdown_frames = max(0, int(round(245 * (1.0 - parameters.swing_speed_scale))))
    total_kick_frames = int(policy.motion_total_steps) - phase_start + 50 + slowdown_frames
    phase_frames = int(round(parameters.contact_phase_offset / runup.control_dt_sec))
    phase_hold_remaining = max(0, phase_frames)
    phase_advance = max(0, -phase_frames)
    phase_adjusted = phase_frames == 0
    last_target = phase_target.copy()
    contact_time: float | None = None
    contact_point: tuple[float, float, float] | None = None
    contact_height_relative_ball_center: float | None = None
    contact_foot_position: tuple[float, float, float] | None = None
    contact_foot_velocity: tuple[float, float, float] | None = None
    contact_normal: tuple[float, float, float] | None = None
    contact_force_world: tuple[float, float, float] | None = None
    contact_peak_force: float | None = None
    launch_velocity: tuple[float, float, float] | None = None
    launch_speed = 0.0
    ball_apex_height = goal.ball_radius_m
    maximum_ball_speed = 0.0
    crossing: tuple[float, float, float] | None = None
    net_capture: tuple[float, float, float] | None = None
    deepest_net_point: tuple[float, float, float] | None = None
    previous_ball = data.qpos[ids.ball_qpos : ids.ball_qpos + 3].copy()

    for frame in range(total_kick_frames):
        _fill_state(state, model, data, ids)
        motion_prior_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        motion_prior_velocity_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        motion_prior_active = False
        motion_prior_velocity_active = False
        target_velocity: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        ballistic_contact_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        ballistic_contact_active = False
        current_policy_frame = max(0, int(policy.time_step) - int(policy.WARMUP_STEPS))
        repeat = _policy_repeat_count(parameters.swing_speed_scale, current_policy_frame, frame)
        # Compress only the pre-swing loading section.  This preserves the
        # learned contact/follow-through clock while removing the visually
        # slow preparation that made a continuous run look like stop-and-kick.
        if 185 <= current_policy_frame < 235:
            repeat += max(
                0,
                _policy_repeat_count(
                    flow.shot_load_speed_scale,
                    current_policy_frame,
                    frame,
                )
                - 1,
            )
        if not phase_adjusted and current_policy_frame >= 185:
            if phase_hold_remaining:
                repeat = 0
                phase_hold_remaining -= 1
                phase_adjusted = phase_hold_remaining == 0
            else:
                repeat += phase_advance
                phase_adjusted = True
        if repeat:
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(repeat):
                    policy.run()
            target = np.asarray(output.actions, dtype=np.float64).copy()
            kp = np.asarray(output.kps, dtype=np.float64) * 0.98
            kd = np.asarray(output.kds, dtype=np.float64) * 0.98
            policy_frame = max(0, int(policy.time_step) - int(policy.WARMUP_STEPS))
            target = _adapt_target(
                target=target,
                default=np.asarray(policy.default_q_mj, dtype=np.float64),
                parameters=parameters,
                policy_frame=policy_frame,
            )
            if (
                loft_teacher_config.enabled
                and loft_teacher_config.start_policy_frame
                <= policy_frame
                <= loft_teacher_config.end_policy_frame
            ):
                target[10] += flow.shot_loft_teacher_foot_pitch_bonus_rad
            if football_motion_prior is not None:
                target, motion_prior_delta, motion_prior_active = (
                    blend_g1_football_motion_prior_target(
                        target=target,
                        prior=football_motion_prior,
                        policy_frame=policy_frame,
                        contact_policy_frame=(flow.football_motion_prior_contact_policy_frame),
                        control_dt_sec=runup.control_dt_sec,
                        blend=flow.football_motion_prior_blend,
                    )
                )
                target_velocity, motion_prior_velocity_delta, motion_prior_velocity_active = (
                    blend_g1_football_motion_prior_velocity(
                        target_velocity=target_velocity,
                        prior=football_motion_prior,
                        policy_frame=policy_frame,
                        contact_policy_frame=(flow.football_motion_prior_contact_policy_frame),
                        control_dt_sec=runup.control_dt_sec,
                        blend=flow.football_motion_prior_velocity_blend,
                    )
                )
                motion_prior_active = motion_prior_active or motion_prior_velocity_active
                football_motion_prior_active_frames += int(motion_prior_active)
                football_motion_prior_velocity_active_frames += int(motion_prior_velocity_active)
                football_motion_prior_peak_target_delta = max(
                    football_motion_prior_peak_target_delta,
                    float(np.max(np.abs(motion_prior_delta))),
                )
                football_motion_prior_peak_target_velocity_delta = max(
                    football_motion_prior_peak_target_velocity_delta,
                    float(np.max(np.abs(motion_prior_velocity_delta))),
                )
            target, ballistic_contact_delta, ballistic_contact_active = (
                blend_g1_ballistic_contact_target(
                    target=target,
                    policy_frame=policy_frame,
                    control_dt_sec=runup.control_dt_sec,
                    config=ballistic_contact_config,
                )
            )
            ballistic_contact_active_frames += int(ballistic_contact_active)
            ballistic_contact_peak_target_delta = max(
                ballistic_contact_peak_target_delta,
                float(np.max(np.abs(ballistic_contact_delta))),
            )
        else:
            target = last_target.copy()
            kp = np.asarray(policy.kps, dtype=np.float64) * 0.98
            kd = np.asarray(policy.kds, dtype=np.float64) * 0.98
            policy_frame = current_policy_frame
        cerebellar_recovery_active = False
        cerebellar_recovery_blend_fraction = 0.0
        if cerebellar_recovery is not None:
            support = _contact_observation(model, data, ids)
            recovery = cerebellar_recovery.adapt_target(
                target=target,
                policy_frame=policy_frame,
                timestamp_sec=float(data.time),
                ball_contact_detected=contact_time is not None,
                left_support=support.left_floor,
                right_support=support.right_floor,
            )
            target = recovery.target
            terminal_slice = {
                "whole_body": slice(None),
                "legs": slice(0, 12),
                "upper_body": slice(12, None),
            }[recovery.terminal_damping_joint_group]
            kp[terminal_slice] *= recovery.terminal_kp_scale
            kd[terminal_slice] *= recovery.terminal_kd_scale
            cerebellar_recovery_active = recovery.active
            cerebellar_recovery_blend_fraction = recovery.blend_fraction
            cerebellar_recovery_active_frames += int(recovery.active)
            cerebellar_recovery_peak_blend_fraction = max(
                cerebellar_recovery_peak_blend_fraction,
                recovery.blend_fraction,
            )
        ballistic_contact_torque, ballistic_contact_torque_active = (
            g1_ballistic_contact_torque_residual(
                policy_frame=policy_frame,
                control_dt_sec=runup.control_dt_sec,
                config=ballistic_contact_torque_config,
            )
        )
        ballistic_contact_torque_active_frames += int(ballistic_contact_torque_active)
        ballistic_contact_torque_peak = max(
            ballistic_contact_torque_peak,
            float(np.max(np.abs(ballistic_contact_torque))),
        )
        scheduled_contact_gain = bool(
            flow.strike_gain_schedule_start_policy_frame > 0
            and policy_frame >= flow.strike_gain_schedule_start_policy_frame
        )
        if contact_time is not None or scheduled_contact_gain:
            gain_scale = np.asarray(
                (
                    flow.follow_through_gain_scales
                    if contact_time is not None
                    else flow.strike_gain_scales
                ),
                dtype=np.float64,
            )
            if contact_time is not None and retry_recovery_executed:
                gain_scale = gain_scale * flow.football_retry_follow_through_gain_scale
            kp = kp * gain_scale
            kd = kd * gain_scale
            if contact_time is not None and flow.post_contact_damping_scale > 1.0:
                # Preserve impact and early follow-through, then smoothly add
                # velocity feedback for recovery.  The torque still passes
                # through the same hard projection and boundary guard.
                recovery_progress = np.clip(
                    (float(data.time) - contact_time - flow.post_contact_damping_delay_sec)
                    / flow.post_contact_damping_ramp_sec,
                    0.0,
                    1.0,
                )
                recovery_blend = (
                    recovery_progress * recovery_progress * (3.0 - 2.0 * recovery_progress)
                )
                kd = kd * (1.0 + (flow.post_contact_damping_scale - 1.0) * recovery_blend)
        last_target = target.copy()
        if contact_time is not None:
            residual_event = G1FootballEventPhase.RECOVERY
        elif policy_frame < 185:
            residual_event = G1FootballEventPhase.LOAD
        else:
            residual_event = G1FootballEventPhase.SWING
        baseline = (target - data.qpos[7:36]) * kp + (target_velocity - data.qvel[6:35]) * kd
        residual, residual_accepted, residual_confidence = _residual_for_frame(
            residual_controller,
            data=data,
            ids=ids,
            target=target,
            event_phase=residual_event,
            baseline_torque=baseline,
        )
        loft_teacher_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        loft_teacher_force = 0.0
        loft_teacher_forward_force = 0.0
        loft_teacher_lateral_force = 0.0
        loft_teacher_foot_vx = 0.0
        loft_teacher_foot_vy = 0.0
        loft_teacher_foot_vz = 0.0
        loft_teacher_active = False
        loft_teacher_pre_action_observation: NDArray[np.float64] = np.zeros(11, dtype=np.float64)
        loft_teacher_pre_action_observation_valid = False
        impulse_actor_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        impulse_actor_lateral_force = 0.0
        impulse_actor_vertical_force = 0.0
        impulse_actor_foot_vy = 0.0
        impulse_actor_foot_vz = 0.0
        impulse_actor_active = False
        episodic_memory_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        episodic_memory_lateral_force = 0.0
        episodic_memory_vertical_force = 0.0
        episodic_memory_active = False
        episodic_memory_evaluated = False
        episodic_memory_context_supported = True
        episodic_memory_launch_supported = True
        episodic_memory_selected_seed_label: int | None = None
        episodic_memory_active_seed_label: int | None = None
        episodic_memory_context_distance = 0.0
        boundary_guard_correction: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        boundary_guard_active = False
        pre_guard_raw: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        commanded_torque_peak_abs: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        contact_in_frame = False
        frame_contact_force_peak = 0.0
        frame_contact_normal: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        frame_contact_force_world: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        for _ in range(substeps):
            loft_effect = g1_loft_teacher_effect(
                model=model,
                data=data,
                right_ankle_body_id=ids.right_ankle,
                config=loft_teacher_config,
                policy_frame=policy_frame,
                contact_observed=contact_time is not None,
                ball_position=np.asarray(
                    data.qpos[ids.ball_qpos : ids.ball_qpos + 3], dtype=np.float64
                ),
            )
            # The teacher can enter and leave the proximity gate inside one
            # 20 ms control frame.  Preserve the strongest 2 ms sample and an
            # any-substep activation bit for evidence, while applying only the
            # current substep torque to physics.  Recording only the final
            # substep previously allowed a teacher-altered trajectory to claim
            # zero active frames and zero peak force.
            loft_effect_torque = loft_effect.torque
            if loft_effect.active:
                if not loft_teacher_pre_action_observation_valid:
                    loft_teacher_pre_action_observation = g1_episodic_contact_observation(
                        data=data,
                        right_ankle_body_id=ids.right_ankle,
                        torso_body_id=ids.torso,
                        ball_position=np.asarray(
                            data.qpos[ids.ball_qpos : ids.ball_qpos + 3],
                            dtype=np.float64,
                        ),
                    )
                    loft_teacher_pre_action_observation_valid = True
                loft_teacher_active = True
                if abs(loft_effect.vertical_force_n) >= abs(loft_teacher_force):
                    loft_teacher_force = loft_effect.vertical_force_n
                    loft_teacher_foot_vz = loft_effect.foot_vertical_speed_mps
                if loft_effect.forward_force_n >= loft_teacher_forward_force:
                    loft_teacher_forward_force = loft_effect.forward_force_n
                    loft_teacher_foot_vx = loft_effect.foot_forward_speed_mps
                if abs(loft_effect.lateral_force_n) >= abs(loft_teacher_lateral_force):
                    loft_teacher_lateral_force = loft_effect.lateral_force_n
                    loft_teacher_foot_vy = loft_effect.foot_lateral_speed_mps
                if float(np.max(np.abs(loft_effect_torque))) >= float(
                    np.max(np.abs(loft_teacher_torque))
                ):
                    loft_teacher_torque = loft_effect_torque.copy()
            impulse_effect = (
                None
                if ballistic_contact_impulse_actor is None
                else g1_ballistic_contact_impulse_effect(
                    model=model,
                    data=data,
                    right_ankle_body_id=ids.right_ankle,
                    actor=ballistic_contact_impulse_actor,
                    policy_frame=policy_frame,
                    contact_observed=contact_time is not None,
                    ball_position=np.asarray(
                        data.qpos[ids.ball_qpos : ids.ball_qpos + 3],
                        dtype=np.float64,
                    ),
                    ball_velocity=np.asarray(
                        data.qvel[ids.ball_qvel : ids.ball_qvel + 3],
                        dtype=np.float64,
                    ),
                    goal_plane_x_m=goal.plane_x_m,
                    target_y_m=goal.target_y_m,
                    target_z_m=goal.target_z_m,
                )
            )
            impulse_effect_torque = (
                np.zeros(29, dtype=np.float64) if impulse_effect is None else impulse_effect.torque
            )
            if impulse_effect is not None and impulse_effect.active:
                impulse_actor_active = True
                if abs(impulse_effect.lateral_force_n) >= abs(impulse_actor_lateral_force):
                    impulse_actor_lateral_force = impulse_effect.lateral_force_n
                    impulse_actor_foot_vy = impulse_effect.foot_lateral_speed_mps
                if abs(impulse_effect.vertical_force_n) >= abs(impulse_actor_vertical_force):
                    impulse_actor_vertical_force = impulse_effect.vertical_force_n
                    impulse_actor_foot_vz = impulse_effect.foot_vertical_speed_mps
                if float(np.max(np.abs(impulse_effect_torque))) >= float(
                    np.max(np.abs(impulse_actor_torque))
                ):
                    impulse_actor_torque = impulse_effect_torque.copy()
            if impulse_effect is not None and impulse_effect.target_conditioned:
                ballistic_contact_impulse_actor_out_of_envelope_frames += int(
                    not impulse_effect.launch_envelope_supported
                )
                ballistic_contact_impulse_actor_peak_desired_lateral_launch_speed = max(
                    ballistic_contact_impulse_actor_peak_desired_lateral_launch_speed,
                    abs(impulse_effect.desired_lateral_launch_speed_mps),
                )
                ballistic_contact_impulse_actor_peak_desired_vertical_launch_speed = max(
                    ballistic_contact_impulse_actor_peak_desired_vertical_launch_speed,
                    abs(impulse_effect.desired_vertical_launch_speed_mps),
                )
            episodic_effect = (
                None
                if episodic_contact_memory is None
                else g1_episodic_contact_effect(
                    model=model,
                    data=data,
                    right_ankle_body_id=ids.right_ankle,
                    torso_body_id=ids.torso,
                    memory=episodic_contact_memory,
                    policy_frame=policy_frame,
                    contact_observed=contact_time is not None,
                    ball_position=np.asarray(
                        data.qpos[ids.ball_qpos : ids.ball_qpos + 3],
                        dtype=np.float64,
                    ),
                    ball_velocity=np.asarray(
                        data.qvel[ids.ball_qvel : ids.ball_qvel + 3],
                        dtype=np.float64,
                    ),
                    goal_plane_x_m=goal.plane_x_m,
                    target_y_m=goal.target_y_m,
                    target_z_m=goal.target_z_m,
                )
            )
            episodic_effect_torque = (
                np.zeros(29, dtype=np.float64)
                if episodic_effect is None
                else episodic_effect.torque
            )
            if (
                episodic_effect is not None
                and episodic_effect.selected_context_seed_label is not None
            ):
                episodic_memory_evaluated = True
                episodic_memory_context_supported = bool(
                    episodic_memory_context_supported and episodic_effect.context_supported
                )
                episodic_memory_launch_supported = bool(
                    episodic_memory_launch_supported and episodic_effect.launch_envelope_supported
                )
                episodic_memory_selected_seed_label = episodic_effect.selected_context_seed_label
                episodic_memory_context_distance = max(
                    episodic_memory_context_distance,
                    episodic_effect.context_distance,
                )
                episodic_contact_memory_peak_desired_lateral_launch_speed = max(
                    episodic_contact_memory_peak_desired_lateral_launch_speed,
                    abs(episodic_effect.desired_lateral_launch_speed_mps),
                )
                episodic_contact_memory_peak_desired_vertical_launch_speed = max(
                    episodic_contact_memory_peak_desired_vertical_launch_speed,
                    abs(episodic_effect.desired_vertical_launch_speed_mps),
                )
            if episodic_effect is not None and episodic_effect.active:
                episodic_memory_active = True
                episodic_memory_active_seed_label = episodic_effect.selected_context_seed_label
                if abs(episodic_effect.lateral_force_n) >= abs(episodic_memory_lateral_force):
                    episodic_memory_lateral_force = episodic_effect.lateral_force_n
                if abs(episodic_effect.vertical_force_n) >= abs(episodic_memory_vertical_force):
                    episodic_memory_vertical_force = episodic_effect.vertical_force_n
                if float(np.max(np.abs(episodic_effect_torque))) >= float(
                    np.max(np.abs(episodic_memory_torque))
                ):
                    episodic_memory_torque = episodic_effect_torque.copy()
            controller_torque = (
                (target - data.qpos[7:36]) * kp
                + (target_velocity - data.qvel[6:35]) * kd
                + residual
                + ballistic_contact_torque
            )
            contact_task_torque = (
                loft_effect_torque + impulse_effect_torque + episodic_effect_torque
            )
            if flow.torque_authority_projection_ratio > 0.0 and np.any(
                np.abs(contact_task_torque) > 1e-12
            ):
                task_projection = project_g1_additive_torque_authority(
                    parent_torque_nm=controller_torque,
                    additive_torque_nm=contact_task_torque,
                    hard_limits_nm=hard_limits,
                    maximum_demand_ratio=flow.torque_authority_projection_ratio,
                )
                contact_task_torque = task_projection.projected_additive_torque_nm
                contact_task_authority_projection_steps += int(task_projection.active)
                contact_task_authority_scale_min = min(
                    contact_task_authority_scale_min,
                    task_projection.scale,
                )
            pre_guard_raw = controller_torque + contact_task_torque
            commanded_torque_peak_abs = np.maximum(
                commanded_torque_peak_abs,
                np.abs(pre_guard_raw),
            )
            raw = pre_guard_raw
            boundary_guard_correction = np.zeros(29, dtype=np.float64)
            boundary_guard_active = False
            # Protect the ankle before an active contact residual can cross a
            # hard limit.  Safety must not depend on whether the force came
            # from a teacher, a distilled actor, or the structured baseline.
            # Keep the correction inside the same audited bound used by the
            # reusable guard policy; the pure projection helper intentionally
            # does not apply that cap itself.
            if (
                residual_controller is not None
                or loft_teacher_config.enabled
                or ballistic_contact_impulse_actor is not None
                or episodic_contact_memory is not None
                or football_motion_prior is not None
                or any(abs(value) > 0.0 for value in flow.ballistic_contact_residual_rad)
                or any(abs(value) > 0.0 for value in flow.ballistic_contact_torque_residual_nm)
                or any(abs(value) > 0.0 for value in flow.ballistic_contact_torque_preload_nm)
                or any(
                    abs(value) > 0.0 for value in flow.ballistic_counterbalance_torque_residual_nm
                )
            ):
                projected, _, _ = project_g1_joint_boundary_torque(
                    joint_position=np.asarray(data.qpos[7:36], dtype=np.float64),
                    joint_velocity=np.asarray(data.qvel[6:35], dtype=np.float64),
                    commanded_torque=pre_guard_raw,
                    joint_lower_limits=joint_lower_limits,
                    joint_upper_limits=joint_upper_limits,
                    protected_joint_indices=right_ankle_pitch_index,
                    config=joint_boundary_guard_config,
                )
                boundary_guard_correction = np.clip(
                    projected - pre_guard_raw,
                    -joint_boundary_guard_config.maximum_correction_nm,
                    joint_boundary_guard_config.maximum_correction_nm,
                )
                raw = pre_guard_raw + boundary_guard_correction
                boundary_guard_active = bool(np.any(np.abs(boundary_guard_correction) > 1e-12))
                boundary_guard_active_steps += int(boundary_guard_active)
                boundary_guard_peak_correction = max(
                    boundary_guard_peak_correction,
                    float(np.max(np.abs(boundary_guard_correction))),
                )
            last_torque = project_authority(raw)
            torque_violation = torque_violation or bool(np.any(np.abs(last_torque) > hard_limits))
            data.ctrl[:] = last_torque
            _apply_compliant_net_force(data, ids, goal, flow)
            mujoco.mj_step(model, data)
            physics_steps += 1
            contacts = _contact_observation(model, data, ids)
            contact_in_frame = contact_in_frame or contacts.ball_right
            if contacts.ball_right and contacts.ball_force_n >= frame_contact_force_peak:
                frame_contact_force_peak = contacts.ball_force_n
                frame_contact_normal = np.asarray(
                    contacts.ball_contact_normal_xyz, dtype=np.float64
                )
                frame_contact_force_world = np.asarray(
                    contacts.ball_contact_force_world_xyz_n, dtype=np.float64
                )
            if contacts.ball_right:
                contact_peak_force = max(contact_peak_force or 0.0, contacts.ball_force_n)
            if contacts.ball_right and contact_time is None:
                contact_time = float(data.time)
                contact_point = contacts.ball_contact_point
                contact_height_relative_ball_center = float(
                    contacts.ball_contact_point[2] - data.qpos[ids.ball_qpos + 2]
                )
                contact_foot_position = (
                    float(data.xpos[ids.right_ankle][0]),
                    float(data.xpos[ids.right_ankle][1]),
                    float(data.xpos[ids.right_ankle][2]),
                )
                contact_foot_velocity = (
                    float(data.cvel[ids.right_ankle][3]),
                    float(data.cvel[ids.right_ankle][4]),
                    float(data.cvel[ids.right_ankle][5]),
                )
                contact_normal = contacts.ball_contact_normal_xyz
                contact_force_world = contacts.ball_contact_force_world_xyz_n
            ball = data.qpos[ids.ball_qpos : ids.ball_qpos + 3].copy()
            if contact_time is not None:
                ball_apex_height = max(ball_apex_height, float(ball[2]))
                ball_linear_velocity = np.asarray(
                    data.qvel[ids.ball_qvel : ids.ball_qvel + 3],
                    dtype=np.float64,
                )
                candidate_launch_speed = float(np.linalg.norm(ball_linear_velocity))
                # Contact impulse is distributed over several 2 ms physics
                # steps.  Keep the fastest early post-contact velocity rather
                # than an arbitrary first/last 20 ms control sample.
                if (
                    float(data.time) - contact_time <= 0.12
                    and candidate_launch_speed > launch_speed
                ):
                    launch_speed = candidate_launch_speed
                    launch_velocity = (
                        float(ball_linear_velocity[0]),
                        float(ball_linear_velocity[1]),
                        float(ball_linear_velocity[2]),
                    )
            if crossing is None and previous_ball[0] < goal.plane_x_m <= ball[0]:
                alpha = (goal.plane_x_m - previous_ball[0]) / max(ball[0] - previous_ball[0], 1e-12)
                point = previous_ball + alpha * (ball - previous_ball)
                crossing = (goal.plane_x_m, float(point[1]), float(point[2]))
            capture_x = _net_capture_plane_x(goal, flow, float(ball[2]))
            if net_capture is None and previous_ball[0] < capture_x <= ball[0]:
                alpha = (capture_x - previous_ball[0]) / max(ball[0] - previous_ball[0], 1e-12)
                point = previous_ball + alpha * (ball - previous_ball)
                net_capture = (capture_x, float(point[1]), float(point[2]))
            deepest_net_point = _deepest_goal_mouth_point(deepest_net_point, ball, goal)
            previous_ball = ball
            maximum_ball_speed = max(
                maximum_ball_speed,
                float(np.linalg.norm(data.qvel[ids.ball_qvel : ids.ball_qvel + 3])),
            )
        roll, pitch = _roll_pitch(data.xquat[ids.torso])
        kick_min_height = min(kick_min_height, float(data.qpos[2]))
        kick_peak_tilt = max(kick_peak_tilt, abs(roll), abs(pitch))
        if contact_time is None:
            pause_run_sec, pause_max_sec = _update_motion_pause(
                data, runup.control_dt_sec, pause_run_sec, pause_max_sec
            )
            (
                handoff_min_forward_speed,
                low_forward_run_sec,
                low_forward_max_sec,
            ) = _update_forward_continuity(
                data,
                runup.control_dt_sec,
                handoff_min_forward_speed,
                low_forward_run_sec,
                low_forward_max_sec,
            )
        finite = finite and _finite(data)
        joint_violation = joint_violation or _joint_violation(model, data)
        ballistic_contact_impulse_actor_active_frames += int(impulse_actor_active)
        ballistic_contact_impulse_actor_peak_torque = max(
            ballistic_contact_impulse_actor_peak_torque,
            float(np.max(np.abs(impulse_actor_torque))),
        )
        ballistic_contact_impulse_actor_peak_lateral_force = max(
            ballistic_contact_impulse_actor_peak_lateral_force,
            abs(impulse_actor_lateral_force),
        )
        ballistic_contact_impulse_actor_peak_vertical_force = max(
            ballistic_contact_impulse_actor_peak_vertical_force,
            abs(impulse_actor_vertical_force),
        )
        episodic_contact_memory_active_frames += int(episodic_memory_active)
        episodic_contact_memory_out_of_support_frames += int(
            episodic_memory_evaluated
            and (not episodic_memory_context_supported or not episodic_memory_launch_supported)
        )
        episodic_contact_memory_peak_torque = max(
            episodic_contact_memory_peak_torque,
            float(np.max(np.abs(episodic_memory_torque))),
        )
        episodic_contact_memory_peak_lateral_force = max(
            episodic_contact_memory_peak_lateral_force,
            abs(episodic_memory_lateral_force),
        )
        episodic_contact_memory_peak_vertical_force = max(
            episodic_contact_memory_peak_vertical_force,
            abs(episodic_memory_vertical_force),
        )
        episodic_contact_memory_peak_context_distance = max(
            episodic_contact_memory_peak_context_distance,
            episodic_memory_context_distance,
        )
        episodic_memory_reported_seed_label = (
            episodic_memory_active_seed_label
            if episodic_memory_active_seed_label is not None
            else episodic_memory_selected_seed_label
        )
        if episodic_memory_reported_seed_label is not None:
            if (
                episodic_contact_memory_selected_seed_label is not None
                and episodic_contact_memory_selected_seed_label
                != episodic_memory_reported_seed_label
            ):
                episodic_contact_memory_mixed_contexts = True
            elif not episodic_contact_memory_mixed_contexts:
                episodic_contact_memory_selected_seed_label = episodic_memory_reported_seed_label
        if episodic_contact_memory_mixed_contexts:
            episodic_contact_memory_selected_seed_label = None
        phase = min(1.0, policy_frame / max(1, int(policy.motion_total_steps) - 1))
        if contact_in_frame:
            event_phase = G1FootballEventPhase.CONTACT
        elif contact_time is not None and policy_frame <= 430:
            event_phase = G1FootballEventPhase.FOLLOW_THROUGH
        elif contact_time is not None and frame >= total_kick_frames - 20:
            event_phase = G1FootballEventPhase.READY
        elif contact_time is not None:
            event_phase = G1FootballEventPhase.RECOVERY
        elif policy_frame < 185:
            event_phase = G1FootballEventPhase.LOAD
        elif policy_frame <= 335:
            event_phase = G1FootballEventPhase.SWING
        elif policy_frame <= 430:
            event_phase = G1FootballEventPhase.FOLLOW_THROUGH
        else:
            event_phase = G1FootballEventPhase.LOAD
        _append_trace(
            trace,
            data,
            ids,
            last_torque,
            target,
            target_velocity,
            4,
            phase,
            crossing is not None,
            event_phase,
            pre_guard_raw,
            residual,
            residual_accepted,
            residual_confidence,
            loft_teacher_torque=loft_teacher_torque,
            loft_teacher_force_n=loft_teacher_force,
            loft_teacher_forward_force_n=loft_teacher_forward_force,
            loft_teacher_lateral_force_n=loft_teacher_lateral_force,
            loft_teacher_foot_vx_mps=loft_teacher_foot_vx,
            loft_teacher_foot_vy_mps=loft_teacher_foot_vy,
            loft_teacher_foot_vz_mps=loft_teacher_foot_vz,
            loft_teacher_active=loft_teacher_active,
            loft_teacher_pre_action_observation=(loft_teacher_pre_action_observation),
            loft_teacher_pre_action_observation_valid=(loft_teacher_pre_action_observation_valid),
            ballistic_contact_impulse_actor_torque=impulse_actor_torque,
            ballistic_contact_impulse_actor_lateral_force_n=(impulse_actor_lateral_force),
            ballistic_contact_impulse_actor_vertical_force_n=(impulse_actor_vertical_force),
            ballistic_contact_impulse_actor_foot_vy_mps=impulse_actor_foot_vy,
            ballistic_contact_impulse_actor_foot_vz_mps=impulse_actor_foot_vz,
            ballistic_contact_impulse_actor_active=impulse_actor_active,
            episodic_contact_memory_torque=episodic_memory_torque,
            episodic_contact_memory_lateral_force_n=(episodic_memory_lateral_force),
            episodic_contact_memory_vertical_force_n=(episodic_memory_vertical_force),
            episodic_contact_memory_active=episodic_memory_active,
            episodic_contact_memory_context_supported=(episodic_memory_context_supported),
            episodic_contact_memory_launch_supported=(episodic_memory_launch_supported),
            episodic_contact_memory_selected_seed_label=(episodic_memory_reported_seed_label),
            episodic_contact_memory_context_distance=(episodic_memory_context_distance),
            joint_boundary_guard_correction=boundary_guard_correction,
            joint_boundary_guard_active=boundary_guard_active,
            commanded_torque_peak_abs=commanded_torque_peak_abs,
            football_motion_prior_target_delta=motion_prior_delta,
            football_motion_prior_active=motion_prior_active,
            ballistic_contact_target_delta=ballistic_contact_delta,
            ballistic_contact_residual_active=ballistic_contact_active,
            ballistic_contact_torque_residual=ballistic_contact_torque,
            ballistic_contact_torque_residual_active=(ballistic_contact_torque_active),
            ball_contact_force_peak_n=frame_contact_force_peak,
            ball_contact_normal=frame_contact_normal,
            ball_contact_force_world=frame_contact_force_world,
            cerebellar_recovery_active=cerebellar_recovery_active,
            cerebellar_recovery_blend_fraction=(cerebellar_recovery_blend_fraction),
        )

    # A compliant net can arrest a valid shot before its centre reaches the
    # nominal capture-depth plane.  In that case the deepest measured point
    # inside the mouth is the physical capture observation.  Requiring the
    # ball to pass through an arbitrary depth plane would misclassify the very
    # damping behaviour that the net is intended to provide.
    if net_capture is None and crossing is not None:
        net_capture = deepest_net_point

    if crossing is None:
        plane_error = None
        lower_corner_distance = None
        upper_corner_distance = None
        declared_corner_distance = None
        mouth_hit = False
    else:
        plane_error = math.hypot(crossing[1] - goal.target_y_m, crossing[2] - goal.target_z_m)
        lower_corner_distance = math.hypot(
            goal.width_m / 2.0 - abs(crossing[1]), crossing[2] - goal.ball_radius_m
        )
        upper_corner_distance = math.hypot(
            goal.width_m / 2.0 - abs(crossing[1]),
            goal.height_m - goal.ball_radius_m - crossing[2],
        )
        declared_corner_distance = (
            upper_corner_distance if "upper" in goal.target_corner else lower_corner_distance
        )
        mouth_hit = bool(
            abs(crossing[1]) <= goal.width_m / 2.0 - goal.ball_radius_m
            and goal.ball_radius_m <= crossing[2] <= goal.height_m - goal.ball_radius_m
        )
    net_error = (
        None
        if net_capture is None
        else math.hypot(
            net_capture[1] - goal.target_y_m,
            net_capture[2] - goal.target_z_m,
        )
    )
    final_ball = (
        float(data.qpos[ids.ball_qpos]),
        float(data.qpos[ids.ball_qpos + 1]),
        float(data.qpos[ids.ball_qpos + 2]),
    )
    final_ball_error = math.hypot(
        final_ball[1] - goal.target_y_m,
        final_ball[2] - goal.target_z_m,
    )
    if not math.isfinite(handoff_min_forward_speed):
        handoff_min_forward_speed = 0.0
    forward_speed_retention = handoff_min_forward_speed / max(
        runup_terminal_speed,
        1e-9,
    )
    final_roll, final_pitch = _roll_pitch(data.xquat[ids.torso])
    fall = bool(
        kick_min_height < 0.55
        or kick_peak_tilt > 0.65
        or float(data.qpos[2]) < 0.55
        or max(abs(final_roll), abs(final_pitch)) > 0.55
    )
    residual_array = np.asarray(trace["learned_residual_torque"], dtype=np.float64)
    loft_teacher_array = np.asarray(trace["loft_teacher_torque"], dtype=np.float64)
    loft_teacher_force_array = np.asarray(trace["loft_teacher_force_n"], dtype=np.float64)
    loft_teacher_active_array = np.asarray(trace["loft_teacher_active"], dtype=bool)
    residual_acceptance = np.asarray(trace["learned_residual_accepted"], dtype=bool)
    accepted_frames = int(np.count_nonzero(residual_acceptance))
    residual_peak = float(np.max(np.abs(residual_array))) if residual_array.size else 0.0
    residual_rms = (
        float(np.sqrt(np.mean(np.square(residual_array[residual_acceptance]))))
        if accepted_frames
        else 0.0
    )
    commanded_array = np.asarray(trace["commanded_torque"], dtype=np.float64)
    baseline_array = commanded_array - residual_array
    baseline_rms = (
        float(np.sqrt(np.mean(np.square(baseline_array[residual_acceptance]))))
        if accepted_frames
        else 0.0
    )
    (
        post_contact_peak_speed,
        post_contact_backward_displacement,
        post_contact_velocity_reversals,
        post_contact_settling_time,
        post_contact_peak_joint_velocity_rms,
        post_contact_final_joint_velocity_rms,
        post_contact_mean_pelvis_speed,
        post_contact_mean_joint_velocity_rms,
    ) = _post_contact_recovery_metrics(trace, contact_time)
    result = G1FreeKickResult(
        finite_state=finite,
        learned_runup_executed=True,
        learned_approach_strike_residual_executed=residual_controller is not None,
        loft_teacher_executed=loft_teacher_config.enabled,
        loft_teacher_active_frames=int(np.count_nonzero(loft_teacher_active_array)),
        loft_teacher_peak_torque_nm=(
            float(np.max(np.abs(loft_teacher_array))) if loft_teacher_array.size else 0.0
        ),
        loft_teacher_peak_force_n=(
            float(np.max(np.abs(loft_teacher_force_array)))
            if loft_teacher_force_array.size
            else 0.0
        ),
        joint_boundary_guard_active_steps=boundary_guard_active_steps,
        joint_boundary_guard_peak_correction_nm=boundary_guard_peak_correction,
        residual_accepted_frames=accepted_frames,
        residual_rejected_frames=(
            0 if residual_controller is None else int(len(residual_acceptance) - accepted_frames)
        ),
        residual_peak_nm=residual_peak,
        residual_rms_nm=residual_rms,
        residual_effect_fraction=residual_rms / max(baseline_rms, 1e-12),
        continuous_single_world=True,
        state_reset_after_start=False,
        initial_ball_distance_m=initial_ball_distance,
        shot_distance_m=goal.plane_x_m - 1.0,
        runup_distance_m=runup_end_x - runup_start_x,
        runup_peak_speed_mps=runup_peak_speed,
        runup_min_pelvis_height_m=runup_min_height,
        runup_peak_tilt_rad=runup_peak_tilt,
        runup_terminal_speed_mps=runup_terminal_speed,
        handoff_yaw_rad=handoff_yaw,
        handoff_roll_rad=handoff_features.abs_pelvis_roll_rad,
        handoff_pitch_rad=handoff_features.abs_pelvis_pitch_rad,
        handoff_pelvis_x_m=handoff_features.pelvis_x_m,
        handoff_pelvis_y_m=handoff_features.pelvis_y_m,
        handoff_joint_velocity_rms_rad_s=(handoff_features.joint_velocity_rms_rad_s),
        selected_kick_phase_start_frame=phase_start,
        contextual_phase_expert_executed=contextual_phase_expert,
        proprioceptive_router_executed=router_selection is not None,
        proprioceptive_router_fallback=(
            False if router_selection is None else router_selection.used_fallback
        ),
        proprioceptive_router_nearest_distance=(
            None if router_selection is None else router_selection.nearest_distance
        ),
        proprioceptive_router_distance_margin=(
            None if router_selection is None else router_selection.distance_margin
        ),
        handoff_to_contact_sec=(None if contact_time is None else contact_time - handoff_time),
        pre_contact_motion_pause_sec=pause_max_sec,
        handoff_min_forward_speed_mps=handoff_min_forward_speed,
        handoff_low_forward_speed_duration_sec=low_forward_max_sec,
        handoff_forward_speed_retention_ratio=forward_speed_retention,
        skill_bridge_max_joint_delta_rad=bridge_max,
        skill_bridge_rms_joint_delta_rad=bridge_rms,
        skill_bridge_entry_velocity_rms_rad_s=bridge_entry_velocity_rms,
        skill_bridge_target_exit_velocity_rms_rad_s=bridge_target_exit_velocity_rms,
        skill_bridge_exit_velocity_error_rms_rad_s=bridge_exit_velocity_error_rms,
        skill_bridge_peak_target_acceleration_rms_rad_s2=(bridge_peak_target_acceleration_rms),
        kick_contact_observed=contact_time is not None,
        contact_time_sec=contact_time,
        kick_contact_point_xyz_m=contact_point,
        kick_contact_height_relative_ball_center_m=(contact_height_relative_ball_center),
        kick_contact_foot_position_xyz_m=contact_foot_position,
        kick_contact_foot_velocity_xyz_mps=contact_foot_velocity,
        kick_contact_normal_xyz=contact_normal,
        kick_contact_force_world_xyz_n=contact_force_world,
        kick_contact_peak_force_n=contact_peak_force,
        ball_launch_velocity_xyz_mps=launch_velocity,
        ball_apex_height_m=ball_apex_height,
        ball_speed_peak_mps=maximum_ball_speed,
        goal_crossed=crossing is not None,
        goal_crossing_xyz_m=crossing,
        goal_mouth_hit=mouth_hit,
        goal_plane_target_error_m=plane_error,
        net_capture_xyz_m=net_capture,
        net_capture_target_error_m=net_error,
        final_ball_xyz_m=final_ball,
        final_ball_yz_target_error_m=final_ball_error,
        ball_retained_in_goal=bool(
            goal.plane_x_m - goal.ball_radius_m
            <= final_ball[0]
            <= goal.plane_x_m + goal.depth_m + goal.ball_radius_m
        ),
        precision_radius_m=goal.precision_radius_m,
        declared_target_corner=goal.target_corner,
        declared_corner_distance_m=declared_corner_distance,
        upper_corner_distance_m=upper_corner_distance,
        lower_corner_distance_m=lower_corner_distance,
        kick_min_pelvis_height_m=kick_min_height,
        kick_peak_tilt_rad=kick_peak_tilt,
        final_pelvis_height_m=float(data.qpos[2]),
        final_speed_mps=float(np.linalg.norm(data.qvel[:2])),
        post_kick_fall=fall,
        joint_limit_violation=joint_violation,
        torque_limit_violation=torque_violation,
        actuator_saturation=saturation,
        actuator_saturation_steps=saturation_steps,
        actuator_saturation_fraction=saturation_steps / max(1, physics_steps),
        actuator_peak_demand_ratio=peak_demand_ratio,
        physics_steps=physics_steps,
        post_contact_peak_pelvis_speed_mps=post_contact_peak_speed,
        post_contact_backward_displacement_m=post_contact_backward_displacement,
        post_contact_forward_velocity_reversals=post_contact_velocity_reversals,
        post_contact_settling_time_sec=post_contact_settling_time,
        post_contact_peak_joint_velocity_rms_rad_s=(post_contact_peak_joint_velocity_rms),
        post_contact_final_joint_velocity_rms_rad_s=(post_contact_final_joint_velocity_rms),
        post_contact_mean_pelvis_speed_mps=post_contact_mean_pelvis_speed,
        post_contact_mean_joint_velocity_rms_rad_s=(post_contact_mean_joint_velocity_rms),
        cerebellar_recovery_executed=cerebellar_recovery is not None,
        cerebellar_recovery_active_frames=cerebellar_recovery_active_frames,
        cerebellar_recovery_peak_blend_fraction=(cerebellar_recovery_peak_blend_fraction),
        football_outcome_model_executed=outcome_decision is not None,
        football_outcome_retry_recommended=initial_retry_recommended,
        football_outcome_predicted_hard_safe_probability=(
            None if outcome_decision is None else outcome_decision.predicted_hard_safe_probability
        ),
        football_outcome_predicted_precision_probability=(
            None if outcome_decision is None else outcome_decision.predicted_precision_probability
        ),
        football_outcome_predicted_penalized_error_m=(
            None if outcome_decision is None else outcome_decision.predicted_penalized_error_m
        ),
        football_outcome_retry_recovery_executed=retry_recovery_executed,
        football_outcome_retry_recovery_duration_sec=(
            flow.football_retry_recovery_duration_sec if retry_recovery_executed else 0.0
        ),
        football_outcome_retry_initial_speed_mps=retry_initial_speed,
        football_outcome_retry_final_speed_mps=retry_final_speed,
        football_motion_prior_executed=football_motion_prior is not None,
        football_motion_prior_active_frames=football_motion_prior_active_frames,
        football_motion_prior_peak_target_delta_rad=(football_motion_prior_peak_target_delta),
        football_motion_prior_velocity_executed=bool(
            football_motion_prior_velocity_active_frames > 0
        ),
        football_motion_prior_velocity_active_frames=(football_motion_prior_velocity_active_frames),
        football_motion_prior_peak_target_velocity_delta_rad_s=(
            football_motion_prior_peak_target_velocity_delta
        ),
        ballistic_contact_residual_executed=ballistic_contact_config.enabled,
        ballistic_contact_residual_active_frames=ballistic_contact_active_frames,
        ballistic_contact_residual_peak_target_delta_rad=(ballistic_contact_peak_target_delta),
        ballistic_contact_torque_residual_executed=(ballistic_contact_torque_config.enabled),
        ballistic_contact_torque_residual_active_frames=(ballistic_contact_torque_active_frames),
        ballistic_contact_torque_residual_peak_nm=ballistic_contact_torque_peak,
        ballistic_contact_impulse_actor_executed=(ballistic_contact_impulse_actor is not None),
        ballistic_contact_impulse_actor_active_frames=(
            ballistic_contact_impulse_actor_active_frames
        ),
        ballistic_contact_impulse_actor_peak_torque_nm=(
            ballistic_contact_impulse_actor_peak_torque
        ),
        ballistic_contact_impulse_actor_peak_lateral_force_n=(
            ballistic_contact_impulse_actor_peak_lateral_force
        ),
        ballistic_contact_impulse_actor_peak_vertical_force_n=(
            ballistic_contact_impulse_actor_peak_vertical_force
        ),
        ballistic_contact_impulse_actor_target_conditioned=bool(
            ballistic_contact_impulse_actor is not None
            and ballistic_contact_impulse_actor.schema_version
            == "rosclaw.growth.g1_ballistic_contact_impulse_actor.v2"
        ),
        ballistic_contact_impulse_actor_out_of_envelope_frames=(
            ballistic_contact_impulse_actor_out_of_envelope_frames
        ),
        ballistic_contact_impulse_actor_peak_desired_lateral_launch_speed_mps=(
            ballistic_contact_impulse_actor_peak_desired_lateral_launch_speed
        ),
        ballistic_contact_impulse_actor_peak_desired_vertical_launch_speed_mps=(
            ballistic_contact_impulse_actor_peak_desired_vertical_launch_speed
        ),
        episodic_contact_memory_executed=(episodic_contact_memory is not None),
        episodic_contact_memory_active_frames=(episodic_contact_memory_active_frames),
        episodic_contact_memory_out_of_support_frames=(
            episodic_contact_memory_out_of_support_frames
        ),
        episodic_contact_memory_peak_torque_nm=(episodic_contact_memory_peak_torque),
        episodic_contact_memory_peak_lateral_force_n=(episodic_contact_memory_peak_lateral_force),
        episodic_contact_memory_peak_vertical_force_n=(episodic_contact_memory_peak_vertical_force),
        episodic_contact_memory_peak_desired_lateral_launch_speed_mps=(
            episodic_contact_memory_peak_desired_lateral_launch_speed
        ),
        episodic_contact_memory_peak_desired_vertical_launch_speed_mps=(
            episodic_contact_memory_peak_desired_vertical_launch_speed
        ),
        episodic_contact_memory_selected_seed_label=(episodic_contact_memory_selected_seed_label),
        episodic_contact_memory_peak_context_distance=(
            episodic_contact_memory_peak_context_distance
        ),
        torque_authority_projection_enabled=(flow.torque_authority_projection_ratio > 0.0),
        torque_authority_projection_steps=torque_authority_projection_steps,
        torque_authority_projection_fraction=(
            torque_authority_projection_steps / max(1, physics_steps)
        ),
        torque_authority_projection_peak_correction_nm=(
            torque_authority_projection_peak_correction
        ),
        torque_authority_preprojection_peak_demand_ratio=(
            torque_authority_preprojection_peak_demand_ratio
        ),
        torque_authority_projection_qualified=bool(
            flow.torque_authority_projection_ratio == 0.0
            or torque_authority_projection_steps / max(1, physics_steps)
            <= flow.torque_authority_projection_max_fraction
        ),
        contact_task_authority_projection_steps=(contact_task_authority_projection_steps),
        contact_task_authority_scale_min=contact_task_authority_scale_min,
        ballistic_skill_memory_executed=ballistic_skill_selection is not None,
        ballistic_skill_id=(
            None
            if ballistic_skill_selection is None
            else ballistic_skill_selection.selected_skill_id
        ),
        ballistic_skill_nearest_distance=(
            None
            if ballistic_skill_selection is None
            else ballistic_skill_selection.nearest_distance
        ),
        ballistic_skill_distance_margin=(
            None if ballistic_skill_selection is None else ballistic_skill_selection.distance_margin
        ),
    )
    trajectory = {key: np.asarray(value) for key, value in trace.items()}
    trajectory["sonic_reference_digest"] = np.asarray(
        "" if sonic is None else sonic.reference_digest
    )
    return result, trajectory


def _runup_command(time_sec: float, config: G1LearnedRunupConfig) -> tuple[np.ndarray, int]:
    if time_sec < config.settle_duration_sec:
        return np.zeros(3, dtype=np.float32), 0
    run_end = config.settle_duration_sec + config.run_duration_sec
    command = np.asarray(
        (
            config.forward_velocity_command_mps,
            config.lateral_velocity_command_mps,
            0.0,
        ),
        dtype=np.float32,
    )
    if time_sec < run_end:
        return command, 1
    deceleration_duration = config.brake_duration_sec - config.plant_duration_sec
    if time_sec < run_end + deceleration_duration:
        progress = (time_sec - run_end) / deceleration_duration
        return command * np.float32(max(0.0, 1.0 - progress)), 2
    return np.zeros(3, dtype=np.float32), 2


def _signed_yaw(quaternion_wxyz: np.ndarray) -> float:
    value = np.asarray(quaternion_wxyz, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("handoff yaw requires one finite WXYZ quaternion")
    w, x, y, z = (float(item) for item in value)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _select_contextual_phase(
    flow: G1FreeKickFlowConfig, handoff_yaw_rad: float
) -> tuple[int, bool]:
    if not math.isfinite(handoff_yaw_rad):
        raise ValueError("contextual phase selection requires finite handoff yaw")
    selected = bool(
        flow.contextual_phase_yaw_threshold_rad > 0.0
        and abs(handoff_yaw_rad) >= flow.contextual_phase_yaw_threshold_rad
    )
    return (
        flow.contextual_high_yaw_kick_phase_start_frame
        if selected
        else flow.kick_phase_start_frame,
        selected,
    )


def _update_motion_pause(
    data: Any,
    dt_sec: float,
    current_pause_sec: float,
    maximum_pause_sec: float,
) -> tuple[float, float]:
    """Measure actual pre-contact stillness instead of inferring it from modes."""

    base_speed = float(np.linalg.norm(data.qvel[:2]))
    joint_rms = float(np.sqrt(np.mean(np.square(data.qvel[6:35]))))
    if base_speed < 0.05 and joint_rms < 0.08:
        current_pause_sec += dt_sec
    else:
        current_pause_sec = 0.0
    return current_pause_sec, max(maximum_pause_sec, current_pause_sec)


def _update_forward_continuity(
    data: Any,
    dt_sec: float,
    minimum_forward_speed_mps: float,
    current_low_speed_sec: float,
    maximum_low_speed_sec: float,
) -> tuple[float, float, float]:
    """Measure the visible pelvis-speed dip between run-up and contact."""

    forward_speed = float(data.qvel[0])
    if not math.isfinite(forward_speed):
        raise FloatingPointError("G1 forward continuity received non-finite speed")
    minimum_forward_speed_mps = min(minimum_forward_speed_mps, forward_speed)
    if forward_speed < 0.20:
        current_low_speed_sec += dt_sec
    else:
        current_low_speed_sec = 0.0
    return (
        minimum_forward_speed_mps,
        current_low_speed_sec,
        max(maximum_low_speed_sec, current_low_speed_sec),
    )


def _deepest_goal_mouth_point(
    current: tuple[float, float, float] | None,
    ball_position: np.ndarray,
    goal: G1TrainingGoalSpec,
) -> tuple[float, float, float] | None:
    """Return the deepest measured ball centre that remains inside the mouth."""

    point = np.asarray(ball_position, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return current
    tolerance = 1e-6
    inside = bool(
        point[0] >= goal.plane_x_m - tolerance
        and abs(point[1]) <= goal.width_m / 2.0 - goal.ball_radius_m + tolerance
        and goal.ball_radius_m - tolerance
        <= point[2]
        <= goal.height_m - goal.ball_radius_m + tolerance
    )
    if not inside or (current is not None and point[0] <= current[0]):
        return current
    return (float(point[0]), float(point[1]), float(point[2]))


def _apply_compliant_net_force(
    data: Any,
    ids: _ModelIds,
    goal: G1TrainingGoalSpec,
    flow: G1FreeKickFlowConfig,
) -> None:
    """Apply a deterministic one-sided soft-net force to the ball body."""
    apply_g1_compliant_goal_net_force(
        data,
        ball_body_id=ids.ball,
        ball_qpos=ids.ball_qpos,
        ball_qvel=ids.ball_qvel,
        spec=goal,
        capture_depth_m=flow.net_capture_depth_m,
        stiffness_n_m=flow.net_stiffness_n_m,
        damping_n_s_m=flow.net_damping_n_s_m,
    )


def _net_capture_plane_x(
    goal: G1TrainingGoalSpec,
    flow: G1FreeKickFlowConfig,
    ball_z_m: float,
) -> float:
    """Return the ball-centre contact plane aligned to the sloped visible net."""

    return g1_goal_net_contact_plane_x(
        goal,
        capture_depth_m=flow.net_capture_depth_m,
        ball_z_m=ball_z_m,
    )


def _configure_surface(model: Any, goal: G1TrainingGoalSpec) -> None:
    import mujoco

    ball_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"))
    floor_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    model.geom_friction[ball_geom] = (
        goal.ball_contact_sliding_friction,
        goal.ball_torsional_friction,
        goal.ball_rolling_friction,
    )
    model.geom_friction[floor_geom, 0] = 0.90
    for index in range(int(model.npair)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, index) or ""
        if name == "ball_floor":
            model.pair_friction[index] = (
                goal.ball_sliding_friction,
                goal.ball_sliding_friction,
                goal.ball_torsional_friction,
                goal.ball_rolling_friction,
                goal.ball_rolling_friction,
            )
        elif name.endswith("_floor"):
            model.pair_friction[index, 0] = 0.90


def _finite(data: Any) -> bool:
    return bool(
        np.all(np.isfinite(data.qpos))
        and np.all(np.isfinite(data.qvel))
        and np.all(np.isfinite(data.ctrl))
    )


def _post_contact_recovery_metrics(
    trace: dict[str, list[Any]],
    contact_time: float | None,
) -> tuple[float, float, int, float, float, float, float, float]:
    if contact_time is None:
        return (0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    time = np.asarray(trace["time"], dtype=np.float64)
    pose = np.asarray(trace["pelvis_pose"], dtype=np.float64)
    velocity = np.asarray(trace["pelvis_velocity"], dtype=np.float64)
    joint_velocity = np.asarray(trace["joint_velocity"], dtype=np.float64)
    if (
        time.ndim != 1
        or pose.shape != (len(time), 7)
        or velocity.shape != (len(time), 6)
        or joint_velocity.shape != (len(time), 29)
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(pose))
        or not np.all(np.isfinite(velocity))
        or not np.all(np.isfinite(joint_velocity))
    ):
        raise FloatingPointError("post-contact recovery trace is invalid")
    indices = np.flatnonzero(time >= contact_time)
    if indices.size == 0:
        return (0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    planar_velocity = velocity[indices, :2]
    planar_speed = np.linalg.norm(planar_velocity, axis=1)
    pelvis_x = pose[indices, 0]
    backward = max(0.0, float(pelvis_x[0] - np.min(pelvis_x)))
    forward = planar_velocity[:, 0]
    signs = np.sign(forward[np.abs(forward) >= 0.03])
    reversals = int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    moving = np.flatnonzero(planar_speed > 0.10)
    settling = (
        0.0 if moving.size == 0 else max(0.0, float(time[indices[int(moving[-1])]] - contact_time))
    )
    joint_rms = np.sqrt(np.mean(np.square(joint_velocity[indices]), axis=1))
    recovery_indices = indices[time[indices] >= contact_time + 0.63]
    if recovery_indices.size == 0:
        recovery_indices = indices
    recovery_speed = np.linalg.norm(velocity[recovery_indices, :2], axis=1)
    recovery_joint_rms = np.sqrt(np.mean(np.square(joint_velocity[recovery_indices]), axis=1))
    return (
        float(np.max(planar_speed)),
        backward,
        reversals,
        settling,
        float(np.max(joint_rms)),
        float(joint_rms[-1]),
        float(np.mean(recovery_speed)),
        float(np.mean(recovery_joint_rms)),
    )


def _joint_violation(model: Any, data: Any) -> bool:
    limited = model.jnt_limited[1:30].astype(bool)
    ranges = model.jnt_range[1:30]
    positions = data.qpos[7:36]
    return bool(
        np.any(positions[limited] < ranges[limited, 0] - 1e-5)
        or np.any(positions[limited] > ranges[limited, 1] + 1e-5)
    )


def _append_trace(
    trace: dict[str, list[Any]],
    data: Any,
    ids: _ModelIds,
    torque: np.ndarray,
    target: np.ndarray,
    target_velocity: np.ndarray,
    mode: int,
    phase: float,
    crossed: bool,
    event_phase: G1FootballEventPhase,
    commanded_torque: np.ndarray,
    learned_residual: np.ndarray,
    learned_residual_accepted: bool,
    learned_residual_confidence: float,
    loft_teacher_torque: np.ndarray | None = None,
    loft_teacher_force_n: float = 0.0,
    loft_teacher_forward_force_n: float = 0.0,
    loft_teacher_lateral_force_n: float = 0.0,
    loft_teacher_foot_vx_mps: float = 0.0,
    loft_teacher_foot_vy_mps: float = 0.0,
    loft_teacher_foot_vz_mps: float = 0.0,
    loft_teacher_active: bool = False,
    loft_teacher_pre_action_observation: np.ndarray | None = None,
    loft_teacher_pre_action_observation_valid: bool = False,
    ballistic_contact_impulse_actor_torque: np.ndarray | None = None,
    ballistic_contact_impulse_actor_lateral_force_n: float = 0.0,
    ballistic_contact_impulse_actor_vertical_force_n: float = 0.0,
    ballistic_contact_impulse_actor_foot_vy_mps: float = 0.0,
    ballistic_contact_impulse_actor_foot_vz_mps: float = 0.0,
    ballistic_contact_impulse_actor_active: bool = False,
    episodic_contact_memory_torque: np.ndarray | None = None,
    episodic_contact_memory_lateral_force_n: float = 0.0,
    episodic_contact_memory_vertical_force_n: float = 0.0,
    episodic_contact_memory_active: bool = False,
    episodic_contact_memory_context_supported: bool = True,
    episodic_contact_memory_launch_supported: bool = True,
    episodic_contact_memory_selected_seed_label: int | None = None,
    episodic_contact_memory_context_distance: float = 0.0,
    joint_boundary_guard_correction: np.ndarray | None = None,
    joint_boundary_guard_active: bool = False,
    commanded_torque_peak_abs: np.ndarray | None = None,
    football_motion_prior_target_delta: np.ndarray | None = None,
    football_motion_prior_active: bool = False,
    ballistic_contact_target_delta: np.ndarray | None = None,
    ballistic_contact_residual_active: bool = False,
    ballistic_contact_torque_residual: np.ndarray | None = None,
    ballistic_contact_torque_residual_active: bool = False,
    cerebellar_recovery_active: bool = False,
    cerebellar_recovery_blend_fraction: float = 0.0,
    ball_contact_force_peak_n: float = 0.0,
    ball_contact_normal: np.ndarray | None = None,
    ball_contact_force_world: np.ndarray | None = None,
) -> None:
    trace["time"].append(float(data.time))
    trace["joint_position"].append(data.qpos[7:36].copy())
    trace["joint_velocity"].append(data.qvel[6:35].copy())
    trace["joint_torque"].append(np.asarray(torque, dtype=np.float64).copy())
    commanded = np.asarray(commanded_torque, dtype=np.float64)
    peak_abs = (
        np.abs(commanded)
        if commanded_torque_peak_abs is None
        else np.asarray(commanded_torque_peak_abs, dtype=np.float64)
    )
    if peak_abs.shape != (29,) or not np.all(np.isfinite(peak_abs)) or np.any(peak_abs < 0.0):
        raise FloatingPointError("G1 commanded torque peak trace is invalid")
    projected = np.asarray(torque, dtype=np.float64)
    trace["commanded_torque"].append(commanded.copy())
    trace["commanded_torque_peak_abs"].append(peak_abs.copy())
    trace["safety_projected_torque"].append(projected.copy())
    trace["executed_torque"].append(np.asarray(data.actuator_force[:29], dtype=np.float64).copy())
    trace["torque_projection_applied"].append(bool(np.any(np.abs(commanded - projected) > 1e-12)))
    trace["learned_residual_torque"].append(np.asarray(learned_residual, dtype=np.float64).copy())
    trace["learned_residual_accepted"].append(bool(learned_residual_accepted))
    trace["learned_residual_confidence"].append(float(learned_residual_confidence))
    teacher_torque = (
        np.zeros(29, dtype=np.float64)
        if loft_teacher_torque is None
        else np.asarray(loft_teacher_torque, dtype=np.float64)
    )
    if teacher_torque.shape != (29,) or not np.all(np.isfinite(teacher_torque)):
        raise FloatingPointError("G1 loft teacher trace contains invalid torque")
    teacher_scalars = (
        loft_teacher_force_n,
        loft_teacher_forward_force_n,
        loft_teacher_lateral_force_n,
        loft_teacher_foot_vx_mps,
        loft_teacher_foot_vy_mps,
        loft_teacher_foot_vz_mps,
    )
    if not all(math.isfinite(value) for value in teacher_scalars):
        raise FloatingPointError("G1 loft teacher trace contains non-finite state")
    trace["loft_teacher_torque"].append(teacher_torque.copy())
    trace["loft_teacher_force_n"].append(float(loft_teacher_force_n))
    trace["loft_teacher_forward_force_n"].append(float(loft_teacher_forward_force_n))
    trace["loft_teacher_lateral_force_n"].append(float(loft_teacher_lateral_force_n))
    trace["loft_teacher_foot_vx_mps"].append(float(loft_teacher_foot_vx_mps))
    trace["loft_teacher_foot_vy_mps"].append(float(loft_teacher_foot_vy_mps))
    trace["loft_teacher_foot_vz_mps"].append(float(loft_teacher_foot_vz_mps))
    trace["loft_teacher_active"].append(bool(loft_teacher_active))
    pre_action_observation = (
        np.zeros(11, dtype=np.float64)
        if loft_teacher_pre_action_observation is None
        else np.asarray(loft_teacher_pre_action_observation, dtype=np.float64)
    )
    if pre_action_observation.shape != (11,) or not np.all(np.isfinite(pre_action_observation)):
        raise FloatingPointError("G1 loft teacher pre-action observation is invalid")
    if loft_teacher_pre_action_observation_valid and not loft_teacher_active:
        raise ValueError("G1 loft teacher pre-action observation lacks an active teacher")
    trace["loft_teacher_pre_action_observation"].append(pre_action_observation.copy())
    trace["loft_teacher_pre_action_observation_valid"].append(
        bool(loft_teacher_pre_action_observation_valid)
    )
    impulse_torque = (
        np.zeros(29, dtype=np.float64)
        if ballistic_contact_impulse_actor_torque is None
        else np.asarray(ballistic_contact_impulse_actor_torque, dtype=np.float64)
    )
    impulse_scalars = (
        ballistic_contact_impulse_actor_lateral_force_n,
        ballistic_contact_impulse_actor_vertical_force_n,
        ballistic_contact_impulse_actor_foot_vy_mps,
        ballistic_contact_impulse_actor_foot_vz_mps,
    )
    if impulse_torque.shape != (29,) or not np.all(np.isfinite(impulse_torque)):
        raise FloatingPointError("G1 contact impulse actor trace contains invalid torque")
    if not all(math.isfinite(value) for value in impulse_scalars):
        raise FloatingPointError("G1 contact impulse actor trace contains non-finite state")
    trace["ballistic_contact_impulse_actor_torque"].append(impulse_torque.copy())
    trace["ballistic_contact_impulse_actor_lateral_force_n"].append(
        float(ballistic_contact_impulse_actor_lateral_force_n)
    )
    trace["ballistic_contact_impulse_actor_vertical_force_n"].append(
        float(ballistic_contact_impulse_actor_vertical_force_n)
    )
    trace["ballistic_contact_impulse_actor_foot_vy_mps"].append(
        float(ballistic_contact_impulse_actor_foot_vy_mps)
    )
    trace["ballistic_contact_impulse_actor_foot_vz_mps"].append(
        float(ballistic_contact_impulse_actor_foot_vz_mps)
    )
    trace["ballistic_contact_impulse_actor_active"].append(
        bool(ballistic_contact_impulse_actor_active)
    )
    episodic_torque = (
        np.zeros(29, dtype=np.float64)
        if episodic_contact_memory_torque is None
        else np.asarray(episodic_contact_memory_torque, dtype=np.float64)
    )
    episodic_scalars = (
        episodic_contact_memory_lateral_force_n,
        episodic_contact_memory_vertical_force_n,
        episodic_contact_memory_context_distance,
    )
    if episodic_torque.shape != (29,) or not np.all(np.isfinite(episodic_torque)):
        raise FloatingPointError("G1 episodic contact memory trace contains invalid torque")
    if not all(math.isfinite(value) for value in episodic_scalars):
        raise FloatingPointError("G1 episodic contact memory trace contains non-finite state")
    trace["episodic_contact_memory_torque"].append(episodic_torque.copy())
    trace["episodic_contact_memory_lateral_force_n"].append(
        float(episodic_contact_memory_lateral_force_n)
    )
    trace["episodic_contact_memory_vertical_force_n"].append(
        float(episodic_contact_memory_vertical_force_n)
    )
    trace["episodic_contact_memory_active"].append(bool(episodic_contact_memory_active))
    trace["episodic_contact_memory_context_supported"].append(
        bool(episodic_contact_memory_context_supported)
    )
    trace["episodic_contact_memory_launch_supported"].append(
        bool(episodic_contact_memory_launch_supported)
    )
    trace["episodic_contact_memory_selected_seed_label"].append(
        -1
        if episodic_contact_memory_selected_seed_label is None
        else int(episodic_contact_memory_selected_seed_label)
    )
    trace["episodic_contact_memory_context_distance"].append(
        float(episodic_contact_memory_context_distance)
    )
    guard_correction = (
        np.zeros(29, dtype=np.float64)
        if joint_boundary_guard_correction is None
        else np.asarray(joint_boundary_guard_correction, dtype=np.float64)
    )
    if guard_correction.shape != (29,) or not np.all(np.isfinite(guard_correction)):
        raise FloatingPointError("G1 joint boundary trace contains invalid correction")
    trace["joint_boundary_guard_correction"].append(guard_correction.copy())
    trace["joint_boundary_guard_active"].append(bool(joint_boundary_guard_active))
    motion_delta = (
        np.zeros(29, dtype=np.float64)
        if football_motion_prior_target_delta is None
        else np.asarray(football_motion_prior_target_delta, dtype=np.float64)
    )
    if motion_delta.shape != (29,) or not np.all(np.isfinite(motion_delta)):
        raise FloatingPointError("G1 football motion prior trace contains invalid target delta")
    trace["football_motion_prior_target_delta"].append(motion_delta.copy())
    trace["football_motion_prior_active"].append(bool(football_motion_prior_active))
    ballistic_delta = (
        np.zeros(29, dtype=np.float64)
        if ballistic_contact_target_delta is None
        else np.asarray(ballistic_contact_target_delta, dtype=np.float64)
    )
    if ballistic_delta.shape != (29,) or not np.all(np.isfinite(ballistic_delta)):
        raise ValueError("ballistic contact target delta must contain 29 finite joints")
    trace["ballistic_contact_target_delta"].append(ballistic_delta.copy())
    trace["ballistic_contact_residual_active"].append(bool(ballistic_contact_residual_active))
    ballistic_torque = (
        np.zeros(29, dtype=np.float64)
        if ballistic_contact_torque_residual is None
        else np.asarray(ballistic_contact_torque_residual, dtype=np.float64)
    )
    if ballistic_torque.shape != (29,) or not np.all(np.isfinite(ballistic_torque)):
        raise ValueError("ballistic contact torque residual must contain 29 finite joints")
    trace["ballistic_contact_torque_residual"].append(ballistic_torque.copy())
    trace["ballistic_contact_torque_residual_active"].append(
        bool(ballistic_contact_torque_residual_active)
    )
    if not math.isfinite(cerebellar_recovery_blend_fraction) or not (
        0.0 <= cerebellar_recovery_blend_fraction <= 1.0
    ):
        raise ValueError("cerebellar recovery blend fraction must be in [0, 1]")
    trace["cerebellar_recovery_active"].append(bool(cerebellar_recovery_active))
    trace["cerebellar_recovery_blend_fraction"].append(float(cerebellar_recovery_blend_fraction))
    trace["policy_action"].append(np.asarray(target, dtype=np.float64).copy())
    trace["controller_target_velocity"].append(np.asarray(target_velocity, dtype=np.float64).copy())
    trace["pelvis_pose"].append(data.qpos[:7].copy())
    trace["pelvis_velocity"].append(data.qvel[:6].copy())
    trace["torso_quaternion"].append(data.xquat[ids.torso].copy())
    trace["ball_pose"].append(data.qpos[ids.ball_qpos : ids.ball_qpos + 7].copy())
    trace["ball_velocity"].append(data.qvel[ids.ball_qvel : ids.ball_qvel + 6].copy())
    trace["right_foot_position"].append(data.xpos[ids.right_ankle].copy())
    trace["right_foot_linear_velocity"].append(data.cvel[ids.right_ankle][3:6].copy())
    normal = (
        np.zeros(3, dtype=np.float64)
        if ball_contact_normal is None
        else np.asarray(ball_contact_normal, dtype=np.float64)
    )
    force_world = (
        np.zeros(3, dtype=np.float64)
        if ball_contact_force_world is None
        else np.asarray(ball_contact_force_world, dtype=np.float64)
    )
    if (
        normal.shape != (3,)
        or force_world.shape != (3,)
        or not np.all(np.isfinite(normal))
        or not np.all(np.isfinite(force_world))
        or not math.isfinite(ball_contact_force_peak_n)
        or ball_contact_force_peak_n < 0.0
    ):
        raise FloatingPointError("G1 contact dynamics trace is invalid")
    trace["ball_contact_force_peak_n"].append(float(ball_contact_force_peak_n))
    trace["ball_contact_normal"].append(normal.copy())
    trace["ball_contact_force_world"].append(force_world.copy())
    trace["controller_mode"].append(mode)
    trace["event_phase"].append(int(event_phase))
    trace["policy_phase"].append(phase)
    trace["goal_crossing"].append(crossed)


def _residual_for_frame(
    controller: G1ApproachStrikeResidualController | None,
    *,
    data: Any,
    ids: _ModelIds,
    target: np.ndarray,
    event_phase: G1FootballEventPhase,
    baseline_torque: np.ndarray,
) -> tuple[np.ndarray, bool, float]:
    if controller is None:
        return np.zeros(29, dtype=np.float64), False, 0.0
    decision = controller.propose(
        data=data,
        ids=ids,
        target=target,
        event_phase=int(event_phase),
        baseline_torque=baseline_torque,
    )
    return decision.residual_torque, decision.accepted, decision.confidence


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _football_experiment_context_hash(
    *,
    flow: G1FreeKickFlowConfig,
    sonic: G1SonicRunupConfig | None,
    runup: G1LearnedRunupConfig,
    goal: G1TrainingGoalSpec,
) -> str:
    flow_value = asdict(flow)
    sonic_value = {} if sonic is None else asdict(sonic)
    runup_value = asdict(runup)
    goal_value = asdict(goal)
    for mapping in (flow_value, sonic_value, runup_value, goal_value):
        mapping.pop("schema_version", None)
    sonic_value.pop("planner_seed", None)
    for key in (
        "kick_phase_start_frame",
        "contextual_phase_yaw_threshold_rad",
        "contextual_high_yaw_kick_phase_start_frame",
        "contextual_phase_calibration_hash",
        "proprioceptive_router_hash",
        "football_outcome_model_hash",
        "football_retry_recovery_duration_sec",
        "football_retry_follow_through_gain_scale",
    ):
        flow_value.pop(key, None)
    # Preserve compatibility with v18 outcome artifacts: the v19 vertical
    # correction is semantically absent when it is exactly zero.
    if float(flow_value.get("aim_bias_z_m", 0.0)) == 0.0:
        flow_value.pop("aim_bias_z_m", None)
    return canonical_hash(
        {
            "flow_config": flow_value,
            "sonic_runup_config": sonic_value,
            "runup_config": runup_value,
            "goal_spec": goal_value,
        }
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "G1FootballEventPhase",
    "G1FreeKickEvidence",
    "G1FreeKickFlowConfig",
    "G1FreeKickResult",
    "run_g1_free_kick_showcase",
]
