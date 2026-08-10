"""Coupled two-G1 pass-to-shot experiment in one MuJoCo world.

Unlike :mod:`g1_two_player_relay`, this experiment does not transfer a measured
ball state between two independent episodes.  Both G1 bodies, both live
RoboNaldo policies, and the one physical ball advance through the same MuJoCo
solver.  The module remains deliberately SIM-only and records machine-checkable
physics traces; rendered pixels are never used as promotion evidence.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import math
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw.growth.ballistic_contact_impulse_actor import (
    G1BallisticContactImpulseActor,
    g1_ballistic_contact_impulse_effect,
    load_g1_ballistic_contact_impulse_actor,
)
from rosclaw.growth.learners import (
    IQLResidualDecision,
    IQLResidualGuardConfig,
    NumpyIQLActor,
    SupportBoundIQLResidualActor,
)
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    _adapt_target,
    _load_robonaldo,
    _policy_repeat_count,
    _quaternion_multiply,
    _roll_pitch,
    trajectory_digest,
)
from rosclaw.simforge.g1_cerebellar_recovery import (
    shared_post_impact_recovery_config,
)
from rosclaw.simforge.g1_stadium_scene import (
    G1TrainingGoalSpec,
    apply_g1_compliant_goal_net_force,
    build_g1_coupled_stadium_model,
    build_g1_three_player_stadium_model,
)
from rosclaw.simforge.g1_two_player_relay import (
    _base_scenario,
    _recovery_config,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    G1_HARD_TORQUE_LIMITS,
    ShotParameters,
    hash_bytes,
    hash_json,
)

_MOTION_REL = Path("policy/robonaldo/model/freekick_motion.npz")
_SCENE_REL = Path("g1_description/scene_with_ball.xml")
_MODEL_REL = Path("g1_description/g1_liao.xml")
_PASSER_ORIGIN = np.asarray((3.7452039279962945, -0.16406006503921598, 0.0))
_PASSER_YAW = math.pi
_SHOOTER_START_SEC = 2.02
_CONTROL_DT = 0.02
_PHYSICS_DT = 0.002
_SUBSTEPS = 10
_TOTAL_TIME_SEC = 15.0
# Frozen from the receiver's foot/ball geometry and kick-prior contact plane,
# not from the later scoring outcome.  It is the physical ball-center pocket
# at first receiver contact.
_PASS_RECEPTION_TARGET = np.asarray((1.00, 0.0, 0.115), dtype=np.float64)


@dataclass(frozen=True)
class G1JointGuardConfig:
    margin_rad: float = 0.04
    prediction_horizon_sec: float = 0.08
    boundary_kp: float = 80.0
    boundary_kd: float = 6.0
    schema_version: str = "rosclaw.growth.g1_joint_guard_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.margin_rad,
            self.prediction_horizon_sec,
            self.boundary_kp,
            self.boundary_kd,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("joint guard config must be finite")
        if not 0.01 <= self.margin_rad <= 0.10:
            raise ValueError("joint guard margin must be in [0.01, 0.10]")
        if not 0.02 <= self.prediction_horizon_sec <= 0.20:
            raise ValueError("joint guard horizon must be in [0.02, 0.20]")
        if not 20.0 <= self.boundary_kp <= 200.0:
            raise ValueError("joint guard kp must be in [20, 200]")
        if not 1.0 <= self.boundary_kd <= 20.0:
            raise ValueError("joint guard kd must be in [1, 20]")


@dataclass(frozen=True)
class G1GoalkeeperConfig:
    """Causal ball-flight tracker around the qualified locomotion policy."""

    depth_from_goal_line_m: float = 0.48
    reaction_delay_sec: float = 0.12
    lateral_position_gain: float = 1.35
    maximum_lateral_speed_mps: float = 0.38
    ready_shuffle_speed_mps: float = 0.10
    ready_shuffle_period_sec: float = 2.4
    arm_spread_rad: float = 0.24
    maximum_waist_lean_rad: float = 0.08
    schema_version: str = "rosclaw.simforge.g1_goalkeeper_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.depth_from_goal_line_m,
            self.reaction_delay_sec,
            self.lateral_position_gain,
            self.maximum_lateral_speed_mps,
            self.ready_shuffle_speed_mps,
            self.ready_shuffle_period_sec,
            self.arm_spread_rad,
            self.maximum_waist_lean_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("goalkeeper config must be finite")
        if not 0.25 <= self.depth_from_goal_line_m <= 0.80:
            raise ValueError("goalkeeper depth must be in [0.25, 0.80] m")
        if not 0.08 <= self.reaction_delay_sec <= 0.35:
            raise ValueError("goalkeeper reaction delay must be in [0.08, 0.35] s")
        if not 0.5 <= self.lateral_position_gain <= 2.5:
            raise ValueError("goalkeeper position gain must be in [0.5, 2.5]")
        if not 0.20 <= self.maximum_lateral_speed_mps <= 0.40:
            raise ValueError("goalkeeper lateral speed must be in [0.20, 0.40] m/s")
        if not 0.0 <= self.ready_shuffle_speed_mps <= 0.20:
            raise ValueError("goalkeeper ready shuffle must be in [0, 0.20] m/s")
        if not 1.5 <= self.ready_shuffle_period_sec <= 4.0:
            raise ValueError("goalkeeper shuffle period must be in [1.5, 4.0] s")
        if not 0.10 <= self.arm_spread_rad <= 0.45:
            raise ValueError("goalkeeper arm spread must be in [0.10, 0.45] rad")
        if not 0.0 <= self.maximum_waist_lean_rad <= 0.15:
            raise ValueError("goalkeeper waist lean must be in [0, 0.15] rad")


def shared_post_impact_simulation_kwargs() -> dict[str, Any]:
    """Symmetric post-impact ownership selected by bounded SIM search."""

    recovery = shared_post_impact_recovery_config()
    guard = G1JointGuardConfig()
    return {
        "passer_recovery_config": recovery,
        "passer_post_policy_frame": 265,
        "passer_post_policy_blend_frames": 2,
        "passer_joint_guard_enabled": True,
        "passer_post_policy_neutral_velocity_enabled": True,
        "passer_joint_guard_config": guard,
        "passer_post_policy_recovery_enabled": True,
        "shooter_recovery_config": recovery,
        "shooter_post_policy_frame": 275,
        "shooter_post_policy_blend_frames": 0,
        "shooter_joint_guard_enabled": True,
        "shooter_post_policy_neutral_velocity_enabled": True,
        "shooter_joint_guard_config": guard,
        "shooter_post_policy_recovery_enabled": True,
    }


def trained_coupled_skill_simulation_kwargs() -> dict[str, Any]:
    """Return the retained pass/shot skill plus its shared recovery control."""

    values = shared_post_impact_simulation_kwargs()
    values.update(
        {
            "shooter_parameter_overrides": {
                "foot_yaw_offset": 0.085,
                "foot_pitch_offset": 0.010,
            },
            # This expert is selected only after the live ball-phase
            # synchronizer has causally latched at least one hold frame.
            "shooter_early_arrival_parameter_overrides": {
                "foot_yaw_offset": 0.115,
                "foot_pitch_offset": 0.025,
            },
        }
    )
    return values


@dataclass(frozen=True)
class G1CoupledRelayResult:
    """Auditable outcome of one simultaneous two-body physics rollout."""

    finite_state: bool
    pass_contact_observed: bool
    shot_contact_observed: bool
    pass_contact_time_sec: float | None
    shot_contact_time_sec: float | None
    pass_peak_ball_speed_mps: float
    shot_peak_ball_speed_mps: float
    goal_crossed: bool
    goal_crossing_y_m: float | None
    goal_crossing_z_m: float | None
    target_error_m: float | None
    passer_min_pelvis_height_m: float
    shooter_min_pelvis_height_m: float
    passer_roll_peak_rad: float
    passer_pitch_peak_rad: float
    shooter_roll_peak_rad: float
    shooter_pitch_peak_rad: float
    passer_tail_wobble_index: float
    shooter_tail_wobble_index: float
    receiver_phase_hold_frames: int
    receiver_phase_advance_frames: int
    receiver_max_ball_phase_error_m: float
    robot_robot_contact_count: int
    joint_limit_violation: bool
    torque_limit_violation: bool
    actuator_saturation: bool
    physics_steps: int
    passer_support_foot_slip_m: float = 0.0
    shooter_support_foot_slip_m: float = 0.0
    passer_post_contact_support_foot_slip_m: float = 0.0
    shooter_post_contact_support_foot_slip_m: float = 0.0
    passer_contact_impulse_ns: float = 0.0
    shooter_contact_impulse_ns: float = 0.0
    passer_post_kick_fall: bool = False
    shooter_post_kick_fall: bool = False
    shooter_learned_torque_fraction: float = 0.0
    shooter_learned_torque_fallback_fraction: float = 0.0
    shooter_learned_torque_mean_confidence: float = 0.0
    shooter_learned_torque_peak_residual_nm: float = 0.0
    shooter_learned_torque_support_rms_peak: float = 0.0
    shooter_joint_guard_fraction: float = 0.0
    shooter_joint_guard_route: str = "disabled"
    pass_delivery_position_m: tuple[float, float, float] | None = None
    pass_delivery_error_m: float | None = None
    pass_delivery_lateral_error_m: float | None = None
    passer_joint_guard_fraction: float = 0.0
    passer_joint_guard_route: str = "disabled"
    passer_recovery_active_fraction: float = 0.0
    shooter_recovery_active_fraction: float = 0.0
    passer_recovery_peak_blend_fraction: float = 0.0
    shooter_recovery_peak_blend_fraction: float = 0.0
    shooter_aim_expert_route: str = "nominal"
    shooter_early_arrival_expert_fraction: float = 0.0
    shooter_ballistic_actor_active_fraction: float = 0.0
    shooter_ballistic_actor_peak_torque_nm: float = 0.0
    shooter_ballistic_actor_hash: str | None = None
    goalkeeper_enabled: bool = False
    goalkeeper_reaction_active_fraction: float = 0.0
    goalkeeper_lateral_displacement_m: float = 0.0
    goalkeeper_peak_lateral_speed_mps: float = 0.0
    goalkeeper_min_pelvis_height_m: float | None = None
    goalkeeper_ball_contact_observed: bool = False
    goalkeeper_ball_contact_time_sec: float | None = None
    passer_joint_limit_violation: bool = False
    shooter_joint_limit_violation: bool = False
    goalkeeper_joint_limit_violation: bool = False
    schema_version: str = "rosclaw.g1_goalforge.coupled_relay_result.v3"

    @property
    def pass_precision_passed(self) -> bool:
        """Whether the pass reaches the frozen receiver contact pocket."""

        return bool(
            self.pass_delivery_error_m is not None
            and self.pass_delivery_error_m <= 0.05
            and self.pass_delivery_lateral_error_m is not None
            and self.pass_delivery_lateral_error_m <= 0.03
        )

    @property
    def passed(self) -> bool:
        return bool(
            self.finite_state
            and self.pass_contact_observed
            and self.shot_contact_observed
            and self.pass_contact_time_sec is not None
            and self.shot_contact_time_sec is not None
            and self.pass_contact_time_sec < self.shot_contact_time_sec
            and self.pass_peak_ball_speed_mps >= 0.55
            and self.pass_precision_passed
            and self.shot_peak_ball_speed_mps >= 6.0
            and self.goal_crossed
            and self.target_error_m is not None
            and self.target_error_m <= 0.48
            and self.passer_min_pelvis_height_m >= 0.55
            and self.shooter_min_pelvis_height_m >= 0.55
            and self.passer_roll_peak_rad <= 0.55
            and self.passer_pitch_peak_rad <= 0.65
            and self.shooter_roll_peak_rad <= 0.55
            and self.shooter_pitch_peak_rad <= 0.65
            and not self.joint_limit_violation
            and not self.torque_limit_violation
            and not self.actuator_saturation
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        value["pass_precision_passed"] = self.pass_precision_passed
        return value


@dataclass(frozen=True)
class G1CoupledRelayRobustnessCase:
    shooter_start_offset_sec: float
    passed: bool
    shot_contact_time_sec: float | None
    goal_crossing_y_m: float | None
    goal_crossing_z_m: float | None
    target_error_m: float | None
    phase_hold_frames: int
    phase_advance_frames: int
    minimum_pelvis_height_m: float
    joint_limit_violation: bool
    torque_limit_violation: bool
    schema_version: str = "rosclaw.g1_goalforge.coupled_relay_robustness_case.v1"


@dataclass(frozen=True)
class G1CoupledRelayEvidence:
    """Strict-replay receipt for the same-world two-G1 experiment."""

    body_hash: str
    kick_prior_hash: str
    standby_policy_hash: str
    backend_commit: str
    implementation_hash: str
    request_hash: str
    trajectory_hash: str
    trajectory_digest: str
    strict_replay: bool
    result: G1CoupledRelayResult
    receiver_timing_parent: tuple[G1CoupledRelayRobustnessCase, ...]
    receiver_timing_robustness: tuple[G1CoupledRelayRobustnessCase, ...]
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    simultaneous_two_body_physics: bool = True
    shared_ball_state: bool = True
    hardware_command_sent: bool = False
    environment_hash: str = ""
    schema_version: str = "rosclaw.g1_goalforge.coupled_relay_evidence.v1"

    @property
    def passed(self) -> bool:
        return bool(
            self.strict_replay
            and self.result.passed
            and len(self.receiver_timing_parent) == 5
            and len(self.receiver_timing_robustness) == 5
            and all(case.passed for case in self.receiver_timing_robustness)
            and sum(case.passed for case in self.receiver_timing_robustness)
            > sum(case.passed for case in self.receiver_timing_parent)
            and self.activation_ceiling == "SIM_ONLY"
            and not self.hardware_command_sent
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["result"] = self.result.to_dict()
        value["receiver_timing_parent"] = [asdict(case) for case in self.receiver_timing_parent]
        value["receiver_timing_robustness"] = [
            asdict(case) for case in self.receiver_timing_robustness
        ]
        value["passed"] = self.passed
        value["claims"] = {
            "simultaneous_two_body_physics": True,
            "single_shared_ball": True,
            "independent_live_policy_instances": True,
            "strict_deterministic_replay": self.strict_replay,
            "receiver_timing_robustness_passed": all(
                case.passed for case in self.receiver_timing_robustness
            ),
            "receiver_timing_successes_before": sum(
                case.passed for case in self.receiver_timing_parent
            ),
            "receiver_timing_successes_after": sum(
                case.passed for case in self.receiver_timing_robustness
            ),
            "pixels_used_for_promotion": False,
            "real_hardware": False,
        }
        return value


@dataclass
class _Robot:
    role: str
    prefix: str
    origin: np.ndarray
    world_from_local_quat: np.ndarray
    qpos_base: int
    qvel_base: int
    joint_qpos: np.ndarray
    joint_qvel: np.ndarray
    actuators: np.ndarray
    joint_ids: np.ndarray
    pelvis_body: int
    torso_body: int
    left_ankle_body: int
    right_ankle_body: int
    state: Any
    output: Any
    policy: Any
    standby_output: Any | None
    standby_policy: Any | None
    parameters: ShotParameters
    start_sec: float
    hold_target: np.ndarray
    last_target: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    recovery_controller: Any | None = None
    phase_hold_frames: int = 0
    phase_hold_remaining: int = 0
    post_policy_frame: int | None = None
    post_policy_blend_frames: int = 0
    post_policy_active: bool = False
    post_policy_transition_step: int = 0
    post_policy_origin_target: np.ndarray | None = None
    post_policy_origin_kp: np.ndarray | None = None
    post_policy_origin_kd: np.ndarray | None = None
    post_policy_blend_fraction: float = 0.0
    post_policy_activation_simulation_frame: int | None = None
    entered: bool = False
    contact_latched: bool = False
    contact_time: float | None = None
    latest_left_support: bool = False
    latest_right_support: bool = False
    phase_hold_count: int = 0
    phase_advance_count: int = 0
    max_ball_phase_error_m: float = 0.0
    last_phase_correction: int = 0
    phase_sync_enabled: bool = False
    left_support_anchor: np.ndarray | None = None
    right_support_anchor: np.ndarray | None = None
    latest_support_slip_m: float = 0.0
    peak_support_slip_m: float = 0.0
    post_contact_peak_support_slip_m: float = 0.0
    contact_impulse_ns: float = 0.0
    recovery_torque_actor: Any | None = None
    learned_torque_frame_count: int = 0
    learned_torque_fallback_count: int = 0
    learned_torque_confidence_sum: float = 0.0
    learned_torque_peak_residual_nm: float = 0.0
    learned_torque_support_rms_peak: float = 0.0
    joint_guard_enabled: bool = False
    joint_guard_frame_count: int = 0
    post_policy_neutral_velocity_enabled: bool = False
    joint_guard_config: G1JointGuardConfig = G1JointGuardConfig()
    joint_guard_late_config: G1JointGuardConfig | None = None
    selected_joint_guard_config: G1JointGuardConfig | None = None
    joint_guard_route: str = "disabled"
    post_policy_recovery_enabled: bool = False
    recovery_active_frame_count: int = 0
    recovery_peak_blend_fraction: float = 0.0
    last_recovery_active: bool = False
    last_recovery_blend_fraction: float = 0.0
    early_arrival_parameters: ShotParameters | None = None
    early_arrival_expert_frame_count: int = 0


def run_g1_coupled_relay(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
) -> G1CoupledRelayEvidence:
    """Run and strictly replay a SIM-only pass and finish in one world."""

    evidence_root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if evidence_root == checkout or checkout in evidence_root.parents:
        raise ValueError("coupled relay evidence must be outside the source checkout")
    evidence_root.mkdir(parents=True, exist_ok=False)

    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    recovery = shared_post_impact_recovery_config()
    controller_kwargs = trained_coupled_skill_simulation_kwargs()
    # The early-arrival aim expert was learned in the v2 stadium regime. It is
    # intentionally excluded from the pinned v1 timing-ablation receipt; using
    # it here narrows the legacy success window instead of testing the phase
    # synchronizer in isolation.
    controller_kwargs.pop("shooter_early_arrival_parameter_overrides")
    standby_policy_hash = _standby_policy_hash(asset_root.expanduser().resolve())
    implementation_hash = hash_bytes(Path(__file__).read_bytes())
    request = {
        "schema_version": "rosclaw.g1_goalforge.coupled_relay_request.v1",
        "body_hash": backend.qualification.body_hash,
        "kick_prior_hash": backend.qualification.kick_prior_hash,
        "implementation_hash": implementation_hash,
        "standby_policy_hash": standby_policy_hash,
        "passer_origin_m": _PASSER_ORIGIN.tolist(),
        "passer_yaw_rad": _PASSER_YAW,
        "shooter_policy_start_sec": _SHOOTER_START_SEC,
        "receiver_timing_robustness_offsets_sec": [-0.04, -0.02, 0.0, 0.02, 0.04],
        "receiver_timing_ablation": {
            "parent": "phase_synchronizer_disabled",
            "candidate": "phase_synchronizer_enabled",
            "promotion_rule": "candidate_successes_must_exceed_parent_and_equal_5",
        },
        "passer_post_policy_frame": 265,
        "passer_post_policy_blend_frames": 2,
        "shooter_post_policy_frame": 275,
        "shooter_post_policy_blend_frames": 0,
        "pass_reception_target_m": _PASS_RECEPTION_TARGET.tolist(),
        "target_m": [5.0, 1.10, 1.09],
        "passer_parameters": {
            "stance_offset_y": -0.04,
            "pelvis_yaw_offset": 0.10,
            "com_shift_y": -0.04,
            "swing_amplitude": 0.75,
            "swing_speed_scale": 0.80,
            "recovery_step_length": 0.03,
        },
        "shooter_parameters": {
            "stance_offset_y": -0.06,
            "pelvis_yaw_offset": 0.175,
            "com_shift_y": -0.065,
            "swing_speed_scale": 0.90,
            "foot_yaw_offset": 0.085,
            "foot_pitch_offset": 0.010,
            "recovery_step_length": 0.055,
        },
        "shooter_early_arrival_expert": "excluded_from_legacy_timing_ablation",
        "recovery_config": asdict(recovery),
        "control_contract": {
            "control_dt_sec": _CONTROL_DT,
            "physics_dt_sec": _PHYSICS_DT,
            "physics_substeps": _SUBSTEPS,
            "total_time_sec": _TOTAL_TIME_SEC,
            "torque_guard_scale": 0.85,
            "ball_ground_friction": 0.10,
            "receiver_phase_synchronizer": {
                "policy_frame_range": [184, 252],
                "deadband_m": 0.025,
                "expected_ball_x_polynomial": [
                    1.32676487e-4,
                    -2.54767831e-2,
                    2.12799256,
                ],
            },
            "standby_policy": "RoboNaldo_LocoMode_zero_velocity",
            "standby_command_is_physical_zero": True,
            "shared_recovery_continues_across_handoff": True,
            "post_contact_joint_guard_both_roles": True,
            "state_transitions": [
                "G1_A_FreeKick_to_LocoMode",
                "G1_B_LocoMode_to_FreeKick_to_LocoMode",
            ],
        },
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "runtime": coupled_runtime_manifest(),
    }
    request["environment_hash"] = hash_json(request["runtime"])
    request_path = evidence_root / "request.json"
    _write_json(request_path, request)
    result, trajectory = _simulate(asset_root, backend, **controller_kwargs)
    replay_result, replay_trajectory = _simulate(asset_root, backend, **controller_kwargs)
    strict_replay = bool(
        result.to_dict() == replay_result.to_dict()
        and trajectory_digest(trajectory) == trajectory_digest(replay_trajectory)
    )
    parent_cases: list[G1CoupledRelayRobustnessCase] = []
    robustness_cases: list[G1CoupledRelayRobustnessCase] = []
    for offset in (-0.04, -0.02, 0.0, 0.02, 0.04):
        parent_result = (
            result
            if offset == 0.0
            else _simulate(
                asset_root,
                backend,
                shooter_start_sec=_SHOOTER_START_SEC + offset,
                receiver_phase_sync_enabled=False,
                **controller_kwargs,
            )[0]
        )
        parent_cases.append(_robustness_case(offset, parent_result))
        case_result = (
            result
            if offset == 0.0
            else _simulate(
                asset_root,
                backend,
                shooter_start_sec=_SHOOTER_START_SEC + offset,
                **controller_kwargs,
            )[0]
        )
        robustness_cases.append(_robustness_case(offset, case_result))
    trajectory_path = evidence_root / "trajectory.npz"
    np.savez_compressed(trajectory_path, **trajectory)  # type: ignore[arg-type]
    evidence = G1CoupledRelayEvidence(
        body_hash=backend.qualification.body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
        standby_policy_hash=standby_policy_hash,
        backend_commit=backend.qualification.backend_commit,
        implementation_hash=implementation_hash,
        request_hash=hash_bytes(request_path.read_bytes()),
        trajectory_hash=_file_hash(trajectory_path),
        trajectory_digest=trajectory_digest(trajectory),
        strict_replay=strict_replay,
        result=result,
        receiver_timing_parent=tuple(parent_cases),
        receiver_timing_robustness=tuple(robustness_cases),
        environment_hash=str(request["environment_hash"]),
    )
    _write_json(evidence_root / "g1-coupled-relay.json", evidence.to_dict())
    return evidence


def _simulate(
    asset_root: Path,
    backend: G1MuJoCoBackend,
    *,
    shooter_start_sec: float = _SHOOTER_START_SEC,
    shooter_target: tuple[float, float, float] = (5.0, 1.10, 1.09),
    shooter_policy_target: tuple[float, float, float] | None = None,
    shooter_parameter_overrides: dict[str, float] | None = None,
    shooter_early_arrival_parameter_overrides: dict[str, float] | None = None,
    passer_parameter_overrides: dict[str, float] | None = None,
    passer_recovery_config: Any | None = None,
    passer_post_policy_frame: int | None = 290,
    passer_post_policy_blend_frames: int = 0,
    passer_joint_guard_enabled: bool = False,
    passer_precontact_joint_guard_enabled: bool = False,
    passer_waist_pitch_target_margin_rad: float = 0.0,
    passer_post_policy_neutral_velocity_enabled: bool = False,
    passer_joint_guard_config: G1JointGuardConfig | None = None,
    passer_post_policy_recovery_enabled: bool = False,
    ball_ground_friction: float = 0.10,
    receiver_phase_sync_enabled: bool = True,
    shooter_recovery_candidate_path: Path | None = None,
    shooter_recovery_residual_config: IQLResidualGuardConfig | None = None,
    shooter_recovery_config: Any | None = None,
    shooter_post_policy_frame: int | None = 430,
    shooter_post_policy_blend_frames: int = 0,
    shooter_joint_guard_enabled: bool = False,
    shooter_post_policy_neutral_velocity_enabled: bool = False,
    shooter_joint_guard_config: G1JointGuardConfig | None = None,
    shooter_joint_guard_late_config: G1JointGuardConfig | None = None,
    shooter_post_policy_recovery_enabled: bool = False,
    shooter_ballistic_actor_path: Path | None = None,
    shooter_ballistic_actor_proximity_m: float | None = None,
    passer_origin: tuple[float, float, float] | None = None,
    passer_ball_local_xy: tuple[float, float] = (1.205, -0.16),
    goal_spec: G1TrainingGoalSpec | None = None,
    goalkeeper_config: G1GoalkeeperConfig | None = None,
    unified_stadium_scene: bool = False,
) -> tuple[G1CoupledRelayResult, dict[str, np.ndarray]]:
    import mujoco

    root = asset_root.expanduser().resolve()
    active_passer_origin = np.asarray(
        _PASSER_ORIGIN if passer_origin is None else passer_origin,
        dtype=np.float64,
    )
    if active_passer_origin.shape != (3,) or not np.all(np.isfinite(active_passer_origin)):
        raise ValueError("passer origin must be a finite xyz vector")
    active_passer_ball_xy = np.asarray(passer_ball_local_xy, dtype=np.float64)
    if (
        active_passer_ball_xy.shape != (2,)
        or not np.all(np.isfinite(active_passer_ball_xy))
        or not 1.05 <= active_passer_ball_xy[0] <= 1.35
        or not -0.30 <= active_passer_ball_xy[1] <= -0.08
    ):
        raise ValueError("passer ball pocket must be finite and inside the qualified envelope")
    active_goal = goal_spec or G1TrainingGoalSpec(
        plane_x_m=shooter_target[0],
        target_y_m=shooter_target[1],
        target_z_m=shooter_target[2],
    )
    if not np.allclose(
        np.asarray(shooter_target, dtype=np.float64),
        np.asarray(
            (active_goal.plane_x_m, active_goal.target_y_m, active_goal.target_z_m),
            dtype=np.float64,
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("shooter target and shared stadium goal target must match")
    active_policy_target = np.asarray(
        shooter_target if shooter_policy_target is None else shooter_policy_target,
        dtype=np.float64,
    )
    if active_policy_target.shape != (3,) or not np.all(np.isfinite(active_policy_target)):
        raise ValueError("shooter policy target must be a finite xyz vector")
    if goalkeeper_config is not None and not unified_stadium_scene:
        raise ValueError("goalkeeper requires the unified stadium physics scene")
    if passer_waist_pitch_target_margin_rad != 0.0 and not (
        0.005 <= passer_waist_pitch_target_margin_rad <= 0.05
    ):
        raise ValueError("passer waist-pitch margin must be zero or in [0.005, 0.05]")
    model = _coupled_model(
        root,
        passer_origin=active_passer_origin,
        goal=active_goal,
        goalkeeper_config=goalkeeper_config,
        unified_stadium_scene=unified_stadium_scene,
    )
    data = mujoco.MjData(model)
    model.opt.timestep = _PHYSICS_DT
    scenario = _base_scenario()
    ball_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    ball_joint = int(model.body_jntadr[ball_body])
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    ball_qvel = int(model.jnt_dofadr[ball_joint])
    floor_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if not 0.03 <= ball_ground_friction <= 0.80:
        raise ValueError("coupled relay ball friction must be in [0.03, 0.80]")
    model.geom_friction[ball_geom] = (
        active_goal.ball_contact_sliding_friction,
        active_goal.ball_torsional_friction,
        active_goal.ball_rolling_friction,
    )
    model.geom_friction[floor_geom, 0] = scenario.support_ground_friction
    for pair_index in range(int(model.npair)):
        pair_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, pair_index) or ""
        if pair_name == "ball_floor":
            model.pair_friction[pair_index] = (
                ball_ground_friction,
                ball_ground_friction,
                active_goal.ball_torsional_friction,
                active_goal.ball_rolling_friction,
                active_goal.ball_rolling_friction,
            )

    state_type, output_type, policy_type, mujoco_to_isaac = _load_robonaldo(root)
    shooter_ballistic_actor: G1BallisticContactImpulseActor | None = None
    if shooter_ballistic_actor_path is not None:
        shooter_ballistic_actor = load_g1_ballistic_contact_impulse_actor(
            shooter_ballistic_actor_path
        )
        if shooter_ballistic_actor_proximity_m is not None:
            shooter_ballistic_actor = replace(
                shooter_ballistic_actor,
                maximum_foot_ball_distance_m=shooter_ballistic_actor_proximity_m,
            )
        if shooter_ballistic_actor.body_hash != backend.qualification.body_hash:
            raise ValueError("ballistic contact actor Body hash does not match coupled G1")
    with np.load(root / _MOTION_REL) as motion:
        initial_position = np.asarray(motion["body_pos_w"][0, 0], dtype=np.float64)
        initial_quaternion = np.asarray(motion["body_quat_w"][0, 0], dtype=np.float64)
        initial_joints = np.asarray(motion["joint_pos"][0][mujoco_to_isaac], dtype=np.float64)
    standby_target = np.asarray(
        (
            -0.2,
            0.0,
            0.0,
            0.42,
            -0.23,
            0.0,
            -0.2,
            0.0,
            0.0,
            0.42,
            -0.23,
            0.0,
            0.0,
            0.0,
            0.0,
            0.35,
            0.18,
            0.0,
            0.87,
            0.0,
            0.0,
            0.0,
            0.35,
            -0.18,
            0.0,
            0.87,
            0.0,
            0.0,
            0.0,
        ),
        dtype=np.float64,
    )
    standby_kp = np.asarray(
        (
            100,
            100,
            100,
            150,
            40,
            40,
            100,
            100,
            100,
            150,
            40,
            40,
            300,
            300,
            300,
            100,
            100,
            50,
            50,
            20,
            20,
            20,
            100,
            100,
            50,
            50,
            20,
            20,
            20,
        ),
        dtype=np.float64,
    )
    standby_kd = np.asarray(
        (
            2,
            2,
            2,
            4,
            2,
            2,
            2,
            2,
            2,
            4,
            2,
            2,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
        ),
        dtype=np.float64,
    )

    shooter_parameters = ShotParameters(
        stance_offset_y=-0.06,
        pelvis_yaw_offset=0.175,
        com_shift_y=-0.065,
        swing_speed_scale=0.90,
        foot_yaw_offset=0.03025,
        recovery_step_length=0.055,
        policy_type="parameter",
    )
    if shooter_parameter_overrides:
        shooter_parameters = replace(
            shooter_parameters,
            **shooter_parameter_overrides,
        )
    shooter_early_arrival_parameters = (
        replace(
            shooter_parameters,
            **shooter_early_arrival_parameter_overrides,
        )
        if shooter_early_arrival_parameter_overrides
        else None
    )
    passer_parameters = ShotParameters(
        stance_offset_y=-0.04,
        pelvis_yaw_offset=0.10,
        com_shift_y=-0.04,
        swing_amplitude=0.75,
        swing_speed_scale=0.80,
        recovery_step_length=0.03,
        policy_type="parameter",
    )
    if passer_parameter_overrides:
        passer_parameters = replace(
            passer_parameters,
            **passer_parameter_overrides,
        )
    passer_scenario = replace(
        scenario,
        scenario_id="goalforge-coupled-g1-a-soft-back-pass",
        ball_x_m=float(active_passer_ball_xy[0]),
        ball_y_m=float(active_passer_ball_xy[1]),
        ball_velocity_x_mps=0.0,
        ball_velocity_y_mps=0.0,
        ball_launch_delay_sec=0.0,
        ball_ground_friction=ball_ground_friction,
        target_y_m=0.0,
        target_z_m=0.20,
    )
    shooter_scenario = replace(
        scenario,
        scenario_id="goalforge-coupled-g1-b-fast-high-finish",
        ball_x_m=1.25,
        ball_y_m=0.0,
        ball_velocity_x_mps=-0.59,
        ball_velocity_y_mps=0.0,
        ball_launch_delay_sec=0.0,
        ball_ground_friction=ball_ground_friction,
        target_y_m=shooter_target[1],
        # GoalForgeScenario v1 caps only the recovery controller's context at
        # 1.2 m.  The separately bound stadium target can be a higher corner.
        target_z_m=min(shooter_target[2], 1.20),
    )
    passer_recovery = backend.build_cerebellar_recovery_controller(
        passer_scenario, passer_recovery_config or _recovery_config()
    )
    shooter_recovery = backend.build_cerebellar_recovery_controller(
        shooter_scenario, shooter_recovery_config or _recovery_config()
    )
    passer_recovery.reset()
    shooter_recovery.reset()
    if shooter_recovery_candidate_path is not None:
        shooter_recovery_actor: NumpyIQLActor | SupportBoundIQLResidualActor | None
        if shooter_recovery_residual_config is None:
            shooter_recovery_actor = NumpyIQLActor.load(shooter_recovery_candidate_path)
        else:
            shooter_recovery_actor = SupportBoundIQLResidualActor.load(
                shooter_recovery_candidate_path,
                shooter_recovery_residual_config,
            )
    else:
        shooter_recovery_actor = None
    shooter = _make_robot(
        model=model,
        data=data,
        role="shooter",
        prefix="",
        origin=np.zeros(3, dtype=np.float64),
        yaw=0.0,
        state_type=state_type,
        output_type=output_type,
        policy_type=policy_type,
        parameters=shooter_parameters,
        start_sec=shooter_start_sec,
        initial_position=initial_position,
        initial_quaternion=initial_quaternion,
        initial_joints=standby_target,
        target_local=np.asarray(active_policy_target, dtype=np.float32),
        phase_hold_frames=0,
        standby_target=standby_target,
        standby_kp=standby_kp,
        standby_kd=standby_kd,
        use_locomotion_standby=True,
        recovery_controller=shooter_recovery,
        post_policy_frame=shooter_post_policy_frame,
        post_policy_blend_frames=shooter_post_policy_blend_frames,
        phase_sync_enabled=receiver_phase_sync_enabled,
        recovery_torque_actor=shooter_recovery_actor,
        joint_guard_enabled=shooter_joint_guard_enabled,
        post_policy_neutral_velocity_enabled=shooter_post_policy_neutral_velocity_enabled,
        joint_guard_config=shooter_joint_guard_config or G1JointGuardConfig(),
        joint_guard_late_config=shooter_joint_guard_late_config,
        post_policy_recovery_enabled=shooter_post_policy_recovery_enabled,
        early_arrival_parameters=shooter_early_arrival_parameters,
    )
    passer = _make_robot(
        model=model,
        data=data,
        role="passer",
        prefix="passer_",
        origin=active_passer_origin,
        yaw=_PASSER_YAW,
        state_type=state_type,
        output_type=output_type,
        policy_type=policy_type,
        parameters=passer_parameters,
        start_sec=0.0,
        initial_position=initial_position,
        initial_quaternion=initial_quaternion,
        initial_joints=initial_joints,
        target_local=np.asarray((5.0, 0.0, 0.20), dtype=np.float32),
        phase_hold_frames=0,
        standby_target=None,
        standby_kp=None,
        standby_kd=None,
        use_locomotion_standby=True,
        recovery_controller=passer_recovery,
        post_policy_frame=passer_post_policy_frame,
        post_policy_blend_frames=passer_post_policy_blend_frames,
        phase_sync_enabled=False,
        recovery_torque_actor=None,
        joint_guard_enabled=passer_joint_guard_enabled,
        post_policy_neutral_velocity_enabled=passer_post_policy_neutral_velocity_enabled,
        joint_guard_config=passer_joint_guard_config or G1JointGuardConfig(),
        joint_guard_late_config=None,
        post_policy_recovery_enabled=passer_post_policy_recovery_enabled,
        early_arrival_parameters=None,
    )
    goalkeeper: _Robot | None = None
    goalkeeper_origin: np.ndarray | None = None
    if goalkeeper_config is not None:
        goalkeeper_origin = np.asarray(
            (
                active_goal.plane_x_m - goalkeeper_config.depth_from_goal_line_m,
                0.0,
                0.0,
            ),
            dtype=np.float64,
        )
        goalkeeper = _make_robot(
            model=model,
            data=data,
            role="goalkeeper",
            prefix="goalkeeper_",
            origin=goalkeeper_origin,
            yaw=math.pi,
            state_type=state_type,
            output_type=output_type,
            policy_type=policy_type,
            parameters=ShotParameters(policy_type="parameter"),
            start_sec=math.inf,
            initial_position=initial_position,
            initial_quaternion=initial_quaternion,
            initial_joints=standby_target,
            target_local=np.asarray((-5.0, 0.0, 0.2), dtype=np.float32),
            phase_hold_frames=0,
            standby_target=standby_target,
            standby_kp=standby_kp,
            standby_kd=standby_kd,
            use_locomotion_standby=True,
            recovery_controller=None,
            post_policy_frame=None,
            post_policy_blend_frames=0,
            phase_sync_enabled=False,
            recovery_torque_actor=None,
            joint_guard_enabled=True,
            post_policy_neutral_velocity_enabled=False,
            joint_guard_config=G1JointGuardConfig(
                margin_rad=0.08,
                prediction_horizon_sec=0.16,
                boundary_kp=140.0,
                boundary_kd=12.0,
            ),
            joint_guard_late_config=None,
            post_policy_recovery_enabled=False,
            early_arrival_parameters=None,
        )
    robots = (passer, shooter) if goalkeeper is None else (passer, shooter, goalkeeper)
    passer_geoms = _robot_geom_ids(model, passer.pelvis_body)
    shooter_geoms = _robot_geom_ids(model, shooter.pelvis_body)
    goalkeeper_geoms = (
        frozenset() if goalkeeper is None else _robot_geom_ids(model, goalkeeper.pelvis_body)
    )
    # One immutable shared ball.  This is the pass scenario's local initial
    # state transformed into the coupled world; there is no later teleport.
    data.qpos[ball_qpos : ball_qpos + 3] = active_passer_origin + _rotate_z(
        np.asarray((*active_passer_ball_xy, active_goal.ball_radius_m), dtype=np.float64),
        _PASSER_YAW,
    )
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[ball_qvel : ball_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)

    for robot in robots:
        _fill_local_state(robot, data, ball_body, ball_qvel)
    _enter_policy(passer)

    hard_limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    guarded_limits = hard_limits * 0.85
    total_frames = int(round(_TOTAL_TIME_SEC / _CONTROL_DT))
    trace: dict[str, list[Any]] = {
        "time": [],
        "ball_pose": [],
        "ball_velocity": [],
        "passer_pelvis_pose": [],
        "shooter_pelvis_pose": [],
        "passer_torso_quaternion": [],
        "shooter_torso_quaternion": [],
        "passer_joint_position": [],
        "shooter_joint_position": [],
        "passer_joint_velocity": [],
        "shooter_joint_velocity": [],
        "passer_joint_torque": [],
        "shooter_joint_torque": [],
        "passer_commanded_torque": [],
        "shooter_commanded_torque": [],
        "passer_safety_projected_torque": [],
        "shooter_safety_projected_torque": [],
        "passer_executed_torque": [],
        "shooter_executed_torque": [],
        "passer_policy_action": [],
        "shooter_policy_action": [],
        "passer_com_position": [],
        "shooter_com_position": [],
        "passer_left_foot_position": [],
        "passer_right_foot_position": [],
        "shooter_left_foot_position": [],
        "shooter_right_foot_position": [],
        "passer_support_foot_slip": [],
        "shooter_support_foot_slip": [],
        "passer_contact_impulse": [],
        "shooter_contact_impulse": [],
        "shooter_learned_torque_active": [],
        "passer_joint_guard_active": [],
        "shooter_joint_guard_active": [],
        "passer_post_policy_blend_fraction": [],
        "shooter_post_policy_blend_fraction": [],
        "passer_recovery_active": [],
        "shooter_recovery_active": [],
        "passer_recovery_blend_fraction": [],
        "shooter_recovery_blend_fraction": [],
        "passer_policy_frame": [],
        "shooter_policy_frame": [],
        "shooter_phase_correction": [],
        "shooter_ballistic_actor_active": [],
        "shooter_ballistic_actor_torque": [],
        "passer_foot_contact": [],
        "shooter_foot_contact": [],
        "ball_contact_role": [],
        "robot_robot_contact_count": [],
    }
    if goalkeeper is not None:
        trace.update(
            {
                "goalkeeper_pelvis_pose": [],
                "goalkeeper_torso_quaternion": [],
                "goalkeeper_joint_position": [],
                "goalkeeper_joint_velocity": [],
                "goalkeeper_joint_torque": [],
                "goalkeeper_policy_action": [],
                "goalkeeper_command_mps": [],
                "goalkeeper_predicted_target_y_m": [],
                "goalkeeper_reaction_active": [],
                "goalkeeper_ball_contact": [],
            }
        )
    finite = True
    joint_violation = False
    role_joint_violation = {robot.role: False for robot in robots}
    torque_violation = False
    actuator_saturation = False
    robot_robot_contact_count = 0
    passer_min_height = math.inf
    shooter_min_height = math.inf
    passer_roll_peak = 0.0
    passer_pitch_peak = 0.0
    shooter_roll_peak = 0.0
    shooter_pitch_peak = 0.0
    pass_peak_speed = 0.0
    shot_peak_speed = 0.0
    goal_crossed = False
    crossing_y: float | None = None
    crossing_z: float | None = None
    pass_delivery_position: np.ndarray | None = None
    goalkeeper_contact_time: float | None = None
    goalkeeper_min_height = math.inf
    goalkeeper_peak_lateral_speed = 0.0
    goalkeeper_reaction_frames = 0
    goalkeeper_initial_y = 0.0 if goalkeeper is None else float(data.qpos[goalkeeper.qpos_base + 1])
    previous_ball_x = float(data.qpos[ball_qpos])

    for frame in range(total_frames):
        if not shooter.entered and data.time + 1e-12 >= shooter.start_sec:
            _fill_local_state(shooter, data, ball_body, ball_qvel)
            _enter_policy(shooter)
        policy_frames: dict[str, int] = {}
        for robot in robots:
            _fill_local_state(robot, data, ball_body, ball_qvel)
        goalkeeper_command_mps = 0.0
        goalkeeper_target_y_m = 0.0
        goalkeeper_reaction_active = False
        if goalkeeper is not None and goalkeeper_config is not None:
            (
                goalkeeper_command_mps,
                goalkeeper_target_y_m,
                goalkeeper_reaction_active,
            ) = _command_goalkeeper(
                goalkeeper,
                data=data,
                ball_qpos=ball_qpos,
                ball_qvel=ball_qvel,
                goal=active_goal,
                config=goalkeeper_config,
                shot_contact_time=shooter.contact_time,
            )
            goalkeeper_reaction_frames += int(goalkeeper_reaction_active)
        for robot in robots:
            if (
                not robot.entered
                and robot.standby_policy is not None
                and robot.role != "goalkeeper"
            ):
                robot.state.vel_cmd = _normalized_zero_locomotion_command(robot.standby_policy)
            policy_frames[robot.role] = _update_policy(robot, frame, timestamp_sec=float(data.time))
            if robot.role == "passer" and passer_waist_pitch_target_margin_rad > 0.0:
                joint_index = 14
                joint_range = model.jnt_range[robot.joint_ids[joint_index]]
                robot.last_target[joint_index] = float(
                    np.clip(
                        robot.last_target[joint_index],
                        joint_range[0] + passer_waist_pitch_target_margin_rad,
                        joint_range[1] - passer_waist_pitch_target_margin_rad,
                    )
                )
            if robot.role == "goalkeeper" and goalkeeper_config is not None:
                _apply_goalkeeper_reach(
                    robot,
                    target_y_m=goalkeeper_target_y_m,
                    current_y_m=float(data.qpos[robot.qpos_base + 1]),
                    reaction_active=goalkeeper_reaction_active,
                    config=goalkeeper_config,
                )
        learned_torque: dict[str, np.ndarray | IQLResidualDecision | None] = {
            robot.role: None for robot in robots
        }
        joint_guard_active: dict[str, bool] = {robot.role: False for robot in robots}
        for robot in robots:
            if robot.recovery_torque_actor is not None and robot.contact_latched:
                actor_state = _recovery_actor_state(
                    robot,
                    data,
                    ball_body=ball_body,
                    timestamp_sec=float(data.time),
                )
                if isinstance(robot.recovery_torque_actor, SupportBoundIQLResidualActor):
                    # A residual is meaningful only after the measured-contact
                    # structured recovery controller has taken ownership.
                    if not robot.post_policy_active:
                        continue
                    q = data.qpos[robot.joint_qpos]
                    dq = data.qvel[robot.joint_qvel]
                    baseline_torque = (robot.last_target - q) * robot.kp - dq * robot.kd
                    decision = robot.recovery_torque_actor.action(
                        actor_state,
                        baseline_torque,
                    )
                    robot.learned_torque_support_rms_peak = max(
                        robot.learned_torque_support_rms_peak,
                        decision.standardized_rms,
                    )
                    if decision.accepted:
                        learned_torque[robot.role] = decision
                        robot.learned_torque_frame_count += 1
                        robot.learned_torque_confidence_sum += decision.confidence
                        robot.learned_torque_peak_residual_nm = max(
                            robot.learned_torque_peak_residual_nm,
                            decision.peak_residual_nm,
                        )
                    else:
                        robot.learned_torque_fallback_count += 1
                else:
                    learned_torque[robot.role] = robot.recovery_torque_actor.action(actor_state)
                    robot.learned_torque_frame_count += 1

        contact_role = 0
        frame_robot_contacts = 0
        commanded_torque: dict[str, NDArray[np.float64]] = {
            robot.role: np.zeros(29, dtype=np.float64) for robot in robots
        }
        projected_torque: dict[str, NDArray[np.float64]] = {
            robot.role: np.zeros(29, dtype=np.float64) for robot in robots
        }
        executed_torque: dict[str, NDArray[np.float64]] = {
            robot.role: np.zeros(29, dtype=np.float64) for robot in robots
        }
        frame_contact_impulse = {robot.role: 0.0 for robot in robots}
        passer_support = (False, False)
        shooter_support = (False, False)
        goalkeeper_support = (False, False)
        frame_ballistic_actor_active = False
        frame_ballistic_actor_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        for _ in range(_SUBSTEPS):
            for robot in robots:
                q = data.qpos[robot.joint_qpos]
                dq = data.qvel[robot.joint_qvel]
                actor_torque = learned_torque[robot.role]
                baseline_torque = (robot.last_target - q) * robot.kp - dq * robot.kd
                if isinstance(actor_torque, IQLResidualDecision):
                    raw_torque = baseline_torque + actor_torque.residual_torque
                elif actor_torque is not None:
                    raw_torque = actor_torque
                else:
                    raw_torque = baseline_torque
                if robot.role == "shooter" and shooter_ballistic_actor is not None:
                    effect = g1_ballistic_contact_impulse_effect(
                        model=model,
                        data=data,
                        right_ankle_body_id=robot.right_ankle_body,
                        actor=shooter_ballistic_actor,
                        policy_frame=policy_frames[robot.role],
                        contact_observed=robot.contact_latched,
                        ball_position=np.asarray(
                            data.qpos[ball_qpos : ball_qpos + 3],
                            dtype=np.float64,
                        ),
                    )
                    raw_torque = raw_torque + effect.torque
                    frame_ballistic_actor_active = frame_ballistic_actor_active or effect.active
                    if float(np.max(np.abs(effect.torque))) >= float(
                        np.max(np.abs(frame_ballistic_actor_torque))
                    ):
                        frame_ballistic_actor_torque = effect.torque.copy()
                safety_projected = raw_torque
                # The candidate is a post-impact recovery module.  Keeping the
                # guard behind the measured contact latch preserves the frozen
                # kick prior and prevents recovery search from gaming accuracy.
                if robot.joint_guard_enabled and (
                    robot.contact_latched
                    or robot.role == "goalkeeper"
                    or (robot.role == "passer" and passer_precontact_joint_guard_enabled)
                ):
                    if robot.selected_joint_guard_config is None:
                        (
                            robot.selected_joint_guard_config,
                            robot.joint_guard_route,
                        ) = _select_joint_guard_config(
                            standard=robot.joint_guard_config,
                            late_arrival=robot.joint_guard_late_config,
                            phase_advance_count=robot.phase_advance_count,
                        )
                    guard_config = robot.selected_joint_guard_config
                    safety_projected, active = _project_joint_safe_torque(
                        joint_position=q,
                        joint_velocity=dq,
                        commanded_torque=raw_torque,
                        joint_ranges=model.jnt_range[robot.joint_ids],
                        limited=model.jnt_limited[robot.joint_ids].astype(bool),
                        margin_rad=guard_config.margin_rad,
                        prediction_horizon_sec=guard_config.prediction_horizon_sec,
                        boundary_kp=guard_config.boundary_kp,
                        boundary_kd=guard_config.boundary_kd,
                    )
                    joint_guard_active[robot.role] = joint_guard_active[robot.role] or active
                torque = np.clip(safety_projected, -guarded_limits, guarded_limits)
                commanded_torque[robot.role] = raw_torque.copy()
                projected_torque[robot.role] = torque.copy()
                torque_violation = torque_violation or bool(np.any(np.abs(torque) > hard_limits))
                actuator_saturation = actuator_saturation or bool(
                    np.any(np.abs(torque) >= hard_limits * 0.999)
                )
                data.ctrl[robot.actuators] = torque
            if unified_stadium_scene:
                apply_g1_compliant_goal_net_force(
                    data,
                    ball_body_id=ball_body,
                    ball_qpos=ball_qpos,
                    ball_qvel=ball_qvel,
                    spec=active_goal,
                    capture_depth_m=max(0.20, 0.80 * active_goal.depth_m),
                    stiffness_n_m=180.0,
                    damping_n_s_m=10.0,
                )
            mujoco.mj_step(model, data)
            for robot in robots:
                executed_torque[robot.role] = data.actuator_force[robot.actuators].copy()
            observation = _contacts(
                model=model,
                data=data,
                ball_geom=ball_geom,
                floor_geom=floor_geom,
                passer_geoms=passer_geoms,
                shooter_geoms=shooter_geoms,
                goalkeeper_geoms=goalkeeper_geoms,
            )
            passer_support = (
                passer_support[0] or observation["passer_left"],
                passer_support[1] or observation["passer_right"],
            )
            shooter_support = (
                shooter_support[0] or observation["shooter_left"],
                shooter_support[1] or observation["shooter_right"],
            )
            goalkeeper_support = (
                goalkeeper_support[0] or observation["goalkeeper_left"],
                goalkeeper_support[1] or observation["goalkeeper_right"],
            )
            frame_robot_contacts += int(observation["robot_robot"])
            if observation["ball_passer"]:
                contact_role = 1
                impulse = float(observation["ball_passer_force_n"]) * _PHYSICS_DT
                passer.contact_impulse_ns += impulse
                frame_contact_impulse["passer"] += impulse
                if not passer.contact_latched:
                    passer.contact_latched = True
                    passer.contact_time = float(data.time)
                    _reset_post_contact_support_anchors(passer)
            if observation["ball_shooter"]:
                contact_role = 2
                impulse = float(observation["ball_shooter_force_n"]) * _PHYSICS_DT
                shooter.contact_impulse_ns += impulse
                frame_contact_impulse["shooter"] += impulse
                if not shooter.contact_latched:
                    pass_delivery_position = np.asarray(
                        data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64
                    ).copy()
                    shooter.contact_latched = True
                    shooter.contact_time = float(data.time)
                    _reset_post_contact_support_anchors(shooter)
            if goalkeeper is not None and observation["ball_goalkeeper"]:
                contact_role = 3
                if goalkeeper_contact_time is None:
                    goalkeeper_contact_time = float(data.time)
        passer.latest_left_support, passer.latest_right_support = passer_support
        shooter.latest_left_support, shooter.latest_right_support = shooter_support
        if goalkeeper is not None:
            goalkeeper.latest_left_support, goalkeeper.latest_right_support = goalkeeper_support
        _update_support_slip(passer, data, passer_support)
        _update_support_slip(shooter, data, shooter_support)
        if goalkeeper is not None:
            _update_support_slip(goalkeeper, data, goalkeeper_support)
        for robot in robots:
            robot.joint_guard_frame_count += int(joint_guard_active[robot.role])
        robot_robot_contact_count += frame_robot_contacts

        ball_position = data.qpos[ball_qpos : ball_qpos + 3].copy()
        ball_velocity = data.qvel[ball_qvel : ball_qvel + 3].copy()
        ball_speed = float(np.linalg.norm(ball_velocity))
        if passer.contact_latched and not shooter.contact_latched:
            pass_peak_speed = max(pass_peak_speed, ball_speed)
        if shooter.contact_latched:
            shot_peak_speed = max(shot_peak_speed, ball_speed)
        if not goal_crossed and previous_ball_x < active_goal.plane_x_m <= float(ball_position[0]):
            goal_crossed = True
            crossing_y = float(ball_position[1])
            crossing_z = float(ball_position[2])
        previous_ball_x = float(ball_position[0])

        heights = (float(data.qpos[passer.qpos_base + 2]), float(data.qpos[2]))
        passer_min_height = min(passer_min_height, heights[0])
        shooter_min_height = min(shooter_min_height, heights[1])
        if goalkeeper is not None:
            goalkeeper_min_height = min(
                goalkeeper_min_height,
                float(data.qpos[goalkeeper.qpos_base + 2]),
            )
            goalkeeper_peak_lateral_speed = max(
                goalkeeper_peak_lateral_speed,
                abs(float(data.qvel[goalkeeper.qvel_base + 1])),
            )
        passer_roll, passer_pitch = _roll_pitch(data.xquat[passer.torso_body])
        shooter_roll, shooter_pitch = _roll_pitch(data.xquat[shooter.torso_body])
        passer_roll_peak = max(passer_roll_peak, abs(passer_roll))
        passer_pitch_peak = max(passer_pitch_peak, abs(passer_pitch))
        shooter_roll_peak = max(shooter_roll_peak, abs(shooter_roll))
        shooter_pitch_peak = max(shooter_pitch_peak, abs(shooter_pitch))
        for robot in robots:
            ranges = model.jnt_range[robot.joint_ids]
            limited = model.jnt_limited[robot.joint_ids].astype(bool)
            q = data.qpos[robot.joint_qpos]
            role_violation = bool(
                np.any(q[limited] < ranges[limited, 0] - 1e-5)
                or np.any(q[limited] > ranges[limited, 1] + 1e-5)
            )
            role_joint_violation[robot.role] = role_joint_violation[robot.role] or role_violation
            joint_violation = joint_violation or role_violation
        finite = finite and all(
            np.all(np.isfinite(value)) for value in (data.qpos, data.qvel, data.ctrl, ball_position)
        )
        trace["shooter_ballistic_actor_active"].append(frame_ballistic_actor_active)
        trace["shooter_ballistic_actor_torque"].append(frame_ballistic_actor_torque)
        if goalkeeper is not None:
            trace["goalkeeper_pelvis_pose"].append(
                data.qpos[goalkeeper.qpos_base : goalkeeper.qpos_base + 7].copy()
            )
            trace["goalkeeper_torso_quaternion"].append(data.xquat[goalkeeper.torso_body].copy())
            trace["goalkeeper_joint_position"].append(data.qpos[goalkeeper.joint_qpos].copy())
            trace["goalkeeper_joint_velocity"].append(data.qvel[goalkeeper.joint_qvel].copy())
            trace["goalkeeper_joint_torque"].append(data.ctrl[goalkeeper.actuators].copy())
            trace["goalkeeper_policy_action"].append(goalkeeper.last_target.copy())
            trace["goalkeeper_command_mps"].append(goalkeeper_command_mps)
            trace["goalkeeper_predicted_target_y_m"].append(goalkeeper_target_y_m)
            trace["goalkeeper_reaction_active"].append(goalkeeper_reaction_active)
            trace["goalkeeper_ball_contact"].append(
                goalkeeper_contact_time is not None
                and abs(float(data.time) - goalkeeper_contact_time) <= _CONTROL_DT + 1e-9
            )
        _append_trace(
            trace,
            data=data,
            ball_qpos=ball_qpos,
            ball_qvel=ball_qvel,
            passer=passer,
            shooter=shooter,
            policy_frames=policy_frames,
            support=(passer_support, shooter_support),
            contact_role=contact_role,
            robot_contacts=frame_robot_contacts,
            commanded_torque=commanded_torque,
            projected_torque=projected_torque,
            executed_torque=executed_torque,
            contact_impulse=frame_contact_impulse,
            learned_torque=learned_torque,
            joint_guard_active=joint_guard_active,
        )
        if not finite:
            break

    target_error = (
        math.hypot(crossing_y - shooter_target[1], crossing_z - shooter_target[2])
        if crossing_y is not None and crossing_z is not None
        else None
    )
    pass_delivery_error = (
        float(np.linalg.norm(pass_delivery_position - _PASS_RECEPTION_TARGET))
        if pass_delivery_position is not None
        else None
    )
    pass_delivery_lateral_error = (
        abs(float(pass_delivery_position[1] - _PASS_RECEPTION_TARGET[1]))
        if pass_delivery_position is not None
        else None
    )
    trajectory = {name: np.asarray(values) for name, values in trace.items()}
    result = G1CoupledRelayResult(
        finite_state=finite,
        pass_contact_observed=passer.contact_latched,
        shot_contact_observed=shooter.contact_latched,
        pass_contact_time_sec=passer.contact_time,
        shot_contact_time_sec=shooter.contact_time,
        pass_peak_ball_speed_mps=pass_peak_speed,
        shot_peak_ball_speed_mps=shot_peak_speed,
        goal_crossed=goal_crossed,
        goal_crossing_y_m=crossing_y,
        goal_crossing_z_m=crossing_z,
        target_error_m=target_error,
        passer_min_pelvis_height_m=passer_min_height,
        shooter_min_pelvis_height_m=shooter_min_height,
        passer_roll_peak_rad=passer_roll_peak,
        passer_pitch_peak_rad=passer_pitch_peak,
        shooter_roll_peak_rad=shooter_roll_peak,
        shooter_pitch_peak_rad=shooter_pitch_peak,
        passer_tail_wobble_index=_tail_wobble(trajectory, "passer"),
        shooter_tail_wobble_index=_tail_wobble(trajectory, "shooter"),
        receiver_phase_hold_frames=shooter.phase_hold_count,
        receiver_phase_advance_frames=shooter.phase_advance_count,
        receiver_max_ball_phase_error_m=shooter.max_ball_phase_error_m,
        robot_robot_contact_count=robot_robot_contact_count,
        joint_limit_violation=joint_violation,
        torque_limit_violation=torque_violation,
        actuator_saturation=actuator_saturation,
        physics_steps=len(trace["time"]) * _SUBSTEPS,
        passer_support_foot_slip_m=passer.peak_support_slip_m,
        shooter_support_foot_slip_m=shooter.peak_support_slip_m,
        passer_post_contact_support_foot_slip_m=passer.post_contact_peak_support_slip_m,
        shooter_post_contact_support_foot_slip_m=shooter.post_contact_peak_support_slip_m,
        passer_contact_impulse_ns=passer.contact_impulse_ns,
        shooter_contact_impulse_ns=shooter.contact_impulse_ns,
        passer_post_kick_fall=passer_min_height < 0.55,
        shooter_post_kick_fall=shooter_min_height < 0.55,
        shooter_learned_torque_fraction=(
            shooter.learned_torque_frame_count
            / max(
                1,
                total_frames - int(round((shooter.contact_time or _TOTAL_TIME_SEC) / _CONTROL_DT)),
            )
        ),
        shooter_learned_torque_fallback_fraction=(
            shooter.learned_torque_fallback_count
            / max(
                1,
                total_frames - int(round((shooter.contact_time or _TOTAL_TIME_SEC) / _CONTROL_DT)),
            )
        ),
        shooter_learned_torque_mean_confidence=(
            shooter.learned_torque_confidence_sum / max(1, shooter.learned_torque_frame_count)
        ),
        shooter_learned_torque_peak_residual_nm=shooter.learned_torque_peak_residual_nm,
        shooter_learned_torque_support_rms_peak=shooter.learned_torque_support_rms_peak,
        shooter_joint_guard_fraction=shooter.joint_guard_frame_count / max(1, total_frames),
        shooter_joint_guard_route=shooter.joint_guard_route,
        pass_delivery_position_m=(
            (
                float(pass_delivery_position[0]),
                float(pass_delivery_position[1]),
                float(pass_delivery_position[2]),
            )
            if pass_delivery_position is not None
            else None
        ),
        pass_delivery_error_m=pass_delivery_error,
        pass_delivery_lateral_error_m=pass_delivery_lateral_error,
        passer_joint_guard_fraction=passer.joint_guard_frame_count / max(1, total_frames),
        passer_joint_guard_route=passer.joint_guard_route,
        passer_recovery_active_fraction=passer.recovery_active_frame_count / max(1, total_frames),
        shooter_recovery_active_fraction=shooter.recovery_active_frame_count / max(1, total_frames),
        passer_recovery_peak_blend_fraction=passer.recovery_peak_blend_fraction,
        shooter_recovery_peak_blend_fraction=shooter.recovery_peak_blend_fraction,
        shooter_aim_expert_route=(
            "early_arrival" if shooter.early_arrival_expert_frame_count > 0 else "nominal"
        ),
        shooter_early_arrival_expert_fraction=(
            shooter.early_arrival_expert_frame_count / max(1, total_frames)
        ),
        shooter_ballistic_actor_active_fraction=(
            float(np.mean(trajectory["shooter_ballistic_actor_active"]))
            if trajectory["shooter_ballistic_actor_active"].size
            else 0.0
        ),
        shooter_ballistic_actor_peak_torque_nm=(
            float(np.max(np.abs(trajectory["shooter_ballistic_actor_torque"])))
            if trajectory["shooter_ballistic_actor_torque"].size
            else 0.0
        ),
        shooter_ballistic_actor_hash=(
            None if shooter_ballistic_actor is None else shooter_ballistic_actor.actor_hash
        ),
        goalkeeper_enabled=goalkeeper is not None,
        goalkeeper_reaction_active_fraction=(goalkeeper_reaction_frames / max(1, total_frames)),
        goalkeeper_lateral_displacement_m=(
            0.0
            if goalkeeper is None
            else abs(float(data.qpos[goalkeeper.qpos_base + 1]) - goalkeeper_initial_y)
        ),
        goalkeeper_peak_lateral_speed_mps=goalkeeper_peak_lateral_speed,
        goalkeeper_min_pelvis_height_m=(None if goalkeeper is None else goalkeeper_min_height),
        goalkeeper_ball_contact_observed=goalkeeper_contact_time is not None,
        goalkeeper_ball_contact_time_sec=goalkeeper_contact_time,
        passer_joint_limit_violation=role_joint_violation["passer"],
        shooter_joint_limit_violation=role_joint_violation["shooter"],
        goalkeeper_joint_limit_violation=role_joint_violation.get("goalkeeper", False),
    )
    return result, trajectory


def _coupled_model(
    root: Path,
    *,
    passer_origin: np.ndarray | None = None,
    goal: G1TrainingGoalSpec | None = None,
    goalkeeper_config: G1GoalkeeperConfig | None = None,
    unified_stadium_scene: bool = False,
) -> Any:
    """Build either the pinned legacy scene or the unified stadium contract."""

    origin = np.asarray(
        _PASSER_ORIGIN if passer_origin is None else passer_origin,
        dtype=np.float64,
    )
    origin_m = (float(origin[0]), float(origin[1]), float(origin[2]))
    if not unified_stadium_scene:
        if goalkeeper_config is not None:
            raise ValueError("legacy coupled scene cannot attach a goalkeeper")
        return _legacy_coupled_model(root, origin)
    if goalkeeper_config is None:
        return build_g1_coupled_stadium_model(
            root,
            passer_origin_m=origin_m,
            spec=goal,
        )
    active_goal = goal or G1TrainingGoalSpec()
    return build_g1_three_player_stadium_model(
        root,
        passer_origin_m=origin_m,
        goalkeeper_origin_m=(
            active_goal.plane_x_m - goalkeeper_config.depth_from_goal_line_m,
            0.0,
            0.0,
        ),
        spec=active_goal,
    )


def _legacy_coupled_model(root: Path, passer_origin: np.ndarray) -> Any:
    """Preserve the published two-G1 skill's v1 physics identity.

    The trained v1 phase synchronizer depends on the upstream ball damping.
    New long-relay skills must opt into ``unified_stadium_scene`` and are not
    allowed to inherit this compatibility route.
    """

    import mujoco

    parent = mujoco.MjSpec.from_file(str(root / _SCENE_REL))
    child = mujoco.MjSpec.from_file(str(root / _MODEL_REL))
    frame = parent.worldbody.add_frame(
        name="passer_frame",
        pos=tuple(float(value) for value in passer_origin),
        quat=(0.0, 0.0, 0.0, 1.0),
    )
    first_body = child.worldbody.first_body()
    if first_body is None:
        raise ValueError("qualified G1 model does not contain a root body")
    frame.attach_body(first_body, prefix="passer_")
    model = parent.compile()
    if model.nu != 58:
        raise ValueError(f"legacy coupled model has {model.nu} actuators, expected 58")
    return model


def _make_robot(
    *,
    model: Any,
    data: Any,
    role: str,
    prefix: str,
    origin: np.ndarray,
    yaw: float,
    state_type: Any,
    output_type: Any,
    policy_type: Any,
    parameters: ShotParameters,
    start_sec: float,
    initial_position: np.ndarray,
    initial_quaternion: np.ndarray,
    initial_joints: np.ndarray,
    target_local: np.ndarray,
    phase_hold_frames: int,
    standby_target: np.ndarray | None,
    standby_kp: np.ndarray | None,
    standby_kd: np.ndarray | None,
    use_locomotion_standby: bool,
    recovery_controller: Any | None,
    post_policy_frame: int | None,
    post_policy_blend_frames: int,
    phase_sync_enabled: bool,
    recovery_torque_actor: Any | None,
    joint_guard_enabled: bool,
    post_policy_neutral_velocity_enabled: bool,
    joint_guard_config: G1JointGuardConfig,
    joint_guard_late_config: G1JointGuardConfig | None,
    post_policy_recovery_enabled: bool,
    early_arrival_parameters: ShotParameters | None,
) -> _Robot:
    import mujoco

    if not 0 <= post_policy_blend_frames <= 200:
        raise ValueError("post-policy blend frames must be in [0, 200]")

    free_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
    qpos_base = int(model.jnt_qposadr[free_joint])
    qvel_base = int(model.jnt_dofadr[free_joint])
    joint_ids = np.asarray(
        [_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name) for name in G1_DDS_JOINT_NAMES],
        dtype=np.int64,
    )
    joint_qpos = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int64)
    joint_qvel = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int64)
    actuators = np.asarray(
        [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name) for name in G1_DDS_JOINT_NAMES],
        dtype=np.int64,
    )
    state = state_type(29)
    output = output_type(29)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = policy_type(state, output)
    standby_output = None
    standby_policy = None
    if use_locomotion_standby:
        standby_type = importlib.import_module("policy.loco_mode.LocoMode").LocoMode
        standby_output = output_type(29)
        with contextlib.redirect_stdout(io.StringIO()):
            standby_policy = standby_type(state, standby_output)
            standby_policy.enter()
    policy.target_pos_w = target_local
    half_yaw = 0.5 * yaw
    frame_quaternion = np.asarray(
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)), dtype=np.float64
    )
    posture_yaw = 0.5 * parameters.pelvis_yaw_offset
    posture_quaternion = np.asarray(
        (math.cos(posture_yaw), 0.0, 0.0, math.sin(posture_yaw)), dtype=np.float64
    )
    local_quaternion = _quaternion_multiply(posture_quaternion, initial_quaternion)
    data.qpos[qpos_base : qpos_base + 3] = origin + _rotate_z(
        initial_position
        + np.asarray(
            (parameters.stance_offset_x, parameters.stance_offset_y, 0.0),
            dtype=np.float64,
        ),
        yaw,
    )
    data.qpos[qpos_base + 3 : qpos_base + 7] = _quaternion_multiply(
        frame_quaternion, local_quaternion
    )
    data.qpos[joint_qpos] = initial_joints
    hold_target = (
        np.asarray(standby_target, dtype=np.float64).copy()
        if standby_target is not None
        else initial_joints.copy()
    )
    kp = (
        np.asarray(standby_kp, dtype=np.float64).copy()
        if standby_kp is not None
        else np.asarray(policy.kps, dtype=np.float64).copy()
    )
    kd = (
        np.asarray(standby_kd, dtype=np.float64).copy()
        if standby_kd is not None
        else np.asarray(policy.kds, dtype=np.float64).copy()
    )
    return _Robot(
        role=role,
        prefix=prefix,
        origin=origin.copy(),
        world_from_local_quat=frame_quaternion,
        qpos_base=qpos_base,
        qvel_base=qvel_base,
        joint_qpos=joint_qpos,
        joint_qvel=joint_qvel,
        actuators=actuators,
        joint_ids=joint_ids,
        pelvis_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis"),
        torso_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "torso_link"),
        left_ankle_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "left_ankle_roll_link"),
        right_ankle_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "right_ankle_roll_link"),
        state=state,
        output=output,
        policy=policy,
        standby_output=standby_output,
        standby_policy=standby_policy,
        parameters=parameters,
        start_sec=start_sec,
        hold_target=hold_target,
        last_target=hold_target.copy(),
        kp=kp,
        kd=kd,
        recovery_controller=recovery_controller,
        phase_hold_frames=phase_hold_frames,
        phase_hold_remaining=phase_hold_frames,
        post_policy_frame=post_policy_frame,
        post_policy_blend_frames=post_policy_blend_frames,
        phase_sync_enabled=phase_sync_enabled,
        recovery_torque_actor=recovery_torque_actor,
        joint_guard_enabled=joint_guard_enabled,
        post_policy_neutral_velocity_enabled=post_policy_neutral_velocity_enabled,
        joint_guard_config=joint_guard_config,
        joint_guard_late_config=joint_guard_late_config,
        post_policy_recovery_enabled=post_policy_recovery_enabled,
        early_arrival_parameters=early_arrival_parameters,
    )


def _fill_local_state(robot: _Robot, data: Any, ball_body: int, ball_qvel: int) -> None:
    inverse_quaternion = robot.world_from_local_quat.copy()
    inverse_quaternion[1:] *= -1.0
    robot.state.q = data.qpos[robot.joint_qpos].copy()
    robot.state.dq = data.qvel[robot.joint_qvel].copy()
    robot.state.tau_est = data.ctrl[robot.actuators].copy()
    # MuJoCo stores a free joint's six tangent velocities in the joint frame.
    # The attached passer frame is therefore already the policy's local frame;
    # rotating these values a second time flips x/y and destabilizes inference.
    robot.state.root_lin_vel_b = data.qvel[robot.qvel_base : robot.qvel_base + 3].copy()
    robot.state.root_ang_vel_b = data.qvel[robot.qvel_base + 3 : robot.qvel_base + 6].copy()
    robot.state.torso_pos_w = _to_local(data.xpos[robot.torso_body], robot)
    robot.state.torso_quat_w = _quaternion_multiply(
        inverse_quaternion, data.xquat[robot.torso_body]
    )
    robot.state.pelvis_pos_w = _to_local(data.qpos[robot.qpos_base : robot.qpos_base + 3], robot)
    robot.state.pelvis_quat_w = _quaternion_multiply(
        inverse_quaternion, data.qpos[robot.qpos_base + 3 : robot.qpos_base + 7]
    )
    robot.state.ball_pos_w = _to_local(data.xpos[ball_body], robot)
    robot.state.ball_vel_w = _rotate_z(data.qvel[ball_qvel : ball_qvel + 3], -_yaw(robot))
    robot.state.ball_valid = True
    quaternion = np.asarray(robot.state.pelvis_quat_w, dtype=np.float64)
    qw, qx, qy, qz = map(float, quaternion)
    robot.state.gravity_ori = np.asarray(
        (
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ),
        dtype=np.float64,
    )
    robot.state.ang_vel = robot.state.root_ang_vel_b.copy()


def _enter_policy(robot: _Robot) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        robot.policy.enter()
    robot.entered = True


def _smooth_policy_handoff(
    *,
    origin_target: np.ndarray,
    origin_kp: np.ndarray,
    origin_kd: np.ndarray,
    destination_target: np.ndarray,
    destination_kp: np.ndarray,
    destination_kd: np.ndarray,
    transition_step: int,
    blend_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Blend a policy handoff without a target or impedance step."""

    if blend_frames <= 0 or not 0 <= transition_step < blend_frames:
        raise ValueError("policy handoff step must be inside a positive blend window")
    values = (
        origin_target,
        origin_kp,
        origin_kd,
        destination_target,
        destination_kp,
        destination_kd,
    )
    if any(value.shape != (29,) or not np.all(np.isfinite(value)) for value in values):
        raise ValueError("policy handoff vectors must be finite 29-joint arrays")
    linear = (transition_step + 1) / blend_frames
    fraction = linear * linear * (3.0 - 2.0 * linear)
    return (
        origin_target + fraction * (destination_target - origin_target),
        origin_kp + fraction * (destination_kp - origin_kp),
        origin_kd + fraction * (destination_kd - origin_kd),
        fraction,
    )


def _project_joint_safe_torque(
    *,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    commanded_torque: np.ndarray,
    joint_ranges: np.ndarray,
    limited: np.ndarray,
    margin_rad: float = 0.04,
    prediction_horizon_sec: float = 0.08,
    boundary_kp: float = 80.0,
    boundary_kd: float = 6.0,
) -> tuple[np.ndarray, bool]:
    """Project outward torque when a velocity-aware joint envelope is threatened."""

    if (
        joint_position.shape != (29,)
        or joint_velocity.shape != (29,)
        or commanded_torque.shape != (29,)
        or joint_ranges.shape != (29, 2)
        or limited.shape != (29,)
    ):
        raise ValueError("joint guard requires the 29-DoF G1 contract")
    if not all(
        np.all(np.isfinite(value))
        for value in (joint_position, joint_velocity, commanded_torque, joint_ranges)
    ):
        raise ValueError("joint guard inputs must be finite")
    if not 0.0 < margin_rad <= 0.10 or not 0.0 < prediction_horizon_sec <= 0.20:
        raise ValueError("joint guard envelope parameters are invalid")
    projected = commanded_torque.copy()
    predicted = joint_position + prediction_horizon_sec * joint_velocity
    lower = joint_ranges[:, 0] + margin_rad
    upper = joint_ranges[:, 1] - margin_rad
    lower_threat = limited & (predicted < lower)
    upper_threat = limited & (predicted > upper)
    lower_brake = boundary_kp * (lower - joint_position) - boundary_kd * joint_velocity
    upper_brake = boundary_kp * (upper - joint_position) - boundary_kd * joint_velocity
    projected[lower_threat] = np.maximum(projected[lower_threat], lower_brake[lower_threat])
    projected[upper_threat] = np.minimum(projected[upper_threat], upper_brake[upper_threat])
    active = not np.array_equal(projected, commanded_torque)
    return projected, active


def _select_joint_guard_config(
    *,
    standard: G1JointGuardConfig,
    late_arrival: G1JointGuardConfig | None,
    phase_advance_count: int,
) -> tuple[G1JointGuardConfig, str]:
    """Latch one guard expert from the causal receiver phase signal."""

    if phase_advance_count < 0:
        raise ValueError("phase advance count must be non-negative")
    if late_arrival is not None and phase_advance_count > 0:
        return late_arrival, "late_arrival"
    return standard, "standard"


def _normalized_zero_locomotion_command(policy: Any) -> np.ndarray:
    """Invert RoboNaldo joystick scaling so the physical command is zero."""

    ranges = np.asarray(
        (policy.range_velx, policy.range_vely, policy.range_velz),
        dtype=np.float64,
    )
    if ranges.shape != (3, 2) or not np.all(np.isfinite(ranges)):
        raise ValueError("locomotion command ranges must be finite [min, max] pairs")
    widths = ranges[:, 1] - ranges[:, 0]
    if np.any(widths <= 0.0) or np.any(ranges[:, 0] > 0.0) or np.any(ranges[:, 1] < 0.0):
        raise ValueError("locomotion command ranges must be ordered and contain zero")
    normalized = -1.0 - 2.0 * ranges[:, 0] / widths
    if np.any(np.abs(normalized) > 1.0 + 1e-12):
        raise ValueError("zero locomotion command is outside the normalized input range")
    return normalized


def _normalized_locomotion_command(policy: Any, physical_command: np.ndarray) -> np.ndarray:
    """Map physical ``vx, vy, yaw`` commands into RoboNaldo joystick space."""

    command = np.asarray(physical_command, dtype=np.float64)
    ranges = np.asarray(
        (policy.range_velx, policy.range_vely, policy.range_velz),
        dtype=np.float64,
    )
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("locomotion command must be a finite physical 3-vector")
    widths = ranges[:, 1] - ranges[:, 0]
    if ranges.shape != (3, 2) or np.any(widths <= 0.0):
        raise ValueError("locomotion command ranges are invalid")
    if np.any(command < ranges[:, 0]) or np.any(command > ranges[:, 1]):
        raise ValueError("physical locomotion command is outside the policy range")
    return -1.0 + 2.0 * (command - ranges[:, 0]) / widths


def _command_goalkeeper(
    robot: _Robot,
    *,
    data: Any,
    ball_qpos: int,
    ball_qvel: int,
    goal: G1TrainingGoalSpec,
    config: G1GoalkeeperConfig,
    shot_contact_time: float | None,
) -> tuple[float, float, bool]:
    """Issue a causal lateral shuffle from measured ball flight only."""

    if robot.standby_policy is None:
        raise RuntimeError("goalkeeper requires the qualified locomotion policy")
    timestamp = float(data.time)
    ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64)
    velocity = np.asarray(data.qvel[ball_qvel : ball_qvel + 3], dtype=np.float64)
    reaction_active = bool(
        shot_contact_time is not None
        and timestamp >= shot_contact_time + config.reaction_delay_sec
        and velocity[0] > 0.10
    )
    if reaction_active:
        horizon = max(0.0, (goal.plane_x_m - float(ball[0])) / float(velocity[0]))
        target_y = float(ball[1] + velocity[1] * horizon)
        mouth_limit = goal.width_m / 2.0 - 0.22
        target_y = float(np.clip(target_y, -mouth_limit, mouth_limit))
    else:
        phase = 2.0 * math.pi * timestamp / config.ready_shuffle_period_sec
        target_y = 0.12 * math.sin(phase)
    current_y = float(data.qpos[robot.qpos_base + 1])
    world_velocity_y = float(
        np.clip(
            config.lateral_position_gain * (target_y - current_y),
            -config.maximum_lateral_speed_mps,
            config.maximum_lateral_speed_mps,
        )
    )
    if not reaction_active:
        world_velocity_y = float(
            np.clip(
                world_velocity_y,
                -config.ready_shuffle_speed_mps,
                config.ready_shuffle_speed_mps,
            )
        )
    # The keeper faces -x, so its local +y points toward world -y.
    robot.state.vel_cmd = _normalized_locomotion_command(
        robot.standby_policy,
        np.asarray((0.0, -world_velocity_y, 0.0), dtype=np.float64),
    )
    return world_velocity_y, target_y, reaction_active


def _apply_goalkeeper_reach(
    robot: _Robot,
    *,
    target_y_m: float,
    current_y_m: float,
    reaction_active: bool,
    config: G1GoalkeeperConfig,
) -> None:
    """Add a bounded reach/lean to the locomotion target after ball reaction."""

    if not reaction_active:
        return
    direction = float(np.sign(target_y_m - current_y_m))
    magnitude = float(np.clip(abs(target_y_m - current_y_m) / 0.9, 0.0, 1.0))
    robot.last_target[13] += direction * config.maximum_waist_lean_rad * magnitude
    robot.last_target[16] += config.arm_spread_rad
    robot.last_target[23] -= config.arm_spread_rad
    # Reach farther with the arm on the predicted side while keeping both
    # elbows flexed enough to avoid self-collision.
    if direction >= 0.0:
        robot.last_target[22] -= 0.14 * magnitude
        robot.last_target[25] = max(robot.last_target[25] - 0.12 * magnitude, 0.35)
    else:
        robot.last_target[15] -= 0.14 * magnitude
        robot.last_target[18] = max(robot.last_target[18] - 0.12 * magnitude, 0.35)


def _update_policy(
    robot: _Robot,
    simulation_frame: int,
    *,
    timestamp_sec: float,
) -> int:
    robot.last_recovery_active = False
    robot.last_recovery_blend_fraction = 0.0
    if not robot.entered:
        if robot.standby_policy is not None and robot.standby_output is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                robot.standby_policy.run()
            robot.last_target = np.asarray(robot.standby_output.actions, dtype=np.float64).copy()
            robot.kp = np.asarray(robot.standby_output.kps, dtype=np.float64).copy()
            robot.kd = np.asarray(robot.standby_output.kds, dtype=np.float64).copy()
        else:
            robot.last_target = robot.hold_target.copy()
        return 0
    current_policy_frame = max(0, int(robot.policy.time_step) - int(robot.policy.WARMUP_STEPS))
    robot.last_phase_correction = 0
    if (
        not robot.post_policy_active
        and robot.post_policy_frame is not None
        and current_policy_frame >= robot.post_policy_frame
        and robot.contact_latched
    ):
        robot.post_policy_active = True
        robot.post_policy_origin_target = robot.last_target.copy()
        robot.post_policy_origin_kp = robot.kp.copy()
        robot.post_policy_origin_kd = robot.kd.copy()
        robot.post_policy_activation_simulation_frame = simulation_frame
    if robot.post_policy_active:
        if robot.standby_policy is None or robot.standby_output is None:
            raise RuntimeError("post-kick locomotion transition is unavailable")
        if robot.post_policy_neutral_velocity_enabled:
            robot.state.vel_cmd = _normalized_zero_locomotion_command(robot.standby_policy)
        with contextlib.redirect_stdout(io.StringIO()):
            robot.standby_policy.run()
        standby_target = np.asarray(robot.standby_output.actions, dtype=np.float64)
        standby_kp = np.asarray(robot.standby_output.kps, dtype=np.float64)
        standby_kd = np.asarray(robot.standby_output.kds, dtype=np.float64)
        activation_frame = robot.post_policy_activation_simulation_frame
        if activation_frame is None:
            raise RuntimeError("post-policy activation frame is unavailable")
        recovery_policy_frame = current_policy_frame + max(0, simulation_frame - activation_frame)
        if robot.post_policy_recovery_enabled:
            standby_target, standby_kp, standby_kd = _apply_recovery_controller(
                robot,
                target=standby_target,
                kp=standby_kp,
                kd=standby_kd,
                policy_frame=recovery_policy_frame,
                timestamp_sec=timestamp_sec,
            )
        if robot.post_policy_blend_frames:
            origin_target = robot.post_policy_origin_target
            origin_kp = robot.post_policy_origin_kp
            origin_kd = robot.post_policy_origin_kd
            if origin_target is None or origin_kp is None or origin_kd is None:
                raise RuntimeError("post-policy transition origin is unavailable")
            transition_step = min(
                robot.post_policy_transition_step,
                robot.post_policy_blend_frames - 1,
            )
            (
                robot.last_target,
                robot.kp,
                robot.kd,
                fraction,
            ) = _smooth_policy_handoff(
                origin_target=origin_target,
                origin_kp=origin_kp,
                origin_kd=origin_kd,
                destination_target=standby_target,
                destination_kp=standby_kp,
                destination_kd=standby_kd,
                transition_step=transition_step,
                blend_frames=robot.post_policy_blend_frames,
            )
            robot.post_policy_transition_step += 1
            robot.post_policy_blend_fraction = fraction
        else:
            robot.last_target = standby_target.copy()
            robot.kp = standby_kp.copy()
            robot.kd = standby_kd.copy()
            robot.post_policy_blend_fraction = 1.0
        return recovery_policy_frame
    if current_policy_frame >= 185 and robot.phase_hold_remaining:
        robot.phase_hold_remaining -= 1
        return current_policy_frame
    repeat = _policy_repeat_count(
        robot.parameters.swing_speed_scale,
        current_policy_frame,
        simulation_frame,
    )
    if (
        robot.phase_sync_enabled
        and robot.role == "shooter"
        and 184 <= current_policy_frame <= 252
        and float(robot.state.ball_vel_w[0]) < -0.05
    ):
        phase = float(current_policy_frame - 184)
        expected_ball_x = 1.32676487e-4 * phase * phase - 2.54767831e-2 * phase + 2.12799256
        phase_error = float(robot.state.ball_pos_w[0]) - expected_ball_x
        robot.max_ball_phase_error_m = max(
            robot.max_ball_phase_error_m,
            abs(phase_error),
        )
        if phase_error > 0.025:
            repeat = 0
            robot.phase_hold_count += 1
            robot.last_phase_correction = -1
        elif phase_error < -0.025:
            repeat = max(2, repeat + 1)
            robot.phase_advance_count += 1
            robot.last_phase_correction = 1
    if repeat:
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(repeat):
                robot.policy.run()
        policy_frame = max(0, int(robot.policy.time_step) - int(robot.policy.WARMUP_STEPS))
        active_parameters = robot.parameters
        if (
            robot.role == "shooter"
            and robot.phase_hold_count > 0
            and robot.early_arrival_parameters is not None
        ):
            active_parameters = robot.early_arrival_parameters
            robot.early_arrival_expert_frame_count += 1
        target = _adapt_target(
            target=np.asarray(robot.output.actions, dtype=np.float64),
            default=np.asarray(robot.policy.default_q_mj, dtype=np.float64),
            parameters=active_parameters,
            policy_frame=policy_frame,
        )
        output_kp = np.asarray(robot.output.kps, dtype=np.float64).copy()
        output_kd = np.asarray(robot.output.kds, dtype=np.float64).copy()
        target, output_kp, output_kd = _apply_recovery_controller(
            robot,
            target=target,
            kp=output_kp,
            kd=output_kd,
            policy_frame=policy_frame,
            timestamp_sec=timestamp_sec,
        )
        robot.last_target = target
        robot.kp = output_kp
        robot.kd = output_kd
        return policy_frame
    return current_policy_frame


def _apply_recovery_controller(
    robot: _Robot,
    *,
    target: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    policy_frame: int,
    timestamp_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one shared contact-gated cerebellar recovery contract.

    The helper is deliberately used on both sides of a policy handoff.  This
    prevents the recovery state machine from disappearing when the retained
    locomotion expert takes control after a pass or shot.
    """

    if robot.recovery_controller is None:
        return target.copy(), kp.copy(), kd.copy()
    recovery = robot.recovery_controller.adapt_target(
        target=target,
        policy_frame=policy_frame,
        timestamp_sec=timestamp_sec,
        ball_contact_detected=robot.contact_latched,
        left_support=robot.latest_left_support,
        right_support=robot.latest_right_support,
    )
    output_kp = kp.copy()
    output_kd = kd.copy()
    terminal_slice = {
        "whole_body": slice(None),
        "legs": slice(0, 12),
        "upper_body": slice(12, None),
    }[recovery.terminal_damping_joint_group]
    output_kp[terminal_slice] *= recovery.terminal_kp_scale
    output_kd[terminal_slice] *= recovery.terminal_kd_scale
    robot.last_recovery_active = recovery.active
    robot.last_recovery_blend_fraction = recovery.blend_fraction
    robot.recovery_active_frame_count += int(recovery.active)
    robot.recovery_peak_blend_fraction = max(
        robot.recovery_peak_blend_fraction,
        recovery.blend_fraction,
    )
    return recovery.target, output_kp, output_kd


def _contacts(
    *,
    model: Any,
    data: Any,
    ball_geom: int,
    floor_geom: int,
    passer_geoms: frozenset[int],
    shooter_geoms: frozenset[int],
    goalkeeper_geoms: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    import mujoco

    result = {
        "passer_left": False,
        "passer_right": False,
        "shooter_left": False,
        "shooter_right": False,
        "goalkeeper_left": False,
        "goalkeeper_right": False,
        "ball_passer": False,
        "ball_shooter": False,
        "ball_goalkeeper": False,
        "ball_passer_force_n": 0.0,
        "ball_shooter_force_n": 0.0,
        "robot_robot": False,
    }
    force: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
        if floor_geom in {geom1, geom2}:
            other = name2 if geom1 == floor_geom else name1
            if other.startswith("passer_left_foot"):
                result["passer_left"] = True
            elif other.startswith("passer_right_foot"):
                result["passer_right"] = True
            elif other.startswith("goalkeeper_left_foot"):
                result["goalkeeper_left"] = True
            elif other.startswith("goalkeeper_right_foot"):
                result["goalkeeper_right"] = True
            elif other.startswith("left_foot"):
                result["shooter_left"] = True
            elif other.startswith("right_foot"):
                result["shooter_right"] = True
        if ball_geom in {geom1, geom2}:
            other_geom = geom2 if geom1 == ball_geom else geom1
            if other_geom in passer_geoms:
                result["ball_passer"] = True
                mujoco.mj_contactForce(model, data, index, force)
                result["ball_passer_force_n"] = max(
                    float(result["ball_passer_force_n"]), float(np.linalg.norm(force[:3]))
                )
            elif other_geom in shooter_geoms:
                result["ball_shooter"] = True
                mujoco.mj_contactForce(model, data, index, force)
                result["ball_shooter_force_n"] = max(
                    float(result["ball_shooter_force_n"]), float(np.linalg.norm(force[:3]))
                )
            elif other_geom in goalkeeper_geoms:
                result["ball_goalkeeper"] = True
        if (geom1 in passer_geoms and geom2 in shooter_geoms) or (
            geom2 in passer_geoms and geom1 in shooter_geoms
        ):
            result["robot_robot"] = True
        if goalkeeper_geoms and (
            (geom1 in goalkeeper_geoms and geom2 in (passer_geoms | shooter_geoms))
            or (geom2 in goalkeeper_geoms and geom1 in (passer_geoms | shooter_geoms))
        ):
            result["robot_robot"] = True
    return result


def _robot_geom_ids(model: Any, root_body: int) -> frozenset[int]:
    values: set[int] = set()
    for geom in range(int(model.ngeom)):
        body = int(model.geom_bodyid[geom])
        while body > 0 and body != root_body:
            body = int(model.body_parentid[body])
        if body == root_body:
            values.add(geom)
    return frozenset(values)


def _append_trace(
    trace: dict[str, list[Any]],
    *,
    data: Any,
    ball_qpos: int,
    ball_qvel: int,
    passer: _Robot,
    shooter: _Robot,
    policy_frames: dict[str, int],
    support: tuple[tuple[bool, bool], tuple[bool, bool]],
    contact_role: int,
    robot_contacts: int,
    commanded_torque: dict[str, np.ndarray],
    projected_torque: dict[str, np.ndarray],
    executed_torque: dict[str, np.ndarray],
    contact_impulse: dict[str, float],
    learned_torque: dict[str, np.ndarray | IQLResidualDecision | None],
    joint_guard_active: dict[str, bool],
) -> None:
    trace["time"].append(float(data.time))
    trace["ball_pose"].append(data.qpos[ball_qpos : ball_qpos + 7].copy())
    trace["ball_velocity"].append(data.qvel[ball_qvel : ball_qvel + 6].copy())
    for robot in (passer, shooter):
        trace[f"{robot.role}_pelvis_pose"].append(
            data.qpos[robot.qpos_base : robot.qpos_base + 7].copy()
        )
        trace[f"{robot.role}_torso_quaternion"].append(data.xquat[robot.torso_body].copy())
        trace[f"{robot.role}_joint_position"].append(data.qpos[robot.joint_qpos].copy())
        trace[f"{robot.role}_joint_velocity"].append(data.qvel[robot.joint_qvel].copy())
        trace[f"{robot.role}_joint_torque"].append(data.ctrl[robot.actuators].copy())
        trace[f"{robot.role}_commanded_torque"].append(commanded_torque[robot.role].copy())
        trace[f"{robot.role}_safety_projected_torque"].append(projected_torque[robot.role].copy())
        trace[f"{robot.role}_executed_torque"].append(executed_torque[robot.role].copy())
        trace[f"{robot.role}_policy_action"].append(robot.last_target.copy())
        trace[f"{robot.role}_com_position"].append(data.subtree_com[robot.pelvis_body].copy())
        trace[f"{robot.role}_left_foot_position"].append(data.xpos[robot.left_ankle_body].copy())
        trace[f"{robot.role}_right_foot_position"].append(data.xpos[robot.right_ankle_body].copy())
        trace[f"{robot.role}_support_foot_slip"].append(robot.latest_support_slip_m)
        trace[f"{robot.role}_contact_impulse"].append(contact_impulse[robot.role])
        trace[f"{robot.role}_joint_guard_active"].append(joint_guard_active[robot.role])
        trace[f"{robot.role}_post_policy_blend_fraction"].append(robot.post_policy_blend_fraction)
        trace[f"{robot.role}_recovery_active"].append(robot.last_recovery_active)
        trace[f"{robot.role}_recovery_blend_fraction"].append(robot.last_recovery_blend_fraction)
        trace[f"{robot.role}_policy_frame"].append(policy_frames[robot.role])
    trace["passer_foot_contact"].append(support[0])
    trace["shooter_foot_contact"].append(support[1])
    trace["shooter_phase_correction"].append(shooter.last_phase_correction)
    trace["shooter_learned_torque_active"].append(learned_torque["shooter"] is not None)
    trace["ball_contact_role"].append(contact_role)
    trace["robot_robot_contact_count"].append(robot_contacts)


def _recovery_actor_state(
    robot: _Robot,
    data: Any,
    *,
    ball_body: int,
    timestamp_sec: float,
) -> np.ndarray:
    """Recreate the frozen 74-D recovery feature contract during simulation."""

    from rosclaw.growth.recovery_dataset import STATE_FEATURES

    pelvis = np.asarray(data.qpos[robot.qpos_base : robot.qpos_base + 3], dtype=np.float64)
    pelvis_velocity = np.asarray(data.qvel[robot.qvel_base : robot.qvel_base + 3], dtype=np.float64)
    roll, pitch = _roll_pitch(np.asarray(data.xquat[robot.torso_body], dtype=np.float64))
    com = np.asarray(data.subtree_com[robot.pelvis_body], dtype=np.float64)
    left_foot = np.asarray(data.xpos[robot.left_ankle_body], dtype=np.float64)
    right_foot = np.asarray(data.xpos[robot.right_ankle_body], dtype=np.float64)
    support_count = int(robot.latest_left_support) + int(robot.latest_right_support)
    support_y = (
        (
            left_foot[1] * int(robot.latest_left_support)
            + right_foot[1] * int(robot.latest_right_support)
        )
        / support_count
        if support_count
        else 0.0
    )
    state = np.concatenate(
        (
            np.asarray(data.qpos[robot.joint_qpos], dtype=np.float64),
            np.asarray(data.qvel[robot.joint_qvel], dtype=np.float64),
            np.asarray((pelvis[2],), dtype=np.float64),
            pelvis_velocity,
            np.asarray((roll, pitch, com[1] - support_y), dtype=np.float64),
            np.asarray(
                (float(robot.latest_left_support), float(robot.latest_right_support)),
                dtype=np.float64,
            ),
            np.asarray(data.xpos[ball_body], dtype=np.float64) - pelvis,
            np.asarray(robot.state.ball_vel_w, dtype=np.float64),
            np.asarray(
                (max(0.0, timestamp_sec - float(robot.contact_time or timestamp_sec)),),
                dtype=np.float64,
            ),
        )
    )
    if state.shape != (len(STATE_FEATURES),) or not np.all(np.isfinite(state)):
        raise RuntimeError("live recovery actor state violates the frozen feature contract")
    return state


def _update_support_slip(
    robot: _Robot,
    data: Any,
    support: tuple[bool, bool],
) -> None:
    slips: list[float] = []
    for side, active, body in (
        ("left", support[0], robot.left_ankle_body),
        ("right", support[1], robot.right_ankle_body),
    ):
        anchor_name = f"{side}_support_anchor"
        if not active:
            setattr(robot, anchor_name, None)
            continue
        position = np.asarray(data.xpos[body], dtype=np.float64).copy()
        anchor = getattr(robot, anchor_name)
        if anchor is None:
            setattr(robot, anchor_name, position)
            anchor = position
        slips.append(float(np.linalg.norm((position - anchor)[:2])))
    robot.latest_support_slip_m = max(slips, default=0.0)
    robot.peak_support_slip_m = max(
        robot.peak_support_slip_m,
        robot.latest_support_slip_m,
    )
    if robot.contact_latched:
        robot.post_contact_peak_support_slip_m = max(
            robot.post_contact_peak_support_slip_m,
            robot.latest_support_slip_m,
        )


def _reset_post_contact_support_anchors(robot: _Robot) -> None:
    """Start the recovery-slip clock at measured ball contact."""

    robot.left_support_anchor = None
    robot.right_support_anchor = None
    robot.latest_support_slip_m = 0.0


def _tail_wobble(trajectory: dict[str, np.ndarray], role: str) -> float:
    time = trajectory["time"]
    if len(time) < 3:
        return math.inf
    mask = time >= max(0.0, float(time[-1]) - 2.0)
    torso = trajectory[f"{role}_torso_quaternion"][mask]
    tilt = np.asarray([_roll_pitch(quat) for quat in torso], dtype=np.float64)
    velocity = trajectory[f"{role}_joint_velocity"][mask]
    return float(np.mean(np.linalg.norm(np.diff(tilt, axis=0), axis=1))) + float(
        np.mean(np.linalg.norm(velocity[:, :12], axis=1)) * 0.01
    )


def _robustness_case(
    offset: float,
    result: G1CoupledRelayResult,
) -> G1CoupledRelayRobustnessCase:
    return G1CoupledRelayRobustnessCase(
        shooter_start_offset_sec=offset,
        passed=result.passed,
        shot_contact_time_sec=result.shot_contact_time_sec,
        goal_crossing_y_m=result.goal_crossing_y_m,
        goal_crossing_z_m=result.goal_crossing_z_m,
        target_error_m=result.target_error_m,
        phase_hold_frames=result.receiver_phase_hold_frames,
        phase_advance_frames=result.receiver_phase_advance_frames,
        minimum_pelvis_height_m=min(
            result.passer_min_pelvis_height_m,
            result.shooter_min_pelvis_height_m,
        ),
        joint_limit_violation=result.joint_limit_violation,
        torque_limit_violation=result.torque_limit_violation,
    )


def _id(model: Any, object_type: Any, name: str) -> int:
    import mujoco

    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"coupled G1 model is missing {name}")
    return value


def _yaw(robot: _Robot) -> float:
    return 2.0 * math.atan2(
        float(robot.world_from_local_quat[3]),
        float(robot.world_from_local_quat[0]),
    )


def _rotate_z(vector: np.ndarray, yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray((cosine * x - sine * y, sine * x + cosine * y, z))


def _to_local(position: np.ndarray, robot: _Robot) -> np.ndarray:
    return _rotate_z(np.asarray(position, dtype=np.float64) - robot.origin, -_yaw(robot))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _standby_policy_hash(root: Path) -> str:
    files = (
        Path("policy/loco_mode/LocoMode.py"),
        Path("policy/loco_mode/config/LocoMode.yaml"),
        Path("policy/loco_mode/model/policy_29dof.pt"),
    )
    missing = [str(item) for item in files if not (root / item).is_file()]
    if missing:
        raise ValueError("missing locomotion standby assets: " + ",".join(missing))
    return hash_json({str(item): _file_hash(root / item) for item in files})


def coupled_runtime_manifest() -> dict[str, str]:
    """Return versions that are part of the coupled-physics evidence identity."""

    import mujoco
    import onnxruntime
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "onnxruntime": onnxruntime.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "G1CoupledRelayEvidence",
    "G1CoupledRelayResult",
    "G1CoupledRelayRobustnessCase",
    "G1GoalkeeperConfig",
    "G1JointGuardConfig",
    "coupled_runtime_manifest",
    "run_g1_coupled_relay",
]
