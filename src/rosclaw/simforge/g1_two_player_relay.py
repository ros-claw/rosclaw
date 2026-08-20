"""Evidence-bound two-player G1 pass-to-shot relay in CPU MuJoCo.

The first G1 executes a deliberately soft back-pass.  A measured rolling-ball
state from that episode is rotated into the second G1's local frame and becomes
the second episode's immutable initial condition.  The two episodes are kept
separate on purpose: this is a causal relay proof, not a claim that two bodies
were simulated in one coupled world.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    GoalForgeEpisode,
    trajectory_digest,
)
from rosclaw.simforge.g1_cerebellar_recovery import G1CerebellarRecoveryConfig
from rosclaw.simforge.g1_recovery_quality import measure_g1_recovery_quality
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters, hash_json
from rosclaw.simforge.tasks.g1_goalforge.scenario import (
    GoalForgeScenario,
    generate_goalforge_scenarios,
)

_SECRET = b"rosclaw-phase8-two-player-relay-v1"
_HANDOFF_SPEED_MIN_MPS = 0.55
_HANDOFF_SPEED_MAX_MPS = 0.60
_RECEIVER_CONTACT_X_M = 1.0
_RECEIVER_BALL_X_M = 1.25
_RECEIVER_CONTACT_TIME_SEC = 5.40


@dataclass(frozen=True)
class G1RelayLeg:
    role: str
    scenario: dict[str, Any]
    parameters: dict[str, Any]
    result: dict[str, Any]
    trajectory_path: str
    trajectory_hash: str
    trajectory_digest: str
    strict_replay: bool
    role_qualified: bool
    recovery_receipt: dict[str, Any]
    recovery_metrics: dict[str, Any]
    schema_version: str = "rosclaw.g1_goalforge.relay_leg.v2"


@dataclass(frozen=True)
class G1RelayHandoff:
    passer_trajectory_digest: str
    source_sample_index: int
    source_time_sec: float
    source_ball_position_m: tuple[float, float, float]
    passer_local_velocity_mps: tuple[float, float]
    passer_yaw_in_shooter_world_rad: float
    shooter_local_velocity_mps: tuple[float, float]
    shooter_ball_x_m: float
    shooter_ball_y_m: float
    shooter_launch_delay_sec: float
    observed_speed_mps: float
    schema_version: str = "rosclaw.g1_goalforge.relay_handoff.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["handoff_hash"] = self.handoff_hash
        return value

    @property
    def handoff_hash(self) -> str:
        return hash_json(asdict(self))


@dataclass(frozen=True)
class G1RelayRecoveryEvolution:
    parent_config_hash: str
    candidate_config_hash: str
    parent_result: dict[str, Any]
    parent_metrics: dict[str, Any]
    candidate_metrics: dict[str, Any]
    parent_trajectory_path: str
    parent_trajectory_hash: str
    parent_trajectory_digest: str
    parent_strict_replay: bool
    tail_tilt_reduction: float
    tail_wobble_reduction: float
    pelvis_path_regression: float
    tail_joint_jerk_regression: float
    decision: str
    reasons: tuple[str, ...]
    evidence_domain: str = "SIM"
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.g1_goalforge.relay_recovery_evolution.v1"


@dataclass(frozen=True)
class G1TwoPlayerRelay:
    body_hash: str
    kick_prior_hash: str
    backend_commit: str
    passer: G1RelayLeg
    handoff: G1RelayHandoff
    shooter: G1RelayLeg
    recovery_evolution: G1RelayRecoveryEvolution
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.g1_goalforge.two_player_relay.v2"

    @property
    def passed(self) -> bool:
        result = self.shooter.result
        scenario = self.shooter.scenario
        return bool(
            self.passer.role_qualified
            and self.passer.strict_replay
            and self.shooter.role_qualified
            and self.shooter.strict_replay
            and self.recovery_evolution.parent_strict_replay
            and self.recovery_evolution.decision == "SIM_CHAMPION"
            and self.recovery_evolution.candidate_config_hash
            == self.shooter.recovery_receipt["config_hash"]
            and result["success"]
            and result["target_zone_hit"]
            and float(result["target_error_m"]) <= 0.25
            and float(scenario["target_z_m"]) >= 0.70
            and self.handoff.observed_speed_mps >= _HANDOFF_SPEED_MIN_MPS
            and not any(
                result[name]
                for name in (
                    "post_kick_fall",
                    "joint_limit_violation",
                    "torque_limit_violation",
                    "actuator_saturation",
                )
            )
            and self.activation_ceiling == "SIM_ONLY"
            and not self.hardware_command_sent
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "backend_commit": self.backend_commit,
            "passer": asdict(self.passer),
            "handoff": self.handoff.to_dict(),
            "shooter": asdict(self.shooter),
            "recovery_evolution": asdict(self.recovery_evolution),
            "passed": self.passed,
            "activation_ceiling": self.activation_ceiling,
            "evidence_domain": self.evidence_domain,
            "physics_authority": self.physics_authority,
            "hardware_command_sent": self.hardware_command_sent,
            "claims": {
                "elevated_target_height_m": self.shooter.scenario["target_z_m"],
                "measured_incoming_ball_speed_mps": self.handoff.observed_speed_mps,
                "pass_to_shot_handoff_hash_bound": True,
                "strict_replay_both_legs": (
                    self.passer.strict_replay and self.shooter.strict_replay
                ),
                "shooter_tail_wobble_index": self.shooter.recovery_metrics[
                    "tail_wobble_index"
                ],
                "shooter_terminal_stable_duration_sec": self.shooter.recovery_metrics[
                    "terminal_stable_duration_sec"
                ],
                "bounded_recovery_self_evolution": (
                    self.recovery_evolution.decision == "SIM_CHAMPION"
                ),
                "sequential_two_episode_physics_relay": True,
                "simultaneous_two_body_physics": False,
                "real_hardware": False,
            },
        }


def run_g1_two_player_relay(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
) -> G1TwoPlayerRelay:
    """Run, replay, and persist the SIM-only back-pass-to-high-shot relay."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("G1 relay evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    qualification = backend.qualification
    base = _base_scenario()
    parent_recovery_config = _parent_recovery_config()
    recovery_config = _recovery_config()

    passer_scenario = replace(
        base,
        scenario_id="goalforge-relay-g1-a-soft-back-pass",
        ball_x_m=1.205,
        ball_y_m=-0.16,
        ball_velocity_x_mps=0.0,
        ball_velocity_y_mps=0.0,
        ball_launch_delay_sec=0.0,
        ball_ground_friction=0.10,
        target_y_m=0.0,
        target_z_m=0.20,
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
    passer_episode, passer_strict, passer_receipt = _run_strict(
        backend,
        passer_scenario,
        passer_parameters,
        recovery_config,
    )
    passer_qualified = _passer_qualified(passer_episode)
    if not passer_qualified:
        raise RuntimeError("G1-A did not produce a safe, measurable pass")
    handoff = _measure_handoff(passer_episode)

    shooter_scenario = replace(
        base,
        scenario_id="goalforge-relay-g1-b-fast-high-finish",
        ball_x_m=handoff.shooter_ball_x_m,
        ball_y_m=handoff.shooter_ball_y_m,
        ball_velocity_x_mps=handoff.shooter_local_velocity_mps[0],
        ball_velocity_y_mps=handoff.shooter_local_velocity_mps[1],
        ball_launch_delay_sec=handoff.shooter_launch_delay_sec,
        ball_ground_friction=0.03,
        target_y_m=1.10,
        target_z_m=0.70,
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
    shooter_episode, shooter_strict, shooter_receipt = _run_strict(
        backend,
        shooter_scenario,
        shooter_parameters,
        recovery_config,
    )
    shooter_qualified = _shooter_qualified(shooter_episode, handoff)
    shooter_parent, shooter_parent_strict, shooter_parent_receipt = _run_strict(
        backend,
        shooter_scenario,
        shooter_parameters,
        parent_recovery_config,
    )

    passer_path = _save_trajectory(root / "g1-a-soft-back-pass.npz", passer_episode)
    shooter_path = _save_trajectory(root / "g1-b-fast-high-finish.npz", shooter_episode)
    shooter_parent_path = _save_trajectory(
        root / "g1-b-recovery-parent.npz",
        shooter_parent,
    )
    recovery_evolution = _evaluate_recovery_evolution(
        parent=shooter_parent,
        candidate=shooter_episode,
        parent_config_hash=str(shooter_parent_receipt["config_hash"]),
        candidate_config_hash=str(shooter_receipt["config_hash"]),
        parent_path=shooter_parent_path,
        parent_strict_replay=shooter_parent_strict,
        candidate_strict_replay=shooter_strict,
    )
    report = G1TwoPlayerRelay(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        backend_commit=qualification.backend_commit,
        passer=_leg(
            role="G1_A_PASSER",
            scenario=passer_scenario,
            parameters=passer_parameters,
            episode=passer_episode,
            trajectory_path=passer_path,
            strict_replay=passer_strict,
            role_qualified=passer_qualified,
            recovery_receipt=passer_receipt,
        ),
        handoff=handoff,
        shooter=_leg(
            role="G1_B_SHOOTER",
            scenario=shooter_scenario,
            parameters=shooter_parameters,
            episode=shooter_episode,
            trajectory_path=shooter_path,
            strict_replay=shooter_strict,
            role_qualified=shooter_qualified,
            recovery_receipt=shooter_receipt,
        ),
        recovery_evolution=recovery_evolution,
    )
    _atomic_json(root / "g1-two-player-relay.json", report.to_dict())
    return report


def _base_scenario() -> GoalForgeScenario:
    return generate_goalforge_scenarios(
        ledger=SeedLedger(task_id="g1_penalty_kick", secret=_SECRET),
        partition=Partition.VALIDATION,
        count=1,
        generation=0,
    )[0]


def _recovery_config() -> G1CerebellarRecoveryConfig:
    return G1CerebellarRecoveryConfig(
        start_policy_frame=300,
        blend_frames=100,
        standing_pose_blend=0.30,
        roll_posture_bias_rad=-0.05,
        settling_start_policy_frame=400,
        settling_blend_frames=100,
        settling_standing_pose_blend=0.45,
        settling_roll_posture_bias_rad=-0.02,
        settling_waist_pitch_bias_rad=0.12,
        target_smoothing_alpha=0.60,
        target_smoothing_start_policy_frame=300,
        target_smoothing_joint_group="upper_body",
    )


def _parent_recovery_config() -> G1CerebellarRecoveryConfig:
    return replace(_recovery_config(), settling_waist_pitch_bias_rad=0.09)


def _run_strict(
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
    parameters: ShotParameters,
    recovery_config: G1CerebellarRecoveryConfig,
) -> tuple[GoalForgeEpisode, bool, dict[str, Any]]:
    controller = backend.build_cerebellar_recovery_controller(scenario, recovery_config)
    episode = backend.run(scenario, parameters, recovery_controller=controller)
    replay_controller = backend.build_cerebellar_recovery_controller(scenario, recovery_config)
    replay = backend.run(scenario, parameters, recovery_controller=replay_controller)
    strict = bool(
        replay.result.summary_dict() == episode.result.summary_dict()
        and trajectory_digest(replay.trajectory) == trajectory_digest(episode.trajectory)
    )
    receipt = controller.build_receipt(
        strict_replay=strict,
        evidence_domain="SIM",
    ).to_dict()
    return episode, strict, receipt


def _passer_qualified(episode: GoalForgeEpisode) -> bool:
    result = episode.result
    return bool(
        result.physics_executed
        and result.finite_state
        and result.kick_foot_contacted
        and result.ball_contact_time_sec is not None
        and result.ball_speed_mps >= 1.0
        and result.com_margin_min_m >= -0.04
        and result.torso_roll_peak_rad <= 0.45
        and result.torso_pitch_peak_rad <= 0.55
        and not result.post_kick_fall
        and not result.joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
    )


def _measure_handoff(episode: GoalForgeEpisode) -> G1RelayHandoff:
    contact_time = episode.result.ball_contact_time_sec
    if contact_time is None:
        raise ValueError("relay handoff requires a measured passer contact")
    time = np.asarray(episode.trajectory["time"], dtype=np.float64)
    position = np.asarray(episode.trajectory["ball_pose"], dtype=np.float64)[:, :3]
    velocity = np.asarray(episode.trajectory["ball_velocity"], dtype=np.float64)[:, :2]
    speed = np.linalg.norm(velocity, axis=1)
    eligible = np.flatnonzero(
        (time > contact_time + 0.05)
        & (velocity[:, 0] > _HANDOFF_SPEED_MIN_MPS)
        & (speed >= _HANDOFF_SPEED_MIN_MPS)
        & (speed <= _HANDOFF_SPEED_MAX_MPS)
    )
    if not len(eligible):
        raise RuntimeError("pass never entered the validated receiver-speed envelope")
    index = int(eligible[0])
    passer_velocity = velocity[index]
    # G1-A stands ahead of G1-B and faces back toward it.  A 180-degree frame
    # rotation turns A's local +x pass into B's local -x incoming velocity.
    shooter_velocity = -passer_velocity
    travel_time = (_RECEIVER_BALL_X_M - _RECEIVER_CONTACT_X_M) / abs(
        float(shooter_velocity[0])
    )
    launch_delay = _RECEIVER_CONTACT_TIME_SEC - travel_time
    shooter_ball_y = -float(shooter_velocity[1]) * travel_time
    if not 0.0 <= launch_delay <= 5.0:
        raise RuntimeError("measured pass cannot be scheduled inside the receiver scenario")
    return G1RelayHandoff(
        passer_trajectory_digest=trajectory_digest(episode.trajectory),
        source_sample_index=index,
        source_time_sec=float(time[index]),
        source_ball_position_m=tuple(float(item) for item in position[index]),
        passer_local_velocity_mps=tuple(float(item) for item in passer_velocity),
        passer_yaw_in_shooter_world_rad=math.pi,
        shooter_local_velocity_mps=tuple(float(item) for item in shooter_velocity),
        shooter_ball_x_m=_RECEIVER_BALL_X_M,
        shooter_ball_y_m=shooter_ball_y,
        shooter_launch_delay_sec=launch_delay,
        observed_speed_mps=float(speed[index]),
    )


def _shooter_qualified(
    episode: GoalForgeEpisode,
    handoff: G1RelayHandoff,
) -> bool:
    result = episode.result
    return bool(
        result.physics_executed
        and result.finite_state
        and result.success
        and result.target_zone_hit
        and result.target_error_m <= 0.25
        and result.com_margin_min_m >= -0.035
        and result.torso_roll_peak_rad <= 0.45
        and result.torso_pitch_peak_rad <= 0.55
        and handoff.observed_speed_mps >= _HANDOFF_SPEED_MIN_MPS
        and not result.post_kick_fall
        and not result.joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
    )


def _evaluate_recovery_evolution(
    *,
    parent: GoalForgeEpisode,
    candidate: GoalForgeEpisode,
    parent_config_hash: str,
    candidate_config_hash: str,
    parent_path: Path,
    parent_strict_replay: bool,
    candidate_strict_replay: bool,
) -> G1RelayRecoveryEvolution:
    parent_metrics = measure_g1_recovery_quality(parent.trajectory)
    candidate_metrics = measure_g1_recovery_quality(candidate.trajectory)
    tail_tilt_reduction = _relative_reduction(
        parent_metrics.tail_torso_tilt_rms_rad,
        candidate_metrics.tail_torso_tilt_rms_rad,
    )
    tail_wobble_reduction = _relative_reduction(
        parent_metrics.tail_wobble_index,
        candidate_metrics.tail_wobble_index,
    )
    pelvis_path_regression = _relative_regression(
        parent_metrics.post_contact_pelvis_path_length_m,
        candidate_metrics.post_contact_pelvis_path_length_m,
    )
    tail_joint_jerk_regression = _relative_regression(
        parent_metrics.tail_joint_jerk_rms_rad_s3,
        candidate_metrics.tail_joint_jerk_rms_rad_s3,
    )
    reasons = []
    if not parent_strict_replay or not candidate_strict_replay:
        reasons.append("strict_replay_failed")
    if not parent.result.success or not candidate.result.success:
        reasons.append("goal_not_preserved")
    if tail_tilt_reduction < 0.05:
        reasons.append("tail_tilt_reduction_below_5pct")
    if tail_wobble_reduction < 0.0:
        reasons.append("tail_wobble_regressed")
    if pelvis_path_regression > 0.05:
        reasons.append("pelvis_path_regression_above_5pct")
    if tail_joint_jerk_regression > 0.15:
        reasons.append("tail_joint_jerk_regression_above_15pct")
    if any(
        getattr(candidate.result, name)
        for name in (
            "post_kick_fall",
            "joint_limit_violation",
            "torque_limit_violation",
            "actuator_saturation",
        )
    ):
        reasons.append("candidate_safety_violation")
    return G1RelayRecoveryEvolution(
        parent_config_hash=parent_config_hash,
        candidate_config_hash=candidate_config_hash,
        parent_result=parent.result.summary_dict(),
        parent_metrics=parent_metrics.to_dict(),
        candidate_metrics=candidate_metrics.to_dict(),
        parent_trajectory_path=str(parent_path),
        parent_trajectory_hash=_file_hash(parent_path),
        parent_trajectory_digest=trajectory_digest(parent.trajectory),
        parent_strict_replay=parent_strict_replay,
        tail_tilt_reduction=tail_tilt_reduction,
        tail_wobble_reduction=tail_wobble_reduction,
        pelvis_path_regression=pelvis_path_regression,
        tail_joint_jerk_regression=tail_joint_jerk_regression,
        decision="SIM_CHAMPION" if not reasons else "REJECTED",
        reasons=tuple(reasons),
    )


def _relative_reduction(parent: float, candidate: float) -> float:
    return (parent - candidate) / max(abs(parent), 1e-12)


def _relative_regression(parent: float, candidate: float) -> float:
    return (candidate - parent) / max(abs(parent), 1e-12)


def _leg(
    *,
    role: str,
    scenario: GoalForgeScenario,
    parameters: ShotParameters,
    episode: GoalForgeEpisode,
    trajectory_path: Path,
    strict_replay: bool,
    role_qualified: bool,
    recovery_receipt: dict[str, Any],
) -> G1RelayLeg:
    return G1RelayLeg(
        role=role,
        scenario=scenario.to_private_dict(),
        parameters=parameters.to_dict(),
        result=episode.result.summary_dict(),
        trajectory_path=str(trajectory_path),
        trajectory_hash=_file_hash(trajectory_path),
        trajectory_digest=trajectory_digest(episode.trajectory),
        strict_replay=strict_replay,
        role_qualified=role_qualified,
        recovery_receipt=recovery_receipt,
        recovery_metrics=measure_g1_recovery_quality(episode.trajectory).to_dict(),
    )


def _save_trajectory(path: Path, episode: GoalForgeEpisode) -> Path:
    np.savez_compressed(path, **episode.trajectory)  # type: ignore[arg-type]
    return path


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


__all__ = [
    "G1RelayHandoff",
    "G1RelayLeg",
    "G1RelayRecoveryEvolution",
    "G1TwoPlayerRelay",
    "run_g1_two_player_relay",
]
