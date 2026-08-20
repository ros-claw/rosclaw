"""Strict three-G1 long relay, precision finish and reactive goalkeeper evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    trajectory_digest,
)
from rosclaw.simforge.g1_coupled_relay import (
    G1CoupledRelayResult,
    G1GoalkeeperConfig,
    _simulate,
    _standby_policy_hash,
    coupled_runtime_manifest,
    trained_coupled_skill_simulation_kwargs,
)
from rosclaw.simforge.g1_stadium_scene import G1TrainingGoalSpec
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_bytes, hash_json

_PASSER_ORIGIN = (5.10, -0.16406006503921598, 0.0)
_PASSER_BALL_LOCAL_XY = (1.205, -0.16)
_PHYSICAL_TARGET = (7.50, 0.89, 0.115)
_CALIBRATED_POLICY_TARGET = (7.50, 0.70, 0.50)


@dataclass(frozen=True)
class G1ThreePlayerShowcaseEvidence:
    """Machine-checkable receipt for one shared-world three-player rollout."""

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
    pass_distance_m: float
    shot_distance_m: float
    pass_speed_start_mps: float
    pass_speed_end_mps: float
    pass_speed_max_positive_step_mps: float
    pass_speed_positive_step_count: int
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    simultaneous_three_body_physics: bool = True
    shared_ball_state: bool = True
    unified_physics_and_render_scene: bool = True
    hardware_command_sent: bool = False
    environment_hash: str = ""
    schema_version: str = "rosclaw.g1_goalforge.three_player_showcase_evidence.v1"

    @property
    def passed(self) -> bool:
        return bool(
            self.strict_replay
            and self.result.passed
            and self.result.goalkeeper_enabled
            and self.result.target_error_m is not None
            and self.result.target_error_m <= 0.10
            and self.result.goalkeeper_reaction_active_fraction > 0.0
            and self.result.goalkeeper_lateral_displacement_m >= 0.75
            and self.result.goalkeeper_min_pelvis_height_m is not None
            and self.result.goalkeeper_min_pelvis_height_m >= 0.65
            and not self.result.goalkeeper_joint_limit_violation
            and self.pass_distance_m >= 2.75
            and self.shot_distance_m >= 6.25
            and self.pass_speed_max_positive_step_mps <= 0.03
            and self.activation_ceiling == "SIM_ONLY"
            and not self.hardware_command_sent
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["result"] = self.result.to_dict()
        value["passed"] = self.passed
        value["claims"] = {
            "simultaneous_three_body_physics": self.simultaneous_three_body_physics,
            "single_shared_ball": self.shared_ball_state,
            "reactive_goalkeeper_uses_measured_ball_state": True,
            "goalkeeper_uses_qualified_locomotion_policy": True,
            "unified_physics_and_render_scene": self.unified_physics_and_render_scene,
            "pass_speed_is_measured_not_render_inferred": True,
            "strict_deterministic_replay": self.strict_replay,
            "pixels_used_for_promotion": False,
            "real_hardware": False,
        }
        return value


def three_player_goal_spec() -> G1TrainingGoalSpec:
    """Return the shared long-range goal and thin-net contract."""

    return G1TrainingGoalSpec(
        plane_x_m=_PHYSICAL_TARGET[0],
        width_m=3.0,
        height_m=2.0,
        depth_m=1.2,
        target_y_m=_PHYSICAL_TARGET[1],
        target_z_m=_PHYSICAL_TARGET[2],
        precision_radius_m=0.10,
        ball_free_joint_damping_n_s_m=0.02,
    )


def three_player_goalkeeper_config() -> G1GoalkeeperConfig:
    """Return the retained safe goalkeeper route from bounded SIM search."""

    return G1GoalkeeperConfig(
        maximum_lateral_speed_mps=0.25,
        ready_shuffle_speed_mps=0.06,
        arm_spread_rad=0.20,
        maximum_waist_lean_rad=0.05,
    )


def three_player_simulation_kwargs() -> dict[str, Any]:
    """Frozen retained candidate; the policy target is inverse-system calibration."""

    # Reuse the retained aim expert as well as the symmetric post-impact
    # cerebellum.  Omitting the calibrated foot yaw/pitch route is an explicit
    # failed ablation in the v1 evidence (0.278 m miss and joint rejection).
    values = trained_coupled_skill_simulation_kwargs()
    values.update(
        {
            "shooter_start_sec": 2.50,
            "shooter_target": _PHYSICAL_TARGET,
            "shooter_policy_target": _CALIBRATED_POLICY_TARGET,
            "passer_origin": _PASSER_ORIGIN,
            "passer_ball_local_xy": _PASSER_BALL_LOCAL_XY,
            "ball_ground_friction": 0.10,
            # The longer shared-world route invalidates the old short-pass
            # phase latch. Fixed causal timing won the bounded comparison.
            "receiver_phase_sync_enabled": False,
            "goal_spec": three_player_goal_spec(),
            "goalkeeper_config": three_player_goalkeeper_config(),
            "unified_stadium_scene": True,
        }
    )
    return values


def run_g1_three_player_showcase(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
) -> G1ThreePlayerShowcaseEvidence:
    """Run and strictly replay the retained three-player SIM-only candidate."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("three-player evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    implementation_hash = hash_json(
        {
            "showcase": hash_bytes(Path(__file__).read_bytes()),
            "coupled_runtime": hash_bytes(
                Path(__file__).with_name("g1_coupled_relay.py").read_bytes()
            ),
            "stadium_scene": hash_bytes(
                Path(__file__).with_name("g1_stadium_scene.py").read_bytes()
            ),
        }
    )
    goal = three_player_goal_spec()
    keeper = three_player_goalkeeper_config()
    kwargs = three_player_simulation_kwargs()
    request = {
        "schema_version": "rosclaw.g1_goalforge.three_player_showcase_request.v1",
        "body_hash": backend.qualification.body_hash,
        "kick_prior_hash": backend.qualification.kick_prior_hash,
        "standby_policy_hash": _standby_policy_hash(asset_root.expanduser().resolve()),
        "implementation_hash": implementation_hash,
        "passer_origin_m": list(_PASSER_ORIGIN),
        "passer_ball_local_xy_m": list(_PASSER_BALL_LOCAL_XY),
        "shooter_start_sec": 2.50,
        "physical_scoring_target_m": list(_PHYSICAL_TARGET),
        "inverse_calibrated_policy_target_m": list(_CALIBRATED_POLICY_TARGET),
        "goal_spec": asdict(goal),
        "goalkeeper_config": asdict(keeper),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "runtime": coupled_runtime_manifest(),
    }
    request["environment_hash"] = hash_json(request["runtime"])
    request_path = root / "request.json"
    _write_json(request_path, request)

    result, trajectory = _simulate(asset_root, backend, **kwargs)
    replay_result, replay_trajectory = _simulate(asset_root, backend, **kwargs)
    strict = bool(
        result.to_dict() == replay_result.to_dict()
        and trajectory_digest(trajectory) == trajectory_digest(replay_trajectory)
    )
    metrics = _trajectory_metrics(trajectory, result)
    trajectory_path = root / "trajectory.npz"
    np.savez_compressed(trajectory_path, **trajectory)  # type: ignore[arg-type]
    evidence = G1ThreePlayerShowcaseEvidence(
        body_hash=backend.qualification.body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
        standby_policy_hash=str(request["standby_policy_hash"]),
        backend_commit=backend.qualification.backend_commit,
        implementation_hash=implementation_hash,
        request_hash=_file_hash(request_path),
        trajectory_hash=_file_hash(trajectory_path),
        trajectory_digest=trajectory_digest(trajectory),
        strict_replay=strict,
        result=result,
        environment_hash=str(request["environment_hash"]),
        pass_distance_m=float(metrics["pass_distance_m"]),
        shot_distance_m=float(metrics["shot_distance_m"]),
        pass_speed_start_mps=float(metrics["pass_speed_start_mps"]),
        pass_speed_end_mps=float(metrics["pass_speed_end_mps"]),
        pass_speed_max_positive_step_mps=float(metrics["pass_speed_max_positive_step_mps"]),
        pass_speed_positive_step_count=int(metrics["pass_speed_positive_step_count"]),
    )
    _write_json(root / "g1-three-player-showcase.json", evidence.to_dict())
    return evidence


def _trajectory_metrics(
    trajectory: dict[str, np.ndarray],
    result: G1CoupledRelayResult,
) -> dict[str, float | int]:
    if result.pass_contact_time_sec is None or result.shot_contact_time_sec is None:
        raise ValueError("three-player metrics require both measured contacts")
    time = np.asarray(trajectory["time"], dtype=np.float64)
    pose = np.asarray(trajectory["ball_pose"], dtype=np.float64)
    velocity = np.asarray(trajectory["ball_velocity"], dtype=np.float64)
    pass_index = int(np.searchsorted(time, result.pass_contact_time_sec, side="left"))
    shot_index = int(np.searchsorted(time, result.shot_contact_time_sec, side="left"))
    if not 0 <= pass_index < shot_index < len(time):
        raise ValueError("three-player contact ordering is inconsistent with trajectory")
    rolling_speed = np.linalg.norm(velocity[pass_index:shot_index, :2], axis=1)
    # Skip the discrete kick impulse. Subsequent positive steps expose
    # recontacts or simulation/render discontinuities directly.
    settled_speed = rolling_speed[min(2, len(rolling_speed) - 1) :]
    steps = np.diff(settled_speed)
    positive_steps = steps[steps > 0.01]
    goal_x = _PHYSICAL_TARGET[0]
    crossing_candidates = np.flatnonzero(pose[:, 0] >= goal_x)
    crossing_index = int(crossing_candidates[0]) if crossing_candidates.size else len(pose) - 1
    return {
        "pass_distance_m": float(np.linalg.norm(pose[shot_index, :2] - pose[pass_index, :2])),
        "shot_distance_m": float(np.linalg.norm(pose[crossing_index, :2] - pose[shot_index, :2])),
        "pass_speed_start_mps": float(settled_speed[0]),
        "pass_speed_end_mps": float(settled_speed[-1]),
        "pass_speed_max_positive_step_mps": (
            0.0 if positive_steps.size == 0 else float(np.max(positive_steps))
        ),
        "pass_speed_positive_step_count": int(positive_steps.size),
    }


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "G1ThreePlayerShowcaseEvidence",
    "run_g1_three_player_showcase",
    "three_player_goal_spec",
    "three_player_goalkeeper_config",
    "three_player_simulation_kwargs",
]
