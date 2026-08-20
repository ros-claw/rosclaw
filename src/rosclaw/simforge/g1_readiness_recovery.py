"""Continuous-world physical recovery after a G1 readiness abstention."""

from __future__ import annotations

import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.growth.proprioceptive_expert_router import G1ProprioceptiveExpertRouter
from rosclaw.growth.proprioceptive_readiness_gate import G1ProprioceptiveReadinessGate
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw.simforge.g1_sonic_runup import (
    G1SonicRunupConfig,
    G1SonicRunupController,
    qualify_g1_sonic,
)
from rosclaw.simforge.g1_stadium_scene import (
    G1TrainingGoalSpec,
    build_g1_stadium_model,
    g1_stadium_scene_hash,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_HARD_TORQUE_LIMITS,
    hash_bytes,
    hash_json,
)


@dataclass(frozen=True)
class G1ReadinessRecoveryConfig:
    neural_deceleration_duration_sec: float = 1.80
    hold_duration_sec: float = 1.20
    gain_scale: float = 0.75
    maximum_peak_tilt_rad: float = 0.65
    minimum_pelvis_height_m: float = 0.65
    maximum_final_speed_mps: float = 0.20
    maximum_final_joint_velocity_rms_rad_s: float = 0.50
    schema_version: str = "rosclaw.simforge.g1_readiness_recovery_config.v3"

    def __post_init__(self) -> None:
        values = (
            self.neural_deceleration_duration_sec,
            self.hold_duration_sec,
            self.gain_scale,
            self.maximum_peak_tilt_rad,
            self.minimum_pelvis_height_m,
            self.maximum_final_speed_mps,
            self.maximum_final_joint_velocity_rms_rad_s,
        )
        if not all(math.isfinite(item) for item in values):
            raise ValueError("readiness recovery config must be finite")
        if not 0.4 <= self.neural_deceleration_duration_sec <= 2.0:
            raise ValueError(
                "readiness neural deceleration duration must be in [0.4, 2.0]"
            )
        if not 1.0 <= self.hold_duration_sec <= 3.0:
            raise ValueError("readiness recovery hold duration must be in [1.0, 3.0]")
        if not 0.5 <= self.gain_scale <= 1.0:
            raise ValueError("readiness recovery gain scale must be in [0.5, 1.0]")


@dataclass(frozen=True)
class G1ReadinessRecoveryResult:
    planner_seed: int
    readiness_abstained: bool
    router_phase_start_frame: int
    safe_supported_phases: tuple[int, ...]
    neighbor_seeds: tuple[int, ...]
    neighbor_distances: tuple[float, ...]
    initial_pelvis_height_m: float
    initial_speed_mps: float
    initial_joint_velocity_rms_rad_s: float
    recovery_min_pelvis_height_m: float
    recovery_peak_tilt_rad: float
    final_pelvis_height_m: float
    final_speed_mps: float
    final_joint_velocity_rms_rad_s: float
    pre_abstention_saturation_steps: int
    pre_abstention_peak_demand_ratio: float
    actuator_saturation_steps: int
    actuator_saturation_fraction: float
    actuator_peak_demand_ratio: float
    finite_state: bool
    post_abstention_fall: bool
    joint_limit_violation: bool
    torque_limit_violation: bool
    physics_steps: int
    passed: bool
    schema_version: str = "rosclaw.simforge.g1_readiness_recovery_result.v3"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "safe_supported_phases": list(self.safe_supported_phases),
            "neighbor_seeds": list(self.neighbor_seeds),
            "neighbor_distances": list(self.neighbor_distances),
        }


@dataclass(frozen=True)
class G1ReadinessRecoveryEvidence:
    body_hash: str
    router_hash: str
    readiness_gate_hash: str
    sonic_qualification_hash: str
    sonic_runup_reference_digest: str
    sonic_reference_digest: str
    stadium_scene_hash: str
    implementation_hash: str
    request_hash: str
    trajectory_path: str
    trajectory_hash: str
    trajectory_digest: str
    strict_replay: bool
    sonic_runup_config: G1SonicRunupConfig
    recovery_config: G1ReadinessRecoveryConfig
    result: G1ReadinessRecoveryResult
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "DEVELOPMENT_READINESS_RECOVERY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.simforge.g1_readiness_recovery_evidence.v3"

    @property
    def passed(self) -> bool:
        return bool(
            self.strict_replay
            and self.result.passed
            and self.activation_ceiling == "SIM_ONLY"
            and not self.hardware_command_sent
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "sonic_runup_config": asdict(self.sonic_runup_config),
            "recovery_config": asdict(self.recovery_config),
            "result": self.result.to_dict(),
            "passed": self.passed,
            "claims": {
                "readiness_abstention_executed": self.result.readiness_abstained,
                "continuous_world_no_state_reset": True,
                "phase_continuous_neural_deceleration": True,
                "frozen_planner_tail_then_terminal_pose_padding": True,
                "neural_feedback_active_during_hold": True,
                "ball_contact_attempted": False,
                "neural_sonic_runup": True,
                "pretrained_neural_recovery_tail": True,
                "rosclaw_trained_recovery_policy": False,
                "promotion_evidence": False,
                "real_hardware": False,
            },
        }


def run_g1_readiness_recovery(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    output_dir: Path,
    source_checkout: Path,
    router: G1ProprioceptiveExpertRouter,
    readiness_gate: G1ProprioceptiveReadinessGate,
    sonic_config: G1SonicRunupConfig,
    recovery_config: G1ReadinessRecoveryConfig | None = None,
    evidence_domain: str = "DEVELOPMENT_READINESS_RECOVERY",
) -> G1ReadinessRecoveryEvidence:
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("readiness recovery evidence must be outside the checkout")
    if evidence_domain not in {
        "DEVELOPMENT_READINESS_RECOVERY",
        "FROZEN_READINESS_RECOVERY_VALIDATION",
    }:
        raise ValueError("readiness recovery evidence domain is invalid")
    root.mkdir(parents=True, exist_ok=False)
    recovery = recovery_config or G1ReadinessRecoveryConfig()
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    sonic_qualification = qualify_g1_sonic(sonic_model_root)
    sonic_qualification.require_eligible()
    if (
        router.body_hash != qualification.body_hash
        or readiness_gate.body_hash != qualification.body_hash
    ):
        raise ValueError("readiness recovery Body hash mismatch")
    if readiness_gate.router_hash != router.router_hash or not readiness_gate.accepted:
        raise ValueError("readiness recovery router/gate binding is invalid")
    implementation_hash = hash_json(
        {
            name: hash_bytes(Path(__file__).with_name(name).read_bytes())
            for name in (
                "g1_readiness_recovery.py",
                "g1_sonic_runup.py",
                "g1_stadium_scene.py",
            )
        }
        | {
            "growth/proprioceptive_expert_router.py": hash_bytes(
                Path(__file__)
                .parents[1]
                .joinpath("growth/proprioceptive_expert_router.py")
                .read_bytes()
            ),
            "growth/proprioceptive_readiness_gate.py": hash_bytes(
                Path(__file__)
                .parents[1]
                .joinpath("growth/proprioceptive_readiness_gate.py")
                .read_bytes()
            ),
        }
    )
    request = {
        "schema_version": "rosclaw.simforge.g1_readiness_recovery_request.v3",
        "body_hash": qualification.body_hash,
        "router_hash": router.router_hash,
        "readiness_gate_hash": readiness_gate.gate_hash,
        "sonic_qualification_hash": sonic_qualification.qualification_hash,
        "stadium_scene_hash": g1_stadium_scene_hash(asset_root, G1TrainingGoalSpec()),
        "implementation_hash": implementation_hash,
        "sonic_runup_config": asdict(sonic_config),
        "recovery_config": asdict(recovery),
        "evidence_domain": evidence_domain,
        "activation_ceiling": "SIM_ONLY",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    request_path = root / "request.json"
    _write_json(request_path, request)
    result, trajectory, runup_reference_digest, reference_digest = _simulate(
        asset_root=asset_root,
        sonic_model_root=sonic_model_root,
        router=router,
        readiness_gate=readiness_gate,
        sonic_config=sonic_config,
        recovery_config=recovery,
    )
    (
        replay_result,
        replay_trajectory,
        replay_runup_reference_digest,
        replay_reference_digest,
    ) = _simulate(
        asset_root=asset_root,
        sonic_model_root=sonic_model_root,
        router=router,
        readiness_gate=readiness_gate,
        sonic_config=sonic_config,
        recovery_config=recovery,
    )
    digest = trajectory_digest(trajectory)
    strict = bool(
        result.to_dict() == replay_result.to_dict()
        and digest == trajectory_digest(replay_trajectory)
        and runup_reference_digest == replay_runup_reference_digest
        and reference_digest == replay_reference_digest
    )
    trajectory_path = root / "g1-readiness-recovery-trajectory.npz"
    np.savez_compressed(trajectory_path, **trajectory)
    evidence = G1ReadinessRecoveryEvidence(
        body_hash=qualification.body_hash,
        router_hash=router.router_hash,
        readiness_gate_hash=readiness_gate.gate_hash,
        sonic_qualification_hash=sonic_qualification.qualification_hash,
        sonic_runup_reference_digest=runup_reference_digest,
        sonic_reference_digest=reference_digest,
        stadium_scene_hash=request["stadium_scene_hash"],
        implementation_hash=implementation_hash,
        request_hash=hash_bytes(request_path.read_bytes()),
        trajectory_path=str(trajectory_path),
        trajectory_hash=hash_bytes(trajectory_path.read_bytes()),
        trajectory_digest=digest,
        strict_replay=strict,
        sonic_runup_config=sonic_config,
        recovery_config=recovery,
        result=result,
        evidence_domain=evidence_domain,
    )
    _write_json(root / "g1-readiness-recovery.json", evidence.to_dict())
    return evidence


def _simulate(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    router: G1ProprioceptiveExpertRouter,
    readiness_gate: G1ProprioceptiveReadinessGate,
    sonic_config: G1SonicRunupConfig,
    recovery_config: G1ReadinessRecoveryConfig,
) -> tuple[G1ReadinessRecoveryResult, dict[str, np.ndarray], str, str]:
    import mujoco

    from rosclaw.growth.proprioceptive_expert_router import strike_handoff_features
    from rosclaw.simforge.g1_free_kick_showcase import (
        _BALL_RADIUS_M,
        _configure_surface,
        _finite,
        _joint_violation,
        _ModelIds,
        _roll_pitch,
    )

    model = build_g1_stadium_model(asset_root, G1TrainingGoalSpec())
    data = mujoco.MjData(model)
    model.opt.timestep = sonic_config.physics_dt_sec
    ids = _ModelIds.from_model(model)
    _configure_surface(model)
    sonic = G1SonicRunupController(sonic_model_root, sonic_config)
    data.qpos[:3] = (-3.4, 0.0, 0.793)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:36] = sonic.default_angles
    data.qpos[ids.ball_qpos : ids.ball_qpos + 3] = (1.0, 0.0, _BALL_RADIUS_M)
    data.qpos[ids.ball_qpos + 3 : ids.ball_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    sonic.reset(data)
    trace: dict[str, list[Any]] = {
        "time": [],
        "joint_position": [],
        "joint_velocity": [],
        "joint_torque": [],
        "pelvis_pose": [],
        "torso_quaternion": [],
        "ball_pose": [],
        "controller_mode": [],
    }
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    substeps = int(round(sonic_config.policy_dt_sec / sonic_config.physics_dt_sec))
    finite = True
    joint_violation = False
    torque_violation = False
    pre_abstention_saturation_steps = 0
    pre_abstention_peak_ratio = 0.0
    physics_steps = 0
    for frame in range(sonic_config.execution_frames):
        sonic.update(data, frame)
        last_torque = np.zeros(29, dtype=np.float64)
        for _ in range(substeps):
            raw = sonic.raw_torque(data)
            ratio = float(np.max(np.abs(raw) / limits))
            pre_abstention_peak_ratio = max(pre_abstention_peak_ratio, ratio)
            pre_abstention_saturation_steps += int(ratio >= 0.999)
            last_torque = np.clip(raw, -limits, limits)
            torque_violation = torque_violation or bool(np.any(np.abs(last_torque) > limits))
            data.ctrl[:] = last_torque
            mujoco.mj_step(model, data)
            physics_steps += 1
        sonic.observe(data)
        finite = finite and _finite(data)
        joint_violation = joint_violation or _joint_violation(model, data)
        _append(trace, data, ids, last_torque, 5)

    features = strike_handoff_features(
        np.asarray(data.qpos[:7], dtype=np.float64),
        np.asarray(data.qvel[6:35], dtype=np.float64),
    )
    decision = readiness_gate.decide(features, router)
    if not decision.abstained:
        raise ValueError("readiness recovery may run only for an abstained handoff state")
    initial_height = float(data.qpos[2])
    initial_speed = float(np.linalg.norm(data.qvel[:2]))
    initial_joint_speed = float(np.sqrt(np.mean(np.square(data.qvel[6:35]))))
    runup_reference_digest = sonic.reference_digest
    neural_deceleration_frames = int(
        round(
            recovery_config.neural_deceleration_duration_sec
            / sonic_config.policy_dt_sec
        )
    )
    hold_frames = int(round(recovery_config.hold_duration_sec / sonic_config.policy_dt_sec))
    sonic.extend_stationary_recovery(neural_deceleration_frames + hold_frames)
    minimum_height = initial_height
    initial_roll, initial_pitch = _roll_pitch(data.xquat[ids.torso])
    peak_tilt = max(abs(initial_roll), abs(initial_pitch))
    saturation_steps = 0
    peak_ratio = 0.0
    for frame in range(neural_deceleration_frames):
        sonic.update_recovery_extension(data, frame)
        last_torque = np.zeros(29, dtype=np.float64)
        for _ in range(substeps):
            raw = sonic.raw_torque(data)
            ratio = float(np.max(np.abs(raw) / limits))
            peak_ratio = max(peak_ratio, ratio)
            saturation_steps += int(ratio >= 0.999)
            last_torque = np.clip(raw, -limits, limits)
            torque_violation = torque_violation or bool(np.any(np.abs(last_torque) > limits))
            data.ctrl[:] = last_torque
            mujoco.mj_step(model, data)
            physics_steps += 1
        sonic.observe(data)
        roll, pitch = _roll_pitch(data.xquat[ids.torso])
        minimum_height = min(minimum_height, float(data.qpos[2]))
        peak_tilt = max(peak_tilt, abs(roll), abs(pitch))
        finite = finite and _finite(data)
        joint_violation = joint_violation or _joint_violation(model, data)
        _append(trace, data, ids, last_torque, 6)
    for frame in range(hold_frames):
        sonic.update_recovery_extension(data, neural_deceleration_frames + frame)
        last_torque = np.zeros(29, dtype=np.float64)
        for _ in range(substeps):
            raw = sonic.raw_torque(data) * recovery_config.gain_scale
            ratio = float(np.max(np.abs(raw) / limits))
            peak_ratio = max(peak_ratio, ratio)
            saturation_steps += int(ratio >= 0.999)
            last_torque = np.clip(raw, -limits, limits)
            torque_violation = torque_violation or bool(np.any(np.abs(last_torque) > limits))
            data.ctrl[:] = last_torque
            mujoco.mj_step(model, data)
            physics_steps += 1
        sonic.observe(data)
        roll, pitch = _roll_pitch(data.xquat[ids.torso])
        minimum_height = min(minimum_height, float(data.qpos[2]))
        peak_tilt = max(peak_tilt, abs(roll), abs(pitch))
        finite = finite and _finite(data)
        joint_violation = joint_violation or _joint_violation(model, data)
        _append(trace, data, ids, last_torque, 7)
    final_height = float(data.qpos[2])
    final_speed = float(np.linalg.norm(data.qvel[:2]))
    final_joint_speed = float(np.sqrt(np.mean(np.square(data.qvel[6:35]))))
    fall = bool(minimum_height < 0.55 or peak_tilt > 1.2)
    passed = bool(
        finite
        and not fall
        and not joint_violation
        and not torque_violation
        and saturation_steps == 0
        and minimum_height >= recovery_config.minimum_pelvis_height_m
        and peak_tilt <= recovery_config.maximum_peak_tilt_rad
        and final_height >= 0.70
        and final_speed <= recovery_config.maximum_final_speed_mps
        and final_joint_speed <= recovery_config.maximum_final_joint_velocity_rms_rad_s
    )
    result = G1ReadinessRecoveryResult(
        planner_seed=sonic_config.planner_seed,
        readiness_abstained=True,
        router_phase_start_frame=decision.router_phase_start_frame,
        safe_supported_phases=decision.safe_supported_phases,
        neighbor_seeds=decision.neighbor_seeds,
        neighbor_distances=decision.neighbor_distances,
        initial_pelvis_height_m=initial_height,
        initial_speed_mps=initial_speed,
        initial_joint_velocity_rms_rad_s=initial_joint_speed,
        recovery_min_pelvis_height_m=minimum_height,
        recovery_peak_tilt_rad=peak_tilt,
        final_pelvis_height_m=final_height,
        final_speed_mps=final_speed,
        final_joint_velocity_rms_rad_s=final_joint_speed,
        pre_abstention_saturation_steps=pre_abstention_saturation_steps,
        pre_abstention_peak_demand_ratio=pre_abstention_peak_ratio,
        actuator_saturation_steps=saturation_steps,
        actuator_saturation_fraction=saturation_steps
        / max(1, (neural_deceleration_frames + hold_frames) * substeps),
        actuator_peak_demand_ratio=peak_ratio,
        finite_state=finite,
        post_abstention_fall=fall,
        joint_limit_violation=joint_violation,
        torque_limit_violation=torque_violation,
        physics_steps=physics_steps,
        passed=passed,
    )
    return (
        result,
        {key: np.asarray(value) for key, value in trace.items()},
        runup_reference_digest,
        sonic.reference_digest,
    )


def _append(
    trace: dict[str, list[Any]], data: Any, ids: Any, torque: np.ndarray, mode: int
) -> None:
    trace["time"].append(float(data.time))
    trace["joint_position"].append(np.asarray(data.qpos[7:36], dtype=np.float64).copy())
    trace["joint_velocity"].append(np.asarray(data.qvel[6:35], dtype=np.float64).copy())
    trace["joint_torque"].append(np.asarray(torque, dtype=np.float64).copy())
    trace["pelvis_pose"].append(np.asarray(data.qpos[:7], dtype=np.float64).copy())
    trace["torso_quaternion"].append(np.asarray(data.xquat[ids.torso], dtype=np.float64).copy())
    trace["ball_pose"].append(
        np.asarray(data.qpos[ids.ball_qpos : ids.ball_qpos + 7], dtype=np.float64).copy()
    )
    trace["controller_mode"].append(mode)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "G1ReadinessRecoveryConfig",
    "G1ReadinessRecoveryEvidence",
    "G1ReadinessRecoveryResult",
    "run_g1_readiness_recovery",
]
