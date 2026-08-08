"""Official G1 MuJoCo + RoboNaldo kick-prior backend for GoalForge.

The public assets remain external to ROSClaw.  This module qualifies their
joint/policy contract, executes them headlessly, and records only hashes and
physics evidence.  No hardware transport is opened here.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import math
import subprocess
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import FeedbackReceipt
from rosclaw.feedback.ilc import ILCFeedforward
from rosclaw.feedback.profiles.g1 import g1_joint_residual_limits
from rosclaw.feedback.runtime import FeedbackRuntime
from rosclaw.simforge.g1_cerebellar_recovery import (
    G1CerebellarRecoveryConfig,
    G1CerebellarRecoveryController,
    G1CerebellarRecoveryReceipt,
    evaluate_g1_cerebellar_recovery_regime,
)
from rosclaw.simforge.g1_contextual_recovery import G1ContextualRecoveryArtifact
from rosclaw.simforge.g1_muscle_memory import (
    G1_MUSCLE_MEMORY_ACTIONS,
    G1_MUSCLE_MEMORY_OBSERVATIONS,
    G1MuscleMemoryArtifact,
)
from rosclaw.simforge.g1_neural_torque import (
    G1TorqueControlFrame,
    G1TorqueObserver,
    G1TorquePolicy,
    G1TorquePolicyReceipt,
)
from rosclaw.simforge.g1_recovery_state_memory import (
    G1_RECOVERY_STATE_OBSERVATIONS,
    G1RecoveryStateArtifact,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    G1_HARD_TORQUE_LIMITS,
    GOALFORGE_TASK_ID,
    GoalForgeResult,
    GoalForgeStatus,
    ShotParameters,
    SimulationReceiptV4,
    hash_bytes,
    hash_json,
)
from rosclaw.simforge.tasks.g1_goalforge.scenario import GoalForgeScenario

_POLICY_REL = Path("policy/robonaldo/model/policy-obs-aic.onnx")
_MOTION_REL = Path("policy/robonaldo/model/freekick_motion.npz")
_SCENE_REL = Path("g1_description/scene_with_ball.xml")
_MODEL_REL = Path("g1_description/g1_liao.xml")
_FREEKICK_REL = Path("policy/robonaldo/FreeKick.py")
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class G1AssetQualification:
    eligible: bool
    asset_root: Path
    body_hash: str
    kick_prior_hash: str
    motion_hash: str
    backend_commit: str
    actuator_count: int
    joint_names: tuple[str, ...]
    policy_input_size: int
    policy_output_size: int
    errors: tuple[str, ...]
    schema_version: str = "rosclaw.g1_goalforge.asset_qualification.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["asset_root"] = str(self.asset_root)
        value["joint_names"] = list(self.joint_names)
        value["errors"] = list(self.errors)
        return value

    def require_eligible(self) -> None:
        if not self.eligible:
            raise ValueError("G1 assets are not eligible: " + "; ".join(self.errors))


@dataclass(frozen=True)
class GoalForgeEpisode:
    scenario: GoalForgeScenario
    parameters: ShotParameters
    result: GoalForgeResult
    receipt: SimulationReceiptV4 | None
    artifact_root: Path | None
    trajectory: dict[str, np.ndarray]
    feedback_receipt: FeedbackReceipt | None = None
    feedforward_hash: str | None = None
    recovery_receipt: G1CerebellarRecoveryReceipt | None = None
    torque_policy_receipt: G1TorquePolicyReceipt | None = None

    @property
    def result_hash(self) -> str:
        return hash_json(self.result.summary_dict())


def qualify_g1_assets(asset_root: Path) -> G1AssetQualification:
    """Fail-closed qualification of an external RoboNaldo deployment checkout."""

    root = asset_root.expanduser().resolve()
    errors: list[str] = []
    required = (_POLICY_REL, _MOTION_REL, _SCENE_REL, _MODEL_REL, _FREEKICK_REL)
    missing = [str(path) for path in required if not (root / path).is_file()]
    if missing:
        errors.append("missing_assets=" + ",".join(missing))
        return G1AssetQualification(
            eligible=False,
            asset_root=root,
            body_hash="sha256:" + "0" * 64,
            kick_prior_hash="sha256:" + "0" * 64,
            motion_hash="sha256:" + "0" * 64,
            backend_commit="unknown",
            actuator_count=0,
            joint_names=(),
            policy_input_size=0,
            policy_output_size=0,
            errors=tuple(errors),
        )

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(root / _SCENE_REL))
    joint_names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)) for index in range(1, 30)
    )
    if model.nu != 29:
        errors.append(f"actuator_count={model.nu},expected=29")
    if joint_names != G1_DDS_JOINT_NAMES:
        errors.append("joint_order_does_not_match_unitree_hg_dds")
    for body_name in ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) < 0:
            errors.append(f"missing_body={body_name}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball") < 0:
        errors.append("missing_ball_body")

    input_size = 0
    output_size = 0
    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(
            str(root / _POLICY_REL),
            providers=["CPUExecutionProvider"],
        )
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        input_size = int(input_shape[-1])
        output_size = int(output_shape[-1])
        if input_size != 547 or output_size != 29:
            errors.append(f"policy_shape={input_size}->{output_size},expected=547->29")
    except Exception as exc:  # noqa: BLE001 - qualification reports dependency failures
        errors.append(f"onnx_qualification={type(exc).__name__}:{exc}")

    try:
        motion = np.load(root / _MOTION_REL)
        if motion["joint_pos"].ndim != 2 or motion["joint_pos"].shape[1] != 29:
            errors.append("motion_joint_shape_invalid")
        if motion["body_pos_w"].ndim != 3 or motion["body_quat_w"].shape[-1] != 4:
            errors.append("motion_body_shape_invalid")
        if not all(np.all(np.isfinite(motion[name])) for name in motion.files):
            errors.append("motion_contains_non_finite_values")
    except Exception as exc:  # noqa: BLE001 - qualification reports malformed assets
        errors.append(f"motion_qualification={type(exc).__name__}:{exc}")

    model_hash = hash_bytes((root / _MODEL_REL).read_bytes())
    scene_hash = hash_bytes((root / _SCENE_REL).read_bytes())
    policy_hash = hash_bytes((root / _POLICY_REL).read_bytes())
    motion_hash = hash_bytes((root / _MOTION_REL).read_bytes())
    body_hash = hash_json(
        {
            "model_hash": model_hash,
            "scene_hash": scene_hash,
            "joint_names": joint_names,
            "hard_torque_limits": G1_HARD_TORQUE_LIMITS,
        }
    )
    return G1AssetQualification(
        eligible=not errors,
        asset_root=root,
        body_hash=body_hash,
        kick_prior_hash=policy_hash,
        motion_hash=motion_hash,
        backend_commit=_git_commit(root),
        actuator_count=int(model.nu),
        joint_names=joint_names,
        policy_input_size=input_size,
        policy_output_size=output_size,
        errors=tuple(errors),
    )


class G1MuJoCoBackend:
    """Headless official G1 kick execution with bounded adapter changes."""

    def __init__(
        self,
        *,
        asset_root: Path,
        trace_stride: int = 1,
        torque_guard_scale: float = 0.85,
    ) -> None:
        if not 1 <= trace_stride <= 50:
            raise ValueError("trace_stride must be in [1, 50]")
        if not 0.5 <= torque_guard_scale <= 0.95:
            raise ValueError("torque_guard_scale must be in [0.5, 0.95]")
        self.qualification = qualify_g1_assets(asset_root)
        self.qualification.require_eligible()
        self.trace_stride = trace_stride
        self.torque_guard_scale = torque_guard_scale
        self._policy_session: Any | None = None

    def build_cerebellar_recovery_controller(
        self,
        scenario: GoalForgeScenario,
        config: G1CerebellarRecoveryConfig | None = None,
        muscle_memory_artifact: G1MuscleMemoryArtifact | None = None,
        contextual_recovery_artifact: G1ContextualRecoveryArtifact | None = None,
        recovery_state_artifact: G1RecoveryStateArtifact | None = None,
        fallback_config: G1CerebellarRecoveryConfig | None = None,
    ) -> G1CerebellarRecoveryController:
        """Bind the recovery segment to the exact qualified Body and motion."""

        resolved = config or G1CerebellarRecoveryConfig()
        regime_eligible, regime_reasons = evaluate_g1_cerebellar_recovery_regime(
            support_friction=scenario.support_ground_friction,
            control_latency_ms=scenario.control_latency_ms,
            disturbance_n=scenario.disturbance_n,
            config=resolved,
        )
        *_, mujoco_to_isaac = _load_robonaldo(self.qualification.asset_root)
        with np.load(self.qualification.asset_root / _MOTION_REL) as motion:
            standing_pose = np.asarray(
                motion["joint_pos"][0][mujoco_to_isaac],
                dtype=np.float64,
            )
        return G1CerebellarRecoveryController(
            body_hash=self.qualification.body_hash,
            motion_hash=self.qualification.motion_hash,
            regime_commitment=scenario.scenario_commitment,
            regime_eligible=regime_eligible,
            regime_reasons=regime_reasons,
            standing_pose=standing_pose,
            config=resolved,
            muscle_memory_artifact=muscle_memory_artifact,
            contextual_recovery_artifact=contextual_recovery_artifact,
            recovery_state_artifact=recovery_state_artifact,
            fallback_config=fallback_config,
        )

    def run(
        self,
        scenario: GoalForgeScenario,
        parameters: ShotParameters,
        *,
        feedback_runtime: FeedbackRuntime | None = None,
        feedforward: ILCFeedforward | None = None,
        recovery_controller: G1CerebellarRecoveryController | None = None,
        torque_policy: G1TorquePolicy | None = None,
        torque_overlay_policy: G1TorquePolicy | None = None,
        torque_observer: G1TorqueObserver | None = None,
    ) -> GoalForgeEpisode:
        if torque_policy is not None and torque_overlay_policy is not None:
            raise ValueError("direct torque policy and target-space torque overlay are exclusive")
        active_torque_policy = torque_overlay_policy or torque_policy
        if active_torque_policy is not None and torque_observer is not None:
            raise ValueError("direct torque policy and read-only torque observer are exclusive")
        if torque_policy is not None and any(
            value is not None for value in (feedback_runtime, feedforward, recovery_controller)
        ):
            raise ValueError(
                "direct neural torque policy cannot be combined with target-space adapters"
            )
        if torque_overlay_policy is not None and all(
            value is None for value in (feedback_runtime, feedforward, recovery_controller)
        ):
            safety_only = bool(
                getattr(torque_overlay_policy, "safety_projection_only", False) is True
                and getattr(torque_overlay_policy, "activation_ceiling", None) == "SIM_ONLY"
            )
            if not safety_only:
                raise ValueError("target-space torque overlay requires a target-space adapter")
        if (
            feedback_runtime is not None
            and feedback_runtime.spec.body_hash != self.qualification.body_hash
        ):
            raise ValueError("FeedbackLoopSpec body hash does not match qualified G1 assets")
        if feedforward is not None:
            feedforward.require_compatible(
                body_hash=self.qualification.body_hash,
                regime_hash=scenario.scenario_commitment,
                joint_names=G1_DDS_JOINT_NAMES,
            )
        if recovery_controller is not None:
            recovery_controller.require_compatible(
                body_hash=self.qualification.body_hash,
                motion_hash=self.qualification.motion_hash,
                regime_commitment=scenario.scenario_commitment,
            )
        if parameters.kick_foot != "right":
            return GoalForgeEpisode(
                scenario=scenario,
                parameters=parameters,
                result=_incompatible_result(),
                receipt=None,
                artifact_root=None,
                trajectory={},
            )
        if not scenario.reachable:
            return GoalForgeEpisode(
                scenario=scenario,
                parameters=parameters,
                result=_unreachable_result(),
                receipt=None,
                artifact_root=None,
                trajectory={},
            )
        return self._run_physics(
            scenario,
            parameters,
            feedback_runtime=feedback_runtime,
            feedforward=feedforward,
            recovery_controller=recovery_controller,
            torque_policy=active_torque_policy,
            torque_observer=torque_observer,
        )

    def run_and_record(
        self,
        *,
        scenario: GoalForgeScenario,
        parameters: ShotParameters,
        output_root: Path,
        source_checkout: Path,
        practice_id: str,
        strict_replay: bool = True,
    ) -> GoalForgeEpisode:
        root = _external_root(output_root, source_checkout)
        episode_id = _episode_id(scenario, parameters)
        episode_root = root / episode_id
        episode_root.mkdir(parents=True, exist_ok=False)
        request = {
            "schema_version": "rosclaw.g1_goalforge.trajectory_request.v1",
            "episode_id": episode_id,
            "practice_id": practice_id,
            "task_id": GOALFORGE_TASK_ID,
            "body_hash": self.qualification.body_hash,
            "kick_prior_hash": self.qualification.kick_prior_hash,
            "scenario": scenario.to_private_dict(),
            "parameters": parameters.to_dict(),
            "policy_hash": parameters.policy_hash,
            "safety": {
                "hard_torque_limits": list(G1_HARD_TORQUE_LIMITS),
                "torque_guard_scale": self.torque_guard_scale,
                "immutable": [
                    "hard_torque_limits",
                    "joint_limits",
                    "permit",
                    "lease",
                    "evidence_semantics",
                ],
            },
        }
        request_path = episode_root / "trajectory-request.json"
        _atomic_json(request_path, request)
        episode = self.run(scenario, parameters)
        trajectory_path = episode_root / "trajectory.npz"
        np.savez_compressed(trajectory_path, **episode.trajectory)  # type: ignore[arg-type]
        result_path = episode_root / "result.json"
        _atomic_json(result_path, episode.result.summary_dict())
        request_hash = hash_bytes(request_path.read_bytes())
        trajectory_hash = _bounded_hash(trajectory_path)
        result_hash = hash_bytes(result_path.read_bytes())
        replay_ok = False
        if strict_replay and episode.result.physics_executed:
            replay = self.run(scenario, parameters)
            replay_ok = (
                replay.result.summary_dict() == episode.result.summary_dict()
                and trajectory_digest(replay.trajectory) == trajectory_digest(episode.trajectory)
            )
        receipt = SimulationReceiptV4(
            episode_id=episode_id,
            body_hash=self.qualification.body_hash,
            policy_hash=parameters.policy_hash,
            kick_prior_hash=self.qualification.kick_prior_hash,
            scenario_commitment=scenario.scenario_commitment,
            seed_commitment=scenario.seed_commitment,
            request_hash=request_hash,
            trajectory_hash=trajectory_hash,
            result_hash=result_hash,
            backend="unitree_g1_mujoco_robonaldo",
            backend_commit=self.qualification.backend_commit,
            physics_steps=episode.result.physics_steps,
            independently_verified=_independent_trace_check(
                episode.trajectory,
                episode.result,
            ),
            strict_replay=replay_ok,
        )
        receipt_path = episode_root / "simulation-receipt.json"
        _atomic_json(receipt_path, receipt.to_dict())
        return GoalForgeEpisode(
            scenario=scenario,
            parameters=parameters,
            result=episode.result,
            receipt=receipt,
            artifact_root=episode_root,
            trajectory=episode.trajectory,
            feedback_receipt=episode.feedback_receipt,
            feedforward_hash=episode.feedforward_hash,
        )

    def _run_physics(
        self,
        scenario: GoalForgeScenario,
        parameters: ShotParameters,
        *,
        feedback_runtime: FeedbackRuntime | None = None,
        feedforward: ILCFeedforward | None = None,
        recovery_controller: G1CerebellarRecoveryController | None = None,
        torque_policy: G1TorquePolicy | None = None,
        torque_observer: G1TorqueObserver | None = None,
    ) -> GoalForgeEpisode:
        import mujoco

        root = self.qualification.asset_root
        state_type, output_type, policy_type, mujoco_to_isaac = _load_robonaldo(root)
        model = mujoco.MjModel.from_xml_path(str(root / _SCENE_REL))
        data = mujoco.MjData(model)
        model.opt.timestep = 0.002
        if feedback_runtime is not None:
            feedback_runtime.reset()
        if recovery_controller is not None:
            recovery_controller.reset()
        if torque_policy is not None:
            torque_policy.reset()
        if torque_observer is not None:
            torque_observer.reset()
        _configure_scene(model, scenario)
        state = state_type(29)
        output = output_type(29)
        policy_module = importlib.import_module(policy_type.__module__)
        session_factory = policy_module.onnxruntime.InferenceSession
        if self._policy_session is not None:
            policy_module.onnxruntime.InferenceSession = lambda *_args, **_kwargs: (
                self._policy_session
            )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                policy = policy_type(state, output)
        finally:
            policy_module.onnxruntime.InferenceSession = session_factory
        if self._policy_session is None:
            self._policy_session = policy.ort_session
        policy.target_pos_w = np.array(
            [5.0, scenario.target_y_m, scenario.target_z_m],
            dtype=np.float32,
        )
        motion = np.load(root / _MOTION_REL)
        data.qpos[:3] = motion["body_pos_w"][0, 0]
        data.qpos[0] += parameters.stance_offset_x
        data.qpos[1] += parameters.stance_offset_y
        data.qpos[3:7] = motion["body_quat_w"][0, 0]
        if parameters.pelvis_yaw_offset:
            half_yaw = parameters.pelvis_yaw_offset * 0.5
            yaw_quaternion = np.asarray(
                [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
                dtype=np.float64,
            )
            data.qpos[3:7] = _quaternion_multiply(
                yaw_quaternion,
                np.asarray(data.qpos[3:7], dtype=np.float64),
            )
        data.qpos[7:36] = motion["joint_pos"][0][mujoco_to_isaac]
        if scenario.joint_zero_bias_rad:
            data.qpos[7:19] += scenario.joint_zero_bias_rad
        _reset_ball(model, data, scenario)
        mujoco.mj_forward(model, data)

        ids = _ModelIds.from_model(model)
        _fill_state(state, model, data, ids)
        with contextlib.redirect_stdout(io.StringIO()):
            policy.enter()
        target_queue: deque[np.ndarray] = deque()
        latency_frames = int(round(scenario.control_latency_ms / 20.0))
        delay_frames = int(round(parameters.kick_trigger_delay / 0.02))
        phase_frames = int(round(parameters.contact_phase_offset / 0.02))
        phase_hold_remaining = max(0, phase_frames)
        phase_advance = max(0, -phase_frames)
        phase_adjusted = phase_frames == 0
        hard_limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
        guarded_limits = hard_limits * self.torque_guard_scale
        trace: dict[str, list[Any]] = {
            "time": [],
            "joint_position": [],
            "joint_velocity": [],
            "joint_torque": [],
            "policy_action": [],
            "pelvis_pose": [],
            "torso_quaternion": [],
            "com": [],
            "left_foot_contact": [],
            "right_foot_contact": [],
            "ground_reaction_force": [],
            "support_foot_slip": [],
            "policy_phase": [],
            "com_y_relative": [],
            "ball_lateral_error_m": [],
            "ball_pose": [],
            "ball_velocity": [],
            "ball_angular_velocity": [],
            "foot_ball_contact_point": [],
            "contact_impulse": [],
            "goal_crossing": [],
        }
        if feedback_runtime is not None:
            trace.update(
                {
                    "feedback_residual": [],
                    "feedback_error_rms": [],
                    "feedback_active": [],
                }
            )
            if "skill:kick_phase_rate" in feedback_runtime.spec.output_limits:
                trace["feedback_phase_rate"] = []
        if feedforward is not None:
            trace.update(
                {
                    "feedforward_residual": [],
                    "combined_residual": [],
                    "combined_residual_saturation": [],
                }
            )
        if recovery_controller is not None:
            trace.update(
                {
                    "recovery_active": [],
                    "recovery_blend_fraction": [],
                    "recovery_settling_fraction": [],
                    "recovery_smoothing_active": [],
                    "recovery_smoothing_residual_rms_rad": [],
                    "recovery_proprioception": [],
                    "recovery_policy_frame": [],
                    "recovery_observation_updated": [],
                    "recovery_state_observation": [],
                }
            )
            if recovery_controller.muscle_memory is not None:
                trace.update(
                    {
                        "muscle_memory_active": [],
                        "muscle_memory_out_of_distribution": [],
                        "muscle_memory_residual_rms_rad": [],
                        "muscle_memory_synergy_actions": [],
                    }
                )
        kick_support_anchor: np.ndarray | None = None
        peak_support_slip = 0.0
        com_margin_min = math.inf
        roll_peak = 0.0
        pitch_peak = 0.0
        peak_torque_scale = 0.0
        actuator_saturation = False
        torque_violation = False
        joint_violation = False
        finite = True
        contact_observed = False
        kick_foot_contacted = False
        wrong_foot_contacted = False
        contact_time: float | None = None
        contact_impulse = 0.0
        goal_crossed = False
        crossing_y = math.nan
        crossing_z = math.nan
        crossing_speed = 0.0
        maximum_ball_speed = 0.0
        previous_ball_x = float(data.qpos[ids.ball_qpos])
        stable_after_contact = 0.0
        slowdown_frames = int(round(245 * max(0.0, 1.0 - parameters.swing_speed_scale)))
        total_control_frames = (
            int(motion["joint_pos"].shape[0])
            + 40
            + delay_frames
            + max(0, phase_frames)
            + slowdown_frames
        )
        if feedforward is not None and feedforward.values.shape[0] != total_control_frames:
            raise ValueError(
                "ILC feedforward frame count does not match the pinned GoalForge trajectory"
            )
        last_target = data.qpos[7:36].copy()
        feedback_residual: np.ndarray = np.zeros(29, dtype=np.float64)
        feedforward_residual: np.ndarray = np.zeros(29, dtype=np.float64)
        combined_residual: np.ndarray = np.zeros(29, dtype=np.float64)
        combined_residual_saturation = 0
        combined_limits = np.asarray(
            g1_joint_residual_limits(G1_DDS_JOINT_NAMES),
            dtype=np.float64,
        )
        feedback_error_rms = math.nan
        feedback_phase_rate = 0.0
        policy_phase = 0.0
        phase_repeat_accumulator = 0.0
        next_feedback_time = 0.0
        latest_support_slip = 0.0
        latest_left_support = False
        latest_right_support = False
        latest_left_ground_force = 0.0
        latest_right_ground_force = 0.0
        moving_ball_launched = scenario.ball_launch_delay_sec <= 0.0
        recovery_active = False
        recovery_blend_fraction = 0.0
        recovery_settling_fraction = 0.0
        recovery_smoothing_active = False
        recovery_smoothing_residual_rms_rad = 0.0
        recovery_proprioception: np.ndarray = np.zeros(
            len(G1_MUSCLE_MEMORY_OBSERVATIONS),
            dtype=np.float64,
        )
        recovery_state_observation: np.ndarray = np.zeros(
            len(G1_RECOVERY_STATE_OBSERVATIONS),
            dtype=np.float64,
        )
        recovery_policy_frame = 0
        recovery_observation_updated = False
        muscle_memory_active = False
        muscle_memory_out_of_distribution = False
        muscle_memory_residual_rms_rad = 0.0
        muscle_memory_synergy_actions: np.ndarray = np.zeros(
            len(G1_MUSCLE_MEMORY_ACTIONS),
            dtype=np.float64,
        )
        joint_limited = model.jnt_limited[1:30].astype(bool)
        joint_ranges = np.asarray(model.jnt_range[1:30], dtype=np.float64)
        joint_lower_limits = np.where(joint_limited, joint_ranges[:, 0], -1.0e6)
        joint_upper_limits = np.where(joint_limited, joint_ranges[:, 1], 1.0e6)

        for frame in range(total_control_frames):
            recovery_observation_updated = False
            if feedforward is not None:
                feedforward_residual = feedforward.value_at(frame)
            _fill_state(state, model, data, ids)
            policy_frame = 0
            if frame < delay_frames:
                target = last_target.copy()
                kp = np.asarray(policy.kps, dtype=np.float64)
                kd = np.asarray(policy.kds, dtype=np.float64)
            else:
                current_policy_frame = max(
                    0,
                    int(policy.time_step) - int(policy.WARMUP_STEPS),
                )
                repeat = _policy_repeat_count(
                    parameters.swing_speed_scale,
                    current_policy_frame,
                    frame,
                )
                if not phase_adjusted and current_policy_frame >= 185:
                    if phase_hold_remaining:
                        repeat = 0
                        phase_hold_remaining -= 1
                        phase_adjusted = phase_hold_remaining == 0
                    else:
                        repeat += phase_advance
                        phase_adjusted = True
                repeat, phase_repeat_accumulator = _apply_feedback_phase_rate(
                    repeat=repeat,
                    phase_rate=feedback_phase_rate,
                    accumulator=phase_repeat_accumulator,
                )
                if repeat:
                    with contextlib.redirect_stdout(io.StringIO()):
                        for _ in range(repeat):
                            policy.run()
                    target = np.asarray(output.actions, dtype=np.float64).copy()
                    kp = np.asarray(output.kps, dtype=np.float64)
                    kd = np.asarray(output.kds, dtype=np.float64)
                    policy_frame = max(
                        0,
                        int(policy.time_step) - int(policy.WARMUP_STEPS),
                    )
                    target = _adapt_target(
                        target=target,
                        default=np.asarray(policy.default_q_mj, dtype=np.float64),
                        parameters=parameters,
                        policy_frame=policy_frame,
                    )
                    if recovery_controller is not None:
                        current_phase = min(
                            1.0,
                            policy_frame / max(1, int(motion["joint_pos"].shape[0]) - 1),
                        )
                        proprioceptive_observation = _muscle_memory_observation(
                            data=data,
                            ids=ids,
                            policy_phase=current_phase,
                            contact_time=contact_time,
                            contact_impulse=contact_impulse,
                            left_support=latest_left_support,
                            right_support=latest_right_support,
                            left_ground_force=latest_left_ground_force,
                            right_ground_force=latest_right_ground_force,
                        )
                        recovery_proprioception = np.asarray(
                            [
                                proprioceptive_observation[name]
                                for name in G1_MUSCLE_MEMORY_OBSERVATIONS
                            ],
                            dtype=np.float64,
                        )
                        state_observation = {
                            **proprioceptive_observation,
                            "ball_relative_position_x_m": float(
                                data.xpos[ids.ball][0] - data.qpos[0]
                            ),
                            "ball_relative_position_y_m": float(
                                data.xpos[ids.ball][1] - data.qpos[1]
                            ),
                            "ball_relative_position_z_m": float(
                                data.xpos[ids.ball][2] - data.qpos[2]
                            ),
                            "ball_relative_velocity_x_m_s": float(
                                data.qvel[ids.ball_qvel] - data.qvel[0]
                            ),
                            "ball_relative_velocity_y_m_s": float(
                                data.qvel[ids.ball_qvel + 1] - data.qvel[1]
                            ),
                            "ball_relative_velocity_z_m_s": float(
                                data.qvel[ids.ball_qvel + 2] - data.qvel[2]
                            ),
                        }
                        recovery_state_observation = np.asarray(
                            [state_observation[name] for name in G1_RECOVERY_STATE_OBSERVATIONS],
                            dtype=np.float64,
                        )
                        recovery_policy_frame = policy_frame
                        recovery_observation_updated = True
                        recovery_effect = recovery_controller.adapt_target(
                            target=target,
                            policy_frame=policy_frame,
                            timestamp_sec=float(data.time),
                            ball_contact_detected=contact_observed,
                            left_support=latest_left_support,
                            right_support=latest_right_support,
                            muscle_memory_observation=(
                                state_observation
                                if recovery_controller.recovery_state is not None
                                else (
                                    proprioceptive_observation
                                    if (
                                        recovery_controller.muscle_memory is not None
                                        or recovery_controller.contextual_recovery is not None
                                    )
                                    else None
                                )
                            ),
                        )
                        target = recovery_effect.target
                        terminal_group = recovery_effect.terminal_damping_joint_group
                        terminal_slice = {
                            "whole_body": slice(None),
                            "legs": slice(0, 12),
                            "upper_body": slice(12, None),
                        }[terminal_group]
                        kp = kp.copy()
                        kd = kd.copy()
                        kp[terminal_slice] *= recovery_effect.terminal_kp_scale
                        kd[terminal_slice] *= recovery_effect.terminal_kd_scale
                        recovery_active = recovery_effect.active
                        recovery_blend_fraction = recovery_effect.blend_fraction
                        recovery_settling_fraction = recovery_effect.settling_fraction
                        recovery_smoothing_active = recovery_effect.smoothing_active
                        recovery_smoothing_residual_rms_rad = (
                            recovery_effect.smoothing_residual_rms_rad
                        )
                        muscle_memory_active = recovery_effect.muscle_memory_active
                        muscle_memory_out_of_distribution = (
                            recovery_effect.muscle_memory_out_of_distribution
                        )
                        muscle_memory_residual_rms_rad = (
                            recovery_effect.muscle_memory_residual_rms_rad
                        )
                        if recovery_effect.muscle_memory_synergy_actions.size:
                            muscle_memory_synergy_actions = (
                                recovery_effect.muscle_memory_synergy_actions
                            )
                else:
                    target = last_target.copy()
                    kp = np.asarray(policy.kps, dtype=np.float64)
                    kd = np.asarray(policy.kds, dtype=np.float64)
                    policy_frame = current_policy_frame
                policy_phase = min(
                    1.0,
                    policy_frame / max(1, int(motion["joint_pos"].shape[0]) - 1),
                )
            target_queue.append(target)
            delayed_target = (
                target_queue.popleft() if len(target_queue) > latency_frames else last_target.copy()
            )
            last_target = delayed_target
            frame_torque: np.ndarray = np.zeros(29, dtype=np.float64)
            left_contact = False
            right_contact = False
            left_ground_force = 0.0
            right_ground_force = 0.0
            ball_contact_point: np.ndarray = np.zeros(3, dtype=np.float64)
            for _ in range(10):
                if not moving_ball_launched and data.time + 1e-12 >= scenario.ball_launch_delay_sec:
                    data.qvel[ids.ball_qvel : ids.ball_qvel + 3] = (
                        scenario.ball_velocity_x_mps,
                        scenario.ball_velocity_y_mps,
                        0.0,
                    )
                    moving_ball_launched = True
                if feedback_runtime is not None and data.time + 1e-12 >= next_feedback_time:
                    feedback_phase = (
                        policy_phase
                        if "contact_phase" in feedback_runtime.spec.observation_signals
                        else min(1.0, frame / max(1, total_control_frames - 1))
                    )
                    feedback_effect = _feedback_effect(
                        runtime=feedback_runtime,
                        timestamp_sec=float(data.time),
                        phase=feedback_phase,
                        data=data,
                        ids=ids,
                        base_target=delayed_target,
                        support_slip=latest_support_slip,
                        contact_detected=contact_observed,
                        control_latency_ms=scenario.control_latency_ms,
                    )
                    feedback_residual = feedback_effect.joint_residual
                    feedback_phase_rate = feedback_effect.kick_phase_rate
                    feedback_error_rms = feedback_runtime.records[-1].error_rms
                    next_feedback_time += 1.0 / feedback_runtime.spec.rate_hz
                raw_combined_residual = feedback_residual + feedforward_residual
                combined_residual = np.clip(
                    raw_combined_residual,
                    -combined_limits,
                    combined_limits,
                )
                combined_residual_saturation += int(
                    np.count_nonzero(np.abs(raw_combined_residual - combined_residual) > 1e-12)
                )
                controlled_target = delayed_target + combined_residual
                raw_torque = (controlled_target - data.qpos[7:36]) * kp - data.qvel[6:35] * kd
                raw_scale = float(np.max(np.abs(raw_torque) / hard_limits))
                peak_torque_scale = max(peak_torque_scale, min(raw_scale, self.torque_guard_scale))
                frame_torque = np.clip(raw_torque, -guarded_limits, guarded_limits)
                control_frame = G1TorqueControlFrame(
                    joint_position=np.asarray(data.qpos[7:36], dtype=np.float64),
                    joint_velocity=np.asarray(data.qvel[6:35], dtype=np.float64),
                    joint_lower_limits=joint_lower_limits,
                    joint_upper_limits=joint_upper_limits,
                    torso_quaternion_wxyz=np.asarray(data.xquat[ids.torso], dtype=np.float64),
                    pelvis_position=np.asarray(data.qpos[:3], dtype=np.float64),
                    base_linear_velocity=np.asarray(data.qvel[:3], dtype=np.float64),
                    base_angular_velocity=np.asarray(data.qvel[3:6], dtype=np.float64),
                    ball_position=np.asarray(
                        data.qpos[ids.ball_qpos : ids.ball_qpos + 3],
                        dtype=np.float64,
                    ),
                    ball_velocity=np.asarray(
                        data.qvel[ids.ball_qvel : ids.ball_qvel + 3],
                        dtype=np.float64,
                    ),
                    target_y_m=scenario.target_y_m,
                    target_z_m=scenario.target_z_m,
                    policy_phase=policy_phase,
                    left_contact=latest_left_support,
                    right_contact=latest_right_support,
                    ball_contact_observed=contact_observed,
                )
                if torque_policy is not None:
                    parent_torque = frame_torque.copy()
                    frame_torque = _apply_direct_torque_policy(
                        policy=torque_policy,
                        frame=control_frame,
                        parent_torque=parent_torque,
                        guarded_limits=guarded_limits,
                    )
                    raw_scale = float(np.max(np.abs(frame_torque) / hard_limits))
                    peak_torque_scale = max(peak_torque_scale, raw_scale)
                if torque_observer is not None:
                    torque_observer.observe(control_frame, frame_torque.copy())
                torque_violation = torque_violation or bool(
                    np.any(np.abs(frame_torque) > hard_limits)
                )
                actuator_saturation = actuator_saturation or bool(
                    np.any(np.abs(frame_torque) >= hard_limits * 0.999)
                )
                data.ctrl[:] = frame_torque
                if scenario.disturbance_n and 4.6 <= data.time <= 4.8:
                    data.xfrc_applied[ids.pelvis, 1] = scenario.disturbance_n
                else:
                    data.xfrc_applied[ids.pelvis] = 0.0
                mujoco.mj_step(model, data)
                step_contacts = _contact_observation(model, data, ids)
                left_contact = left_contact or step_contacts.left_floor
                right_contact = right_contact or step_contacts.right_floor
                left_ground_force = max(
                    left_ground_force,
                    step_contacts.left_ground_force_n,
                )
                right_ground_force = max(
                    right_ground_force,
                    step_contacts.right_ground_force_n,
                )
                if step_contacts.ball_right:
                    contact_observed = True
                    kick_foot_contacted = True
                    if contact_time is None:
                        contact_time = float(data.time)
                    contact_impulse += step_contacts.ball_force_n * model.opt.timestep
                    ball_contact_point = np.asarray(
                        step_contacts.ball_contact_point,
                        dtype=np.float64,
                    )
                if step_contacts.ball_left:
                    contact_observed = True
                    wrong_foot_contacted = True
            support_slip = 0.0
            single_support = left_contact and not right_contact
            if 210 <= policy_frame <= 335 and contact_time is None and single_support:
                if kick_support_anchor is None:
                    kick_support_anchor = data.xpos[ids.left_ankle].copy()
                support_slip = float(
                    np.linalg.norm((data.xpos[ids.left_ankle] - kick_support_anchor)[:2])
                )
            elif contact_time is None:
                kick_support_anchor = None
            peak_support_slip = max(peak_support_slip, support_slip)
            latest_support_slip = support_slip
            latest_left_support = left_contact
            latest_right_support = right_contact
            latest_left_ground_force = left_ground_force
            latest_right_ground_force = right_ground_force
            com = data.subtree_com[ids.pelvis].copy()
            support_y = float(data.xpos[ids.left_ankle][1])
            com_margin = 0.11 - abs(float(com[1]) - support_y)
            if 210 <= policy_frame <= 335 and single_support:
                com_margin_min = min(com_margin_min, com_margin)
            roll, pitch = _roll_pitch(data.xquat[ids.torso])
            roll_peak = max(roll_peak, abs(roll))
            pitch_peak = max(pitch_peak, abs(pitch))
            ball_position = data.qpos[ids.ball_qpos : ids.ball_qpos + 3].copy()
            ball_velocity = data.qvel[ids.ball_qvel : ids.ball_qvel + 3].copy()
            ball_speed = float(np.linalg.norm(ball_velocity))
            maximum_ball_speed = max(maximum_ball_speed, ball_speed)
            if not goal_crossed and previous_ball_x < 5.0 <= float(ball_position[0]):
                crossing_y = float(ball_position[1])
                crossing_z = float(ball_position[2])
                crossing_speed = ball_speed
                goal_crossed = True
            previous_ball_x = float(ball_position[0])
            stable_now = float(data.qpos[2]) >= 0.62 and abs(roll) <= 0.30 and abs(pitch) <= 0.35
            if contact_time is not None:
                stable_after_contact = stable_after_contact + 0.02 if stable_now else 0.0
            limited = model.jnt_limited[1:30].astype(bool)
            ranges = model.jnt_range[1:30]
            qj = data.qpos[7:36]
            joint_violation = joint_violation or bool(
                np.any(qj[limited] < ranges[limited, 0] - 1e-5)
                or np.any(qj[limited] > ranges[limited, 1] + 1e-5)
            )
            finite = finite and all(
                np.all(np.isfinite(array))
                for array in (data.qpos, data.qvel, data.ctrl, ball_position, com)
            )
            if frame % self.trace_stride == 0:
                _append_trace(
                    trace,
                    time_sec=float(data.time),
                    data=data,
                    ids=ids,
                    torque=frame_torque,
                    policy_action=delayed_target,
                    com=com,
                    left_contact=left_contact,
                    right_contact=right_contact,
                    ground_reaction_force=(left_ground_force, right_ground_force),
                    support_slip=support_slip,
                    ball_contact_point=ball_contact_point,
                    contact_impulse=contact_impulse,
                    goal_crossed=goal_crossed,
                    feedback_residual=feedback_residual,
                    feedback_error_rms=feedback_error_rms,
                    feedback_phase_rate=feedback_phase_rate,
                    policy_phase=policy_phase,
                    feedforward_residual=feedforward_residual,
                    combined_residual=combined_residual,
                    combined_residual_saturation=combined_residual_saturation,
                    recovery_active=recovery_active,
                    recovery_blend_fraction=recovery_blend_fraction,
                    recovery_settling_fraction=recovery_settling_fraction,
                    recovery_smoothing_active=recovery_smoothing_active,
                    recovery_smoothing_residual_rms_rad=(recovery_smoothing_residual_rms_rad),
                    recovery_proprioception=recovery_proprioception,
                    recovery_policy_frame=recovery_policy_frame,
                    recovery_observation_updated=recovery_observation_updated,
                    recovery_state_observation=recovery_state_observation,
                    muscle_memory_active=muscle_memory_active,
                    muscle_memory_out_of_distribution=muscle_memory_out_of_distribution,
                    muscle_memory_residual_rms_rad=muscle_memory_residual_rms_rad,
                    muscle_memory_synergy_actions=muscle_memory_synergy_actions,
                )
            if not finite:
                break

        if not trace["time"] or not math.isclose(
            float(trace["time"][-1]),
            float(data.time),
            abs_tol=model.opt.timestep * 0.5,
        ):
            _append_trace(
                trace,
                time_sec=float(data.time),
                data=data,
                ids=ids,
                torque=frame_torque,
                policy_action=last_target,
                com=data.subtree_com[ids.pelvis].copy(),
                left_contact=left_contact,
                right_contact=right_contact,
                ground_reaction_force=(left_ground_force, right_ground_force),
                support_slip=support_slip,
                ball_contact_point=ball_contact_point,
                contact_impulse=contact_impulse,
                goal_crossed=goal_crossed,
                feedback_residual=feedback_residual,
                feedback_error_rms=feedback_error_rms,
                feedback_phase_rate=feedback_phase_rate,
                policy_phase=policy_phase,
                feedforward_residual=feedforward_residual,
                combined_residual=combined_residual,
                combined_residual_saturation=combined_residual_saturation,
                recovery_active=recovery_active,
                recovery_blend_fraction=recovery_blend_fraction,
                recovery_settling_fraction=recovery_settling_fraction,
                recovery_smoothing_active=recovery_smoothing_active,
                recovery_smoothing_residual_rms_rad=recovery_smoothing_residual_rms_rad,
                recovery_proprioception=recovery_proprioception,
                recovery_policy_frame=recovery_policy_frame,
                recovery_observation_updated=recovery_observation_updated,
                recovery_state_observation=recovery_state_observation,
                muscle_memory_active=muscle_memory_active,
                muscle_memory_out_of_distribution=muscle_memory_out_of_distribution,
                muscle_memory_residual_rms_rad=muscle_memory_residual_rms_rad,
                muscle_memory_synergy_actions=muscle_memory_synergy_actions,
            )
        target_error = (
            math.hypot(crossing_y - scenario.target_y_m, crossing_z - scenario.target_z_m)
            if goal_crossed
            else math.inf
        )
        target_hit = bool(goal_crossed and target_error <= 0.48)
        fall = bool(data.qpos[2] < 0.55 or roll_peak > 0.55 or pitch_peak > 0.65)
        status = _classify(
            finite=finite,
            kick_foot_contacted=kick_foot_contacted,
            wrong_foot_contacted=wrong_foot_contacted,
            goal_crossed=goal_crossed,
            target_hit=target_hit,
            target_error=target_error,
            crossing_y=crossing_y,
            target_y=scenario.target_y_m,
            maximum_ball_speed=maximum_ball_speed,
            peak_support_slip=peak_support_slip,
            com_margin_min=com_margin_min,
            roll_peak=roll_peak,
            pitch_peak=pitch_peak,
            fall=fall,
            joint_violation=joint_violation,
            torque_violation=torque_violation,
            actuator_saturation=actuator_saturation,
        )
        success = status is GoalForgeStatus.SUCCESS
        robustness = min(
            (0.48 - target_error) if math.isfinite(target_error) else -1.0,
            0.08 - peak_support_slip,
            com_margin_min,
            0.55 - roll_peak,
            0.65 - pitch_peak,
            1.0 - peak_torque_scale,
        )
        result = GoalForgeResult(
            status=status,
            success=success,
            physics_executed=True,
            contact_observed=contact_observed,
            kick_foot_contacted=kick_foot_contacted,
            goal_crossed=goal_crossed,
            target_zone_hit=target_hit,
            target_error_m=target_error,
            ball_speed_mps=max(crossing_speed, maximum_ball_speed),
            ball_contact_time_sec=contact_time,
            contact_impulse_ns=contact_impulse,
            support_foot_slip_m=peak_support_slip,
            com_margin_min_m=com_margin_min,
            torso_roll_peak_rad=roll_peak,
            torso_pitch_peak_rad=pitch_peak,
            peak_torque_scale=peak_torque_scale,
            joint_limit_violation=joint_violation,
            torque_limit_violation=torque_violation,
            actuator_saturation=actuator_saturation,
            post_kick_fall=fall,
            post_kick_stability_time_sec=stable_after_contact,
            final_pelvis_height_m=float(data.qpos[2]),
            physics_steps=int(round(float(data.time) / model.opt.timestep)),
            finite_state=finite,
            robustness=robustness,
        )
        arrays = {name: np.asarray(values) for name, values in trace.items()}
        feedback_receipt = (
            feedback_runtime.build_receipt(
                action_id=_episode_id(scenario, parameters) + ":feedback",
                strict_replay=False,
                evidence_domain="SIM",
            )
            if feedback_runtime is not None
            else None
        )
        recovery_receipt = (
            recovery_controller.build_receipt(strict_replay=False, evidence_domain="SIM")
            if recovery_controller is not None
            else None
        )
        receipt_builder = getattr(torque_policy, "build_receipt", None)
        torque_policy_receipt = receipt_builder() if callable(receipt_builder) else None
        if torque_policy_receipt is not None and not isinstance(
            torque_policy_receipt, G1TorquePolicyReceipt
        ):
            raise ValueError("direct torque policy returned an invalid receipt")
        return GoalForgeEpisode(
            scenario=scenario,
            parameters=parameters,
            result=result,
            receipt=None,
            artifact_root=None,
            trajectory=arrays,
            feedback_receipt=feedback_receipt,
            feedforward_hash=feedforward.trajectory_hash if feedforward is not None else None,
            recovery_receipt=recovery_receipt,
            torque_policy_receipt=torque_policy_receipt,
        )


@dataclass(frozen=True)
class _ModelIds:
    pelvis: int
    torso: int
    left_ankle: int
    right_ankle: int
    ball: int
    ball_geom: int
    ball_qpos: int
    ball_qvel: int

    @classmethod
    def from_model(cls, model: Any) -> _ModelIds:
        import mujoco

        body = lambda name: int(  # noqa: E731 - compact checked lookup
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        )
        ball = body("ball")
        joint = int(model.body_jntadr[ball])
        return cls(
            pelvis=body("pelvis"),
            torso=body("torso_link"),
            left_ankle=body("left_ankle_roll_link"),
            right_ankle=body("right_ankle_roll_link"),
            ball=ball,
            ball_geom=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")),
            ball_qpos=int(model.jnt_qposadr[joint]),
            ball_qvel=int(model.jnt_dofadr[joint]),
        )


@dataclass(frozen=True)
class _Contacts:
    left_floor: bool
    right_floor: bool
    ball_any: bool
    ball_left: bool
    ball_right: bool
    ball_force_n: float
    ball_contact_point: tuple[float, float, float]
    left_ground_force_n: float
    right_ground_force_n: float


@dataclass(frozen=True)
class _FeedbackEffect:
    joint_residual: np.ndarray
    kick_phase_rate: float


def _apply_direct_torque_policy(
    *,
    policy: G1TorquePolicy,
    frame: G1TorqueControlFrame,
    parent_torque: np.ndarray,
    guarded_limits: np.ndarray,
) -> np.ndarray:
    """Apply a SIM-only torque callback behind the backend's final hard guard."""

    try:
        proposed = np.asarray(policy.command(frame, parent_torque), dtype=np.float64)
        if proposed.shape != (29,) or not np.all(np.isfinite(proposed)):
            raise ValueError("direct torque callback returned an invalid 29-vector")
    except (ValueError, FloatingPointError):
        proposed = parent_torque
    applied = np.clip(proposed, -guarded_limits, guarded_limits)
    policy.note_applied(applied)
    return applied


def _load_robonaldo(root: Path) -> tuple[Any, Any, Any, np.ndarray]:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    ctrl = importlib.import_module("common.ctrlcomp")
    module = importlib.import_module("policy.robonaldo.FreeKick")
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError("RoboNaldo module does not expose an import path")
    loaded = Path(module_file).resolve()
    if root not in loaded.parents:
        raise RuntimeError(f"RoboNaldo module resolved outside qualified root: {loaded}")
    return (
        ctrl.StateAndCmd,
        ctrl.PolicyOutput,
        module.FreeKick,
        np.asarray(module.MUJOCO_TO_ISAAC, dtype=np.int64),
    )


def _configure_scene(model: Any, scenario: GoalForgeScenario) -> None:
    import mujoco

    ball = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball"))
    ball_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"))
    original_mass = float(model.body_mass[ball])
    ratio = scenario.ball_mass_kg / max(original_mass, 1e-9)
    model.body_mass[ball] = scenario.ball_mass_kg
    model.body_inertia[ball] *= ratio
    model.geom_friction[ball_geom, 0] = scenario.ball_ground_friction
    floor = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    model.geom_friction[floor, 0] = scenario.support_ground_friction
    for pair_index in range(int(model.npair)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, pair_index) or ""
        if name == "ball_floor":
            model.pair_friction[pair_index, 0] = scenario.ball_ground_friction
            model.pair_solref[pair_index, 1] = max(0.05, 1.0 - scenario.restitution)
        elif name.endswith("_floor"):
            model.pair_friction[pair_index, 0] = scenario.support_ground_friction


def _reset_ball(model: Any, data: Any, scenario: GoalForgeScenario) -> None:
    import mujoco

    ball = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball"))
    joint = int(model.body_jntadr[ball])
    qpos = int(model.jnt_qposadr[joint])
    qvel = int(model.jnt_dofadr[joint])
    data.qpos[qpos : qpos + 3] = (scenario.ball_x_m, scenario.ball_y_m, 0.115)
    data.qpos[qpos + 3 : qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[qvel : qvel + 3] = (
        (
            scenario.ball_velocity_x_mps,
            scenario.ball_velocity_y_mps,
            0.0,
        )
        if scenario.ball_launch_delay_sec <= 0.0
        else (0.0, 0.0, 0.0)
    )
    data.qvel[qvel + 3 : qvel + 6] = 0.0


def _fill_state(state: Any, model: Any, data: Any, ids: _ModelIds) -> None:
    state.q = data.qpos[7:36].copy()
    state.dq = data.qvel[6:35].copy()
    state.tau_est = data.ctrl.copy()
    state.root_lin_vel_b = data.qvel[0:3].copy()
    state.root_ang_vel_b = data.qvel[3:6].copy()
    state.torso_pos_w = data.xpos[ids.torso].copy()
    state.torso_quat_w = data.xquat[ids.torso].copy()
    state.pelvis_pos_w = data.qpos[0:3].copy()
    state.pelvis_quat_w = data.qpos[3:7].copy()
    state.ball_pos_w = data.xpos[ids.ball].copy()
    state.ball_vel_w = data.qvel[ids.ball_qvel : ids.ball_qvel + 3].copy()
    state.ball_valid = True


def _policy_repeat_count(
    speed_scale: float,
    policy_frame: int,
    simulation_frame: int,
) -> int:
    if not 185 <= policy_frame <= 430:
        return 1
    if speed_scale < 1.0:
        hold_period = max(2, int(round(1.0 / (1.0 - speed_scale))))
        return 0 if simulation_frame % hold_period == 0 else 1
    if speed_scale == 1.0:
        return 1
    extra_period = max(2, int(round(1.0 / (speed_scale - 1.0))))
    return 2 if policy_frame % extra_period == 0 else 1


def _adapt_target(
    *,
    target: np.ndarray,
    default: np.ndarray,
    parameters: ShotParameters,
    policy_frame: int,
) -> np.ndarray:
    adapted = target.copy()
    if 185 <= policy_frame <= 335:
        leg = slice(6, 12)
        adapted[leg] = default[leg] + parameters.swing_amplitude * (adapted[leg] - default[leg])
        adapted[1] += parameters.com_shift_y
        adapted[7] += parameters.com_shift_y * 0.5
        adapted[8] += parameters.foot_yaw_offset
        adapted[11] += parameters.foot_yaw_offset * 0.25
        adapted[10] += parameters.foot_pitch_offset
        # One bounded operational-space-inspired synergy: at the nominal
        # strike contact, MuJoCo's right-foot vertical Jacobian is dominated
        # by hip pitch/yaw while knee velocity opposes lift. This low-
        # dimensional target residual raises the foot path without bypassing
        # the learned prior, PD loop, or final torque projector.
        adapted[6] -= parameters.loft_synergy
        adapted[8] -= parameters.loft_synergy * 0.60
        adapted[9] += parameters.loft_synergy
        adapted[10] -= parameters.loft_synergy * 0.25
        adapted[12] += parameters.pelvis_yaw_offset
    if 335 < policy_frame <= 430:
        adapted[6] -= parameters.recovery_step_length * 0.4
        adapted[8] += parameters.recovery_step_yaw
    return adapted


def _contact_observation(model: Any, data: Any, ids: _ModelIds) -> _Contacts:
    import mujoco

    left_floor = False
    right_floor = False
    ball_any = False
    ball_left = False
    ball_right = False
    ball_force = 0.0
    ball_contact_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_ground_force = 0.0
    right_ground_force = 0.0
    force: np.ndarray = np.zeros(6, dtype=np.float64)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
        names = (name1, name2)
        if "floor" in names:
            other = name2 if name1 == "floor" else name1
            left_floor = left_floor or other.startswith("left_foot")
            right_floor = right_floor or other.startswith("right_foot")
            mujoco.mj_contactForce(model, data, index, force)
            contact_force = float(np.linalg.norm(force[:3]))
            if other.startswith("left_foot"):
                left_ground_force = max(left_ground_force, contact_force)
            if other.startswith("right_foot"):
                right_ground_force = max(right_ground_force, contact_force)
        if ids.ball_geom not in {geom1, geom2}:
            continue
        ball_any = True
        other = name2 if geom1 == ids.ball_geom else name1
        ball_left = ball_left or other.startswith("left_foot")
        ball_right = ball_right or other.startswith("right_foot")
        mujoco.mj_contactForce(model, data, index, force)
        ball_force = max(ball_force, float(np.linalg.norm(force[:3])))
        ball_contact_point = (
            float(contact.pos[0]),
            float(contact.pos[1]),
            float(contact.pos[2]),
        )
    return _Contacts(
        left_floor=left_floor,
        right_floor=right_floor,
        ball_any=ball_any,
        ball_left=ball_left,
        ball_right=ball_right,
        ball_force_n=ball_force,
        ball_contact_point=ball_contact_point,
        left_ground_force_n=left_ground_force,
        right_ground_force_n=right_ground_force,
    )


def _roll_pitch(quaternion_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = map(float, quaternion_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_w, left_x, left_y, left_z = map(float, left)
    right_w, right_x, right_y, right_z = map(float, right)
    value = np.asarray(
        (
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        ),
        dtype=np.float64,
    )
    return value / np.linalg.norm(value)


def _append_trace(
    trace: dict[str, list[Any]],
    *,
    time_sec: float,
    data: Any,
    ids: _ModelIds,
    torque: np.ndarray,
    policy_action: np.ndarray,
    com: np.ndarray,
    left_contact: bool,
    right_contact: bool,
    ground_reaction_force: tuple[float, float],
    support_slip: float,
    ball_contact_point: np.ndarray,
    contact_impulse: float,
    goal_crossed: bool,
    feedback_residual: np.ndarray | None = None,
    feedback_error_rms: float = math.nan,
    feedback_phase_rate: float = 0.0,
    policy_phase: float = 0.0,
    feedforward_residual: np.ndarray | None = None,
    combined_residual: np.ndarray | None = None,
    combined_residual_saturation: int = 0,
    recovery_active: bool = False,
    recovery_blend_fraction: float = 0.0,
    recovery_settling_fraction: float = 0.0,
    recovery_smoothing_active: bool = False,
    recovery_smoothing_residual_rms_rad: float = 0.0,
    recovery_proprioception: np.ndarray | None = None,
    recovery_policy_frame: int = 0,
    recovery_observation_updated: bool = False,
    recovery_state_observation: np.ndarray | None = None,
    muscle_memory_active: bool = False,
    muscle_memory_out_of_distribution: bool = False,
    muscle_memory_residual_rms_rad: float = 0.0,
    muscle_memory_synergy_actions: np.ndarray | None = None,
) -> None:
    trace["time"].append(time_sec)
    trace["joint_position"].append(data.qpos[7:36].copy())
    trace["joint_velocity"].append(data.qvel[6:35].copy())
    trace["joint_torque"].append(torque.copy())
    trace["policy_action"].append(policy_action.copy())
    trace["pelvis_pose"].append(data.qpos[:7].copy())
    trace["torso_quaternion"].append(data.xquat[ids.torso].copy())
    trace["com"].append(com.copy())
    trace["left_foot_contact"].append(left_contact)
    trace["right_foot_contact"].append(right_contact)
    trace["ground_reaction_force"].append(ground_reaction_force)
    trace["support_foot_slip"].append(support_slip)
    trace["policy_phase"].append(policy_phase)
    trace["com_y_relative"].append(float(com[1]) - float(data.xpos[ids.left_ankle][1]))
    trace["ball_lateral_error_m"].append(
        float(data.qpos[ids.ball_qpos + 1]) - float(data.xpos[ids.right_ankle][1])
    )
    trace["ball_pose"].append(data.qpos[ids.ball_qpos : ids.ball_qpos + 7].copy())
    trace["ball_velocity"].append(data.qvel[ids.ball_qvel : ids.ball_qvel + 3].copy())
    trace["ball_angular_velocity"].append(data.qvel[ids.ball_qvel + 3 : ids.ball_qvel + 6].copy())
    trace["foot_ball_contact_point"].append(ball_contact_point.copy())
    trace["contact_impulse"].append(contact_impulse)
    trace["goal_crossing"].append(goal_crossed)
    if "feedback_residual" in trace:
        assert feedback_residual is not None
        trace["feedback_residual"].append(feedback_residual.copy())
        trace["feedback_error_rms"].append(feedback_error_rms)
        trace["feedback_active"].append(bool(np.any(np.abs(feedback_residual) > 0.0)))
    if "feedback_phase_rate" in trace:
        trace["feedback_phase_rate"].append(feedback_phase_rate)
    if "feedforward_residual" in trace:
        assert feedforward_residual is not None
        assert combined_residual is not None
        trace["feedforward_residual"].append(feedforward_residual.copy())
        trace["combined_residual"].append(combined_residual.copy())
        trace["combined_residual_saturation"].append(combined_residual_saturation)
    if "recovery_active" in trace:
        assert recovery_proprioception is not None
        trace["recovery_active"].append(recovery_active)
        trace["recovery_blend_fraction"].append(recovery_blend_fraction)
        trace["recovery_settling_fraction"].append(recovery_settling_fraction)
        trace["recovery_smoothing_active"].append(recovery_smoothing_active)
        trace["recovery_smoothing_residual_rms_rad"].append(recovery_smoothing_residual_rms_rad)
        trace["recovery_proprioception"].append(recovery_proprioception.copy())
        trace["recovery_policy_frame"].append(recovery_policy_frame)
        trace["recovery_observation_updated"].append(recovery_observation_updated)
        assert recovery_state_observation is not None
        trace["recovery_state_observation"].append(recovery_state_observation.copy())
    if "muscle_memory_active" in trace:
        assert muscle_memory_synergy_actions is not None
        trace["muscle_memory_active"].append(muscle_memory_active)
        trace["muscle_memory_out_of_distribution"].append(muscle_memory_out_of_distribution)
        trace["muscle_memory_residual_rms_rad"].append(muscle_memory_residual_rms_rad)
        trace["muscle_memory_synergy_actions"].append(muscle_memory_synergy_actions.copy())


def _muscle_memory_observation(
    *,
    data: Any,
    ids: _ModelIds,
    policy_phase: float,
    contact_time: float | None,
    contact_impulse: float,
    left_support: bool,
    right_support: bool,
    left_ground_force: float,
    right_ground_force: float,
) -> dict[str, float]:
    roll, pitch = _roll_pitch(data.xquat[ids.torso])
    com = data.subtree_com[ids.pelvis]
    return {
        "post_contact_time_sec": (
            max(0.0, float(data.time) - contact_time) if contact_time is not None else 0.0
        ),
        "policy_phase": policy_phase,
        "pelvis_velocity_x_m_s": float(data.qvel[0]),
        "pelvis_velocity_y_m_s": float(data.qvel[1]),
        "pelvis_velocity_z_m_s": float(data.qvel[2]),
        "torso_roll_rad": roll,
        "torso_pitch_rad": pitch,
        "torso_angular_velocity_x_rad_s": float(data.cvel[ids.torso][0]),
        "torso_angular_velocity_y_rad_s": float(data.cvel[ids.torso][1]),
        "torso_angular_velocity_z_rad_s": float(data.cvel[ids.torso][2]),
        "com_y_relative_m": float(com[1]) - float(data.xpos[ids.left_ankle][1]),
        "left_support": float(left_support),
        "right_support": float(right_support),
        "left_ground_force_scale": left_ground_force / 500.0,
        "right_ground_force_scale": right_ground_force / 500.0,
        "contact_impulse_ns": contact_impulse,
    }


def _feedback_effect(
    *,
    runtime: FeedbackRuntime,
    timestamp_sec: float,
    phase: float,
    data: Any,
    ids: _ModelIds,
    base_target: np.ndarray,
    support_slip: float,
    contact_detected: bool,
    control_latency_ms: float,
) -> _FeedbackEffect:
    """Run one feedback tick from MuJoCo state without crossing the EventBus."""

    roll, pitch = _roll_pitch(data.xquat[ids.torso])
    com = data.subtree_com[ids.pelvis]
    support_y = float(data.xpos[ids.left_ankle][1])
    ball_position = np.asarray(data.xpos[ids.ball], dtype=np.float64)
    foot_position = np.asarray(data.xpos[ids.right_ankle], dtype=np.float64)
    ball_velocity = np.asarray(data.qvel[ids.ball_qvel : ids.ball_qvel + 3], dtype=np.float64)
    foot_velocity = np.asarray(data.cvel[ids.right_ankle][3:6], dtype=np.float64)
    relative_position = ball_position - foot_position
    relative_velocity = ball_velocity - foot_velocity
    actual: dict[str, float] = {
        "torso_roll": roll,
        "torso_pitch": pitch,
        "com_y_relative": float(com[1]) - support_y,
        "support_slip_m": support_slip,
        "contact_phase": phase,
        "ball_lateral_error_m": float(data.qpos[ids.ball_qpos + 1])
        - float(data.xpos[ids.right_ankle][1]),
        "contact_detected": float(contact_detected),
        "ball_relative_x_m": float(relative_position[0]),
        "ball_relative_y_m": float(relative_position[1]),
        "ball_relative_z_m": float(relative_position[2]),
        "ball_relative_vx_mps": float(relative_velocity[0]),
        "ball_relative_vy_mps": float(relative_velocity[1]),
        "ball_relative_vz_mps": float(relative_velocity[2]),
        "control_latency_ms": control_latency_ms,
        "energy_margin": float(
            np.clip(
                1.0
                - np.max(
                    np.abs(np.asarray(data.ctrl, dtype=np.float64))
                    / np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
                ),
                0.0,
                1.0,
            )
        ),
        "sensor_quality": float(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.ctrl))
        ),
    }
    actual.update(
        {
            joint_name: float(data.qpos[7 + index])
            for index, joint_name in enumerate(G1_DDS_JOINT_NAMES)
        }
    )
    reference = dict.fromkeys(runtime.spec.reference_signals, 0.0)
    base_action = {
        "joint:" + joint_name: float(base_target[index])
        for index, joint_name in enumerate(G1_DDS_JOINT_NAMES)
    }
    timestamp_ns = int(round(timestamp_sec * 1_000_000_000.0))
    command = runtime.tick(
        timestamp_ns=timestamp_ns,
        observation_timestamp_ns=timestamp_ns,
        phase=phase,
        reference=reference,
        actual=actual,
        base_action=base_action,
    )
    residual: np.ndarray = np.zeros(29, dtype=np.float64)
    kick_phase_rate = 0.0
    joint_index = {name: index for index, name in enumerate(G1_DDS_JOINT_NAMES)}
    for output, value in command.projected.items():
        if output == "skill:kick_phase_rate":
            kick_phase_rate = value
            continue
        joint_name = output.removeprefix("joint:")
        if joint_name in joint_index:
            residual[joint_index[joint_name]] = value
    return _FeedbackEffect(
        joint_residual=residual,
        kick_phase_rate=kick_phase_rate,
    )


def _apply_feedback_phase_rate(
    *,
    repeat: int,
    phase_rate: float,
    accumulator: float,
) -> tuple[int, float]:
    """Convert a bounded L2 phase-rate directive into discrete policy-clock steps."""

    if repeat < 0:
        raise ValueError("policy repeat count must be non-negative")
    if not math.isfinite(phase_rate) or not -1.0 <= phase_rate <= 1.0:
        raise ValueError("feedback phase rate must be finite and in [-1, 1]")
    if not math.isfinite(accumulator):
        raise ValueError("feedback phase accumulator must be finite")
    accumulator += phase_rate
    if accumulator >= 1.0:
        extra = int(math.floor(accumulator))
        repeat += extra
        accumulator -= extra
    elif accumulator <= -1.0 and repeat > 0:
        held = min(repeat, int(math.floor(-accumulator)))
        repeat -= held
        accumulator += held
    accumulator = min(0.999999, max(-0.999999, accumulator))
    return repeat, accumulator


def _classify(
    *,
    finite: bool,
    kick_foot_contacted: bool,
    wrong_foot_contacted: bool,
    goal_crossed: bool,
    target_hit: bool,
    target_error: float,
    crossing_y: float,
    target_y: float,
    maximum_ball_speed: float,
    peak_support_slip: float,
    com_margin_min: float,
    roll_peak: float,
    pitch_peak: float,
    fall: bool,
    joint_violation: bool,
    torque_violation: bool,
    actuator_saturation: bool,
) -> GoalForgeStatus:
    if not finite:
        return GoalForgeStatus.NON_FINITE_STATE
    if joint_violation:
        return GoalForgeStatus.JOINT_LIMIT_EXCEEDED
    if torque_violation:
        return GoalForgeStatus.TORQUE_LIMIT_EXCEEDED
    if fall:
        return GoalForgeStatus.POST_KICK_FALL
    if peak_support_slip > 0.08:
        return GoalForgeStatus.SUPPORT_FOOT_SLIP
    if com_margin_min < -0.04:
        return GoalForgeStatus.COM_OUTSIDE_SUPPORT
    if roll_peak > 0.45 or pitch_peak > 0.55:
        return GoalForgeStatus.TORSO_OVERSHOOT
    if wrong_foot_contacted and not kick_foot_contacted:
        return GoalForgeStatus.WRONG_FOOT_CONTACT
    if not kick_foot_contacted:
        return GoalForgeStatus.BALL_NOT_CONTACTED
    if maximum_ball_speed < 1.0:
        return GoalForgeStatus.SHOT_TOO_WEAK
    if maximum_ball_speed > 16.0:
        return GoalForgeStatus.SHOT_TOO_STRONG
    if goal_crossed and target_hit:
        return GoalForgeStatus.SUCCESS
    if goal_crossed and math.isfinite(target_error):
        return (
            GoalForgeStatus.TARGET_MISS_LEFT
            if crossing_y > target_y
            else GoalForgeStatus.TARGET_MISS_RIGHT
        )
    return GoalForgeStatus.SHOT_TOO_WEAK


def _incompatible_result() -> GoalForgeResult:
    return _empty_result(GoalForgeStatus.POLICY_BODY_INCOMPATIBLE)


def _unreachable_result() -> GoalForgeResult:
    return _empty_result(GoalForgeStatus.BALL_OUT_OF_REACH)


def _empty_result(status: GoalForgeStatus) -> GoalForgeResult:
    return GoalForgeResult(
        status=status,
        success=False,
        physics_executed=False,
        contact_observed=False,
        kick_foot_contacted=False,
        goal_crossed=False,
        target_zone_hit=False,
        target_error_m=math.inf,
        ball_speed_mps=0.0,
        ball_contact_time_sec=None,
        contact_impulse_ns=0.0,
        support_foot_slip_m=0.0,
        com_margin_min_m=0.0,
        torso_roll_peak_rad=0.0,
        torso_pitch_peak_rad=0.0,
        peak_torque_scale=0.0,
        joint_limit_violation=False,
        torque_limit_violation=False,
        actuator_saturation=False,
        post_kick_fall=False,
        post_kick_stability_time_sec=0.0,
        final_pelvis_height_m=0.0,
        physics_steps=0,
        finite_state=True,
        robustness=-1.0,
    )


def _episode_id(scenario: GoalForgeScenario, parameters: ShotParameters) -> str:
    digest = hashlib.sha256(
        f"{scenario.scenario_commitment}\0{parameters.policy_hash}".encode()
    ).hexdigest()
    return f"g1-kick-{digest[:24]}"


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip()


def _external_root(output_root: Path, source_checkout: Path) -> Path:
    root = output_root.expanduser().resolve()
    checkout = source_checkout.resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("raw GoalForge evidence must stay outside the source checkout")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _bounded_hash(path: Path) -> str:
    if path.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("GoalForge trajectory artifact exceeds size limit")
    return hash_bytes(path.read_bytes())


def trajectory_digest(trajectory: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(trajectory):
        value = np.ascontiguousarray(trajectory[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return "sha256:" + digest.hexdigest()


def _independent_trace_check(
    trajectory: dict[str, np.ndarray],
    result: GoalForgeResult,
) -> bool:
    required = {
        "time",
        "joint_position",
        "joint_velocity",
        "joint_torque",
        "policy_action",
        "pelvis_pose",
        "torso_quaternion",
        "com",
        "left_foot_contact",
        "right_foot_contact",
        "support_foot_slip",
        "ball_pose",
        "ball_velocity",
        "ball_angular_velocity",
        "goal_crossing",
    }
    if required - set(trajectory):
        return False
    lengths = {len(np.asarray(value)) for value in trajectory.values()}
    joint_contract = all(
        np.asarray(trajectory[name]).ndim == 2 and np.asarray(trajectory[name]).shape[1] == 29
        for name in ("joint_position", "joint_velocity", "joint_torque", "policy_action")
    )
    finite = all(
        np.all(np.isfinite(np.asarray(value)))
        for value in trajectory.values()
        if np.asarray(value).dtype.kind in "fiu"
    )
    return bool(
        len(lengths) == 1
        and joint_contract
        and finite
        and result.physics_executed
        and result.physics_steps >= next(iter(lengths), 0)
    )


__all__ = [
    "G1AssetQualification",
    "G1MuJoCoBackend",
    "GoalForgeEpisode",
    "qualify_g1_assets",
    "trajectory_digest",
]
