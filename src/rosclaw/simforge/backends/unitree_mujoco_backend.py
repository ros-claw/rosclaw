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

    def run(
        self,
        scenario: GoalForgeScenario,
        parameters: ShotParameters,
    ) -> GoalForgeEpisode:
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
        return self._run_physics(scenario, parameters)

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
                and _trajectory_digest(replay.trajectory) == _trajectory_digest(episode.trajectory)
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
        )

    def _run_physics(
        self,
        scenario: GoalForgeScenario,
        parameters: ShotParameters,
    ) -> GoalForgeEpisode:
        import mujoco

        root = self.qualification.asset_root
        state_type, output_type, policy_type, mujoco_to_isaac = _load_robonaldo(root)
        model = mujoco.MjModel.from_xml_path(str(root / _SCENE_REL))
        data = mujoco.MjData(model)
        model.opt.timestep = 0.002
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
            "ball_pose": [],
            "ball_velocity": [],
            "ball_angular_velocity": [],
            "foot_ball_contact_point": [],
            "contact_impulse": [],
        }
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
        last_target = data.qpos[7:36].copy()

        for frame in range(total_control_frames):
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
                else:
                    target = last_target.copy()
                    kp = np.asarray(policy.kps, dtype=np.float64)
                    kd = np.asarray(policy.kds, dtype=np.float64)
                    policy_frame = current_policy_frame
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
                raw_torque = (delayed_target - data.qpos[7:36]) * kp - data.qvel[6:35] * kd
                raw_scale = float(np.max(np.abs(raw_torque) / hard_limits))
                peak_torque_scale = max(peak_torque_scale, min(raw_scale, self.torque_guard_scale))
                frame_torque = np.clip(raw_torque, -guarded_limits, guarded_limits)
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
        return GoalForgeEpisode(
            scenario=scenario,
            parameters=parameters,
            result=result,
            receipt=None,
            artifact_root=None,
            trajectory=arrays,
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
        scenario.ball_velocity_x_mps,
        scenario.ball_velocity_y_mps,
        0.0,
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
    trace["ball_pose"].append(data.qpos[ids.ball_qpos : ids.ball_qpos + 7].copy())
    trace["ball_velocity"].append(data.qvel[ids.ball_qvel : ids.ball_qvel + 3].copy())
    trace["ball_angular_velocity"].append(data.qvel[ids.ball_qvel + 3 : ids.ball_qvel + 6].copy())
    trace["foot_ball_contact_point"].append(ball_contact_point.copy())
    trace["contact_impulse"].append(contact_impulse)


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


def _trajectory_digest(trajectory: dict[str, np.ndarray]) -> str:
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
]
