"""Build event-bound approach-to-strike transitions from strict G1 evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.adapters.g1_free_kick import triage_g1_free_kick_trajectory
from rosclaw.growth.approach_strike_contracts import COST_NAMES, REWARD_NAMES, STATE_FEATURES
from rosclaw.growth.routing import DataProfile, GrowthProblemSignals, route_learners
from rosclaw.simforge.g1_free_kick_showcase import G1FootballEventPhase
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

MINIMUM_TRAINING_EPISODES = 8


@dataclass(frozen=True)
class ApproachStrikeDatasetReceipt:
    output_dir: str
    array_path: str
    array_file_hash: str
    array_content_hash: str
    manifest_path: str
    manifest_hash: str
    transition_count: int
    episode_count: int
    minimum_training_episodes: int
    training_eligible: bool
    learner_ids: tuple[str, ...]
    schema_version: str = "rosclaw.growth.approach_strike_dataset_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "learner_ids": list(self.learner_ids),
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }


def build_g1_approach_strike_dataset(
    *,
    trajectory_paths: tuple[Path, ...],
    evidence_paths: tuple[Path, ...],
    output_dir: Path,
    source_checkout: Path,
) -> ApproachStrikeDatasetReceipt:
    """Extract safe-action transition tuples while preserving source lineage."""

    if not trajectory_paths or len(trajectory_paths) != len(evidence_paths):
        raise ValueError("approach-strike dataset requires paired trajectory/evidence paths")
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("approach-strike dataset must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    chunks: list[dict[str, np.ndarray]] = []
    sources: list[dict[str, Any]] = []
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    trajectory_hashes: set[str] = set()
    for episode, (trajectory_path, evidence_path) in enumerate(
        zip(trajectory_paths, evidence_paths, strict=True)
    ):
        report = triage_g1_free_kick_trajectory(
            trajectory_path=trajectory_path, evidence_path=evidence_path
        )
        if report.source_hash in trajectory_hashes:
            raise ValueError("approach-strike dataset requires independent trajectory hashes")
        trajectory_hashes.add(report.source_hash)
        if not report.strict_replay:
            raise ValueError("approach-strike dataset requires strict replay sources")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        body_hash = str(evidence.get("body_hash", ""))
        if not body_hash.startswith("sha256:"):
            raise ValueError("approach-strike evidence is missing its body hash")
        implementation_hash = str(evidence.get("implementation_hash", ""))
        if not implementation_hash.startswith("sha256:"):
            raise ValueError("approach-strike evidence is missing its implementation hash")
        body_hashes.add(body_hash)
        implementation_hashes.add(implementation_hash)
        with np.load(trajectory_path, allow_pickle=False) as archive:
            values = {name: np.asarray(archive[name]) for name in archive.files}
        chunk = _extract_episode(values, evidence["result"], episode)
        chunks.append(chunk)
        sources.append(
            {
                "trajectory_path": report.source_path,
                "trajectory_hash": report.source_hash,
                "evidence_path": report.evidence_path,
                "evidence_hash": report.evidence_hash,
                "triage_hash": report.report_hash,
                "request_hash": evidence.get("request_hash"),
                "transition_count": len(chunk["state"]),
                "evidence_passed": report.evidence_passed,
                "loft_teacher_executed": evidence["result"].get("loft_teacher_executed") is True,
            }
        )
    if len(body_hashes) != 1:
        raise ValueError("approach-strike dataset cannot mix G1 body hashes")
    if len(implementation_hashes) != 1:
        raise ValueError("approach-strike dataset cannot mix implementation hashes")
    arrays = {name: np.concatenate([chunk[name] for chunk in chunks], axis=0) for name in chunks[0]}
    _validate_arrays(arrays)
    array_path = root / "approach_strike_transitions.npz"
    np.savez_compressed(array_path, **arrays)
    content_hash = _array_content_hash(arrays)
    data_profile = DataProfile(
        has_state=True,
        has_executed_action=True,
        has_next_state=True,
        has_reward_vector=True,
        has_cost_vector=True,
        has_kinematic_reference=True,
        has_chunk_feedback=False,
        fixed_dataset=True,
        online_rollout_allowed=False,
    )
    route = route_learners(
        GrowthProblemSignals(
            repeated_error=0.9,
            local_physics_residual=0.9,
            safety_model_complete=True,
        ),
        data_profile,
    )
    eligible = len(chunks) >= MINIMUM_TRAINING_EPISODES
    environment = {
        "body_hash": next(iter(body_hashes)),
        "implementation_hash": next(iter(implementation_hashes)),
        "physics_authority": "CPU_MUJOCO",
        "control_dt_sec": 0.02,
    }
    manifest = {
        "schema_version": "rosclaw.growth.g1_approach_strike_transition_dataset.v2",
        "task_id": "g1_approach_strike_transition",
        "evidence_domain": "SIM_ONLY",
        "sources": sources,
        "body_hash": next(iter(body_hashes)),
        "environment": environment,
        "environment_hash": canonical_hash(environment),
        "arrays": {
            "path": str(array_path),
            "file_hash": _file_hash(array_path),
            "content_hash": content_hash,
            "transition_count": len(arrays["state"]),
            "episode_count": len(chunks),
            "state_features": list(STATE_FEATURES),
            "reward_names": list(REWARD_NAMES),
            "cost_names": list(COST_NAMES),
            "scalarization": {
                "reward_weights": {
                    "phase_progress": 0.5,
                    "ball_distance_progress": 0.5,
                    "upright": 0.2,
                    "action_smoothness": 0.002,
                    "contact_speed": 1.0,
                    "terminal_precision": 2.0,
                },
                "cost_weights": {
                    "torque_projection": 100.0,
                    "torque_overdrive": 20.0,
                    "low_pelvis": 10.0,
                    "tilt_excess": 10.0,
                    "episode_safety_violation": 100.0,
                },
            },
            "action_semantics": {
                "commanded_action": "unclipped_pd_torque_nm",
                "safety_projected_action": "hard_authority_projected_torque_nm",
                "executed_action": "mujoco_actuator_force_nm",
                "teacher_residual_action": ("sim_only_operational_space_teacher_joint_torque_nm"),
                "joint_boundary_guard_correction": (
                    "post_contact_safety_projection_delta_torque_nm"
                ),
                "kinematic_reference": "base_joint_position_target_rad",
            },
        },
        "data_profile": data_profile.to_dict(),
        "learner_route": route.to_dict(),
        "minimum_training_episodes": MINIMUM_TRAINING_EPISODES,
        "training_eligible": eligible,
        "training_blockers": ([] if eligible else ["insufficient_independent_episodes"]),
        "promotion_truth_allowed": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ApproachStrikeDatasetReceipt(
        output_dir=str(root),
        array_path=str(array_path),
        array_file_hash=str(manifest["arrays"]["file_hash"]),
        array_content_hash=content_hash,
        manifest_path=str(manifest_path),
        manifest_hash=str(manifest["manifest_hash"]),
        transition_count=len(arrays["state"]),
        episode_count=len(chunks),
        minimum_training_episodes=MINIMUM_TRAINING_EPISODES,
        training_eligible=eligible,
        learner_ids=route.learner_ids,
    )


def _extract_episode(
    values: dict[str, np.ndarray], result: dict[str, Any], episode: int
) -> dict[str, np.ndarray]:
    required = {
        "time",
        "joint_position",
        "joint_velocity",
        "pelvis_pose",
        "torso_quaternion",
        "ball_pose",
        "ball_velocity",
        "event_phase",
        "policy_action",
        "commanded_torque",
        "safety_projected_torque",
        "executed_torque",
        "torque_projection_applied",
    }
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"approach-strike source is missing transition fields: {missing}")
    time = np.asarray(values["time"], dtype=np.float64)
    count = len(time)
    phases = np.asarray(values["event_phase"], dtype=np.int64)
    selected = np.flatnonzero(
        np.isin(
            phases[:-1],
            (
                int(G1FootballEventPhase.ALIGN_BRAKE),
                int(G1FootballEventPhase.PLANT_BRIDGE),
                int(G1FootballEventPhase.LOAD),
                int(G1FootballEventPhase.SWING),
                int(G1FootballEventPhase.CONTACT),
            ),
        )
    )
    if len(selected) < 2:
        raise ValueError("approach-strike source has too few transition-phase frames")
    state_all = _state_features(values, time)
    commanded = np.asarray(values["commanded_torque"], dtype=np.float64)
    projected = np.asarray(values["safety_projected_torque"], dtype=np.float64)
    executed = np.asarray(values["executed_torque"], dtype=np.float64)
    projection = np.asarray(values["torque_projection_applied"], dtype=bool)
    if any(array.shape != (count, 29) for array in (commanded, projected, executed)):
        raise ValueError("approach-strike action arrays must be [T, 29]")
    rewards = _reward_vector(values, result, time)
    costs = _cost_vector(values, result)
    teacher_action = np.asarray(
        values.get("loft_teacher_torque", np.zeros((count, 29))),
        dtype=np.float64,
    )
    teacher_active = np.asarray(
        values.get("loft_teacher_active", np.zeros(count, dtype=bool)),
        dtype=bool,
    )
    boundary_correction = np.asarray(
        values.get("joint_boundary_guard_correction", np.zeros((count, 29))),
        dtype=np.float64,
    )
    if teacher_action.shape != (count, 29) or boundary_correction.shape != (count, 29):
        raise ValueError("approach-strike teacher action arrays must be [T, 29]")
    if teacher_active.shape != (count,):
        raise ValueError("approach-strike teacher active flags must be [T]")
    return {
        "state": state_all[selected],
        "commanded_action": commanded[selected],
        "safety_projected_action": projected[selected],
        "executed_action": executed[selected],
        "next_state": state_all[selected + 1],
        "reward_vector": rewards[selected],
        "cost_vector": costs[selected],
        "terminal": (phases[selected + 1] >= int(G1FootballEventPhase.FOLLOW_THROUGH)),
        "episode_index": np.full(len(selected), episode, dtype=np.int64),
        "frame_index": selected.astype(np.int64),
        "event_phase": phases[selected],
        "projection_applied": projection[selected],
        "teacher_residual_action": teacher_action[selected],
        "teacher_active": teacher_active[selected],
        "joint_boundary_guard_correction": boundary_correction[selected],
        "kinematic_reference": np.asarray(values["policy_action"], dtype=np.float64)[selected],
    }


def _state_features(values: dict[str, np.ndarray], time: np.ndarray) -> np.ndarray:
    joint_position = np.asarray(values["joint_position"], dtype=np.float64)
    joint_velocity = np.asarray(values["joint_velocity"], dtype=np.float64)
    pelvis = np.asarray(values["pelvis_pose"], dtype=np.float64)
    torso = np.asarray(values["torso_quaternion"], dtype=np.float64)
    ball_pose = np.asarray(values["ball_pose"], dtype=np.float64)
    ball_velocity = np.asarray(values["ball_velocity"], dtype=np.float64)[:, :3]
    phase = np.asarray(values["event_phase"], dtype=np.int64)
    target = np.asarray(values["policy_action"], dtype=np.float64)
    base_velocity = np.gradient(pelvis[:, :3], time, axis=0)
    relative_ball = ball_pose[:, :3] - pelvis[:, :3]
    one_hot = np.eye(len(G1FootballEventPhase), dtype=np.float64)[phase]
    state = np.concatenate(
        (
            joint_position,
            joint_velocity,
            pelvis[:, 2:3],
            base_velocity,
            torso,
            relative_ball,
            ball_velocity,
            one_hot,
            target,
        ),
        axis=1,
    )
    if state.shape[1] != len(STATE_FEATURES):
        raise AssertionError("approach-strike state feature contract drifted")
    return state


def _reward_vector(
    values: dict[str, np.ndarray], result: dict[str, Any], time: np.ndarray
) -> np.ndarray:
    phase = np.asarray(values["event_phase"], dtype=np.float64)
    pelvis = np.asarray(values["pelvis_pose"], dtype=np.float64)
    torso = np.asarray(values["torso_quaternion"], dtype=np.float64)
    ball = np.asarray(values["ball_pose"], dtype=np.float64)[:, :3]
    ball_velocity = np.asarray(values["ball_velocity"], dtype=np.float64)[:, :3]
    target = np.asarray(values["policy_action"], dtype=np.float64)
    distance = np.linalg.norm(ball - pelvis[:, :3], axis=1)
    distance_progress = np.r_[distance[:-1] - distance[1:], 0.0]
    phase_progress = np.r_[np.maximum(phase[1:] - phase[:-1], 0.0) / 8.0, 0.0]
    tilt = _tilt(torso)
    upright = np.clip((pelvis[:, 2] - 0.55) / 0.25, 0.0, 1.0) * np.clip(1.0 - tilt / 0.65, 0.0, 1.0)
    target_velocity = np.gradient(target, time, axis=0)
    smoothness = -np.sqrt(np.mean(np.square(target_velocity), axis=1))
    contact = phase == int(G1FootballEventPhase.CONTACT)
    contact_speed = contact * np.clip(np.linalg.norm(ball_velocity, axis=1) / 10.0, 0.0, 1.5)
    error = result.get("goal_plane_target_error_m")
    radius = result.get("precision_radius_m", 0.16)
    precision = (
        math.exp(-float(error) / max(float(radius), 1e-6))
        if isinstance(error, (int, float)) and math.isfinite(float(error))
        else 0.0
    )
    terminal_precision = contact.astype(np.float64) * precision
    return np.column_stack(
        (
            phase_progress,
            distance_progress,
            upright,
            smoothness,
            contact_speed,
            terminal_precision,
        )
    )


def _cost_vector(values: dict[str, np.ndarray], result: dict[str, Any]) -> np.ndarray:
    commanded = np.asarray(values["commanded_torque"], dtype=np.float64)
    projection = np.asarray(values["torque_projection_applied"], dtype=bool)
    pelvis = np.asarray(values["pelvis_pose"], dtype=np.float64)
    torso = np.asarray(values["torso_quaternion"], dtype=np.float64)
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    overdrive = np.maximum(np.max(np.abs(commanded) / limits[None, :], axis=1) - 1.0, 0.0)
    low_pelvis = np.maximum(0.68 - pelvis[:, 2], 0.0)
    tilt_excess = np.maximum(_tilt(torso) - 0.40, 0.0)
    unsafe = float(
        any(
            result.get(name) is True
            for name in (
                "post_kick_fall",
                "joint_limit_violation",
                "torque_limit_violation",
            )
        )
    )
    return np.column_stack(
        (
            projection.astype(np.float64),
            overdrive,
            low_pelvis,
            tilt_excess,
            np.full(len(pelvis), unsafe, dtype=np.float64),
        )
    )


def _tilt(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion.T
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.maximum(np.abs(roll), np.abs(pitch))


def _validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    count = len(arrays["state"])
    expected = {
        "state": (count, len(STATE_FEATURES)),
        "commanded_action": (count, 29),
        "safety_projected_action": (count, 29),
        "executed_action": (count, 29),
        "next_state": (count, len(STATE_FEATURES)),
        "reward_vector": (count, len(REWARD_NAMES)),
        "cost_vector": (count, len(COST_NAMES)),
        "terminal": (count,),
        "episode_index": (count,),
        "frame_index": (count,),
        "event_phase": (count,),
        "projection_applied": (count,),
        "teacher_residual_action": (count, 29),
        "teacher_active": (count,),
        "joint_boundary_guard_correction": (count, 29),
        "kinematic_reference": (count, 29),
    }
    invalid = [name for name, shape in expected.items() if arrays[name].shape != shape]
    if invalid:
        raise ValueError(f"approach-strike dataset shapes are invalid: {invalid}")
    numeric = [value for value in arrays.values() if value.dtype != np.dtype(bool)]
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("approach-strike dataset contains non-finite values")
    if np.any(arrays["cost_vector"] < 0.0):
        raise ValueError("approach-strike safety costs must be non-negative")


def _array_content_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return "sha256:" + digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "ApproachStrikeDatasetReceipt",
    "COST_NAMES",
    "MINIMUM_TRAINING_EPISODES",
    "REWARD_NAMES",
    "STATE_FEATURES",
    "build_g1_approach_strike_dataset",
]
