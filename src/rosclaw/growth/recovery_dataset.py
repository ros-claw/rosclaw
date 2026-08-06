"""Build physics-qualified post-impact recovery transitions for offline learning."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.adapters.g1_coupled import (
    FootballPhase,
    triage_g1_coupled_trajectory,
    verified_coupled_evidence_context,
)
from rosclaw.growth.routing import DataProfile, GrowthProblemSignals, route_learners

STATE_FEATURES = (
    *(f"joint_position.{index}" for index in range(29)),
    *(f"joint_velocity.{index}" for index in range(29)),
    "pelvis_height",
    "pelvis_velocity_x",
    "pelvis_velocity_y",
    "pelvis_velocity_z",
    "torso_roll",
    "torso_pitch",
    "com_relative_support_y",
    "left_foot_contact",
    "right_foot_contact",
    "ball_relative_x",
    "ball_relative_y",
    "ball_relative_z",
    "ball_velocity_x",
    "ball_velocity_y",
    "ball_velocity_z",
    "time_since_contact",
)
REWARD_NAMES = (
    "upright",
    "momentum_unloading",
    "low_joint_speed",
    "low_action_jerk",
    "support_centering",
    "readiness",
    "energy_efficiency",
)
COST_NAMES = (
    "fall",
    "support_slip_excess",
    "torque_projection",
    "lost_support",
    "episode_safety_violation",
)


@dataclass(frozen=True)
class RecoveryDatasetReceipt:
    output_dir: str
    array_path: str
    array_file_hash: str
    array_content_hash: str
    manifest_path: str
    manifest_hash: str
    transition_count: int
    episode_count: int
    training_eligible: bool
    learner_ids: tuple[str, ...]
    schema_version: str = "rosclaw.growth.recovery_dataset_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "learner_ids": list(self.learner_ids),
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }


def build_g1_recovery_dataset(
    *,
    trajectory_paths: tuple[Path, ...],
    evidence_path: Path,
    output_dir: Path,
    source_checkout: Path,
) -> RecoveryDatasetReceipt:
    """Extract transition tuples while preserving action and environment lineage."""

    if not trajectory_paths:
        raise ValueError("recovery dataset requires at least one trajectory")
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("recovery dataset evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    chunks: list[dict[str, np.ndarray]] = []
    sources: list[dict[str, Any]] = []
    environments: set[str] = set()
    runtimes: list[dict[str, str]] = []
    for episode_index, source in enumerate(trajectory_paths):
        context = verified_coupled_evidence_context(evidence_path, source)
        if not context.strict_replay:
            raise ValueError("recovery dataset requires strict replay for every source")
        if context.environment_hash is None or context.runtime is None:
            raise ValueError("recovery dataset requires a runtime-bound evidence receipt")
        environments.add(context.environment_hash)
        runtimes.append(dict(context.runtime))
        report = triage_g1_coupled_trajectory(
            source,
            role="shooter",
            evidence_context=context,
        )
        chunk = _extract_episode(source, report.contact_index, episode_index, context.result)
        chunks.append(chunk)
        sources.append(
            {
                "trajectory_path": report.source_path,
                "trajectory_hash": report.source_hash,
                "evidence_hash": context.evidence_hash,
                "environment_hash": context.environment_hash,
                "contact_index": report.contact_index,
                "transition_count": len(chunk["state"]),
                "triage_hash": report.report_hash,
            }
        )
    if len(environments) != 1:
        raise ValueError("recovery dataset cannot mix runtime environments")
    if any(runtime != runtimes[0] for runtime in runtimes[1:]):
        raise ValueError("recovery dataset runtime manifests disagree")
    arrays = {name: np.concatenate([chunk[name] for chunk in chunks], axis=0) for name in chunks[0]}
    _validate_dataset_arrays(arrays)
    array_path = root / "recovery_transitions.npz"
    np.savez_compressed(array_path, **arrays)
    array_content_hash = _array_content_hash(arrays)
    profile = DataProfile(
        has_state=True,
        has_executed_action=True,
        has_next_state=True,
        has_reward_vector=True,
        has_cost_vector=True,
        has_kinematic_reference=True,
        fixed_dataset=True,
        online_rollout_allowed=False,
    )
    route = route_learners(
        GrowthProblemSignals(repeated_error=0.8, local_physics_residual=0.8),
        profile,
    )
    manifest = {
        "schema_version": "rosclaw.growth.g1_recovery_transition_dataset.v1",
        "task_id": "g1_post_impact_recovery",
        "evidence_domain": "SIM_ONLY",
        "sources": sources,
        "environment": runtimes[0],
        "environment_hash": next(iter(environments)),
        "arrays": {
            "path": str(array_path),
            "file_hash": _file_hash(array_path),
            "content_hash": array_content_hash,
            "transition_count": len(arrays["state"]),
            "episode_count": len(chunks),
            "state_features": list(STATE_FEATURES),
            "reward_names": list(REWARD_NAMES),
            "cost_names": list(COST_NAMES),
            "action_semantics": {
                "commanded_action": "unclipped_pd_torque_nm",
                "safety_projected_action": "torque_guard_clipped_torque_nm",
                "executed_action": "mujoco_actuator_force_nm",
            },
        },
        "data_profile": profile.to_dict(),
        "learner_route": route.to_dict(),
        "training_eligible": True,
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
    return RecoveryDatasetReceipt(
        output_dir=str(root),
        array_path=str(array_path),
        array_file_hash=str(manifest["arrays"]["file_hash"]),
        array_content_hash=array_content_hash,
        manifest_path=str(manifest_path),
        manifest_hash=str(manifest["manifest_hash"]),
        transition_count=len(arrays["state"]),
        episode_count=len(chunks),
        training_eligible=True,
        learner_ids=route.learner_ids,
    )


def _extract_episode(
    path: Path,
    contact_index: int,
    episode_index: int,
    result: Any,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "time",
        "ball_pose",
        "ball_velocity",
        "shooter_pelvis_pose",
        "shooter_torso_quaternion",
        "shooter_joint_position",
        "shooter_joint_velocity",
        "shooter_foot_contact",
        "shooter_com_position",
        "shooter_left_foot_position",
        "shooter_right_foot_position",
        "shooter_support_foot_slip",
        "shooter_commanded_torque",
        "shooter_safety_projected_torque",
        "shooter_executed_torque",
    }
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"recovery source is missing transition fields: {missing}")
    time = np.asarray(values["time"], dtype=np.float64)
    pelvis = np.asarray(values["shooter_pelvis_pose"], dtype=np.float64)
    torso = np.asarray(values["shooter_torso_quaternion"], dtype=np.float64)
    joint_position = np.asarray(values["shooter_joint_position"], dtype=np.float64)
    joint_velocity = np.asarray(values["shooter_joint_velocity"], dtype=np.float64)
    foot_contact = np.asarray(values["shooter_foot_contact"], dtype=bool)
    com = np.asarray(values["shooter_com_position"], dtype=np.float64)
    left_foot = np.asarray(values["shooter_left_foot_position"], dtype=np.float64)
    right_foot = np.asarray(values["shooter_right_foot_position"], dtype=np.float64)
    slip = np.asarray(values["shooter_support_foot_slip"], dtype=np.float64)
    commanded = np.asarray(values["shooter_commanded_torque"], dtype=np.float64)
    projected = np.asarray(values["shooter_safety_projected_torque"], dtype=np.float64)
    executed = np.asarray(values["shooter_executed_torque"], dtype=np.float64)
    ball_pose = np.asarray(values["ball_pose"], dtype=np.float64)
    ball_velocity = np.asarray(values["ball_velocity"], dtype=np.float64)[:, :3]
    count = len(time)
    expected_29 = (count, 29)
    if any(
        array.shape != expected_29
        for array in (joint_position, joint_velocity, commanded, projected, executed)
    ):
        raise ValueError("recovery joint/action arrays must be [T, 29]")
    if contact_index < 0 or contact_index >= count - 2:
        raise ValueError("recovery contact index cannot form transitions")
    numeric = (
        time,
        pelvis,
        torso,
        joint_position,
        joint_velocity,
        com,
        left_foot,
        right_foot,
        slip,
        commanded,
        projected,
        executed,
        ball_pose,
        ball_velocity,
    )
    if not all(np.all(np.isfinite(array)) for array in numeric):
        raise ValueError("recovery transition source contains non-finite values")
    pelvis_velocity = np.gradient(pelvis[:, :3], time, axis=0)
    roll, pitch = _roll_pitch(torso)
    support_count = np.maximum(np.sum(foot_contact, axis=1), 1)
    support_y = (
        left_foot[:, 1] * foot_contact[:, 0] + right_foot[:, 1] * foot_contact[:, 1]
    ) / support_count
    com_relative = com[:, 1] - support_y
    elapsed = time - time[contact_index]
    state = np.column_stack(
        (
            joint_position,
            joint_velocity,
            pelvis[:, 2],
            pelvis_velocity,
            roll,
            pitch,
            com_relative,
            foot_contact.astype(np.float64),
            ball_pose[:, :3] - pelvis[:, :3],
            ball_velocity,
            elapsed,
        )
    )
    joint_speed = np.sqrt(np.mean(np.square(joint_velocity), axis=1))
    action_rate = np.linalg.norm(np.gradient(executed, time, axis=0), axis=1) / math.sqrt(29)
    energy = np.mean(np.abs(executed * joint_velocity), axis=1)
    tilt = np.hypot(roll, pitch)
    com_speed = np.linalg.norm(np.gradient(com[:, :2], time, axis=0), axis=1)
    readiness = (
        (tilt <= 0.35)
        & (np.linalg.norm(pelvis_velocity[:, :2], axis=1) <= 0.25)
        & (joint_speed <= 0.55)
        & np.any(foot_contact, axis=1)
    )
    reward = np.column_stack(
        (
            1.0 - np.minimum(tilt / 0.35, 1.0),
            -com_speed,
            -joint_speed,
            -action_rate,
            -np.abs(com_relative),
            readiness.astype(np.float64),
            -energy,
        )
    )
    episode_violation = float(
        bool(result.get("joint_limit_violation"))
        or bool(result.get("torque_limit_violation"))
        or bool(result.get("actuator_saturation"))
        or bool(result.get("shooter_post_kick_fall", result.get("post_kick_fall", False)))
    )
    cost = np.column_stack(
        (
            (pelvis[:, 2] < 0.55).astype(np.float64),
            np.maximum(slip - 0.04, 0.0),
            np.linalg.norm(commanded - projected, axis=1) / math.sqrt(29),
            (~np.any(foot_contact, axis=1)).astype(np.float64),
            np.full(count, episode_violation),
        )
    )
    start = contact_index
    end = count - 1
    phase = np.full(count, list(FootballPhase).index(FootballPhase.RECOVERY), dtype=np.int16)
    phase[readiness] = list(FootballPhase).index(FootballPhase.READY)
    return {
        "state": state[start:end].astype(np.float32),
        "next_state": state[start + 1 : end + 1].astype(np.float32),
        "commanded_action": commanded[start:end].astype(np.float32),
        "safety_projected_action": projected[start:end].astype(np.float32),
        "executed_action": executed[start:end].astype(np.float32),
        "reward_vector": reward[start + 1 : end + 1].astype(np.float32),
        "cost_vector": cost[start + 1 : end + 1].astype(np.float32),
        "phase": phase[start:end],
        "episode_index": np.full(end - start, episode_index, dtype=np.int32),
        "terminal": np.concatenate((np.zeros(end - start - 1, dtype=bool), np.ones(1, dtype=bool))),
    }


def _validate_dataset_arrays(arrays: dict[str, np.ndarray]) -> None:
    count = len(arrays["state"])
    if count < 2 or any(len(array) != count for array in arrays.values()):
        raise ValueError("recovery transition arrays have inconsistent lengths")
    if arrays["state"].shape != (count, len(STATE_FEATURES)):
        raise ValueError("recovery state feature contract mismatch")
    if arrays["next_state"].shape != arrays["state"].shape:
        raise ValueError("recovery next-state contract mismatch")
    for name in ("commanded_action", "safety_projected_action", "executed_action"):
        if arrays[name].shape != (count, 29):
            raise ValueError(f"recovery {name} contract mismatch")
    if arrays["reward_vector"].shape != (count, len(REWARD_NAMES)):
        raise ValueError("recovery reward vector contract mismatch")
    if arrays["cost_vector"].shape != (count, len(COST_NAMES)):
        raise ValueError("recovery cost vector contract mismatch")
    if not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("recovery dataset arrays must be finite")
    if np.any(arrays["cost_vector"] < 0.0):
        raise ValueError("recovery safety costs must be non-negative")


def _roll_pitch(quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("recovery torso quaternion has zero norm")
    w, x, y, z = (quaternion / norm).T
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def _array_content_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "COST_NAMES",
    "REWARD_NAMES",
    "STATE_FEATURES",
    "RecoveryDatasetReceipt",
    "build_g1_recovery_dataset",
]
