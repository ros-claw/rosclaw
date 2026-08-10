"""Derive a parent-conditioned, whole-body G1 shooting style prior.

MotionDecode supplies kinematics, not synchronized ball state, actions, rewards,
or transitions.  This module therefore uses only Q1 repaired shooting clips as
a bounded position-reference teacher.  A strict-replay ROSClaw trajectory is
used to select nearby motion islands; its outcome is not used as a reward.
The resulting prior remains SIM_ONLY and cannot promote or command hardware.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.collective.sources.motiondecode.audit import load_g1_joint_contract
from rosclaw.collective.sources.motiondecode.manifest import (
    MotionDecodeRegistration,
    verify_registered_files,
)
from rosclaw.collective.sources.motiondecode.parser import parse_motion_csv
from rosclaw.collective.sources.motiondecode.repair import clean_motiondecode_spans
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.football_motion_prior import (
    G1FootballMotionPrior,
    G1FootballStyleEvent,
)
from rosclaw.simforge.backends.unitree_mujoco_backend import qualify_g1_assets
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    hash_bytes,
)

_REFERENCE_TIMES_SEC = (
    -0.36,
    -0.28,
    -0.20,
    -0.12,
    -0.06,
    0.0,
    0.06,
    0.12,
    0.20,
    0.32,
    0.48,
    0.64,
)
_SHOOTING_TOKEN = "/3.3.3.3.Shooting/"
_MAX_JSON_BYTES = 128 * 1024 * 1024


def derive_motiondecode_g1_football_skill_prior(
    *,
    registration_path: Path,
    repair_report_path: Path,
    dataset_root: Path,
    target_model_path: Path,
    asset_root: Path,
    parent_evidence_path: Path,
    parent_trajectory_path: Path,
    output_path: Path,
    source_checkout: Path,
    selected_event_count: int = 16,
) -> G1FootballMotionPrior:
    """Select and distil nearby Q1 shooting styles from the training split."""

    if not 8 <= selected_event_count <= 32:
        raise ValueError("MotionDecode football style event count must be in [8, 32]")
    output = _external_output(output_path, source_checkout)
    registration_artifact = _bounded_json(registration_path)
    registration_value = registration_artifact.get("registration")
    if not isinstance(registration_value, dict):
        raise ValueError("MotionDecode registration artifact lacks a registration")
    registration = MotionDecodeRegistration.from_dict(registration_value)
    if registration_artifact.get("registration_hash") != registration.registration_hash:
        raise ValueError("MotionDecode registration hash does not replay")
    if not registration.manifest.license_snapshot.training_permitted:
        raise ValueError("MotionDecode registration does not permit research training")

    repair_artifact = _bounded_json(repair_report_path)
    repair = repair_artifact.get("report")
    if not isinstance(repair, dict):
        raise ValueError("MotionDecode repair artifact lacks a report")
    repair_hash = str(repair_artifact.get("report_hash", ""))
    if repair_hash != canonical_hash(repair):
        raise ValueError("MotionDecode repair report hash does not replay")
    if repair.get("schema_version") != "rosclaw.collective.motiondecode_repair_report.v1":
        raise ValueError("MotionDecode football prior requires a v1 repair report")
    if (
        repair.get("registration_hash") != registration.registration_hash
        or repair.get("source_manifest_hash") != registration.manifest.manifest_hash
        or repair.get("license_decision") != "permitted"
        or repair.get("hardware_authorized") is not False
        or repair.get("training_eligible") is not False
    ):
        raise ValueError("MotionDecode repair lineage or safety boundary is invalid")

    joint_limits, dataset_body_hash, target_model_hash = load_g1_joint_contract(target_model_path)
    if (
        repair.get("target_body_hash") != dataset_body_hash
        or repair.get("target_model_file_hash") != target_model_hash
    ):
        raise ValueError("MotionDecode repair report does not match the target model")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.joint_names != G1_DDS_JOINT_NAMES:
        raise ValueError("qualified football body has an incompatible joint contract")

    parent = _bounded_json(parent_evidence_path)
    trajectory = parent_trajectory_path.expanduser().resolve(strict=True)
    parent_trajectory_hash = hash_bytes(trajectory.read_bytes())
    if (
        parent.get("strict_replay") is not True
        or parent.get("body_hash") != qualification.body_hash
        or parent.get("trajectory_hash") != parent_trajectory_hash
        or parent.get("hardware_command_sent") is not False
        or parent.get("activation_ceiling") != "SIM_ONLY"
    ):
        raise ValueError("parent football evidence is not a strict SIM_ONLY replay")
    parent_reference = _parent_reference(trajectory)

    verified = verify_registered_files(registration, dataset_root)
    records = {
        record.relative_path: record
        for record in registration.manifest.files
        if _SHOOTING_TOKEN in f"/{record.relative_path}"
    }
    raw_results = repair.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("MotionDecode repair report results are missing")
    results = {
        str(item.get("relative_path", "")): item
        for item in raw_results
        if isinstance(item, dict)
        and item.get("q1_after") is True
        and _SHOOTING_TOKEN in f"/{item.get('relative_path', '')}"
    }
    if not results or not set(results).issubset(records):
        raise ValueError("MotionDecode repair report has no registered Q1 shooting clips")

    lower = np.asarray([joint_limits[name][0] for name in G1_DDS_JOINT_NAMES])
    upper = np.asarray([joint_limits[name][1] for name in G1_DDS_JOINT_NAMES])
    candidates: list[tuple[float, str, np.ndarray, G1FootballStyleEvent]] = []
    heldout: dict[str, str] = {}
    train_files = 0
    for relative, result in sorted(results.items()):
        record = records[relative]
        if _heldout(record.content_hash):
            heldout[relative] = record.content_hash
            continue
        train_files += 1
        episode = parse_motion_csv(
            verified[relative],
            source_manifest_hash=registration.manifest.manifest_hash,
            expected_file_hash=record.content_hash,
            target_body_hash=dataset_body_hash,
            sample_rate_hz=registration.manifest.sample_rate_hz,
        )
        stop = _retained_frame_count(result, int(episode.time.shape[0]))
        q = np.asarray(episode.joint_position[:stop], dtype=np.float64)
        root_position = np.asarray(episode.root_position[:stop], dtype=np.float64)
        root_quaternion = np.asarray(episode.root_quaternion[:stop], dtype=np.float64)
        if stop != episode.time.shape[0]:
            time = np.arange(stop, dtype=np.float64) / episode.sample_rate_hz
            velocity = np.gradient(q, time, axis=0, edge_order=2)
        else:
            velocity = np.asarray(episode.joint_velocity, dtype=np.float64)
        spans = clean_motiondecode_spans(
            episode,
            joint_lower=lower,
            joint_upper=upper,
            minimum_frames=140,
        )
        spans = tuple((start, min(end, stop)) for start, end in spans if start < stop)
        candidate = _best_episode_event(
            relative_path=relative,
            source_hash=record.content_hash,
            q=q,
            velocity=velocity,
            root_position=root_position,
            root_quaternion=root_quaternion,
            fps=episode.sample_rate_hz,
            spans=spans,
            parent_reference=parent_reference,
            model_path=target_model_path,
        )
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < selected_event_count:
        raise ValueError("too few Q1 MotionDecode shooting events survived style screening")
    selected = candidates[:selected_event_count]
    sequences = np.asarray([item[2] for item in selected], dtype=np.float64)
    reference = np.median(sequences, axis=0)
    iqr = np.quantile(sequences, 0.75, axis=0) - np.quantile(sequences, 0.25, axis=0)
    right_reference = reference[:, 6:12]
    right_iqr = iqr[:, 6:12]
    prior = G1FootballMotionPrior(
        body_hash=qualification.body_hash,
        dataset_readme_hash=_hash_file(dataset_root / "README.md"),
        split_manifest_hash=registration.manifest.manifest_hash,
        joint_order_contract_hash=target_model_hash,
        train_partition_hash=canonical_hash({item[1]: item[3].source_hash for item in selected}),
        heldout_partition_commitment=canonical_hash(heldout),
        joint_names=G1_DDS_JOINT_NAMES[6:12],
        reference_times_sec=_REFERENCE_TIMES_SEC,
        right_leg_reference_rad=tuple(
            tuple(float(value) for value in row) for row in right_reference
        ),
        right_leg_iqr_rad=tuple(tuple(float(value) for value in row) for row in right_iqr),
        selected_events=(),
        train_files_considered=train_files,
        qualified_event_count=len(candidates),
        whole_body_reference_rad=tuple(tuple(float(value) for value in row) for row in reference),
        whole_body_iqr_rad=tuple(tuple(float(value) for value in row) for row in iqr),
        whole_body_maximum_target_correction_rad=((0.30,) * 12 + (0.16,) * 3 + (0.12,) * 14),
        motiondecode_source_manifest_hash=registration.manifest.manifest_hash,
        motiondecode_repair_report_hash=repair_hash,
        parent_trajectory_hash=parent_trajectory_hash,
        style_events=tuple(item[3] for item in selected),
        source_dataset="MotionDecode",
        maximum_target_correction_rad=0.30,
        schema_version="rosclaw.growth.g1_football_motion_prior.v2",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(prior.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prior


def _best_episode_event(
    *,
    relative_path: str,
    source_hash: str,
    q: np.ndarray,
    velocity: np.ndarray,
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    fps: float,
    spans: tuple[tuple[int, int], ...],
    parent_reference: np.ndarray,
    model_path: Path,
) -> tuple[float, str, np.ndarray, G1FootballStyleEvent] | None:
    if not spans:
        return None
    right_position, left_position = _foot_positions(
        q=q,
        root_position=root_position,
        root_quaternion=root_quaternion,
        model_path=model_path,
    )
    right_speed = np.linalg.norm(np.gradient(right_position, 1.0 / fps, axis=0), axis=1)
    left_speed = np.linalg.norm(np.gradient(left_position, 1.0 / fps, axis=0), axis=1)
    right_leg_speed = np.linalg.norm(velocity[:, 6:12], axis=1)
    offsets = np.rint(np.asarray(_REFERENCE_TIMES_SEC) * fps).astype(int)
    weights = np.asarray((2.0,) * 12 + (1.5,) * 3 + (0.25,) * 14)
    options: list[tuple[float, int, np.ndarray, float, float, float]] = []
    for start, end in spans:
        lower = max(start, -int(offsets[0]) + 2)
        upper = min(end, q.shape[0] - int(offsets[-1]) - 2)
        if upper <= lower:
            continue
        threshold = float(np.quantile(right_speed[lower:upper], 0.80))
        for frame in range(lower + 1, upper - 1):
            if (
                right_speed[frame] < max(3.0, threshold)
                or right_leg_speed[frame] < 8.0
                or right_speed[frame] < left_speed[frame] * 1.05
                or right_speed[frame] < right_speed[frame - 1]
                or right_speed[frame] < right_speed[frame + 1]
            ):
                continue
            sequence = q[frame + offsets]
            style_distance = float(
                np.sqrt(np.mean(np.square(sequence - parent_reference) * weights[None, :]))
            )
            support = float(
                np.quantile(
                    left_speed[max(0, frame - round(0.15 * fps)) : frame + round(0.15 * fps) + 1],
                    0.95,
                )
            )
            post = velocity[frame + round(0.30 * fps) : min(q.shape[0], frame + round(0.60 * fps))]
            recovery = float(np.sqrt(np.mean(np.square(post)))) if post.size else math.inf
            selectivity = right_speed[frame] / max(left_speed[frame], 0.20)
            rank = (
                style_distance + 0.015 * support + 0.020 * recovery - 0.006 * min(selectivity, 4.0)
            )
            options.append((rank, frame, sequence.copy(), support, recovery, right_speed[frame]))
    if not options:
        return None
    rank, frame, sequence, support, recovery, peak = min(options, key=lambda item: item[0])
    score = max(0.0, 1.0 / (1.0 + rank))
    event = G1FootballStyleEvent(
        relative_path=relative_path,
        source_hash=source_hash,
        reference_frame=frame,
        frame_count=int(q.shape[0]),
        fps=fps,
        score=score,
        right_foot_peak_speed_mps=peak,
        support_foot_p95_speed_mps=support,
        post_event_joint_velocity_rms_rad_s=recovery,
    )
    return rank, relative_path, sequence, event


def _foot_positions(
    *,
    q: np.ndarray,
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path.expanduser().resolve()))
    data = mujoco.MjData(model)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    if min(pelvis, left, right) < 0:
        raise ValueError("target model lacks G1 pelvis or ankle bodies")
    qpos_addresses = tuple(
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in G1_DDS_JOINT_NAMES
    )
    output = [np.empty((q.shape[0], 3)), np.empty((q.shape[0], 3))]
    for frame in range(q.shape[0]):
        data.qpos[:3] = root_position[frame]
        data.qpos[3:7] = root_quaternion[frame]
        data.qpos[list(qpos_addresses)] = q[frame]
        mujoco.mj_forward(model, data)
        rotation = np.asarray(data.xmat[pelvis]).reshape(3, 3)
        origin = np.asarray(data.xpos[pelvis])
        output[0][frame] = rotation.T @ (np.asarray(data.xpos[right]) - origin)
        output[1][frame] = rotation.T @ (np.asarray(data.xpos[left]) - origin)
    return output[0], output[1]


def _parent_reference(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        required = {"time", "policy_action", "ball_contact_force_peak_n"}
        if not required.issubset(data.files):
            raise ValueError("parent football trajectory lacks action or contact evidence")
        time = np.asarray(data["time"], dtype=np.float64)
        action = np.asarray(data["policy_action"], dtype=np.float64)
        force = np.asarray(data["ball_contact_force_peak_n"], dtype=np.float64)
    if (
        time.ndim != 1
        or action.shape != (time.shape[0], 29)
        or force.shape != time.shape
        or not np.isfinite(time).all()
        or not np.isfinite(action).all()
        or not np.isfinite(force).all()
        or float(np.max(force)) <= 0.0
    ):
        raise ValueError("parent football trajectory arrays are invalid")
    contact = int(np.argmax(force))
    contact_time = time[contact]
    indices = [
        int(np.argmin(np.abs(time - (contact_time + offset)))) for offset in _REFERENCE_TIMES_SEC
    ]
    return action[indices]


def _retained_frame_count(result: dict[str, Any], original: int) -> int:
    manifest = result.get("repair_manifest")
    if manifest is None:
        if result.get("disposition") != "not_required_q1":
            raise ValueError("Q1 MotionDecode result has an invalid repair disposition")
        return original
    if not isinstance(manifest, dict) or result.get("disposition") != "repaired_q1":
        raise ValueError("MotionDecode repair manifest is invalid")
    retained = int(manifest.get("retained_frame_count", 0))
    if not 3 <= retained < original:
        raise ValueError("MotionDecode repair retained-frame count is invalid")
    return retained


def _heldout(source_hash: str) -> bool:
    return int(source_hash.removeprefix("sha256:")[:8], 16) % 5 == 0


def _external_output(path: Path, checkout: Path) -> Path:
    output = path.expanduser().resolve()
    source = checkout.expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("MotionDecode football prior output must be outside the checkout")
    if output.exists():
        raise FileExistsError("MotionDecode football prior output already exists")
    return output


def _bounded_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError("football prior JSON evidence is missing or too large")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("football prior JSON evidence must be an object")
    return value


def _hash_file(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=True)
    return hash_bytes(resolved.read_bytes())


__all__ = ["derive_motiondecode_g1_football_skill_prior"]
