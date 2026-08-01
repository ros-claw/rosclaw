"""Kinematic and MuJoCo-geometry qualification for MotionDecode pilots."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.collective.sources.motiondecode.manifest import (
    MotionDecodeSourceManifest,
    inspect_motiondecode_source,
    manifest_hash,
)
from rosclaw.collective.sources.motiondecode.parser import (
    CanonicalMotionEpisode,
    parse_motiondecode_csv,
)
from rosclaw.collective.sources.motiondecode.taxonomy import (
    MotionDecodeStratum,
    select_motiondecode_pilot,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    hash_bytes,
    hash_json,
)


@dataclass(frozen=True)
class MotionDecodeEpisodeAudit:
    relative_path: str
    stratum: str
    source_hash: str | None
    frame_count: int
    duration_sec: float
    finite: bool
    quaternion_norm_error_max: float | None
    quaternion_sign_flip_count: int
    joint_limit_violation_fraction: float | None
    joint_limit_excess_max_rad: float | None
    joint_speed_p99_rad_s: float | None
    joint_speed_max_rad_s: float | None
    joint_acceleration_p99_rad_s2: float | None
    root_speed_max_m_s: float | None
    duplicate_frame_fraction: float | None
    clean_window_count: int
    clean_frame_fraction: float | None
    maximum_clean_window_frames: int
    mujoco_frames_checked: int
    mujoco_finite: bool
    contact_frame_fraction: float | None
    penetration_depth_max_m: float | None
    qualification: str
    training_eligible: bool
    repair_eligible: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    schema_version: str = "rosclaw.collective.motiondecode_episode_audit.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionDecodePilotReport:
    source_manifest: MotionDecodeSourceManifest
    source_manifest_hash: str
    body_hash: str
    kinematic_body_hash: str
    model_hash: str
    selection_seed: int
    requested_limit: int
    selection_requested: dict[str, int]
    selection_counts: dict[str, int]
    selection_shortages: dict[str, int]
    selection_substitutions: dict[str, int]
    episodes: tuple[MotionDecodeEpisodeAudit, ...]
    aggregates: dict[str, Any]
    decision: str
    blockers: tuple[str, ...]
    pipeline_failures: tuple[str, ...]
    evidence_domain: str = "SIM_ONLY"
    hardware_authorized: bool = False
    raw_data_exported: bool = False
    promotion_evidence_eligible: bool = False
    schema_version: str = "rosclaw.collective.motiondecode_pilot.v1"

    @property
    def pipeline_passed(self) -> bool:
        return not self.pipeline_failures and len(self.episodes) == self.requested_limit

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["pipeline_passed"] = self.pipeline_passed
        value["claims"] = {
            "football_contact_training": False,
            "direct_torque_labels": False,
            "reward_labels": False,
            "time_axis": "implicit_120hz",
            "coordinate_convention": "UNVERIFIED",
            "maximum_truth_level": "T4_SOURCE/T1_GEOMETRY_CHECK_ONLY",
            "candidate_promoted": False,
            "real_robot_evidence": False,
        }
        return value


@dataclass(frozen=True)
class _MuJoCoContract:
    model: Any
    data: Any
    qpos_addresses: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    robot_body_ids: frozenset[int]
    ground_geom_ids: frozenset[int]
    body_hash: str
    model_hash: str


def run_motiondecode_pilot(
    *,
    dataset_root: Path,
    revision: str,
    model_path: Path,
    asset_root: Path | None = None,
    output_dir: Path,
    source_checkout: Path,
    requested_usage: str = "research",
    limit: int = 400,
    seed: int = 20260801,
) -> MotionDecodePilotReport:
    """Run a deterministic pilot and persist only hashes and audit summaries."""

    root = _external_output(output_dir, source_checkout)
    root.mkdir(parents=True, exist_ok=False)
    manifest, relative_paths = inspect_motiondecode_source(
        dataset_root,
        revision=revision,
        requested_usage=requested_usage,
    )
    selection = select_motiondecode_pilot(relative_paths, limit=limit, seed=seed)
    contract = _load_mujoco_contract(model_path)
    execution_body_hash = contract.body_hash
    if asset_root is not None:
        from rosclaw.simforge.backends.unitree_mujoco_backend import qualify_g1_assets

        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        expected_scene = qualification.asset_root / "g1_description" / "scene_with_ball.xml"
        if model_path.expanduser().resolve() != expected_scene.resolve():
            raise ValueError("MotionDecode audit model is not the qualified execution scene")
        execution_body_hash = qualification.body_hash
    dataset = dataset_root.expanduser().resolve()
    episodes: list[MotionDecodeEpisodeAudit] = []
    pipeline_failures: list[str] = []
    for stratum, relative_path in selection.selected:
        episodes.append(
            _audit_path(
                dataset / relative_path,
                dataset_root=dataset,
                stratum=stratum,
                contract=contract,
            )
        )
    if len(selection.selected) != limit:
        pipeline_failures.append(
            f"pilot_selected={len(selection.selected)},requested={limit}"
        )
    aggregates = _aggregate(episodes)
    blockers: list[str] = []
    if manifest.source.revision_binding != "VERIFIED":
        blockers.append("local payload is not cryptographically bound to the stated remote revision")
    if manifest.football_files == 0:
        blockers.append("local snapshot has no football or ball-game CSV")
    if manifest.object_pose_files == 0:
        blockers.append("no synchronized ball/object pose exists for contact learning")
    if int(aggregates["q0_invalid_count"]) > 0:
        blockers.append("one or more pilot episodes failed the kinematic contract")
    if int(aggregates["training_eligible_count"]) < max(1, int(0.8 * len(episodes))):
        blockers.append("fewer than 80% of selected episodes are motion-prior eligible")
    decision = (
        "MOTION_PRIOR_ONLY"
        if not pipeline_failures and int(aggregates["training_eligible_count"]) > 0
        else "REJECTED"
    )
    report = MotionDecodePilotReport(
        source_manifest=manifest,
        source_manifest_hash=manifest_hash(manifest),
        body_hash=execution_body_hash,
        kinematic_body_hash=contract.body_hash,
        model_hash=contract.model_hash,
        selection_seed=seed,
        requested_limit=limit,
        selection_requested=dict(selection.requested),
        selection_counts=dict(selection.selected_counts),
        selection_shortages=dict(selection.shortages),
        selection_substitutions=dict(selection.substitutions),
        episodes=tuple(episodes),
        aggregates=aggregates,
        decision=decision,
        blockers=tuple(blockers),
        pipeline_failures=tuple(pipeline_failures),
    )
    _atomic_json(root / "source-manifest.json", manifest.to_dict())
    _atomic_json(root / "motiondecode-pilot-report.json", report.to_dict())
    _atomic_json(
        root / "experience-capsule.json",
        replace(
            manifest.capsule,
            target_body_mapping=(
                f"exact_29dof_names:{contract.body_hash};execution_body:{execution_body_hash}"
            ),
            quality=(
                "KINEMATIC_AUDITED_MOTION_PRIOR"
                if decision == "MOTION_PRIOR_ONLY"
                else "REJECTED"
            ),
            applicability="MOTION_PRIOR_ONLY_NO_FOOTBALL_CONTACT",
            training_eligible=decision == "MOTION_PRIOR_ONLY",
        ).to_dict(),
    )
    return report


def _audit_path(
    path: Path,
    *,
    dataset_root: Path,
    stratum: MotionDecodeStratum,
    contract: _MuJoCoContract,
) -> MotionDecodeEpisodeAudit:
    try:
        episode = parse_motiondecode_csv(path, dataset_root=dataset_root)
        return _audit_episode(episode, stratum=stratum, contract=contract)
    except (OSError, ValueError) as exc:
        return MotionDecodeEpisodeAudit(
            relative_path=path.relative_to(dataset_root).as_posix(),
            stratum=stratum.value,
            source_hash=None,
            frame_count=0,
            duration_sec=0.0,
            finite=False,
            quaternion_norm_error_max=None,
            quaternion_sign_flip_count=0,
            joint_limit_violation_fraction=None,
            joint_limit_excess_max_rad=None,
            joint_speed_p99_rad_s=None,
            joint_speed_max_rad_s=None,
            joint_acceleration_p99_rad_s2=None,
            root_speed_max_m_s=None,
            duplicate_frame_fraction=None,
            clean_window_count=0,
            clean_frame_fraction=None,
            maximum_clean_window_frames=0,
            mujoco_frames_checked=0,
            mujoco_finite=False,
            contact_frame_fraction=None,
            penetration_depth_max_m=None,
            qualification="Q0_INVALID",
            training_eligible=False,
            repair_eligible=False,
            errors=(f"{type(exc).__name__}:{exc}",),
            warnings=(),
        )


def _audit_episode(
    episode: CanonicalMotionEpisode,
    *,
    stratum: MotionDecodeStratum,
    contract: _MuJoCoContract,
) -> MotionDecodeEpisodeAudit:
    quaternion_norm = np.linalg.norm(episode.root_quaternion, axis=1)
    quaternion_error = float(np.max(np.abs(quaternion_norm - 1.0)))
    quaternion_dots = np.sum(
        episode.root_quaternion[1:] * episode.root_quaternion[:-1], axis=1
    )
    sign_flips = int(np.count_nonzero(quaternion_dots < 0.0))
    low_excess = np.maximum(contract.joint_lower - episode.joint_position, 0.0)
    high_excess = np.maximum(episode.joint_position - contract.joint_upper, 0.0)
    excess = np.maximum(low_excess, high_excess)
    violation_fraction = float(np.count_nonzero(excess > 1e-6) / excess.size)
    excess_max = float(np.max(excess))
    speed = np.abs(episode.joint_velocity)
    acceleration = np.abs(episode.joint_acceleration)
    root_velocity = np.gradient(episode.root_position, 1.0 / episode.sample_rate_hz, axis=0)
    root_speed = np.linalg.norm(root_velocity, axis=1)
    differences = np.max(np.abs(np.diff(episode.joint_position, axis=0)), axis=1)
    root_differences = np.max(np.abs(np.diff(episode.root_position, axis=0)), axis=1)
    duplicate_fraction = float(np.mean((differences < 1e-9) & (root_differences < 1e-9)))
    clean_spans = clean_motiondecode_spans(
        episode,
        joint_lower=contract.joint_lower,
        joint_upper=contract.joint_upper,
    )
    clean_frames = sum(stop - start for start, stop in clean_spans)
    clean_window_count = sum(max(0, 1 + (stop - start - 32) // 16) for start, stop in clean_spans)
    clean_fraction = clean_frames / episode.frame_count
    maximum_clean = max((stop - start for start, stop in clean_spans), default=0)
    geometry = _mujoco_geometry(episode, contract)
    errors: list[str] = []
    warnings: list[str] = []
    if quaternion_error > 0.10:
        errors.append("quaternion_norm_error_gt_0p10")
    elif quaternion_error > 0.02:
        warnings.append("quaternion_norm_error_gt_0p02")
    if violation_fraction > 0.01 or excess_max > 0.25:
        warnings.append("joint_limit_frames_excluded_from_clean_windows")
    elif violation_fraction > 0.0:
        warnings.append("minor_joint_limit_violation")
    joint_speed_max = float(np.max(speed))
    root_speed_max = float(np.max(root_speed))
    if joint_speed_max > 50.0:
        warnings.append("joint_speed_spikes_segmented")
    elif joint_speed_max > 25.0:
        warnings.append("joint_speed_gt_25_rad_s")
    if root_speed_max > 12.0:
        warnings.append("root_position_jumps_segmented")
    elif root_speed_max > 6.0:
        warnings.append("root_speed_gt_6_m_s")
    if duplicate_fraction > 0.95:
        warnings.append("mostly_duplicate_frames")
    if not geometry["finite"]:
        errors.append("mujoco_geometry_non_finite")
    penetration = float(geometry["penetration_depth_max_m"])
    if penetration > 0.08:
        warnings.append("root_height_alignment_required_gt_0p08_m")
    elif penetration > 0.03:
        warnings.append("mujoco_penetration_gt_0p03_m")
    if clean_window_count == 0:
        errors.append("no_continuous_32_frame_training_window")
    elif clean_fraction < 0.50:
        errors.append("clean_frame_fraction_lt_0p50")
    repair_eligible = bool(clean_window_count) and bool(geometry["finite"])
    training_eligible = not errors
    return MotionDecodeEpisodeAudit(
        relative_path=episode.relative_path.as_posix(),
        stratum=stratum.value,
        source_hash=episode.source_hash,
        frame_count=episode.frame_count,
        duration_sec=float(episode.time_sec[-1]),
        finite=True,
        quaternion_norm_error_max=quaternion_error,
        quaternion_sign_flip_count=sign_flips,
        joint_limit_violation_fraction=violation_fraction,
        joint_limit_excess_max_rad=excess_max,
        joint_speed_p99_rad_s=float(np.quantile(speed, 0.99)),
        joint_speed_max_rad_s=joint_speed_max,
        joint_acceleration_p99_rad_s2=float(np.quantile(acceleration, 0.99)),
        root_speed_max_m_s=root_speed_max,
        duplicate_frame_fraction=duplicate_fraction,
        clean_window_count=clean_window_count,
        clean_frame_fraction=clean_fraction,
        maximum_clean_window_frames=maximum_clean,
        mujoco_frames_checked=int(geometry["frames_checked"]),
        mujoco_finite=bool(geometry["finite"]),
        contact_frame_fraction=float(geometry["contact_frame_fraction"]),
        penetration_depth_max_m=penetration,
        qualification="Q1_KINEMATIC_ONLY" if training_eligible else "Q0_INVALID",
        training_eligible=training_eligible,
        repair_eligible=repair_eligible,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _load_mujoco_contract(model_path: Path) -> _MuJoCoContract:
    import mujoco

    path = model_path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("G1 MuJoCo model must be a regular non-symlink file")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in G1_DDS_JOINT_NAMES],
        dtype=np.int32,
    )
    if np.any(ids < 0):
        raise ValueError("G1 MuJoCo model does not contain all MotionDecode joints")
    names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(index))) for index in ids
    )
    if names != G1_DDS_JOINT_NAMES:
        raise ValueError("G1 MuJoCo joint order does not match MotionDecode")
    limited = model.jnt_limited[ids].astype(bool)
    if not np.all(limited):
        raise ValueError("G1 MuJoCo joint limits are incomplete")
    qpos = model.jnt_qposadr[ids].astype(np.int32)
    robot_body_ids = frozenset(_descendants(model, 1))
    ground_geom_ids = frozenset(
        index for index in range(model.ngeom) if int(model.geom_bodyid[index]) == 0
    )
    model_hash = hash_bytes(path.read_bytes())
    body_hash = hash_json(
        {
            "model_hash": model_hash,
            "joint_names": names,
            "joint_lower": model.jnt_range[ids, 0].tolist(),
            "joint_upper": model.jnt_range[ids, 1].tolist(),
        }
    )
    return _MuJoCoContract(
        model=model,
        data=data,
        qpos_addresses=qpos,
        joint_lower=np.asarray(model.jnt_range[ids, 0], dtype=np.float64),
        joint_upper=np.asarray(model.jnt_range[ids, 1], dtype=np.float64),
        robot_body_ids=robot_body_ids,
        ground_geom_ids=ground_geom_ids,
        body_hash=body_hash,
        model_hash=model_hash,
    )


def _mujoco_geometry(
    episode: CanonicalMotionEpisode,
    contract: _MuJoCoContract,
) -> dict[str, float | int | bool]:
    import mujoco

    frames = np.unique(np.linspace(0, episode.frame_count - 1, num=min(16, episode.frame_count), dtype=int))
    contact_frames = 0
    maximum_penetration = 0.0
    finite = True
    root_xy = episode.root_position[0, :2]
    for frame in frames:
        mujoco.mj_resetData(contract.model, contract.data)
        root = episode.root_position[frame].copy()
        root[:2] -= root_xy
        quaternion = episode.root_quaternion[frame]
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12 or not math.isfinite(norm):
            finite = False
            continue
        contract.data.qpos[:3] = root
        contract.data.qpos[3:7] = quaternion / norm
        contract.data.qpos[contract.qpos_addresses] = episode.joint_position[frame]
        mujoco.mj_forward(contract.model, contract.data)
        finite = finite and bool(
            np.all(np.isfinite(contract.data.qpos))
            and np.all(np.isfinite(contract.data.xpos))
        )
        grounded = False
        for contact_index in range(contract.data.ncon):
            contact = contract.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(contract.model.geom_bodyid[geom1])
            body2 = int(contract.model.geom_bodyid[geom2])
            ground_robot = (
                geom1 in contract.ground_geom_ids and body2 in contract.robot_body_ids
            ) or (geom2 in contract.ground_geom_ids and body1 in contract.robot_body_ids)
            if ground_robot:
                grounded = True
                maximum_penetration = max(maximum_penetration, max(0.0, -float(contact.dist)))
        contact_frames += int(grounded)
    return {
        "frames_checked": len(frames),
        "finite": finite,
        "contact_frame_fraction": contact_frames / len(frames),
        "penetration_depth_max_m": maximum_penetration,
    }


def _descendants(model: Any, root_body: int) -> tuple[int, ...]:
    values = []
    for body in range(model.nbody):
        current = body
        while current > 0 and current != root_body:
            current = int(model.body_parentid[current])
        if current == root_body:
            values.append(body)
    return tuple(values)


def _aggregate(episodes: list[MotionDecodeEpisodeAudit]) -> dict[str, Any]:
    eligible = [item for item in episodes if item.training_eligible]
    frame_counts = [item.frame_count for item in episodes if item.frame_count]
    durations = [item.duration_sec for item in episodes if item.frame_count]
    return {
        "episode_count": len(episodes),
        "training_eligible_count": len(eligible),
        "training_eligible_fraction": len(eligible) / max(1, len(episodes)),
        "q0_invalid_count": sum(item.qualification == "Q0_INVALID" for item in episodes),
        "q1_kinematic_only_count": sum(
            item.qualification == "Q1_KINEMATIC_ONLY" for item in episodes
        ),
        "q2_or_higher_count": 0,
        "total_frames": sum(frame_counts),
        "clean_training_window_count": sum(item.clean_window_count for item in episodes),
        "clean_frame_fraction_weighted": (
            sum(
                item.frame_count * float(item.clean_frame_fraction or 0.0)
                for item in episodes
            )
            / max(1, sum(frame_counts))
        ),
        "total_duration_sec": sum(durations),
        "frame_count_median": float(np.median(frame_counts)) if frame_counts else 0.0,
        "duration_sec_median": float(np.median(durations)) if durations else 0.0,
        "episodes_with_joint_limit_warning_or_error": sum(
            bool(item.joint_limit_violation_fraction) for item in episodes
        ),
        "episodes_with_mujoco_contact": sum(
            bool(item.contact_frame_fraction) for item in episodes
        ),
        "source_hash_commitment": hash_json(
            [item.source_hash for item in episodes if item.source_hash is not None]
        ),
        "qualification_ceiling": "Q1_KINEMATIC_ONLY",
    }


def clean_motiondecode_spans(
    episode: CanonicalMotionEpisode,
    *,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    minimum_frames: int = 32,
) -> tuple[tuple[int, int], ...]:
    """Return half-open, physically continuous spans for representation learning.

    This does not repair or relabel frames.  It excludes reset boundaries,
    non-unit quaternions, joint-limit excursions, and implausible 120 Hz
    derivatives, so a long clip can contribute clean windows without teaching
    the network its capture discontinuities.
    """

    if minimum_frames < 2:
        raise ValueError("clean MotionDecode span must contain at least two frames")
    quaternion_error = np.abs(np.linalg.norm(episode.root_quaternion, axis=1) - 1.0)
    in_limits = np.all(
        (episode.joint_position >= joint_lower - 1e-4)
        & (episode.joint_position <= joint_upper + 1e-4),
        axis=1,
    )
    frame_valid = (quaternion_error <= 0.02) & in_limits
    root_step_speed = (
        np.linalg.norm(np.diff(episode.root_position, axis=0), axis=1) * episode.sample_rate_hz
    )
    joint_step_speed = (
        np.max(np.abs(np.diff(episode.joint_position, axis=0)), axis=1)
        * episode.sample_rate_hz
    )
    edge_valid = (root_step_speed <= 8.0) & (joint_step_speed <= 25.0)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for frame in range(episode.frame_count):
        valid = bool(frame_valid[frame]) and (frame == 0 or bool(edge_valid[frame - 1]))
        if valid and start is None:
            start = frame
        if not valid and start is not None:
            if frame - start >= minimum_frames:
                spans.append((start, frame))
            start = None
    if start is not None and episode.frame_count - start >= minimum_frames:
        spans.append((start, episode.frame_count))
    return tuple(spans)


def _external_output(path: Path, source_checkout: Path) -> Path:
    target = path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if target == checkout or checkout in target.parents:
        raise ValueError("MotionDecode evidence output must be outside the source checkout")
    return target


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
