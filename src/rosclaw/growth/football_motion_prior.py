"""Dataset-qualified, support-bounded G1 football motion prior.

The artifact produced here is deliberately not a motor policy.  It distils
right-foot contact coordination from the *training* partition of
OmniContact into a short joint-position reference.  Runtime use is SIM_ONLY,
is blended into an already qualified kick policy, and remains bounded by a
per-joint correction limit.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.backends.unitree_mujoco_backend import qualify_g1_assets
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES

_SOCCER_CASES = (
    "case1_kick_forward",
    "case2_kick_left",
    "case3_kick_right",
)
_ISAACLAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
_RIGHT_LEG_NAMES = G1_DDS_JOINT_NAMES[6:12]
_RIGHT_LEG_ISAAC_INDICES = tuple(_ISAACLAB_JOINT_NAMES.index(name) for name in _RIGHT_LEG_NAMES)
_REFERENCE_TIMES_SEC = (-0.18, -0.12, -0.06, 0.0, 0.06, 0.12)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _literal_assignment(path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        if node.value is None:
            break
        value = ast.literal_eval(node.value)
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
            break
        return tuple(value)
    raise ValueError(f"joint-order contract does not define literal {name}")


def _contact_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    runs: list[tuple[int, int]] = []
    for start_value in starts:
        start = int(start_value)
        end = start
        while end + 1 < mask.shape[0] and bool(mask[end + 1]):
            end += 1
        runs.append((start, end))
    return tuple(runs)


@dataclass(frozen=True)
class G1FootballMotionEvent:
    relative_path: str
    source_hash: str
    capture_id: str
    contact_start_frame: int
    contact_end_frame: int
    reference_contact_frame: int
    fps: float
    score: float
    outgoing_planar_speed_mps: float
    outgoing_vertical_speed_mps: float
    vertical_speed_delta_mps: float
    right_foot_peak_speed_mps: float

    def __post_init__(self) -> None:
        if not self.relative_path or not self.capture_id:
            raise ValueError("football motion event identity must not be empty")
        if not self.source_hash.startswith("sha256:"):
            raise ValueError("football motion event requires a SHA-256 source hash")
        if not (
            0 <= self.contact_start_frame <= self.reference_contact_frame
            and self.contact_start_frame <= self.contact_end_frame
        ):
            raise ValueError("football motion event frame order is invalid")
        values = (
            self.fps,
            self.score,
            self.outgoing_planar_speed_mps,
            self.outgoing_vertical_speed_mps,
            self.vertical_speed_delta_mps,
            self.right_foot_peak_speed_mps,
        )
        if not all(math.isfinite(value) for value in values) or self.fps <= 0.0:
            raise ValueError("football motion event metrics must be finite")


@dataclass(frozen=True)
class G1FootballStyleEvent:
    """One Q1 MotionDecode shooting event used as a kinematic style teacher."""

    relative_path: str
    source_hash: str
    reference_frame: int
    frame_count: int
    fps: float
    score: float
    right_foot_peak_speed_mps: float
    support_foot_p95_speed_mps: float
    post_event_joint_velocity_rms_rad_s: float

    def __post_init__(self) -> None:
        if not self.relative_path or not self.source_hash.startswith("sha256:"):
            raise ValueError("football style event identity is invalid")
        if not 0 <= self.reference_frame < self.frame_count or self.frame_count < 3:
            raise ValueError("football style event frame contract is invalid")
        values = (
            self.fps,
            self.score,
            self.right_foot_peak_speed_mps,
            self.support_foot_p95_speed_mps,
            self.post_event_joint_velocity_rms_rad_s,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("football style event metrics must be finite and non-negative")
        if self.fps <= 0.0:
            raise ValueError("football style event FPS must be positive")


@dataclass(frozen=True)
class G1FootballMotionPrior:
    body_hash: str
    dataset_readme_hash: str
    split_manifest_hash: str
    joint_order_contract_hash: str
    train_partition_hash: str
    heldout_partition_commitment: str
    joint_names: tuple[str, ...]
    reference_times_sec: tuple[float, ...]
    right_leg_reference_rad: tuple[tuple[float, ...], ...]
    right_leg_iqr_rad: tuple[tuple[float, ...], ...]
    selected_events: tuple[G1FootballMotionEvent, ...]
    train_files_considered: int
    qualified_event_count: int
    whole_body_reference_rad: tuple[tuple[float, ...], ...] = ()
    whole_body_iqr_rad: tuple[tuple[float, ...], ...] = ()
    whole_body_maximum_target_correction_rad: tuple[float, ...] = ()
    motiondecode_source_manifest_hash: str | None = None
    motiondecode_repair_report_hash: str | None = None
    parent_trajectory_hash: str | None = None
    style_events: tuple[G1FootballStyleEvent, ...] = ()
    source_dataset: str = "OmniContact"
    maximum_target_correction_rad: float = 0.45
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.g1_football_motion_prior.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("body_hash", self.body_hash),
            ("dataset_readme_hash", self.dataset_readme_hash),
            ("split_manifest_hash", self.split_manifest_hash),
            ("joint_order_contract_hash", self.joint_order_contract_hash),
            ("train_partition_hash", self.train_partition_hash),
            ("heldout_partition_commitment", self.heldout_partition_commitment),
        ):
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"{label} must be a sha256: content hash")
        if self.joint_names != _RIGHT_LEG_NAMES:
            raise ValueError("football motion prior must use canonical G1 right-leg order")
        if self.reference_times_sec != tuple(sorted(self.reference_times_sec)):
            raise ValueError("football motion prior times must be sorted")
        if len(self.reference_times_sec) < 3 or 0.0 not in self.reference_times_sec:
            raise ValueError("football motion prior must span a contact-centred sequence")
        expected = (len(self.reference_times_sec), len(self.joint_names))
        for label, values in (
            ("reference", self.right_leg_reference_rad),
            ("iqr", self.right_leg_iqr_rad),
        ):
            array = np.asarray(values, dtype=np.float64)
            if array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f"football motion prior {label} has invalid shape or values")
            if label == "iqr" and np.any(array < 0.0):
                raise ValueError("football motion prior IQR must be non-negative")
        if not 0.05 <= self.maximum_target_correction_rad <= 0.60:
            raise ValueError("football motion prior correction must be in [0.05, 0.60] rad")
        if self.train_files_considered <= 0 or self.qualified_event_count < len(
            self.selected_events
        ):
            raise ValueError("football motion prior training counts are inconsistent")
        if (
            self.schema_version == "rosclaw.growth.g1_football_motion_prior.v1"
            and not self.selected_events
        ):
            raise ValueError("football motion prior requires selected training events")
        if self.schema_version == "rosclaw.growth.g1_football_motion_prior.v1":
            if (
                self.whole_body_reference_rad
                or self.whole_body_iqr_rad
                or self.whole_body_maximum_target_correction_rad
                or self.motiondecode_source_manifest_hash is not None
                or self.motiondecode_repair_report_hash is not None
                or self.parent_trajectory_hash is not None
                or self.style_events
                or self.source_dataset != "OmniContact"
            ):
                raise ValueError("football motion prior v1 cannot contain a whole-body style")
        elif self.schema_version == "rosclaw.growth.g1_football_motion_prior.v2":
            expected_whole_body = (len(self.reference_times_sec), len(G1_DDS_JOINT_NAMES))
            for label, values in (
                ("whole-body reference", self.whole_body_reference_rad),
                ("whole-body IQR", self.whole_body_iqr_rad),
            ):
                array = np.asarray(values, dtype=np.float64)
                if array.shape != expected_whole_body or not np.isfinite(array).all():
                    raise ValueError(f"football motion prior {label} is invalid")
                if label.endswith("IQR") and np.any(array < 0.0):
                    raise ValueError("football motion prior whole-body IQR must be non-negative")
            correction = np.asarray(
                self.whole_body_maximum_target_correction_rad,
                dtype=np.float64,
            )
            if (
                correction.shape != (len(G1_DDS_JOINT_NAMES),)
                or not np.isfinite(correction).all()
                or np.any(correction < 0.02)
                or np.any(correction > 0.45)
            ):
                raise ValueError("football motion prior whole-body bounds are invalid")
            for label, artifact_hash in (
                ("motiondecode_source_manifest_hash", self.motiondecode_source_manifest_hash),
                ("motiondecode_repair_report_hash", self.motiondecode_repair_report_hash),
                ("parent_trajectory_hash", self.parent_trajectory_hash),
            ):
                if (
                    artifact_hash is None
                    or not artifact_hash.startswith("sha256:")
                    or len(artifact_hash) != 71
                ):
                    raise ValueError(f"{label} must bind a sha256: evidence artifact")
            if self.source_dataset != "MotionDecode" or not self.style_events:
                raise ValueError("football motion prior v2 requires MotionDecode style events")
        else:
            raise ValueError("unsupported football motion prior schema")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
        ):
            raise ValueError("football motion prior must remain SIM_ONLY and non-promotable")

    @property
    def prior_hash(self) -> str:
        return canonical_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        payload = {
            **asdict(self),
            "selected_events": [asdict(event) for event in self.selected_events],
            "heldout_metrics_accessed": False,
            "direct_torque_output": False,
            "training_partition_only": True,
        }
        if self.schema_version == "rosclaw.growth.g1_football_motion_prior.v1":
            for field in (
                "whole_body_reference_rad",
                "whole_body_iqr_rad",
                "whole_body_maximum_target_correction_rad",
                "motiondecode_source_manifest_hash",
                "motiondecode_repair_report_hash",
                "parent_trajectory_hash",
                "style_events",
                "source_dataset",
            ):
                payload.pop(field, None)
        else:
            payload["style_events"] = [asdict(event) for event in self.style_events]
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "prior_hash": self.prior_hash}


def load_g1_football_motion_prior(path: Path) -> G1FootballMotionPrior:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    declared_hash = str(payload.pop("prior_hash", ""))
    payload.pop("heldout_metrics_accessed", None)
    payload.pop("direct_torque_output", None)
    payload.pop("training_partition_only", None)
    payload["joint_names"] = tuple(payload["joint_names"])
    payload["reference_times_sec"] = tuple(payload["reference_times_sec"])
    payload["right_leg_reference_rad"] = tuple(
        tuple(row) for row in payload["right_leg_reference_rad"]
    )
    payload["right_leg_iqr_rad"] = tuple(tuple(row) for row in payload["right_leg_iqr_rad"])
    payload["selected_events"] = tuple(
        G1FootballMotionEvent(**event) for event in payload["selected_events"]
    )
    if "whole_body_reference_rad" in payload:
        payload["whole_body_reference_rad"] = tuple(
            tuple(row) for row in payload["whole_body_reference_rad"]
        )
        payload["whole_body_iqr_rad"] = tuple(tuple(row) for row in payload["whole_body_iqr_rad"])
        payload["whole_body_maximum_target_correction_rad"] = tuple(
            payload["whole_body_maximum_target_correction_rad"]
        )
        payload["style_events"] = tuple(
            G1FootballStyleEvent(**event) for event in payload["style_events"]
        )
    prior = G1FootballMotionPrior(**payload)
    if declared_hash != prior.prior_hash:
        raise ValueError("football motion prior hash mismatch")
    return prior


def blend_g1_football_motion_prior_target(
    *,
    target: np.ndarray,
    prior: G1FootballMotionPrior,
    policy_frame: int,
    contact_policy_frame: int,
    control_dt_sec: float,
    blend: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Blend a data prior into the right leg without bypassing the PD loop."""

    if target.shape != (29,) or not np.isfinite(target).all():
        raise ValueError("football motion prior target must contain 29 finite joints")
    if not 0.0 <= blend <= 0.50 or not math.isfinite(blend):
        raise ValueError("football motion prior blend must be in [0, 0.50]")
    if control_dt_sec <= 0.0 or not math.isfinite(control_dt_sec):
        raise ValueError("football motion prior control clock must be positive")
    delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    relative_time = (policy_frame - contact_policy_frame) * control_dt_sec
    times = np.asarray(prior.reference_times_sec, dtype=np.float64)
    if blend == 0.0 or relative_time < times[0] or relative_time > times[-1]:
        return target.copy(), delta, False
    whole_body = bool(prior.whole_body_reference_rad)
    reference = np.asarray(
        prior.whole_body_reference_rad if whole_body else prior.right_leg_reference_rad,
        dtype=np.float64,
    )
    desired = np.asarray(
        [
            np.interp(relative_time, times, reference[:, joint])
            for joint in range(reference.shape[1])
        ],
        dtype=np.float64,
    )
    progress = (relative_time - times[0]) / max(times[-1] - times[0], 1e-9)
    envelope = math.sin(math.pi * min(1.0, max(0.0, progress))) ** 2
    indices = np.arange(29) if whole_body else np.arange(6, 12)
    maximum = (
        np.asarray(prior.whole_body_maximum_target_correction_rad, dtype=np.float64)
        if whole_body
        else prior.maximum_target_correction_rad
    )
    bounded = np.clip(
        desired - target[indices],
        -maximum,
        maximum,
    )
    delta[indices] = blend * envelope * bounded
    adapted = target.astype(np.float64, copy=True) + delta
    return adapted, delta, bool(np.any(np.abs(delta) > 1e-12))


def derive_g1_football_motion_prior(
    *,
    omnicontact_root: Path,
    joint_order_contract: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    selected_event_count: int = 24,
) -> G1FootballMotionPrior:
    """Distil high, fast right-foot contacts from train data only."""

    root = omnicontact_root.expanduser().resolve()
    contract = joint_order_contract.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("football motion prior output must be outside the source checkout")
    if output.exists():
        raise ValueError("football motion prior output already exists")
    if not 8 <= selected_event_count <= 64:
        raise ValueError("selected football event count must be in [8, 64]")
    readme = root / "README.md"
    splits_path = root / "metadata" / "splits.csv"
    if not readme.is_file() or not splits_path.is_file() or not contract.is_file():
        raise ValueError("OmniContact README, splits, and joint contract are required")
    if _literal_assignment(contract, "ISAACLAB_JOINT_NAMES") != _ISAACLAB_JOINT_NAMES:
        raise ValueError("joint-order contract has an incompatible IsaacLab G1 order")
    if _literal_assignment(contract, "MUJOCO_JOINT_NAMES") != G1_DDS_JOINT_NAMES:
        raise ValueError("joint-order contract has an incompatible MuJoCo G1 order")
    split_rows = tuple(csv.DictReader(splits_path.open(encoding="utf-8", newline="")))
    split_by_capture = {row["capture_id"]: row["split"] for row in split_rows}
    if set(split_by_capture.values()) != {"train", "val", "test"}:
        raise ValueError("OmniContact split manifest must contain train/val/test")
    paths = tuple(
        sorted(
            path
            for case in _SOCCER_CASES
            for path in (root / "npz" / "soccer" / case).glob("*_with_contact.npz")
        )
    )
    routed: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for path in paths:
        capture = path.stem.removesuffix("_with_contact")
        split = split_by_capture.get(capture)
        if split not in routed:
            raise ValueError(f"soccer capture {capture} is absent from the split manifest")
        routed[split].append(path)
    heldout_commitment = canonical_hash(
        {
            split: [str(path.relative_to(root)) for path in routed[split]]
            for split in ("val", "test")
        }
    )
    candidates: list[tuple[float, Path, int, int, int, dict[str, float]]] = []
    for path in routed["train"]:
        per_capture: list[tuple[float, Path, int, int, int, dict[str, float]]] = []
        with np.load(path, allow_pickle=False) as data:
            required = {
                "fps",
                "joint_pos",
                "object_lin_vel_w",
                "ee_pos_w",
                "contact_info",
            }
            if not required.issubset(data.files):
                raise ValueError(f"OmniContact training clip is missing fields: {path}")
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            joints = np.asarray(data["joint_pos"], dtype=np.float64)
            object_velocity = np.asarray(data["object_lin_vel_w"], dtype=np.float64)
            ee_position = np.asarray(data["ee_pos_w"], dtype=np.float64)
            contact = np.asarray(data["contact_info"])
            frame_count = joints.shape[0]
            if not 30.0 <= fps <= 240.0:
                raise ValueError(f"OmniContact clip FPS is outside [30, 240]: {path}")
            if (
                joints.shape != (frame_count, 29)
                or object_velocity.shape != (frame_count, 3)
                or ee_position.shape != (frame_count, 5, 3)
                or contact.shape != (frame_count, 4, 1)
                or not np.isfinite(joints).all()
                or not np.isfinite(object_velocity).all()
                or not np.isfinite(ee_position).all()
            ):
                raise ValueError(f"OmniContact clip has invalid shapes or non-finite data: {path}")
            right_contact = contact[:, 1, 0].astype(bool)
            for start, end in _contact_runs(right_contact):
                if (end - start + 1) / fps > 0.25:
                    continue
                pre = max(0, start - round(0.12 * fps))
                post = min(frame_count, end + 1 + round(0.15 * fps))
                before = (
                    np.mean(object_velocity[pre:start], axis=0)
                    if start > pre
                    else object_velocity[start]
                )
                outgoing = object_velocity[start:post]
                deltas = outgoing - before
                frame_scores = 2.0 * deltas[:, 2] + 0.35 * np.linalg.norm(deltas[:, :2], axis=1)
                offset = int(np.argmax(frame_scores))
                reference_frame = start + offset
                after = object_velocity[reference_frame]
                velocity_delta = after - before
                foot_lo = max(1, start - round(0.08 * fps))
                foot_hi = min(frame_count - 1, end + round(0.04 * fps))
                foot_velocity = (
                    ee_position[foot_lo + 1 : foot_hi + 1, 3]
                    - ee_position[foot_lo - 1 : foot_hi - 1, 3]
                ) * (fps / 2.0)
                foot_peak = (
                    float(np.max(np.linalg.norm(foot_velocity, axis=1)))
                    if foot_velocity.size
                    else 0.0
                )
                planar = float(np.linalg.norm(after[:2]))
                vertical_delta = float(velocity_delta[2])
                if not (
                    vertical_delta >= 0.30
                    and planar >= 1.50
                    and float(after[2]) >= 0.20
                    and foot_peak >= 0.50
                ):
                    continue
                score = float(
                    2.0 * vertical_delta
                    + 0.35 * np.linalg.norm(velocity_delta[:2])
                    + 0.15 * foot_peak
                )
                per_capture.append(
                    (
                        score,
                        path,
                        start,
                        end,
                        reference_frame,
                        {
                            "fps": fps,
                            "planar": planar,
                            "vertical": float(after[2]),
                            "vertical_delta": vertical_delta,
                            "foot_peak": foot_peak,
                        },
                    )
                )
        per_capture.sort(key=lambda item: item[0], reverse=True)
        candidates.extend(per_capture[:2])
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) < selected_event_count:
        raise ValueError("OmniContact train split has too few qualified right-foot events")
    selected = tuple(candidates[:selected_event_count])
    sequences: list[np.ndarray] = []
    events: list[G1FootballMotionEvent] = []
    for score, path, start, end, contact_frame, metrics in selected:
        with np.load(path, allow_pickle=False) as data:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            joints = np.asarray(data["joint_pos"], dtype=np.float64)
            indices = np.clip(
                np.rint(contact_frame + np.asarray(_REFERENCE_TIMES_SEC) * fps).astype(int),
                0,
                joints.shape[0] - 1,
            )
            sequences.append(joints[indices][:, _RIGHT_LEG_ISAAC_INDICES])
        capture = path.stem.removesuffix("_with_contact")
        events.append(
            G1FootballMotionEvent(
                relative_path=str(path.relative_to(root)),
                source_hash=_hash_file(path),
                capture_id=capture,
                contact_start_frame=start,
                contact_end_frame=end,
                reference_contact_frame=contact_frame,
                fps=metrics["fps"],
                score=score,
                outgoing_planar_speed_mps=metrics["planar"],
                outgoing_vertical_speed_mps=metrics["vertical"],
                vertical_speed_delta_mps=metrics["vertical_delta"],
                right_foot_peak_speed_mps=metrics["foot_peak"],
            )
        )
    array = np.asarray(sequences, dtype=np.float64)
    reference = np.median(array, axis=0)
    iqr = np.quantile(array, 0.75, axis=0) - np.quantile(array, 0.25, axis=0)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    prior = G1FootballMotionPrior(
        body_hash=qualification.body_hash,
        dataset_readme_hash=_hash_file(readme),
        split_manifest_hash=_hash_file(splits_path),
        joint_order_contract_hash=_hash_file(contract),
        train_partition_hash=canonical_hash(
            {
                event.relative_path: event.source_hash
                for event in sorted(events, key=lambda event: event.relative_path)
            }
        ),
        heldout_partition_commitment=heldout_commitment,
        joint_names=_RIGHT_LEG_NAMES,
        reference_times_sec=_REFERENCE_TIMES_SEC,
        right_leg_reference_rad=tuple(tuple(float(value) for value in row) for row in reference),
        right_leg_iqr_rad=tuple(tuple(float(value) for value in row) for row in iqr),
        selected_events=tuple(events),
        train_files_considered=len(routed["train"]),
        qualified_event_count=len(candidates),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(prior.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prior


__all__ = [
    "G1FootballMotionEvent",
    "G1FootballMotionPrior",
    "G1FootballStyleEvent",
    "blend_g1_football_motion_prior_target",
    "derive_g1_football_motion_prior",
    "load_g1_football_motion_prior",
]
