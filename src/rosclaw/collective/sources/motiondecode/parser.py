"""Bounded parser for the Unitree G1 MotionDecode CSV representation."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES

MOTIONDECODE_HZ = 120.0
MOTIONDECODE_HEADER = (
    "root_pos_x(m)",
    "root_pos_y(m)",
    "root_pos_z(m)",
    "root_rot_w",
    "root_rot_x",
    "root_rot_y",
    "root_rot_z",
    *(f"dof_{name}(rad)" for name in G1_DDS_JOINT_NAMES),
)
_MAX_CSV_BYTES = 128 * 1024 * 1024
_MAX_FRAMES = 120 * 60 * 20


@dataclass(frozen=True)
class CanonicalMotionEpisode:
    relative_path: Path
    source_hash: str
    time_sec: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_acceleration: np.ndarray
    sample_rate_hz: float = MOTIONDECODE_HZ
    time_semantics: str = "IMPLICIT_FIXED_RATE_FROM_DATA_CARD"
    action_semantics: str = "ABSENT"
    reward_semantics: str = "ABSENT"
    schema_version: str = "rosclaw.canonical_motion_episode.v1"

    @property
    def frame_count(self) -> int:
        return len(self.time_sec)


def parse_motiondecode_csv(
    path: Path,
    *,
    dataset_root: Path,
    max_bytes: int = _MAX_CSV_BYTES,
    max_frames: int = _MAX_FRAMES,
) -> CanonicalMotionEpisode:
    root = dataset_root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("MotionDecode CSV escapes the dataset root") from exc
    if path.is_symlink() or not resolved.is_file() or resolved.suffix.lower() != ".csv":
        raise ValueError("MotionDecode payload must be a regular non-symlink CSV")
    size = resolved.stat().st_size
    if not 1 <= size <= max_bytes:
        raise ValueError("MotionDecode CSV is empty or exceeds the byte ceiling")
    if not 2 <= max_frames <= _MAX_FRAMES:
        raise ValueError("MotionDecode frame ceiling is invalid")

    rows: list[list[float]] = []
    digest = hashlib.sha256()
    with resolved.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(chunk)
    with resolved.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError("MotionDecode CSV is empty") from exc
        if header != MOTIONDECODE_HEADER:
            raise ValueError("MotionDecode CSV header does not match the 29-DoF G1 contract")
        for frame_index, row in enumerate(reader):
            if frame_index >= max_frames:
                raise ValueError("MotionDecode CSV exceeds the frame ceiling")
            if len(row) != len(MOTIONDECODE_HEADER):
                raise ValueError(f"MotionDecode frame {frame_index} has the wrong column count")
            try:
                values = [float(item) for item in row]
            except ValueError as exc:
                raise ValueError(f"MotionDecode frame {frame_index} is not numeric") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"MotionDecode frame {frame_index} contains non-finite values")
            rows.append(values)
    if len(rows) < 2:
        raise ValueError("MotionDecode episode must contain at least two frames")
    value = np.asarray(rows, dtype=np.float64)
    time = np.arange(len(value), dtype=np.float64) / MOTIONDECODE_HZ
    joint_position = value[:, 7:]
    joint_velocity = np.gradient(joint_position, 1.0 / MOTIONDECODE_HZ, axis=0)
    joint_acceleration = np.gradient(joint_velocity, 1.0 / MOTIONDECODE_HZ, axis=0)
    return CanonicalMotionEpisode(
        relative_path=relative,
        source_hash="sha256:" + digest.hexdigest(),
        time_sec=time,
        root_position=value[:, :3],
        root_quaternion=value[:, 3:7],
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        joint_acceleration=joint_acceleration,
    )
