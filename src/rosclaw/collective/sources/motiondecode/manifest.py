"""Inventory and provenance manifest for a local MotionDecode snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.collective.capsule import ExperienceCapsule, SourceDescriptor
from rosclaw.collective.sources.motiondecode.license import (
    MotionDecodeLicenseDecision,
    inspect_motiondecode_license,
)

_MAX_SOURCE_FILES = 2_000_000
_MAX_INDEX_BYTES = 16 * 1024 * 1024
_EXPECTED_REMOTE_PRIMARY = frozenset(
    {
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "1.5",
        "1.6",
        "1.7",
        "1.8",
        "1.9",
        "1.10",
        "1.11",
        "1.12",
        "1.13",
        "1.14",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "3.1",
        "3.2",
        "3.3",
        "4",
        "5",
    }
)


@dataclass(frozen=True)
class MotionDecodeSourceManifest:
    source: SourceDescriptor
    license: MotionDecodeLicenseDecision
    csv_file_count: int
    csv_total_bytes: int
    local_primary_categories: tuple[str, ...]
    indexed_primary_categories: tuple[str, ...]
    absent_indexed_categories: tuple[str, ...]
    expected_remote_categories_absent: tuple[str, ...]
    football_files: int
    object_pose_files: int
    video_files: int
    semantic_label_files: int
    local_snapshot_complete: bool
    safe_to_read: bool
    capsule: ExperienceCapsule
    warnings: tuple[str, ...]
    schema_version: str = "rosclaw.collective.motiondecode_source_manifest.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_motiondecode_source(
    dataset_root: Path,
    *,
    revision: str,
    requested_usage: str = "research",
) -> tuple[MotionDecodeSourceManifest, tuple[Path, ...]]:
    """Inspect without loading motion payloads or trusting README claims."""

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("MotionDecode revision must be a full git commit")
    root = dataset_root.expanduser().resolve()
    samples = root / "samples"
    if not root.is_dir() or not samples.is_dir() or samples.is_symlink():
        raise ValueError("MotionDecode root must contain a non-symlink samples directory")
    license_decision = inspect_motiondecode_license(root, requested_usage=requested_usage)
    if not license_decision.permitted:
        raise ValueError("requested MotionDecode usage is not permitted by the local terms")

    paths: list[Path] = []
    total_bytes = 0
    other_counts = {"object": 0, "video": 0, "labels": 0}
    inventory = hashlib.sha256()
    for directory, names, files in os.walk(samples, followlinks=False):
        directory_path = Path(directory)
        names[:] = sorted(
            name for name in names if not (directory_path / name).is_symlink()
        )
        for name in sorted(files):
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"unsafe MotionDecode payload: {path.relative_to(root)}")
            relative = path.relative_to(root)
            suffix = path.suffix.lower()
            if suffix == ".csv":
                paths.append(relative)
                size = path.stat().st_size
                total_bytes += size
                inventory.update(relative.as_posix().encode())
                inventory.update(b"\0")
                inventory.update(str(size).encode())
                inventory.update(b"\n")
            elif suffix in {".mp4", ".mov", ".mkv"}:
                other_counts["video"] += 1
            elif suffix in {".json", ".jsonl"}:
                other_counts["labels"] += 1
            if suffix == ".csv" and _looks_like_object_pose(relative):
                other_counts["object"] += 1
            if len(paths) > _MAX_SOURCE_FILES:
                raise ValueError("MotionDecode source exceeds the bounded file-count ceiling")
    if not paths:
        raise ValueError("MotionDecode source contains no CSV payloads")
    paths.sort(key=Path.as_posix)
    inventory_hash = "sha256:" + inventory.hexdigest()

    # Preserve numeric labels exactly (1.10 must not collapse to 1.1).
    local_primary = tuple(sorted({_category_label(path.parts[1]) for path in paths}))
    indexed = _indexed_primary_categories(root / "metadata" / "index.csv")
    absent_indexed = tuple(sorted(set(indexed) - set(local_primary)))
    expected_absent = tuple(sorted(_EXPECTED_REMOTE_PRIMARY - set(local_primary)))
    football_files = sum(
        "football" in path.as_posix().lower() or "ball_game" in path.as_posix().lower()
        for path in paths
    )
    warnings = list(license_decision.warnings)
    if absent_indexed:
        warnings.append("local samples omit categories declared by metadata/index.csv")
    if football_files == 0:
        warnings.append("no local Football or Ball_Game CSV is available")
    if not other_counts["object"]:
        warnings.append("no synchronized object-pose CSV was identified")
    if not other_counts["video"]:
        warnings.append("no local multi-view video payload was identified")
    if not other_counts["labels"]:
        warnings.append("no local per-episode semantic label payload was identified")
    source = SourceDescriptor(
        provider="ChingMu",
        dataset="CMRobot/MotionDecode",
        revision=revision,
        inventory_hash=inventory_hash,
        license_hash=license_decision.license_hash,
        attribution="Motion data from ChingMu (CMRobot/MotionDecode)",
        # A ModelScope/HF materialized folder has no local commit binding.  A
        # remote HEAD observation must never be represented as content proof.
        revision_binding="UNVERIFIED_LOCAL_SNAPSHOT",
    )
    capsule = ExperienceCapsule(
        source=source,
        source_body="human_optical_mocap_retargeted_to_unitree_g1",
        target_body="unitree_g1_29dof",
        target_body_mapping="header_exact_joint_name_mapping_pending_body_hash",
        task_semantics=tuple(sorted({path.parts[-2] for path in paths})),
        observation_semantics=(
            "root_position_m",
            "root_quaternion_wxyz",
            "joint_position_rad",
            "implicit_time_120hz",
        ),
        action_semantics=(),
        modalities=("kinematic_trajectory",),
        quality="UNVERIFIED_LOCAL_SNAPSHOT",
        truth_level="T4",
        applicability="MOTION_REFERENCE_ONLY_PENDING_KINEMATIC_AUDIT",
        training_eligible=False,
    )
    complete = not expected_absent and not absent_indexed
    return (
        MotionDecodeSourceManifest(
            source=source,
            license=license_decision,
            csv_file_count=len(paths),
            csv_total_bytes=total_bytes,
            local_primary_categories=local_primary,
            indexed_primary_categories=indexed,
            absent_indexed_categories=absent_indexed,
            expected_remote_categories_absent=expected_absent,
            football_files=football_files,
            object_pose_files=other_counts["object"],
            video_files=other_counts["video"],
            semantic_label_files=other_counts["labels"],
            local_snapshot_complete=complete,
            safe_to_read=True,
            capsule=capsule,
            warnings=tuple(warnings),
        ),
        tuple(paths),
    )


def manifest_hash(manifest: MotionDecodeSourceManifest) -> str:
    payload = json.dumps(
        manifest.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _category_label(directory_name: str) -> str:
    match = re.match(r"^(\d+(?:\.\d+)?)\.", directory_name)
    return match.group(1) if match else directory_name


def _looks_like_object_pose(relative_path: Path) -> bool:
    """Avoid confusing an action *about* an object with an object-pose stream."""

    value = relative_path.as_posix().lower()
    return any(token in value for token in ("object_pose", "object_6d", "6dop", "6d_pose"))


def _indexed_primary_categories(index_path: Path) -> tuple[str, ...]:
    if not index_path.is_file() or index_path.is_symlink():
        raise ValueError("MotionDecode metadata/index.csv is missing or unsafe")
    if not 1 <= index_path.stat().st_size <= _MAX_INDEX_BYTES:
        raise ValueError("MotionDecode metadata/index.csv is empty or too large")
    labels: set[str] = set()
    with index_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Label" not in reader.fieldnames:
            raise ValueError("MotionDecode index does not contain the Label column")
        for row in reader:
            label = (row.get("Label") or "").strip()
            match = re.match(r"^(\d+(?:\.\d+)?)", label)
            if match:
                value = match.group(1)
                parts = value.split(".")
                labels.add(".".join(parts[:2]) if len(parts) > 1 else parts[0])
    if not labels:
        raise ValueError("MotionDecode index contains no parseable labels")
    return tuple(sorted(labels))
