"""Immutable contracts emitted by the task-neutral dataset doctor."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


def _require_hash(label: str, value: str) -> None:
    if not _HASH.fullmatch(value):
        raise ValueError(f"{label} must be a sha256: content hash")


def _require_identifier(label: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a normalized identifier")


def _require_dataset_name(label: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or len(value) > 128
        or "/" in value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{label} must be a bounded dataset name")


def _unique_identifiers(
    values: tuple[str, ...],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(values)
    if not allow_empty and not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique")
    for value in normalized:
        _require_identifier(label, value)
    return normalized


def _safe_relative_path(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("relative_path must be a safe relative path")
    normalized = value.replace("\\", "/")
    if (
        not normalized
        or normalized.startswith("/")
        or ".." in normalized.split("/")
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError("relative_path must be a safe relative path")
    return normalized


def _bounded_strings(
    values: tuple[str, ...],
    *,
    label: str,
    maximum_length: int,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    normalized = tuple(values)
    if not allow_empty and not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique")
    if any(
        not isinstance(value, str)
        or not value
        or len(value) > maximum_length
        or any(ord(character) < 32 for character in value)
        for value in normalized
    ):
        raise ValueError(f"{label} contains an invalid bounded string")
    return normalized


def _count_mapping(values: Mapping[str, int], *, label: str) -> Mapping[str, int]:
    normalized: dict[str, int] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError(f"{label} keys must be bounded non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} values must be non-negative integers")
        normalized[key] = value
    return MappingProxyType(dict(sorted(normalized.items())))


class FileHashMode(StrEnum):
    NONE = "none"
    METADATA = "metadata"
    SHA256 = "sha256"


class DatasetSnapshotState(StrEnum):
    READY = "ready"
    EMPTY = "empty"
    PARTIAL = "partial"
    TRANSFERRING = "transferring"
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class DatasetSourceDescriptor:
    """Provenance and label vocabulary declared by a downstream source plugin."""

    source_id: str
    dataset_ids: tuple[str, ...]
    label_ids: tuple[str, ...]
    source_uri: str
    revision: str
    manifest_hash: str | None = None
    schema_version: str = "rosclaw.dataset.source_descriptor.v1"

    def __post_init__(self) -> None:
        _require_identifier("source_id", self.source_id)
        dataset_ids = tuple(self.dataset_ids)
        if not dataset_ids or len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("dataset_ids must contain unique bounded dataset names")
        for value in dataset_ids:
            _require_dataset_name("dataset_ids", value)
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(
            self,
            "label_ids",
            _unique_identifiers(self.label_ids, label="label_ids"),
        )
        for label in ("source_uri", "revision"):
            value = getattr(self, label)
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 2048
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError(f"{label} must be a bounded non-empty string")
        if self.manifest_hash is not None:
            _require_hash("manifest_hash", self.manifest_hash)

    @property
    def descriptor_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "dataset_ids": list(self.dataset_ids),
            "label_ids": list(self.label_ids),
            "source_uri": self.source_uri,
            "revision": self.revision,
            "manifest_hash": self.manifest_hash,
        }


@dataclass(frozen=True)
class DatasetFileRecord:
    relative_path: str
    size_bytes: int
    mtime_ns: int
    suffix: str
    digest: str | None
    hash_status: str
    issue_codes: tuple[str, ...] = ()
    schema_version: str = "rosclaw.dataset.file_record.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _safe_relative_path(self.relative_path))
        for label in ("size_bytes", "mtime_ns"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if self.digest is not None:
            _require_hash("digest", self.digest)
        if not isinstance(self.hash_status, str) or not self.hash_status:
            raise ValueError("hash_status must not be empty")
        object.__setattr__(
            self,
            "issue_codes",
            _unique_identifiers(
                self.issue_codes,
                label="issue_codes",
                allow_empty=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "suffix": self.suffix,
            "digest": self.digest,
            "hash_status": self.hash_status,
            "issue_codes": list(self.issue_codes),
        }


@dataclass(frozen=True)
class DatasetFileAnnotation:
    relative_path: str
    source_id: str
    label_ids: tuple[str, ...]
    schema_version: str = "rosclaw.dataset.file_annotation.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _safe_relative_path(self.relative_path))
        _require_identifier("source_id", self.source_id)
        object.__setattr__(
            self,
            "label_ids",
            _unique_identifiers(self.label_ids, label="label_ids"),
        )

    @property
    def annotation_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "source_id": self.source_id,
            "label_ids": list(self.label_ids),
        }


@dataclass(frozen=True)
class DatasetInventory:
    dataset_id: str
    state: DatasetSnapshotState
    total_file_count: int
    total_size_bytes: int
    extension_counts: Mapping[str, int]
    issue_counts: Mapping[str, int]
    license_files: tuple[str, ...]
    label_match_count: int
    annotation_count: int
    label_counts: Mapping[str, int]
    annotations: tuple[DatasetFileAnnotation, ...]
    duplicate_content_groups: tuple[tuple[str, ...], ...]
    files: tuple[DatasetFileRecord, ...]
    source_error_count: int = 0
    source_errors: tuple[str, ...] = ()
    scan_error_count: int = 0
    scan_errors: tuple[str, ...] = ()
    schema_version: str = "rosclaw.dataset.inventory.v3"

    def __post_init__(self) -> None:
        _require_dataset_name("dataset_id", self.dataset_id)
        if not isinstance(self.state, DatasetSnapshotState):
            raise ValueError("state must be a DatasetSnapshotState")
        for label in (
            "total_file_count",
            "total_size_bytes",
            "label_match_count",
            "annotation_count",
            "source_error_count",
            "scan_error_count",
        ):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        files = tuple(self.files)
        if any(not isinstance(value, DatasetFileRecord) for value in files):
            raise ValueError("files must contain DatasetFileRecord values")
        if self.total_file_count != len(files):
            raise ValueError("total_file_count must match files")
        if self.total_size_bytes != sum(value.size_bytes for value in files):
            raise ValueError("total_size_bytes must match files")
        paths = [value.relative_path for value in files]
        if len(paths) != len(set(paths)):
            raise ValueError("file paths must be unique")
        object.__setattr__(self, "files", files)
        object.__setattr__(
            self,
            "extension_counts",
            _count_mapping(self.extension_counts, label="extension_counts"),
        )
        object.__setattr__(
            self,
            "issue_counts",
            _count_mapping(self.issue_counts, label="issue_counts"),
        )
        object.__setattr__(
            self,
            "label_counts",
            _count_mapping(self.label_counts, label="label_counts"),
        )
        license_files = tuple(_safe_relative_path(value) for value in self.license_files)
        if len(license_files) != len(set(license_files)) or any(
            value not in paths for value in license_files
        ):
            raise ValueError("license_files must be unique inventory paths")
        object.__setattr__(self, "license_files", license_files)
        annotations = tuple(self.annotations)
        if any(not isinstance(value, DatasetFileAnnotation) for value in annotations):
            raise ValueError("annotations must contain DatasetFileAnnotation values")
        if any(value.relative_path not in paths for value in annotations):
            raise ValueError("annotations must reference inventory paths")
        annotation_keys = [(value.relative_path, value.source_id) for value in annotations]
        if len(annotation_keys) != len(set(annotation_keys)):
            raise ValueError("annotations must be unique by path and source")
        if self.label_match_count < len({value.relative_path for value in annotations}):
            raise ValueError("label_match_count must cover retained annotations")
        if self.annotation_count < len(annotations):
            raise ValueError("annotation_count must cover retained annotations")
        object.__setattr__(self, "annotations", annotations)
        groups = tuple(
            tuple(_safe_relative_path(path) for path in group)
            for group in self.duplicate_content_groups
        )
        if len(groups) != len(set(groups)) or any(
            len(group) < 2
            or len(group) != len(set(group))
            or any(path not in paths for path in group)
            for group in groups
        ):
            raise ValueError("duplicate content groups must contain inventory paths")
        object.__setattr__(self, "duplicate_content_groups", groups)
        for count_label, values_label in (
            ("source_error_count", "source_errors"),
            ("scan_error_count", "scan_errors"),
        ):
            values = _bounded_strings(
                tuple(getattr(self, values_label)),
                label=values_label,
                maximum_length=1024,
            )
            if getattr(self, count_label) < len(values):
                raise ValueError(f"{count_label} must cover retained errors")
            object.__setattr__(self, values_label, values)

    @property
    def snapshot_complete(self) -> bool:
        return self.state is DatasetSnapshotState.READY

    @property
    def license_evidence_present(self) -> bool:
        return bool(self.license_files)

    @property
    def annotations_truncated(self) -> bool:
        return self.annotation_count > len(self.annotations)

    @property
    def source_errors_truncated(self) -> bool:
        return self.source_error_count > len(self.source_errors)

    @property
    def scan_errors_truncated(self) -> bool:
        return self.scan_error_count > len(self.scan_errors)

    @property
    def training_eligible(self) -> bool:
        return False

    @property
    def inventory_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "state": self.state.value,
            "total_file_count": self.total_file_count,
            "total_size_bytes": self.total_size_bytes,
            "extension_counts": dict(self.extension_counts),
            "issue_counts": dict(self.issue_counts),
            "license_files": list(self.license_files),
            "license_evidence_present": self.license_evidence_present,
            "label_match_count": self.label_match_count,
            "annotation_count": self.annotation_count,
            "label_counts": dict(self.label_counts),
            "annotations": [value.to_dict() for value in self.annotations],
            "annotations_truncated": self.annotations_truncated,
            "duplicate_content_groups": [list(group) for group in self.duplicate_content_groups],
            "files": [value.to_dict() for value in self.files],
            "source_error_count": self.source_error_count,
            "source_errors": list(self.source_errors),
            "source_errors_truncated": self.source_errors_truncated,
            "scan_error_count": self.scan_error_count,
            "scan_errors": list(self.scan_errors),
            "scan_errors_truncated": self.scan_errors_truncated,
            "snapshot_complete": self.snapshot_complete,
            "training_eligible": self.training_eligible,
        }


@dataclass(frozen=True)
class DatasetDoctorReport:
    root: str
    started_at: str
    finished_at: str
    hash_mode: FileHashMode
    transfer_active: bool
    sources: tuple[DatasetSourceDescriptor, ...]
    extension_errors: tuple[str, ...]
    inventories: tuple[DatasetInventory, ...]
    schema_version: str = "rosclaw.dataset.doctor_report.v2"

    def __post_init__(self) -> None:
        for label in ("root", "started_at", "finished_at"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value or len(value) > 4096:
                raise ValueError(f"{label} must be a bounded non-empty string")
        if not isinstance(self.hash_mode, FileHashMode):
            raise ValueError("hash_mode must be a FileHashMode")
        if not isinstance(self.transfer_active, bool):
            raise ValueError("transfer_active must be boolean")
        sources = tuple(self.sources)
        if any(not isinstance(value, DatasetSourceDescriptor) for value in sources):
            raise ValueError("sources must contain DatasetSourceDescriptor values")
        if len({value.source_id for value in sources}) != len(sources):
            raise ValueError("source ids must be unique")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(
            self,
            "extension_errors",
            _bounded_strings(
                self.extension_errors,
                label="extension_errors",
                maximum_length=1024,
            ),
        )
        inventories = tuple(self.inventories)
        identifiers = [value.dataset_id for value in inventories]
        if (
            not inventories
            or any(not isinstance(value, DatasetInventory) for value in inventories)
            or len(identifiers) != len(set(identifiers))
        ):
            raise ValueError("inventories must be non-empty with unique dataset ids")
        object.__setattr__(self, "inventories", inventories)

    @property
    def snapshot_complete(self) -> bool:
        return (
            not self.transfer_active
            and not self.extension_errors
            and all(value.snapshot_complete for value in self.inventories)
        )

    @property
    def training_eligible(self) -> bool:
        return False

    @property
    def report_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "root": self.root,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "hash_mode": self.hash_mode.value,
            "transfer_active": self.transfer_active,
            "sources": [value.to_dict() for value in self.sources],
            "extension_errors": list(self.extension_errors),
            "inventories": [value.to_dict() for value in self.inventories],
            "snapshot_complete": self.snapshot_complete,
            "training_eligible": self.training_eligible,
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }


__all__ = [
    "DatasetDoctorReport",
    "DatasetFileAnnotation",
    "DatasetFileRecord",
    "DatasetInventory",
    "DatasetSnapshotState",
    "DatasetSourceDescriptor",
    "FileHashMode",
]
