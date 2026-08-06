"""Immutable contracts emitted by the generic dataset doctor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash


class FileHashMode(StrEnum):
    """How file identity was established for a point-in-time inventory."""

    NONE = "none"
    METADATA = "metadata"
    SHA256 = "sha256"


class DatasetSnapshotState(StrEnum):
    """Technical state only; this is deliberately separate from licensing."""

    READY = "ready"
    EMPTY = "empty"
    PARTIAL = "partial"
    TRANSFERRING = "transferring"
    UNREADABLE = "unreadable"


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
        path = self.relative_path.replace("\\", "/")
        if not path or path.startswith("/") or ".." in path.split("/"):
            raise ValueError("relative_path must be a safe relative path")
        if self.size_bytes < 0 or self.mtime_ns < 0:
            raise ValueError("file size and mtime must be non-negative")
        if self.digest is not None and not self.digest.startswith("sha256:"):
            raise ValueError("digest must use the sha256: prefix")
        if not self.hash_status:
            raise ValueError("hash_status must not be empty")
        issues = tuple(self.issue_codes)
        if len(issues) != len(set(issues)) or any(not value for value in issues):
            raise ValueError("issue_codes must contain unique non-empty values")
        object.__setattr__(self, "relative_path", path)
        object.__setattr__(self, "issue_codes", issues)

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
class DatasetInventory:
    dataset_id: str
    state: DatasetSnapshotState
    total_file_count: int
    total_size_bytes: int
    extension_counts: Mapping[str, int]
    issue_counts: Mapping[str, int]
    license_files: tuple[str, ...]
    football_matches: tuple[str, ...]
    duplicate_content_groups: tuple[tuple[str, ...], ...]
    files: tuple[DatasetFileRecord, ...]
    scan_errors: tuple[str, ...] = ()
    schema_version: str = "rosclaw.dataset.inventory.v1"

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be empty")
        if not isinstance(self.state, DatasetSnapshotState):
            raise ValueError("state must be a DatasetSnapshotState")
        if self.total_file_count < 0 or self.total_size_bytes < 0:
            raise ValueError("inventory counts must be non-negative")
        files = tuple(self.files)
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
            MappingProxyType(
                {str(key): int(value) for key, value in self.extension_counts.items()}
            ),
        )
        object.__setattr__(
            self,
            "issue_counts",
            MappingProxyType({str(key): int(value) for key, value in self.issue_counts.items()}),
        )
        for label in ("license_files", "football_matches", "scan_errors"):
            values = tuple(getattr(self, label))
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
            object.__setattr__(self, label, values)
        groups = tuple(tuple(group) for group in self.duplicate_content_groups)
        if any(len(group) < 2 for group in groups):
            raise ValueError("duplicate content groups must contain at least two paths")
        object.__setattr__(self, "duplicate_content_groups", groups)

    @property
    def snapshot_complete(self) -> bool:
        return self.state is DatasetSnapshotState.READY

    @property
    def license_evidence_present(self) -> bool:
        return bool(self.license_files)

    @property
    def training_eligible(self) -> bool:
        """A doctor never adjudicates legal terms or target applicability."""

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
            "extension_counts": dict(sorted(self.extension_counts.items())),
            "issue_counts": dict(sorted(self.issue_counts.items())),
            "license_files": list(self.license_files),
            "license_evidence_present": self.license_evidence_present,
            "football_matches": list(self.football_matches),
            "duplicate_content_groups": [list(group) for group in self.duplicate_content_groups],
            "files": [value.to_dict() for value in self.files],
            "scan_errors": list(self.scan_errors),
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
    inventories: tuple[DatasetInventory, ...]
    schema_version: str = "rosclaw.dataset.doctor_report.v1"

    def __post_init__(self) -> None:
        if not self.root:
            raise ValueError("root must not be empty")
        if not isinstance(self.hash_mode, FileHashMode):
            raise ValueError("hash_mode must be a FileHashMode")
        inventories = tuple(self.inventories)
        identifiers = [value.dataset_id for value in inventories]
        if not inventories or len(identifiers) != len(set(identifiers)):
            raise ValueError("inventories must be non-empty with unique dataset ids")
        object.__setattr__(self, "inventories", inventories)

    @property
    def snapshot_complete(self) -> bool:
        return not self.transfer_active and all(
            value.snapshot_complete for value in self.inventories
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
            "inventories": [value.to_dict() for value in self.inventories],
            "snapshot_complete": self.snapshot_complete,
            "training_eligible": self.training_eligible,
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }


__all__ = [
    "DatasetDoctorReport",
    "DatasetFileRecord",
    "DatasetInventory",
    "DatasetSnapshotState",
    "FileHashMode",
]
