"""Content-addressed local snapshot manifests for MotionDecode."""

from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.collective.contracts import CollectiveSourceIdentity, LicenseDecision, LicenseUse
from rosclaw.collective.sources.motiondecode.attribution import MotionDecodeAttribution
from rosclaw.collective.sources.motiondecode.license import (
    MotionDecodeLicenseSnapshot,
    snapshot_license,
)
from rosclaw.collective.sources.motiondecode.taxonomy import (
    MotionDecodeCatalogAudit,
    MotionFamily,
    classify_motion,
    parse_catalog,
)
from rosclaw.feedback.contracts import canonical_hash

MOTIONDECODE_PROVIDER = "CMRobot"
MOTIONDECODE_DATASET = "MotionDecode"
MOTIONDECODE_SOURCE_URI = "https://huggingface.co/datasets/CMRobot/MotionDecode"
MOTIONDECODE_SOURCE_BODY_ID = "unitree_g1_retargeted_29dof"
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_REGISTERED_MOTION_BYTES = 128 * 1024 * 1024
_MAX_REGISTERED_SAMPLES = 10_000


@dataclass(frozen=True)
class MotionDecodeFileRecord:
    relative_path: str
    content_hash: str
    size_bytes: int
    media_type: str
    family: MotionFamily
    schema_version: str = "rosclaw.collective.motiondecode_file.v1"

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or self.relative_path.startswith(".")
        ):
            raise ValueError("relative_path must be a safe dataset-relative path")
        if not _SHA256.fullmatch(self.content_hash):
            raise ValueError("content_hash must be a sha256: content hash")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not self.media_type.strip():
            raise ValueError("media_type must not be empty")
        if not isinstance(self.family, MotionFamily):
            raise ValueError("family must be a recognized MotionFamily")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "family": self.family.value,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MotionDecodeFileRecord:
        return cls(
            relative_path=str(value["relative_path"]),
            content_hash=str(value["content_hash"]),
            size_bytes=int(value["size_bytes"]),
            media_type=str(value["media_type"]),
            family=MotionFamily(str(value["family"])),
        )


@dataclass(frozen=True)
class MotionDecodeSourceManifest:
    revision: str
    files: tuple[MotionDecodeFileRecord, ...]
    license_snapshot: MotionDecodeLicenseSnapshot
    attribution: MotionDecodeAttribution
    catalog_audit_hash: str
    local_discovered_sample_count: int
    selected_sample_count: int
    local_selection_complete: bool
    requested_families: tuple[MotionFamily, ...]
    source_uri: str = MOTIONDECODE_SOURCE_URI
    sample_rate_hz: float = 120.0
    root_position_unit: str = "header_declared_m"
    root_quaternion_order: str = "wxyz"
    schema_version: str = "rosclaw.collective.motiondecode_source_manifest.v1"

    def __post_init__(self) -> None:
        if not _REVISION.fullmatch(self.revision):
            raise ValueError("revision must be a pinned 40-character lowercase commit hash")
        if not self.source_uri.strip():
            raise ValueError("source_uri must not be empty")
        files = tuple(self.files)
        if not files or any(not isinstance(item, MotionDecodeFileRecord) for item in files):
            raise ValueError("files must contain MotionDecodeFileRecord entries")
        paths = tuple(item.relative_path for item in files)
        if len(paths) != len(set(paths)) or paths != tuple(sorted(paths)):
            raise ValueError("files must be unique and sorted by relative_path")
        if "metadata/index.csv" not in paths:
            raise ValueError("files must include metadata/index.csv")
        object.__setattr__(self, "files", files)
        if not isinstance(self.license_snapshot, MotionDecodeLicenseSnapshot):
            raise ValueError("license_snapshot must be MotionDecodeLicenseSnapshot")
        if self.license_snapshot.source_revision != self.revision:
            raise ValueError("license snapshot revision does not match source revision")
        if not isinstance(self.attribution, MotionDecodeAttribution):
            raise ValueError("attribution must be MotionDecodeAttribution")
        if self.attribution.revision != self.revision:
            raise ValueError("attribution revision does not match source revision")
        if self.attribution.license_snapshot_hash != self.license_snapshot.snapshot_hash:
            raise ValueError("attribution is not bound to the license snapshot")
        if not _SHA256.fullmatch(self.catalog_audit_hash):
            raise ValueError("catalog_audit_hash must be a sha256: content hash")
        if self.local_discovered_sample_count < 0 or self.selected_sample_count < 0:
            raise ValueError("sample counts must be non-negative")
        if self.selected_sample_count > self.local_discovered_sample_count:
            raise ValueError("selected_sample_count cannot exceed local_discovered_sample_count")
        sample_records = sum(item.relative_path.startswith("samples/") for item in files)
        if sample_records != self.selected_sample_count:
            raise ValueError("selected_sample_count must match sample file records")
        families = tuple(self.requested_families)
        if len(families) != len(set(families)):
            raise ValueError("requested_families must be unique")
        if any(not isinstance(item, MotionFamily) for item in families):
            raise ValueError("requested_families contains an unknown family")
        object.__setattr__(self, "requested_families", families)
        if self.sample_rate_hz != 120.0:
            raise ValueError("MotionDecode source manifests currently require declared 120 Hz")

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def file_hashes(self) -> dict[str, str]:
        return {item.relative_path: item.content_hash for item in self.files}

    @property
    def source_identity(self) -> CollectiveSourceIdentity:
        return CollectiveSourceIdentity(
            provider=MOTIONDECODE_PROVIDER,
            dataset=MOTIONDECODE_DATASET,
            revision=self.revision,
            file_hashes=self.file_hashes,
            license_evidence=self.license_snapshot.evidence,
            source_body_id=MOTIONDECODE_SOURCE_BODY_ID,
        )

    @property
    def hardware_authorized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": MOTIONDECODE_PROVIDER,
            "dataset": MOTIONDECODE_DATASET,
            "source_uri": self.source_uri,
            "revision": self.revision,
            "source_body_id": MOTIONDECODE_SOURCE_BODY_ID,
            "files": [item.to_dict() for item in self.files],
            "license_snapshot": self.license_snapshot.to_dict(),
            "attribution": self.attribution.to_dict(),
            "catalog_audit_hash": self.catalog_audit_hash,
            "inventory_scope": "operator_managed_local_snapshot",
            "upstream_inventory_verified": False,
            "local_discovered_sample_count": self.local_discovered_sample_count,
            "selected_sample_count": self.selected_sample_count,
            "local_selection_complete": self.local_selection_complete,
            "requested_families": [item.value for item in self.requested_families],
            "sample_rate_hz": self.sample_rate_hz,
            "root_position_unit": self.root_position_unit,
            "root_quaternion_order": self.root_quaternion_order,
            "source_identity_hash": self.source_identity.source_hash,
            "hardware_authorized": self.hardware_authorized,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MotionDecodeSourceManifest:
        if value.get("provider") != MOTIONDECODE_PROVIDER:
            raise ValueError("unexpected MotionDecode provider")
        if value.get("dataset") != MOTIONDECODE_DATASET:
            raise ValueError("unexpected MotionDecode dataset")
        if value.get("hardware_authorized") not in (None, False):
            raise ValueError("MotionDecode source evidence cannot authorize hardware")
        if value.get("inventory_scope") != "operator_managed_local_snapshot":
            raise ValueError("MotionDecode manifest has an unknown inventory scope")
        if value.get("upstream_inventory_verified") is not False:
            raise ValueError("local registration cannot claim a verified upstream inventory")
        files_value = value["files"]
        if not isinstance(files_value, list) or any(
            not isinstance(item, dict) for item in files_value
        ):
            raise ValueError("files must be an array")
        requested_families = value["requested_families"]
        if not isinstance(requested_families, list):
            raise ValueError("requested_families must be an array")
        local_selection_complete = value["local_selection_complete"]
        if not isinstance(local_selection_complete, bool):
            raise ValueError("local_selection_complete must be a boolean")
        license_value = value["license_snapshot"]
        attribution_value = value["attribution"]
        if not isinstance(license_value, dict) or not isinstance(attribution_value, dict):
            raise ValueError("license_snapshot and attribution must be objects")
        manifest = cls(
            revision=str(value["revision"]),
            files=tuple(MotionDecodeFileRecord.from_dict(item) for item in files_value),
            license_snapshot=MotionDecodeLicenseSnapshot.from_dict(license_value),
            attribution=MotionDecodeAttribution.from_dict(attribution_value),
            catalog_audit_hash=str(value["catalog_audit_hash"]),
            local_discovered_sample_count=int(value["local_discovered_sample_count"]),
            selected_sample_count=int(value["selected_sample_count"]),
            local_selection_complete=local_selection_complete,
            requested_families=tuple(MotionFamily(str(item)) for item in requested_families),
            source_uri=str(value.get("source_uri", MOTIONDECODE_SOURCE_URI)),
            sample_rate_hz=float(value["sample_rate_hz"]),
            root_position_unit=str(value["root_position_unit"]),
            root_quaternion_order=str(value["root_quaternion_order"]),
        )
        claimed = value.get("source_identity_hash")
        if claimed is not None and claimed != manifest.source_identity.source_hash:
            raise ValueError("source_identity_hash does not replay")
        return manifest


@dataclass(frozen=True)
class MotionDecodeRegistration:
    manifest: MotionDecodeSourceManifest
    catalog_audit: MotionDecodeCatalogAudit
    schema_version: str = "rosclaw.collective.motiondecode_registration.v1"

    def __post_init__(self) -> None:
        if self.manifest.catalog_audit_hash != self.catalog_audit.audit_hash:
            raise ValueError("catalog audit does not match the source manifest")

    @property
    def registration_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def source_registered(self) -> bool:
        return self.catalog_audit.schema_valid

    @property
    def training_eligible(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest": self.manifest.to_dict(),
            "catalog_audit": self.catalog_audit.to_dict(),
            "source_registered": self.source_registered,
            "training_eligible": self.training_eligible,
            "training_blockers": _registration_blockers(self),
            "hardware_authorized": False,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MotionDecodeRegistration:
        if value.get("hardware_authorized") not in (None, False):
            raise ValueError("MotionDecode registration cannot authorize hardware")
        manifest_value = value["manifest"]
        audit_value = value["catalog_audit"]
        if not isinstance(manifest_value, dict) or not isinstance(audit_value, dict):
            raise ValueError("manifest and catalog_audit must be objects")
        return cls(
            manifest=MotionDecodeSourceManifest.from_dict(manifest_value),
            catalog_audit=MotionDecodeCatalogAudit.from_dict(audit_value),
        )


def register_motiondecode_source(
    dataset_root: Path,
    *,
    revision: str,
    requested_use: LicenseUse,
    license_decision: LicenseDecision = LicenseDecision.PENDING,
    terms_path: Path | None = None,
    terms_uri: str | None = None,
    attribution_text: str = "ChingMu / CMRobot MotionDecode",
    families: tuple[MotionFamily, ...] = (),
    limit: int = 400,
) -> MotionDecodeRegistration:
    """Hash a bounded, operator-managed local snapshot without downloading it."""

    if limit <= 0 or limit > _MAX_REGISTERED_SAMPLES:
        raise ValueError(f"limit must be in [1, {_MAX_REGISTERED_SAMPLES}]")
    root = dataset_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("dataset_root must be a directory")
    catalog_path = _safe_file(root, Path("metadata/index.csv"))
    _, catalog_audit = parse_catalog(catalog_path)
    license_snapshot = snapshot_license(
        source_revision=revision,
        requested_use=requested_use,
        decision=license_decision,
        terms_path=terms_path,
        terms_uri=terms_uri,
        attribution=attribution_text,
    )
    attribution = MotionDecodeAttribution(
        provider=MOTIONDECODE_PROVIDER,
        dataset=MOTIONDECODE_DATASET,
        source_uri=MOTIONDECODE_SOURCE_URI,
        revision=revision,
        attribution_text=attribution_text,
        license_snapshot_hash=license_snapshot.snapshot_hash,
    )
    requested = tuple(families)
    discovered = _discover_samples(root, requested)
    selected = _stratified_sample(discovered, limit=limit)
    records = [_file_record(root, catalog_path, MotionFamily.OTHER, "text/csv")]
    records.extend(
        _file_record(root, path, classify_motion(path.as_posix()), "text/csv") for path in selected
    )
    records.sort(key=lambda item: item.relative_path)
    manifest = MotionDecodeSourceManifest(
        revision=revision,
        files=tuple(records),
        license_snapshot=license_snapshot,
        attribution=attribution,
        catalog_audit_hash=catalog_audit.audit_hash,
        local_discovered_sample_count=len(discovered),
        selected_sample_count=len(selected),
        local_selection_complete=len(selected) == len(discovered),
        requested_families=requested,
    )
    return MotionDecodeRegistration(manifest=manifest, catalog_audit=catalog_audit)


def verify_registered_files(
    registration: MotionDecodeRegistration, dataset_root: Path
) -> dict[str, Path]:
    """Replay every registered size/hash against a local root."""

    root = dataset_root.expanduser().resolve(strict=True)
    verified: dict[str, Path] = {}
    for record in registration.manifest.files:
        path = _safe_file(root, Path(record.relative_path))
        actual_hash, actual_size = _hash_file(path)
        if actual_size != record.size_bytes:
            raise ValueError(f"registered file size changed: {record.relative_path}")
        if actual_hash != record.content_hash:
            raise ValueError(f"registered file hash changed: {record.relative_path}")
        verified[record.relative_path] = path
    return verified


def _registration_blockers(registration: MotionDecodeRegistration) -> list[str]:
    blockers = ["PHYSICS_QUALIFICATION_REQUIRED"]
    if not registration.catalog_audit.schema_valid:
        blockers.append("CATALOG_SCHEMA_INVALID")
    if not registration.manifest.license_snapshot.training_permitted:
        blockers.append("LICENSE_NOT_PERMITTED")
    if registration.manifest.selected_sample_count == 0:
        blockers.append("NO_LOCAL_MOTION_SAMPLES")
    return blockers


def _discover_samples(root: Path, families: tuple[MotionFamily, ...]) -> list[Path]:
    samples_root = root / "samples"
    if not samples_root.exists():
        return []
    allowed = set(families)
    paths: list[Path] = []
    for candidate in samples_root.rglob("*.csv"):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        relative = candidate.relative_to(root)
        family = classify_motion(relative.as_posix())
        if allowed and family not in allowed:
            continue
        paths.append(relative)
    return sorted(paths, key=lambda item: item.as_posix())


def _stratified_sample(paths: list[Path], *, limit: int) -> list[Path]:
    """Select a deterministic family- and skill-balanced bounded pilot.

    MotionDecode paths are ordered by taxonomy, so taking the first ``limit``
    files can silently select only one leaf skill (for example Short_Pass).
    Round-robin first across motion families and then across leaf directories.
    The returned set is sorted so the content-addressed manifest remains
    canonical and independent of filesystem traversal order.
    """

    if limit <= 0 or not paths:
        return []
    grouped: dict[MotionFamily, dict[str, deque[Path]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    for path in sorted(paths, key=lambda item: item.as_posix()):
        family = classify_motion(path.as_posix())
        grouped[family][path.parent.as_posix()].append(path)

    family_streams: dict[MotionFamily, deque[Path]] = {}
    for family, leaf_buckets in grouped.items():
        leaf_names = sorted(leaf_buckets)
        stream: deque[Path] = deque()
        while any(leaf_buckets[name] for name in leaf_names):
            for name in leaf_names:
                if leaf_buckets[name]:
                    stream.append(leaf_buckets[name].popleft())
        family_streams[family] = stream

    families = sorted(family_streams, key=lambda item: item.value)
    selected: list[Path] = []
    bounded_limit = min(limit, len(paths))
    while len(selected) < bounded_limit:
        progressed = False
        for family in families:
            stream = family_streams[family]
            if not stream:
                continue
            selected.append(stream.popleft())
            progressed = True
            if len(selected) == bounded_limit:
                break
        if not progressed:
            break
    return sorted(selected, key=lambda item: item.as_posix())


def _safe_file(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("dataset path must stay below dataset_root")
    unresolved = root / relative
    if unresolved.is_symlink():
        raise ValueError(f"dataset evidence cannot be a symlink: {relative.as_posix()}")
    resolved = unresolved.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"dataset evidence must be a regular file below root: {relative}")
    return resolved


def _file_record(
    root: Path, path: Path, family: MotionFamily, media_type: str
) -> MotionDecodeFileRecord:
    resolved = path if path.is_absolute() else _safe_file(root, path)
    relative = resolved.relative_to(root).as_posix()
    if relative.startswith("samples/") and resolved.stat().st_size > _MAX_REGISTERED_MOTION_BYTES:
        raise ValueError(f"motion CSV exceeds the 128 MB safety limit: {relative}")
    content_hash, size_bytes = _hash_file(resolved)
    return MotionDecodeFileRecord(
        relative_path=relative,
        content_hash=content_hash,
        size_bytes=size_bytes,
        media_type=media_type,
        family=family,
    )


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if (
        before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError(f"dataset evidence changed while it was hashed: {path.name}")
    return "sha256:" + digest.hexdigest(), before.st_size


__all__ = [
    "MOTIONDECODE_DATASET",
    "MOTIONDECODE_PROVIDER",
    "MOTIONDECODE_SOURCE_BODY_ID",
    "MOTIONDECODE_SOURCE_URI",
    "MotionDecodeFileRecord",
    "MotionDecodeRegistration",
    "MotionDecodeSourceManifest",
    "register_motiondecode_source",
    "verify_registered_files",
]
