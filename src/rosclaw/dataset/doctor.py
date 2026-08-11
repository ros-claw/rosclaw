"""Point-in-time, transfer-aware inspection for operator-managed datasets."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.dataset.contracts import (
    DatasetDoctorReport,
    DatasetFileAnnotation,
    DatasetFileRecord,
    DatasetInventory,
    DatasetSnapshotState,
    FileHashMode,
)
from rosclaw.dataset.registry import DatasetSourceRegistry
from rosclaw.feedback.contracts import canonical_hash

_PARTIAL_MARKERS = (
    ".aria2",
    ".download",
    ".incomplete",
    ".part",
    ".partial",
    ".tmp",
)
_LICENSE_NAMES = {
    "copying",
    "copying.md",
    "license",
    "license.md",
    "license.txt",
    "terms",
    "terms.md",
    "terms.txt",
}
_MAX_RETAINED_ERRORS = 100


def inspect_dataset_root(
    root: Path,
    *,
    hash_mode: FileHashMode = FileHashMode.METADATA,
    transfer_active: bool = False,
    annotation_limit: int = 100,
    source_registry: DatasetSourceRegistry | None = None,
    discover_sources: bool = True,
) -> DatasetDoctorReport:
    """Inspect top-level datasets without following symbolic links.

    ``transfer_active`` is an operator assertion. It prevents an otherwise
    clean prefix of a download from being mislabeled as a complete snapshot.
    Source plugins receive names only and cannot read the inspected root.
    """

    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"dataset root is not a readable directory: {root}")
    if (
        isinstance(annotation_limit, bool)
        or not isinstance(annotation_limit, int)
        or not 1 <= annotation_limit <= 100_000
    ):
        raise ValueError("annotation_limit must be in [1, 100000]")
    if not isinstance(hash_mode, FileHashMode):
        raise ValueError("hash_mode must be a FileHashMode")
    if not isinstance(transfer_active, bool) or not isinstance(discover_sources, bool):
        raise ValueError("dataset doctor flags must be boolean")
    registry = source_registry or DatasetSourceRegistry()
    discovery_errors: tuple[str, ...] = ()
    if discover_sources:
        discovery_errors = tuple(
            dict.fromkeys(_bounded_text(value) for value in registry.discover().errors)
        )
    started = _timestamp()
    children = sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.is_symlink()),
        key=lambda path: path.name,
    )
    if not children:
        raise ValueError("dataset root contains no top-level dataset directories")
    inventories = tuple(
        _inspect_one(
            path,
            hash_mode=hash_mode,
            transfer_active=transfer_active,
            annotation_limit=annotation_limit,
            source_registry=registry,
        )
        for path in children
    )
    return DatasetDoctorReport(
        root=str(root),
        started_at=started,
        finished_at=_timestamp(),
        hash_mode=hash_mode,
        transfer_active=transfer_active,
        sources=registry.descriptors,
        extension_errors=discovery_errors,
        inventories=inventories,
    )


def write_dataset_doctor_artifacts(
    report: DatasetDoctorReport,
    output_dir: Path,
) -> dict[str, str]:
    """Write review artifacts atomically; never mutate the dataset root."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "inventory": output_dir / "dataset_inventory.json",
        "quality_report": output_dir / "dataset_quality_report.html",
        "license_manifest": output_dir / "license_manifest.json",
        "source_manifest": output_dir / "dataset_source_manifest.json",
        "label_matrix": output_dir / "dataset_label_matrix.csv",
    }
    inventory = report.to_dict()
    inventory["report_hash"] = report.report_hash
    _atomic_write_text(
        paths["inventory"],
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
    )
    license_manifest = {
        "schema_version": "rosclaw.dataset.license_manifest.v1",
        "report_hash": report.report_hash,
        "datasets": [
            {
                "dataset_id": value.dataset_id,
                "license_files": list(value.license_files),
                "license_evidence": [
                    {
                        "path": record.relative_path,
                        "sha256": record.digest,
                        "size_bytes": record.size_bytes,
                        "hash_status": record.hash_status,
                    }
                    for record in value.files
                    if record.relative_path in value.license_files
                ],
                "decision": "pending_operator_review",
                "training_eligible": False,
                "public_demo_allowed": False,
                "commercial_promotion_allowed": False,
            }
            for value in report.inventories
        ],
    }
    _atomic_write_text(
        paths["license_manifest"],
        json.dumps(license_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    source_manifest = {
        "schema_version": "rosclaw.dataset.source_manifest.v1",
        "report_hash": report.report_hash,
        "sources": [value.to_dict() for value in report.sources],
        "extension_errors": list(report.extension_errors),
        "provenance_review_required": True,
        "training_eligible": False,
    }
    _atomic_write_text(
        paths["source_manifest"],
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(paths["label_matrix"], _label_matrix_csv(report))
    _atomic_write_text(paths["quality_report"], _quality_html(report))
    return {key: str(value) for key, value in paths.items()}


def _inspect_one(
    root: Path,
    *,
    hash_mode: FileHashMode,
    transfer_active: bool,
    annotation_limit: int,
    source_registry: DatasetSourceRegistry,
) -> DatasetInventory:
    files: list[DatasetFileRecord] = []
    scan_error_count = 0
    scan_errors: list[str] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            path = Path(directory) / name
            if path.is_symlink():
                scan_error_count += 1
                _retain_unique(
                    scan_errors,
                    f"{path.relative_to(root).as_posix()}: symbolic_directory_not_followed",
                )
            else:
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for file_name in sorted(file_names):
            path = Path(directory) / file_name
            relative = path.relative_to(root).as_posix()
            try:
                files.append(_inspect_file(path, relative=relative, hash_mode=hash_mode))
            except (OSError, PermissionError) as exc:
                scan_error_count += 1
                _retain_unique(
                    scan_errors,
                    _bounded_error(relative, exc),
                )
    files.sort(key=lambda value: value.relative_path)
    extension_counts = Counter(value.suffix or "<none>" for value in files)
    issue_counts = Counter(issue for value in files for issue in value.issue_codes)
    license_files = tuple(
        value.relative_path
        for value in files
        if Path(value.relative_path).name.lower() in _LICENSE_NAMES
    )
    digests: dict[str, list[str]] = defaultdict(list)
    if hash_mode is FileHashMode.SHA256:
        for value in files:
            if value.digest is not None and value.hash_status == "computed":
                digests[value.digest].append(value.relative_path)
    duplicate_groups = tuple(tuple(paths) for _, paths in sorted(digests.items()) if len(paths) > 1)

    annotations: list[DatasetFileAnnotation] = []
    annotation_count = 0
    matched_paths: set[str] = set()
    label_counts: Counter[str] = Counter()
    source_error_count = 0
    source_errors: list[str] = []
    for record in files:
        resolution = source_registry.classify(
            dataset_id=root.name,
            relative_path=record.relative_path,
        )
        if resolution.annotations:
            matched_paths.add(record.relative_path)
        annotation_count += len(resolution.annotations)
        for annotation in resolution.annotations:
            label_counts.update(annotation.label_ids)
            if len(annotations) < annotation_limit:
                annotations.append(annotation)
        source_error_count += len(resolution.errors)
        for error in resolution.errors:
            _retain_unique(source_errors, error)

    state = _state(
        files,
        scan_error_count=scan_error_count,
        source_error_count=source_error_count,
        transfer_active=transfer_active,
    )
    return DatasetInventory(
        dataset_id=root.name,
        state=state,
        total_file_count=len(files),
        total_size_bytes=sum(value.size_bytes for value in files),
        extension_counts=dict(extension_counts),
        issue_counts=dict(issue_counts),
        license_files=license_files,
        label_match_count=len(matched_paths),
        annotation_count=annotation_count,
        label_counts=dict(label_counts),
        annotations=tuple(annotations),
        duplicate_content_groups=duplicate_groups,
        files=tuple(files),
        source_error_count=source_error_count,
        source_errors=tuple(source_errors),
        scan_error_count=scan_error_count,
        scan_errors=tuple(scan_errors),
    )


def _inspect_file(path: Path, *, relative: str, hash_mode: FileHashMode) -> DatasetFileRecord:
    before = path.lstat()
    issues: list[str] = []
    digest: str | None = None
    hash_status = "skipped"
    if path.is_symlink():
        issues.append("symbolic_link_not_followed")
    elif not path.is_file():
        issues.append("not_regular_file")
    else:
        if before.st_size == 0:
            issues.append("zero_byte_file")
        if _is_partial_name(path.name):
            issues.append("partial_transfer_file")
        if before.st_size <= 1024 and _is_lfs_pointer(path):
            issues.append("git_lfs_pointer")
        if hash_mode is FileHashMode.METADATA and path.name.lower() not in _LICENSE_NAMES:
            digest = canonical_hash(
                {
                    "relative_path": relative,
                    "size_bytes": before.st_size,
                    "mtime_ns": before.st_mtime_ns,
                }
            )
            hash_status = "metadata_only"
        elif hash_mode is FileHashMode.SHA256 or path.name.lower() in _LICENSE_NAMES:
            digest = _sha256_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                digest = None
                hash_status = "changed_during_scan"
                issues.append("changed_during_scan")
            else:
                hash_status = (
                    "computed_license" if hash_mode is not FileHashMode.SHA256 else "computed"
                )
    return DatasetFileRecord(
        relative_path=relative,
        size_bytes=before.st_size,
        mtime_ns=before.st_mtime_ns,
        suffix=path.suffix.lower(),
        digest=digest,
        hash_status=hash_status,
        issue_codes=tuple(dict.fromkeys(issues)),
    )


def _state(
    files: list[DatasetFileRecord],
    *,
    scan_error_count: int,
    source_error_count: int,
    transfer_active: bool,
) -> DatasetSnapshotState:
    if scan_error_count and not files:
        return DatasetSnapshotState.UNREADABLE
    if transfer_active:
        return DatasetSnapshotState.TRANSFERRING
    if not files:
        return DatasetSnapshotState.EMPTY
    issues = {issue for value in files for issue in value.issue_codes}
    if "partial_transfer_file" in issues or "changed_during_scan" in issues:
        return DatasetSnapshotState.TRANSFERRING
    if (
        scan_error_count
        or source_error_count
        or "git_lfs_pointer" in issues
        or "not_regular_file" in issues
        or "zero_byte_file" in issues
        or "symbolic_link_not_followed" in issues
    ):
        return DatasetSnapshotState.PARTIAL
    return DatasetSnapshotState.READY


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1\n")
    except OSError:
        return False


def _is_partial_name(name: str) -> bool:
    lowered = name.lower()
    if re.search(r"\.part_[a-z]{2,}(?:\.metadata)?$", lowered):
        return False
    return any(lowered.endswith(marker) for marker in _PARTIAL_MARKERS)


def _label_matrix_csv(report: DatasetDoctorReport) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "dataset_id",
            "snapshot_state",
            "file_count",
            "size_bytes",
            "label_match_count",
            "label_counts_json",
            "license_file_count",
            "source_error_count",
            "scan_error_count",
            "snapshot_complete",
            "training_eligible",
        ]
    )
    for value in report.inventories:
        writer.writerow(
            [
                value.dataset_id,
                value.state.value,
                value.total_file_count,
                value.total_size_bytes,
                value.label_match_count,
                json.dumps(dict(value.label_counts), sort_keys=True),
                len(value.license_files),
                value.source_error_count,
                value.scan_error_count,
                str(value.snapshot_complete).lower(),
                "false",
            ]
        )
    return stream.getvalue()


def _quality_html(report: DatasetDoctorReport) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(value.dataset_id)}</td>"
        f"<td><code>{value.state.value}</code></td>"
        f"<td>{value.total_file_count:,}</td>"
        f"<td>{value.total_size_bytes / (1024**3):.3f}</td>"
        f"<td>{value.label_match_count}</td>"
        f"<td>{html.escape(json.dumps(dict(value.label_counts), sort_keys=True))}</td>"
        f"<td>{len(value.license_files)}</td>"
        f"<td>{value.source_error_count + value.scan_error_count}</td>"
        "</tr>"
        for value in report.inventories
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROSClaw Dataset Doctor</title>
<style>body{{font:15px/1.55 system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:#171717}}code{{background:#f2f2f2;padding:2px 6px;border-radius:5px}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;text-align:left;padding:9px;vertical-align:top}}th{{background:#f7f7f7}}.warn{{padding:14px;border-left:4px solid #d97706;background:#fff7ed}}</style></head>
<body><h1>ROSClaw Dataset Doctor</h1>
<p class="warn"><strong>Evidence boundary:</strong> This is a point-in-time snapshot. transfer_active={str(report.transfer_active).lower()}. Technical completeness never establishes license approval, target applicability, promotion truth, or hardware authority.</p>
<p>Root: <code>{html.escape(report.root)}</code><br>Hash mode: <code>{report.hash_mode.value}</code><br>Report hash: <code>{report.report_hash}</code></p>
<table><thead><tr><th>Dataset</th><th>State</th><th>Files</th><th>GiB</th><th>Label matches</th><th>Labels</th><th>License files</th><th>Errors</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Conclusion</h2><p>snapshot_complete={str(report.snapshot_complete).lower()}; training_eligible=false; hardware_authorized=false.</p>
</body></html>\n"""


def _retain_unique(values: list[str], value: str) -> None:
    if len(values) < _MAX_RETAINED_ERRORS and value not in values:
        values.append(value)


def _bounded_error(relative: str, exc: Exception) -> str:
    detail = " ".join(str(exc).split())[:768]
    return f"{relative}: {type(exc).__name__}: {detail}"


def _bounded_text(value: str) -> str:
    return " ".join(str(value).split())[:1024]


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["inspect_dataset_root", "write_dataset_doctor_artifacts"]
