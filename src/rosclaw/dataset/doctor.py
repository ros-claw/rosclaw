"""Point-in-time, transfer-aware inventory for operator-managed datasets."""

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
    DatasetFileRecord,
    DatasetInventory,
    DatasetSnapshotState,
    FileHashMode,
)
from rosclaw.feedback.contracts import canonical_hash

_PARTIAL_MARKERS = (
    ".aria2",
    ".download",
    ".incomplete",
    ".part",
    ".partial",
    ".tmp",
)
_FOOTBALL_SPORT_TOKENS = frozenset(
    {"football", "soccer", "futsal", "goalkeeper", "goalkeeping"}
)
_FOOTBALL_BALL_ACTIONS = frozenset({"kick", "kicking", "shoot", "shooting", "pass"})
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


def inspect_dataset_root(
    root: Path,
    *,
    hash_mode: FileHashMode = FileHashMode.METADATA,
    transfer_active: bool = False,
    football_match_limit: int = 100,
) -> DatasetDoctorReport:
    """Inspect each top-level dataset without following symbolic links.

    ``transfer_active`` is an operator assertion.  It prevents an otherwise
    clean prefix of a download from being mislabeled as a complete snapshot.
    Content hashing re-stats each file and marks it unstable if it changes
    while being read.
    """

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a readable directory: {root}")
    if football_match_limit <= 0:
        raise ValueError("football_match_limit must be positive")
    if not isinstance(hash_mode, FileHashMode):
        raise ValueError("hash_mode must be a FileHashMode")
    started = _timestamp()
    children = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name)
    if not children:
        raise ValueError("dataset root contains no top-level dataset directories")
    inventories = tuple(
        _inspect_one(
            path,
            hash_mode=hash_mode,
            transfer_active=transfer_active,
            football_match_limit=football_match_limit,
        )
        for path in children
    )
    return DatasetDoctorReport(
        root=str(root),
        started_at=started,
        finished_at=_timestamp(),
        hash_mode=hash_mode,
        transfer_active=transfer_active,
        inventories=inventories,
    )


def write_dataset_doctor_artifacts(report: DatasetDoctorReport, output_dir: Path) -> dict[str, str]:
    """Write the four review artifacts promised by the dataset surface."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "inventory": output_dir / "dataset_inventory.json",
        "quality_report": output_dir / "dataset_quality_report.html",
        "license_manifest": output_dir / "license_manifest.json",
        "football_asset_matrix": output_dir / "football_asset_matrix.csv",
    }
    inventory = report.to_dict()
    inventory["report_hash"] = report.report_hash
    _atomic_write_text(
        paths["inventory"], json.dumps(inventory, ensure_ascii=False, indent=2) + "\n"
    )
    license_manifest = {
        "schema_version": "rosclaw.dataset.license_manifest.v1",
        "report_hash": report.report_hash,
        "datasets": [
            {
                "dataset_id": value.dataset_id,
                "license_files": list(value.license_files),
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
    _atomic_write_text(paths["football_asset_matrix"], _asset_matrix_csv(report))
    _atomic_write_text(paths["quality_report"], _quality_html(report))
    return {key: str(value) for key, value in paths.items()}


def _inspect_one(
    root: Path,
    *,
    hash_mode: FileHashMode,
    transfer_active: bool,
    football_match_limit: int,
) -> DatasetInventory:
    files: list[DatasetFileRecord] = []
    scan_errors: list[str] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = sorted(directory_names)
        for file_name in sorted(file_names):
            path = Path(directory) / file_name
            relative = path.relative_to(root).as_posix()
            try:
                files.append(_inspect_file(path, relative=relative, hash_mode=hash_mode))
            except (OSError, PermissionError) as exc:
                scan_errors.append(f"{relative}: {type(exc).__name__}: {exc}")
    files.sort(key=lambda value: value.relative_path)
    extension_counts = Counter(value.suffix or "<none>" for value in files)
    issue_counts = Counter(issue for value in files for issue in value.issue_codes)
    license_files = tuple(
        value.relative_path
        for value in files
        if Path(value.relative_path).name.lower() in _LICENSE_NAMES
    )
    football_matches = tuple(
        value.relative_path for value in files if _is_football_path(value.relative_path)
    )[:football_match_limit]
    digests: dict[str, list[str]] = defaultdict(list)
    if hash_mode is FileHashMode.SHA256:
        for value in files:
            if value.digest is not None and value.hash_status == "computed":
                digests[value.digest].append(value.relative_path)
    duplicate_groups = tuple(tuple(paths) for _, paths in sorted(digests.items()) if len(paths) > 1)
    state = _state(
        files,
        scan_errors=scan_errors,
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
        football_matches=football_matches,
        duplicate_content_groups=duplicate_groups,
        files=tuple(files),
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
        if hash_mode is FileHashMode.METADATA:
            digest = canonical_hash(
                {
                    "relative_path": relative,
                    "size_bytes": before.st_size,
                    "mtime_ns": before.st_mtime_ns,
                }
            )
            hash_status = "metadata_only"
        elif hash_mode is FileHashMode.SHA256:
            digest = _sha256_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                digest = None
                hash_status = "changed_during_scan"
                issues.append("changed_during_scan")
            else:
                hash_status = "computed"
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
    scan_errors: list[str],
    transfer_active: bool,
) -> DatasetSnapshotState:
    if scan_errors and not files:
        return DatasetSnapshotState.UNREADABLE
    if transfer_active:
        return DatasetSnapshotState.TRANSFERRING
    if not files:
        return DatasetSnapshotState.EMPTY
    issues = {issue for value in files for issue in value.issue_codes}
    if "partial_transfer_file" in issues or "changed_during_scan" in issues:
        return DatasetSnapshotState.TRANSFERRING
    if scan_errors or "git_lfs_pointer" in issues or "not_regular_file" in issues:
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
    return ".part_" in lowered or any(lowered.endswith(marker) for marker in _PARTIAL_MARKERS)


def _is_football_path(path: str) -> bool:
    lowered = path.lower()
    # Hugging Face download metadata mirrors the real path and must not count
    # as a second training asset.  Generic words such as ``pass`` and
    # ``dribble`` are not football evidence on their own: MotionDecode also
    # contains object passing and basketball dribbling.
    components = lowered.split("/")
    if ".cache" in components or lowered.endswith(".metadata"):
        return False
    tokens = tuple(re.findall(r"[a-z0-9]+", lowered))
    if _FOOTBALL_SPORT_TOKENS.intersection(tokens):
        return True
    return any(
        first in _FOOTBALL_BALL_ACTIONS and second == "ball"
        or first == "ball" and second in _FOOTBALL_BALL_ACTIONS
        for first, second in zip(tokens, tokens[1:], strict=False)
    )


def _asset_matrix_csv(report: DatasetDoctorReport) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "dataset_id",
            "snapshot_state",
            "file_count",
            "size_bytes",
            "football_match_count",
            "license_file_count",
            "partial_file_count",
            "lfs_pointer_count",
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
                len(value.football_matches),
                len(value.license_files),
                value.issue_counts.get("partial_transfer_file", 0),
                value.issue_counts.get("git_lfs_pointer", 0),
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
        f"<td>{len(value.football_matches)}</td>"
        f"<td>{len(value.license_files)}</td>"
        f"<td>{html.escape(', '.join(f'{key}={count}' for key, count in sorted(value.issue_counts.items())) or 'none')}</td>"
        "</tr>"
        for value in report.inventories
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROSClaw Dataset Doctor</title>
<style>body{{font:15px/1.55 system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:#171717}}code{{background:#f2f2f2;padding:2px 6px;border-radius:5px}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;text-align:left;padding:9px;vertical-align:top}}th{{background:#f7f7f7}}.warn{{padding:14px;border-left:4px solid #d97706;background:#fff7ed}}</style></head>
<body><h1>ROSClaw Dataset Doctor</h1>
<p class="warn"><strong>证据边界：</strong>这是点时快照。transfer_active={str(report.transfer_active).lower()}；任何数据集即使技术完整，也必须完成许可证和目标身体适用性审查后才能训练，且不能作为 Promotion 物理真值。</p>
<p>Root: <code>{html.escape(report.root)}</code><br>Hash mode: <code>{report.hash_mode.value}</code><br>Report hash: <code>{report.report_hash}</code></p>
<table><thead><tr><th>数据集</th><th>状态</th><th>文件</th><th>GiB</th><th>足球命中</th><th>许可文件</th><th>问题</th></tr></thead><tbody>{rows}</tbody></table>
<h2>结论</h2><p>snapshot_complete={str(report.snapshot_complete).lower()}；training_eligible=false；hardware_authorized=false。</p>
</body></html>\n"""


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
