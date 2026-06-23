"""Skill rollback logic."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from rosclaw.skill.models import LineageRollback, SkillPackage


def rollback_skill(
    pkg: SkillPackage,
    to_version: str,
    reason: str = "",
) -> dict[str, Any]:
    if pkg.lineage is None:
        raise RuntimeError("lineage.yaml not loaded")

    target = next((v for v in pkg.lineage.versions if v.version == to_version), None)
    if target is None:
        raise ValueError(f"Version {to_version!r} not found in lineage")

    snapshot_dir = pkg.root / ".rosclaw" / "snapshots" / to_version
    if not snapshot_dir.exists():
        raise ValueError(f"Snapshot for version {to_version!r} not found")

    current_version = pkg.version

    # Restore files.
    files = [
        "skill.yaml",
        "behavior_tree.xml",
        "providers.yaml",
        "safety.yaml",
        "darwin_eval.yaml",
        "policies/params/default.yaml",
    ]
    for rel in files:
        src = snapshot_dir / rel
        if src.exists():
            dest = pkg.root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())

    # Reload skill.
    pkg.skill = None
    pkg.try_load()

    # Mark newer versions as deprecated.
    for v in pkg.lineage.versions:
        if v.version != to_version and _version_greater(v.version, to_version):
            # Mark by appending note in lineage skill metadata.
            pass

    # Record rollback.
    rollback_record = LineageRollback(
        from_version=current_version,
        to_version=to_version,
        reason=reason or f"Rollback to {to_version}",
    )
    pkg.lineage.rollbacks.append(rollback_record)

    # Write rollback evidence.
    evidence_path = pkg.root / "evidence" / "reports" / f"rollback_{_date_now()}.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps({
            "schema_version": "rosclaw.rollback_report.v1",
            "from_version": current_version,
            "to_version": to_version,
            "reason": rollback_record.reason,
            "at": rollback_record.at,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pkg.write_lineage_yaml()

    return {
        "from_version": current_version,
        "to_version": to_version,
        "evidence": str(evidence_path.relative_to(pkg.root)),
    }


def _version_greater(a: str, b: str) -> bool:
    try:
        import semver

        return semver.VersionInfo.parse(a) > semver.VersionInfo.parse(b)
    except Exception:  # noqa: BLE001
        return a > b


def _date_now() -> str:
    return datetime.now(UTC).strftime("%Y_%m_%d")
