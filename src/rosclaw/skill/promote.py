"""Skill candidate promotion logic."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import semver

from rosclaw.skill.evidence import load_eval_report_dict
from rosclaw.skill.hash import compute_skill_hashes
from rosclaw.skill.models import LineageVersion, SkillPackage


def promote_candidate(
    pkg: SkillPackage,
    candidate_id: str,
    to_version: str,
    stage: str = "validated",
    require_eval_pass: bool = True,
) -> dict[str, Any]:
    if pkg.skill is None or pkg.lineage is None:
        raise RuntimeError("skill.yaml or lineage.yaml not loaded")

    candidate = next((c for c in pkg.lineage.candidates if c.id == candidate_id), None)
    if candidate is None:
        raise ValueError(f"Candidate {candidate_id!r} not found in lineage")

    eval_report = load_eval_report_dict(pkg.root, candidate_id)
    if eval_report is None:
        raise ValueError(f"No eval report found for {candidate_id}; run `rosclaw skill eval` first")

    if require_eval_pass and eval_report.get("decision") != "pass":
        raise ValueError(f"Eval did not pass for {candidate_id}; promotion blocked")

    # Safety gate: no_fall_rate must be perfect if required.
    metrics = eval_report.get("metrics", {})
    if pkg.darwin_eval and "no_fall_rate" in pkg.darwin_eval.metrics:
        threshold = pkg.darwin_eval.metrics["no_fall_rate"].promote_threshold or 1.0
        if metrics.get("no_fall_rate", 0.0) < threshold:
            raise ValueError(f"no_fall_rate below threshold {threshold}; promotion blocked")

    # Validate semver.
    semver.VersionInfo.parse(to_version)

    # Compute hashes and snapshot current state.
    hashes = compute_skill_hashes(pkg.root, include_evidence=False)
    pkg.write_hashes_json(hashes)

    _snapshot_files(pkg, to_version)

    # Update skill.yaml.
    pkg.skill.metadata.version = to_version
    pkg.skill.metadata.stage = stage
    pkg.skill.metadata.candidate_id = None
    pkg.skill.status.promotion_state = stage
    pkg.skill.status.last_eval_passed = True
    pkg.skill.status.safe_to_run_on_real_robot = stage == "validated" and metrics.get("no_fall_rate", 0.0) >= 1.0
    pkg.skill.evidence.latest_eval_report = f"evidence/reports/{candidate_id}_eval.json"

    # Update lineage.
    pkg.lineage.skill["current_version"] = to_version
    pkg.lineage.skill["current_stage"] = stage
    candidate.status = stage
    pkg.lineage.versions.append(
        LineageVersion(
            version=to_version,
            candidate_id=candidate_id,
            package_hash=hashes["package_hash"],
            promoted_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            promoted_by="local",
        )
    )

    # Update CHANGELOG.
    _update_changelog(pkg, to_version, candidate_id, metrics)

    # Atomic writes.
    pkg.write_skill_yaml()
    pkg.write_lineage_yaml()

    # Update lock.
    import yaml

    lock_path = pkg.root / ".rosclaw" / "lock.yaml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {"schema_version": "rosclaw.lock.v1"}
    lock["hashes"] = hashes["files"]
    pkg.write_lock_yaml(lock)

    return {
        "version": to_version,
        "stage": stage,
        "candidate_id": candidate_id,
        "package_hash": hashes["package_hash"],
    }


def _snapshot_files(pkg: SkillPackage, version: str) -> None:
    snapshot_dir = pkg.root / ".rosclaw" / "snapshots" / version
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "skill.yaml",
        "behavior_tree.xml",
        "providers.yaml",
        "safety.yaml",
        "darwin_eval.yaml",
        "policies/params/default.yaml",
    ]
    for rel in files:
        src = pkg.root / rel
        if src.exists():
            dest = snapshot_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())


def _update_changelog(pkg: SkillPackage, version: str, candidate_id: str, metrics: dict[str, Any]) -> None:
    changelog = pkg.root / "CHANGELOG.md"
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    lines = [
        "",
        f"## [{version}] - {date}",
        "",
        "### Added",
        f"- Promoted from {candidate_id}.",
        "",
        "### Evidence",
    ]
    for k, v in metrics.items():
        lines.append(f"- {k}: {v}")
    lines.extend([
        "",
        "### Safety",
        "- Requires sandbox_first mode.",
        "",
    ])
    existing = changelog.read_text(encoding="utf-8") if changelog.exists() else "# Changelog\n"
    # Insert after first line.
    parts = existing.split("\n", 1)
    new_text = parts[0] + "\n" + "\n".join(lines) + ("\n" + parts[1] if len(parts) > 1 else "")
    changelog.write_text(new_text, encoding="utf-8")
