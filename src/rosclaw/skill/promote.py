"""Skill candidate promotion logic."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import semver

from rosclaw.skill.evidence import load_eval_report_dict
from rosclaw.skill.hash import candidate_artifact_paths, compute_skill_hashes, validate_candidate_id
from rosclaw.skill.models import STAGES, LineageVersion, SkillPackage


def promote_candidate(
    pkg: SkillPackage,
    candidate_id: str,
    to_version: str,
    stage: str = "validated",
    require_eval_pass: bool = True,
) -> dict[str, Any]:
    if pkg.skill is None or pkg.lineage is None:
        raise RuntimeError("skill.yaml or lineage.yaml not loaded")
    candidate_id = validate_candidate_id(candidate_id)
    if not require_eval_pass:
        raise ValueError("Evaluation bypass is not supported for skill promotion")
    if stage not in STAGES:
        raise ValueError(f"Unsupported promotion stage: {stage}")

    candidate = next((c for c in pkg.lineage.candidates if c.id == candidate_id), None)
    if candidate is None:
        raise ValueError(f"Candidate {candidate_id!r} not found in lineage")

    eval_report = load_eval_report_dict(pkg.root, candidate_id)
    if eval_report is None:
        raise ValueError(f"No eval report found for {candidate_id}; run `rosclaw skill eval` first")
    if (
        not isinstance(eval_report, dict)
        or eval_report.get("schema_version") != "rosclaw.eval_report.v1"
        or eval_report.get("candidate_id") != candidate_id
    ):
        raise ValueError(f"Eval report contract is invalid for {candidate_id}; promotion blocked")

    if eval_report.get("decision") != "pass":
        raise ValueError(f"Eval did not pass for {candidate_id}; promotion blocked")

    # A persisted decision is only a cache. Re-run all package, replay, Darwin,
    # and promotion checks so editing the report cannot authorize promotion.
    from rosclaw.skill.eval import evaluate_skill

    fresh_report = evaluate_skill(
        pkg,
        candidate_id=candidate_id,
        mode=str(eval_report.get("mode") or "replay"),
        save_evidence=True,
    )
    eval_report = fresh_report.to_dict()
    if eval_report.get("decision") != "pass":
        raise ValueError(f"Fresh eval did not pass for {candidate_id}; promotion blocked")
    if eval_report.get("evidence_domain") == "SIMULATION" and stage != "validated":
        raise ValueError("Simulation evidence may only promote a skill to the validated stage")

    # Safety gate: no_fall_rate must be perfect if required.
    metrics = eval_report.get("metrics", {})
    if pkg.darwin_eval and "no_fall_rate" in pkg.darwin_eval.metrics:
        threshold = pkg.darwin_eval.metrics["no_fall_rate"].promote_threshold or 1.0
        if metrics.get("no_fall_rate", 0.0) < threshold:
            raise ValueError(f"no_fall_rate below threshold {threshold}; promotion blocked")

    # Validate semver.
    semver.VersionInfo.parse(to_version)

    # Activate exactly the candidate files bound into the replayed receipts.
    _activate_candidate_files(pkg, candidate_id)

    # Bind lineage to the evaluated candidate state. Final package hashes are
    # recomputed after promotion metadata and changelog writes below.
    source_hashes = compute_skill_hashes(pkg.root, include_evidence=False)

    # Update skill.yaml.
    pkg.skill.metadata.version = to_version
    pkg.skill.metadata.stage = stage
    pkg.skill.metadata.candidate_id = None
    pkg.skill.status.promotion_state = stage
    pkg.skill.status.last_eval_passed = True
    # A simulation-validated skill may be packaged and deployed back to a
    # sandbox, but simulation evidence never authorizes hardware execution.
    pkg.skill.status.safe_to_run_on_real_robot = bool(
        stage == "validated"
        and eval_report.get("evidence_domain") == "HARDWARE"
        and eval_report.get("promotion_ceiling") == "REAL"
        and (eval_report.get("checks") or {}).get("hardware_verification") is True
    )
    pkg.skill.evidence.latest_eval_report = f"evidence/reports/{candidate_id}_eval.json"

    # Update lineage.
    pkg.lineage.skill["current_version"] = to_version
    pkg.lineage.skill["current_stage"] = stage
    candidate.status = stage
    pkg.lineage.versions.append(
        LineageVersion(
            version=to_version,
            candidate_id=candidate_id,
            package_hash=source_hashes["package_hash"],
            promoted_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            promoted_by="local",
        )
    )

    # Update CHANGELOG.
    _update_changelog(pkg, to_version, candidate_id, metrics)

    # Each metadata write is atomic; then hash the resulting promoted package.
    pkg.write_skill_yaml()
    pkg.write_lineage_yaml()

    hashes = compute_skill_hashes(pkg.root, include_evidence=False)
    pkg.write_hashes_json(hashes)
    _snapshot_files(pkg, to_version)

    # Update lock.
    import yaml

    lock_path = pkg.root / ".rosclaw" / "lock.yaml"
    lock = (
        yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        if lock_path.exists()
        else {"schema_version": "rosclaw.lock.v1"}
    )
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


def _activate_candidate_files(pkg: SkillPackage, candidate_id: str) -> None:
    candidates = candidate_artifact_paths(pkg.root, candidate_id)
    destinations = {
        "parameters": pkg.root / "policies" / "params" / "default.yaml",
        "behavior_tree": pkg.root / "behavior_tree.xml",
    }
    content: dict[str, bytes] = {}
    for name, path in candidates.items():
        if not path.is_file():
            raise ValueError(f"Candidate artifact is missing: {path.relative_to(pkg.root)}")
        content[name] = path.read_bytes()
    for name, destination in destinations.items():
        temporary = destination.with_suffix(destination.suffix + ".promote.tmp")
        temporary.write_bytes(content[name])
        temporary.replace(destination)


def _update_changelog(
    pkg: SkillPackage, version: str, candidate_id: str, metrics: dict[str, Any]
) -> None:
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
    lines.extend(
        [
            "",
            "### Safety",
            "- Requires sandbox_first mode.",
            "",
        ]
    )
    existing = changelog.read_text(encoding="utf-8") if changelog.exists() else "# Changelog\n"
    # Insert after first line.
    parts = existing.split("\n", 1)
    new_text = parts[0] + "\n" + "\n".join(lines) + ("\n" + parts[1] if len(parts) > 1 else "")
    changelog.write_text(new_text, encoding="utf-8")
