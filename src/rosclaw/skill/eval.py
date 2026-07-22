"""Skill evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rosclaw.body.resolver import BodyResolver
from rosclaw.skill.evidence import write_eval_report
from rosclaw.skill.hash import compute_skill_hashes
from rosclaw.skill.models import EvalReport, SkillPackage
from rosclaw.skill.validators import validate_package


def evaluate_skill(
    pkg: SkillPackage,
    candidate_id: str | None = None,
    mode: str = "replay",
    save_evidence: bool = True,
) -> EvalReport:
    if pkg.skill is None:
        raise RuntimeError("skill.yaml not loaded")

    if candidate_id is None:
        candidate_id = pkg.skill.metadata.candidate_id

    report = EvalReport(
        skill=pkg.name,
        candidate_id=candidate_id,
        version=pkg.version,
        stage=pkg.skill.metadata.stage,
        mode=mode,
    )

    # 1. Schema/package validation.
    validation = validate_package(pkg)
    report.checks["schema_lint"] = validation.checks.get("skill_schema", False)
    report.checks["package_integrity_check"] = validation.checks.get("package_integrity", False)
    report.checks["behavior_tree_lint"] = validation.checks.get("behavior_tree_lint", False)
    report.checks["provider_route_check"] = validation.checks.get("providers_schema", False)
    report.checks["e_urdf_compat_check"] = validation.checks.get("eurdf_compat_schema", False)
    report.checks["safety_policy_check"] = validation.checks.get("safety_schema", False)

    # 2. e-URDF compatibility against linked body (best-effort).
    try:
        resolver = BodyResolver()
        if resolver.is_linked() and pkg.eurdf_compat:
            report.checks["e_urdf_compat_check"] = True
    except Exception:  # noqa: BLE001
        report.checks["e_urdf_compat_check"] = False

    # 3. Sandbox receipt and strict replay evidence. Missing evidence is not a
    # pass: it yields NEED_MORE_EVIDENCE below.
    receipt = _load_simulation_receipt(pkg, candidate_id)
    sandbox_ok = _simulation_receipt_check(pkg, receipt)
    report.checks["sandbox_eval"] = sandbox_ok
    replay_ok = _replay_check(receipt)
    report.checks["replay_check"] = replay_ok

    # 4. Darwin eval against thresholds. Metrics must come from a
    # physics-backed report, never a mining-count heuristic.
    metrics, darwin_ok = _darwin_eval(pkg, candidate_id)
    report.metrics = metrics
    report.checks["darwin_eval"] = darwin_ok

    # 6. Promotion gate.
    report.checks["promotion_gate_check"] = all(
        [
            report.checks.get("schema_lint", False),
            report.checks.get("behavior_tree_lint", False),
            report.checks.get("provider_route_check", False),
            report.checks.get("e_urdf_compat_check", False),
            report.checks.get("safety_policy_check", False),
            sandbox_ok,
            replay_ok,
            darwin_ok,
        ]
    )

    if report.checks["promotion_gate_check"]:
        report.decision = "pass"
    elif all(
        report.checks.get(name, False)
        for name in (
            "schema_lint",
            "package_integrity_check",
            "behavior_tree_lint",
            "provider_route_check",
            "e_urdf_compat_check",
            "safety_policy_check",
        )
    ) and not (sandbox_ok and replay_ok and darwin_ok):
        report.decision = "need_more_evidence"
    else:
        report.decision = "fail"

    if save_evidence:
        report.artifacts = {"report": str(write_eval_report(pkg.root, report))}

    return report


def _load_simulation_receipt(pkg: SkillPackage, candidate_id: str | None) -> dict[str, Any] | None:
    if not candidate_id:
        return None
    receipt_path = pkg.root / "evidence" / "receipts" / f"{candidate_id}.json"
    if not receipt_path.is_file():
        return None
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _artifact_path(pkg: SkillPackage, reference: str) -> Path | None:
    parsed = urlparse(reference)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    if parsed.scheme:
        return None
    return (pkg.root / reference).resolve()


def _simulation_receipt_check(pkg: SkillPackage, receipt: dict[str, Any] | None) -> bool:
    if not receipt:
        return False
    simulation = receipt.get("simulation_result") or {}
    dispatch = receipt.get("dispatch_result") or {}
    quality = (receipt.get("verification_result") or {}).get("data_quality") or {}
    artifact_hashes = simulation.get("artifact_hashes")
    if not (
        receipt.get("execution_mode", receipt.get("mode")) == "SIMULATION"
        and receipt.get("evidence_domain") == "SIMULATION"
        and receipt.get("body_snapshot_hash")
        and simulation.get("has_physics") is True
        and simulation.get("physics_executed") is True
        and dispatch.get("physics_executed") is True
        and simulation.get("model_hash")
        and simulation.get("action_hash")
        and isinstance(artifact_hashes, dict)
        and artifact_hashes
        and quality.get("body_snapshot_match") is True
    ):
        return False

    artifacts = receipt.get("artifacts") or []
    paths = {
        path.name: path
        for reference in artifacts
        if isinstance(reference, str) and (path := _artifact_path(pkg, reference)) is not None
    }
    for name, expected in artifact_hashes.items():
        path = paths.get(str(name))
        if path is None or not path.is_file():
            return False
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(expected).removeprefix("sha256:"):
            return False
    return True


def _replay_check(receipt: dict[str, Any] | None) -> bool:
    if not receipt:
        return False
    verification = receipt.get("verification_result") or {}
    quality = verification.get("data_quality") or {}
    replay = receipt.get("replay") or {}
    return bool(
        (quality.get("replayable") is True or replay.get("verified") is True)
        and replay.get("environment_match", True) is True
        and replay.get("hashes_verified", True) is True
    )


def _darwin_eval(pkg: SkillPackage, candidate_id: str | None) -> tuple[dict[str, Any], bool]:
    if pkg.darwin_eval is None or not pkg.darwin_eval.metrics:
        return {}, False
    if not candidate_id:
        return {}, False
    report_path = pkg.root / "evidence" / "reports" / f"{candidate_id}_darwin.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, False
    per_seed = report.get("per_seed")
    regression = report.get("regression") or {}
    if not (
        report.get("evidence_domain") == "SIMULATION"
        and report.get("physics_executed") is True
        and isinstance(per_seed, dict)
        and len(per_seed) >= 2
        and all(
            isinstance(item, dict)
            and isinstance(item.get("baseline"), dict)
            and isinstance(item.get("candidate"), dict)
            for item in per_seed.values()
        )
        and regression.get("passed") is True
        and not regression.get("critical_regressions")
    ):
        return {}, False
    metrics = report.get("candidate_metrics") or report.get("metrics") or {}
    if not isinstance(metrics, dict):
        return {}, False

    ok = True
    for metric_name, threshold in pkg.darwin_eval.metrics.items():
        value = metrics.get(metric_name)
        if value is None:
            if threshold.required:
                ok = False
            continue
        if threshold.promote_threshold is not None and value < threshold.promote_threshold:
            ok = False
        if threshold.max_allowed is not None and value > threshold.max_allowed:
            ok = False
        if threshold.max_mean is not None and value > threshold.max_mean:
            ok = False
    return metrics, ok


def refresh_hashes(pkg: SkillPackage) -> None:
    hashes = compute_skill_hashes(pkg.root, include_evidence=False)
    pkg.write_hashes_json(hashes)
