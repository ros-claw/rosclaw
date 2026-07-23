"""Skill evaluation pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rosclaw.auto.promotion.gate import PromotionGate
from rosclaw.body.resolver import BodyResolver
from rosclaw.sandbox.evidence import artifacts_within, verify_promotion_receipt
from rosclaw.skill.evidence import write_eval_report
from rosclaw.skill.hash import (
    compute_candidate_evidence_hash,
    compute_skill_hashes,
    validate_candidate_id,
)
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
    if candidate_id is not None:
        candidate_id = validate_candidate_id(candidate_id)

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
    sandbox_ok = _simulation_receipt_check(pkg, receipt, candidate_id)
    report.checks["sandbox_eval"] = sandbox_ok
    replay_ok = sandbox_ok
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
            report.checks.get("package_integrity_check", False),
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
        report.evidence_domain = "SIMULATION"
        report.promotion_ceiling = "SIM"
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
    return _load_json_mapping(receipt_path)


def _load_json_mapping(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> dict[str, Any] | None:
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _simulation_receipt_check(
    pkg: SkillPackage,
    receipt: dict[str, Any] | None,
    candidate_id: str | None,
) -> bool:
    if not receipt or not candidate_id:
        return False
    if not artifacts_within(receipt, pkg.root):
        return False
    if receipt.get("evaluation_variant") != "candidate":
        return False
    if not _receipt_robot_compatible(pkg, receipt) or not _receipt_candidate_compatible(
        pkg, receipt, candidate_id
    ):
        return False
    try:
        return verify_promotion_receipt(receipt).verified
    except Exception:  # noqa: BLE001 - skill promotion must fail closed
        return False


def _receipt_robot_compatible(pkg: SkillPackage, receipt: dict[str, Any]) -> bool:
    if pkg.eurdf_compat is None:
        return False
    compatible = {item.robot for item in pkg.eurdf_compat.compatible_robots}
    request = receipt.get("request")
    if not isinstance(request, dict):
        return False
    scenario = request.get("scenario")
    if not isinstance(scenario, dict):
        return False
    return str(scenario.get("robot_id") or "") in compatible


def _receipt_candidate_compatible(
    pkg: SkillPackage, receipt: dict[str, Any], candidate_id: str
) -> bool:
    try:
        expected_hash = compute_candidate_evidence_hash(pkg.root, candidate_id)
    except (OSError, ValueError):
        return False
    request = receipt.get("request")
    scenario = request.get("scenario") if isinstance(request, dict) else None
    metadata = scenario.get("metadata") if isinstance(scenario, dict) else None
    return bool(
        isinstance(metadata, dict)
        and metadata.get("skill_candidate_id") == candidate_id
        and metadata.get("skill_candidate_hash") == expected_hash
    )


def _darwin_eval(pkg: SkillPackage, candidate_id: str | None) -> tuple[dict[str, Any], bool]:
    if pkg.darwin_eval is None or not pkg.darwin_eval.metrics:
        return {}, False
    if not candidate_id:
        return {}, False
    report_path = pkg.root / "evidence" / "reports" / f"{candidate_id}_darwin.json"
    report = _load_json_mapping(report_path)
    if report is None:
        return {}, False
    per_seed = report.get("per_seed")
    receipts = report.get("simulation_receipts")
    regression = report.get("regression_results") or {}
    baseline_metrics = report.get("baseline_metrics")
    candidate_metrics = report.get("candidate_metrics")
    if not (
        report.get("evidence_domain") == "SIMULATION"
        and report.get("physics_executed") is True
        and isinstance(baseline_metrics, dict)
        and isinstance(candidate_metrics, dict)
        and isinstance(per_seed, dict)
        and len(per_seed) >= 2
        and isinstance(receipts, list)
        and receipts
        and all(
            isinstance(receipt, dict)
            and artifacts_within(receipt, pkg.root)
            and _receipt_robot_compatible(pkg, receipt)
            and _receipt_candidate_compatible(pkg, receipt, candidate_id)
            for receipt in receipts
        )
    ):
        return {}, False
    gate = PromotionGate().evaluate(
        baseline_metrics,
        candidate_metrics,
        current_level="baseline",
        per_seed=per_seed,
        sandbox_risk_score=candidate_metrics.get("collision_rate"),
        simulation_receipts=receipts,
        regression_results=regression,
    )
    if not gate.passed:
        return {}, False

    ok = True
    metrics: dict[str, float] = {}
    for metric_name, threshold in pkg.darwin_eval.metrics.items():
        value = candidate_metrics.get(metric_name)
        if value is None:
            if threshold.required:
                ok = False
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            ok = False
            continue
        metrics[metric_name] = float(value)
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
