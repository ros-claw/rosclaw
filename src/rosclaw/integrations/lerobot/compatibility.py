"""LeRobot policy compatibility matrix for ROSClaw.

This module maps LeRobot policy families to the bridge capabilities that are
expected to work in the current P1.1 implementation.  It also consumes a real
smoke report to upgrade a policy from "listed" to "validated" once a
`rosclaw lerobot smoke-policy` run has succeeded.

The matrix must remain free of torch/lerobot imports so it can be used from the
ROSClaw core interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rosclaw.integrations.lerobot.smoke_report import (
    SmokeReport,
    read_latest_smoke_report,
)

COMPATIBILITY_LEVELS = [
    "unsupported",
    "listed",
    "inspect_ok",
    "load_ok",
    "infer_ok",
    "validated",
]


@dataclass(frozen=True)
class PolicyCompatibility:
    """Compatibility row for a single LeRobot policy family."""

    policy_type: str
    inspect: bool
    load_test: bool
    infer: bool
    body_mapping_required: bool = True
    body_compatible: bool = False
    notes: str = ""


# P1.1 compatibility matrix.  Keep this conservative: only ACT has been
# validated through a real smoke target in this round. Other known families are
# listed as inspect-only until a real checkpoint is load-tested/inferred.
POLICY_COMPATIBILITY_MATRIX: dict[str, PolicyCompatibility] = {
    "act": PolicyCompatibility(
        policy_type="act",
        inspect=True,
        load_test=True,
        infer=True,
        notes="Tested ALOHA-style ACT policies; action chunks supported.",
    ),
    "diffusion": PolicyCompatibility(
        policy_type="diffusion",
        inspect=True,
        load_test=False,
        infer=False,
        notes="Pending real smoke; diffusion observation/action preprocessing is not validated in P1.1.",
    ),
    "vqbet": PolicyCompatibility(
        policy_type="vqbet",
        inspect=True,
        load_test=False,
        infer=False,
        notes="Pending real smoke; VQ-BeT loading and inference are not validated in P1.1.",
    ),
    "tdmpc": PolicyCompatibility(
        policy_type="tdmpc",
        inspect=True,
        load_test=False,
        infer=False,
        notes="Pending real smoke; TDMPC loading and observation preprocessing are not validated in P1.1.",
    ),
}


def get_policy_compatibility(policy_type: str | None) -> PolicyCompatibility:
    """Return the compatibility row for a policy family.

    Unknown families are treated as ``listed`` (inspect only, no guarantees).
    ``None`` returns an unsupported placeholder.
    """
    if not policy_type:
        return PolicyCompatibility(
            policy_type="unknown",
            inspect=False,
            load_test=False,
            infer=False,
            notes="No policy type provided.",
        )

    normalized = policy_type.lower().strip()
    if normalized in POLICY_COMPATIBILITY_MATRIX:
        return POLICY_COMPATIBILITY_MATRIX[normalized]

    return PolicyCompatibility(
        policy_type=policy_type,
        inspect=True,
        load_test=False,
        infer=False,
        notes="Policy type is not in the P1.1 compatibility matrix; inspect only.",
    )


def classify_compatibility_level(
    policy_type: str | None,
    report: SmokeReport | None = None,
) -> str:
    """Return the compatibility level for ``policy_type``.

    Levels are ordered from least to most supported:
    ``unsupported < listed < inspect_ok < load_ok < infer_ok < validated``.
    """
    compat = get_policy_compatibility(policy_type)

    if not compat.inspect:
        return "unsupported"

    # If we have a successful smoke report for the same policy type, treat it
    # as validated for this runtime.
    if report is not None and report.status == "ok":
        report_type = (report.policy.get("policy_type") or "").lower()
        if report_type == compat.policy_type.lower():
            return "validated"

    if compat.infer:
        return "infer_ok"
    if compat.load_test:
        return "load_ok"
    return "inspect_ok"


def build_compatibility_report(
    policy_type: str | None = None,
    report: SmokeReport | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable compatibility report.

    If ``policy_type`` is given, report on that family.  Otherwise report on
    every family in the matrix plus the latest validated policy (if any).
    """
    if report is None:
        report = read_latest_smoke_report()

    if policy_type:
        compat = get_policy_compatibility(policy_type)
        level = classify_compatibility_level(policy_type, report=report)
        return {
            "policy_type": policy_type,
            "level": level,
            "inspect": compat.inspect,
            "load_test": compat.load_test,
            "infer": compat.infer,
            "body_mapping_required": compat.body_mapping_required,
            "body_compatible": compat.body_compatible,
            "notes": compat.notes,
            "validated_by_report": level == "validated",
        }

    rows: list[dict[str, Any]] = []
    for compat in POLICY_COMPATIBILITY_MATRIX.values():
        level = classify_compatibility_level(compat.policy_type, report=report)
        rows.append(
            {
                "policy_type": compat.policy_type,
                "level": level,
                "inspect": compat.inspect,
                "load_test": compat.load_test,
                "infer": compat.infer,
                "body_mapping_required": compat.body_mapping_required,
                "body_compatible": compat.body_compatible,
                "notes": compat.notes,
                "validated_by_report": level == "validated",
            }
        )

    latest: dict[str, Any] | None = None
    if report is not None:
        latest = {
            "policy_type": report.policy.get("policy_type"),
            "policy": report.policy.get("repo_id") or report.policy.get("local_path"),
            "status": report.status,
            "validated": report.status == "ok",
            "time": report.created_at,
        }

    return {
        "matrix": rows,
        "latest_smoke": latest,
        "levels": COMPATIBILITY_LEVELS,
    }


def format_compatibility_text(report: dict[str, Any]) -> str:
    """Render a compatibility report as human-readable text."""
    lines: list[str] = []
    lines.append("LeRobot Policy Compatibility Matrix (P1.1)")
    lines.append("")

    matrix = report.get("matrix", [])
    single = report.get("policy_type")

    if single and not matrix:
        # Single-policy-type report: render the focused row.
        lines.append(
            f"{'Policy Type':<12} {'Level':<12} {'Inspect':<8} {'Load':<8} {'Infer':<8} Notes"
        )
        lines.append("-" * 90)
        lines.append(
            f"{report['policy_type']:<12} "
            f"{report['level']:<12} "
            f"{'yes' if report['inspect'] else 'no':<8} "
            f"{'yes' if report['load_test'] else 'no':<8} "
            f"{'yes' if report['infer'] else 'no':<8} "
            f"{report.get('notes', '')}"
        )
    elif matrix:
        lines.append(
            f"{'Policy Type':<12} {'Level':<12} {'Inspect':<8} {'Load':<8} {'Infer':<8} Notes"
        )
        lines.append("-" * 90)
        for row in matrix:
            lines.append(
                f"{row['policy_type']:<12} "
                f"{row['level']:<12} "
                f"{'yes' if row['inspect'] else 'no':<8} "
                f"{'yes' if row['load_test'] else 'no':<8} "
                f"{'yes' if row['infer'] else 'no':<8} "
                f"{row.get('notes', '')}"
            )
    else:
        lines.append("No compatibility data available.")

    latest = report.get("latest_smoke")
    if latest:
        lines.append("")
        lines.append("Latest Smoke Report")
        lines.append(f"  Policy:     {latest.get('policy') or 'N/A'}")
        lines.append(f"  Type:       {latest.get('policy_type') or 'N/A'}")
        lines.append(f"  Status:     {latest.get('status')}")
        lines.append(f"  Validated:  {'yes' if latest.get('validated') else 'no'}")
        lines.append(f"  Time:       {latest.get('time') or 'N/A'}")

    return "\n".join(lines)


__all__ = [
    "COMPATIBILITY_LEVELS",
    "POLICY_COMPATIBILITY_MATRIX",
    "PolicyCompatibility",
    "build_compatibility_report",
    "classify_compatibility_level",
    "format_compatibility_text",
    "get_policy_compatibility",
]
