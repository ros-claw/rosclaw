"""Validate a body action mapping against safety and compatibility rules."""

from __future__ import annotations

from typing import Any

from rosclaw.body.action_mapping.schema import ActionMapping, MappedAction, MappingCompatibility


def validate_action_mapping(mapping: ActionMapping) -> dict[str, Any]:
    """Return a validation report for an action mapping.

    The report is meant for preflight checks: it is conservative and blocks
    anything that is not ``exact`` or ``convertible`` unless ``allow_partial``
    is enabled.
    """
    blocked = mapping.is_blocked()
    status = "blocked" if blocked else "ok"
    if mapping.compatibility == MappingCompatibility.EXACT:
        status = "ok"
    elif mapping.compatibility == MappingCompatibility.CONVERTIBLE:
        status = "ok_with_conversion"
    elif mapping.compatibility == MappingCompatibility.PARTIAL:
        status = "blocked" if not mapping.allow_partial else "partial"
    elif mapping.compatibility == MappingCompatibility.INCOMPATIBLE:
        status = "blocked"
    elif mapping.compatibility == MappingCompatibility.UNKNOWN:
        status = "blocked"

    joint_issues = [
        {
            "policy_name": j.policy_name,
            "body_name": j.body_name,
            "compatible": j.compatible,
            "reason": j.reason,
        }
        for j in mapping.joints
        if not j.compatible or j.policy_index < 0
    ]

    return {
        "status": status,
        "blocked": blocked,
        "compatibility": mapping.compatibility.value,
        "allow_partial": mapping.allow_partial,
        "joint_count": len(mapping.joints),
        "matched_joints": sum(1 for j in mapping.joints if j.policy_index >= 0),
        "block_reasons": list(mapping.block_reasons),
        "warnings": list(mapping.warnings),
        "joint_issues": joint_issues,
    }


def validate_mapped_action(
    mapped: MappedAction,
    *,
    joint_limits: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Check a mapped action for NaN/Inf and optional joint-limit violations."""
    import math

    issues: list[str] = []
    if mapped.blocked:
        issues.extend(mapped.block_reasons)

    values = mapped.body_action_values
    for i, value in enumerate(values):
        if math.isnan(value) or math.isinf(value):
            issues.append(
                f"Invalid value at index {i} (joint {mapped.body_joint_names[i]}): {value}"
            )

    if joint_limits:
        dim = len(mapped.body_joint_names)
        chunks = mapped.chunk_size or 1
        for chunk_idx in range(chunks):
            offset = chunk_idx * dim
            for j_idx, joint_name in enumerate(mapped.body_joint_names):
                value = values[offset + j_idx]
                limits = joint_limits.get(joint_name, {})
                lower = limits.get("lower")
                upper = limits.get("upper")
                if lower is not None and value < lower:
                    issues.append(
                        f"Joint {joint_name} chunk {chunk_idx} below lower limit: "
                        f"{value} < {lower}"
                    )
                if upper is not None and value > upper:
                    issues.append(
                        f"Joint {joint_name} chunk {chunk_idx} above upper limit: "
                        f"{value} > {upper}"
                    )

    return {
        "blocked": mapped.blocked or bool(issues),
        "compatibility": mapped.compatibility.value,
        "issues": issues,
        "warnings": mapped.warnings,
    }
