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
    elif (
        mapping.compatibility == MappingCompatibility.INCOMPATIBLE
        or mapping.compatibility == MappingCompatibility.UNKNOWN
    ):
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
    names = mapped.body_joint_names
    dim = len(names)
    chunks = 1 if mapped.chunk_size is None else mapped.chunk_size
    valid_chunks = type(chunks) is int and chunks > 0
    if not valid_chunks:
        issues.append("Invalid layout at $.chunk_size: expected a positive integer or null")
    # The mapper's blocked empty result is a sentinel, not an executable layout.
    empty_blocked = mapped.blocked and not values and not names
    valid_names = bool(names) and all(isinstance(n, str) and n.strip() for n in names)
    if valid_names:
        valid_names = len(set(names)) == dim
    if not valid_names and not empty_blocked:
        issues.append("Invalid layout at $.body_joint_names: expected nonempty unique joint names")
    valid_size = valid_chunks and len(values) == dim * chunks
    if not valid_size:
        issues.append(
            f"Invalid layout at $.body_action_values: {len(values)} values for "
            f"{dim} joints and chunk_size={mapped.chunk_size}"
        )

    for i, value in enumerate(values):
        if not math.isfinite(value):
            location = f"joint {names[i % dim]}, chunk {i // dim}" if dim else "no joint name"
            issues.append(
                f"Invalid value at index {i} ($.body_action_values[{i}], {location}): {value}"
            )
            continue
        # Do not interpret limits against an ambiguous or incomplete layout.
        if joint_limits and valid_names and valid_size:
            joint_name = names[i % dim]
            chunk_idx = i // dim
            limits = joint_limits.get(joint_name, {})
            lower = limits.get("lower")
            upper = limits.get("upper")
            if lower is not None and value < lower:
                issues.append(
                    f"Joint {joint_name} chunk {chunk_idx} below lower limit: {value} < {lower}"
                )
            if upper is not None and value > upper:
                issues.append(
                    f"Joint {joint_name} chunk {chunk_idx} above upper limit: {value} > {upper}"
                )

    return {
        "blocked": mapped.blocked or bool(issues),
        "compatibility": mapped.compatibility.value,
        "issues": issues,
        "warnings": mapped.warnings,
    }
