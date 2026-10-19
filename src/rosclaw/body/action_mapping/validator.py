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
        if isinstance(mapped.block_reasons, (list, tuple)):
            issues.extend(str(reason) for reason in mapped.block_reasons)
        else:
            issues.append("Malformed block reasons")

    values = mapped.body_action_values
    names = mapped.body_joint_names
    if not isinstance(values, (list, tuple)):
        issues.append("Body action values must be a flat sequence")
        values = []
    if not isinstance(names, (list, tuple)):
        issues.append("Body joint names must be a sequence")
        names = []
    dim = len(names)
    if not dim:
        issues.append("Body has no controllable joints")
    elif any(not isinstance(name, str) or not name for name in names):
        issues.append("Body joint names must be nonempty strings")
    elif len(set(names)) != dim:
        issues.append("Duplicate body joint names are not allowed")

    chunks = 1 if mapped.chunk_size is None else mapped.chunk_size
    valid_chunks = type(chunks) is int and chunks > 0
    if not valid_chunks:
        issues.append("Chunk size must be a positive integer or None")
    elif len(values) != dim * chunks:
        issues.append(f"Action length mismatch: expected {dim * chunks} values, got {len(values)}")

    if joint_limits is not None and not isinstance(joint_limits, dict):
        issues.append("Joint limits must be a dictionary")
        joint_limits = None

    # Iterate actual values rather than indexing an assumed complete chunk.
    # This checks even excess values without indexing beyond the name list.
    for i, value in enumerate(values):
        joint_name = names[i % dim] if dim else "unknown"
        try:
            finite = math.isfinite(value)
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            issues.append(f"Invalid value at index {i} (joint {joint_name}): {value}")
            continue

        if not joint_limits or not dim or not isinstance(joint_name, str):
            continue
        limits = joint_limits.get(joint_name, {})
        if not isinstance(limits, dict):
            issues.append(f"Malformed limits for joint {joint_name}")
            continue
        chunk_idx = i // dim
        for key, direction in (("lower", "below"), ("upper", "above")):
            bound = limits.get(key)
            if bound is None:
                continue
            try:
                if not math.isfinite(bound):
                    issues.append(f"Invalid {key} limit for joint {joint_name}")
                    continue
                violated = value < bound if key == "lower" else value > bound
            except (TypeError, ValueError, OverflowError):
                issues.append(f"Invalid {key} limit for joint {joint_name}")
                continue
            if violated:
                operator = "<" if key == "lower" else ">"
                issues.append(
                    f"Joint {joint_name} chunk {chunk_idx} {direction} {key} limit: "
                    f"{value} {operator} {bound}"
                )

    return {
        "blocked": mapped.blocked or bool(issues),
        "compatibility": mapped.compatibility.value,
        "issues": issues,
        "warnings": mapped.warnings,
    }
