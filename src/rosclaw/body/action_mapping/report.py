"""Human- and machine-readable reports for body action mapping."""

from __future__ import annotations

from typing import Any

from rosclaw.body.action_mapping.schema import ActionMapping, MappedAction


def format_mapping_report(mapping: ActionMapping) -> dict[str, Any]:
    """Build a structured report for a mapping preflight check."""
    return {
        "compatibility": mapping.compatibility.value,
        "blocked": mapping.is_blocked(),
        "allow_partial": mapping.allow_partial,
        "policy": mapping.policy_space.to_dict(),
        "body": mapping.body_space.to_dict(),
        "joints": [j.to_dict() for j in mapping.joints],
        "block_reasons": list(mapping.block_reasons),
        "warnings": list(mapping.warnings),
    }


def format_mapping_text(mapping: ActionMapping) -> str:
    """Render a mapping report as human-readable text."""
    lines = [
        "[rosclaw-lerobot] Body action mapping",
        f"  Compatibility:   {mapping.compatibility.value}",
        f"  Blocked:         {mapping.is_blocked()}",
        f"  Policy names:    {len(mapping.policy_space.names)}",
        f"  Body joints:     {len(mapping.body_space.joint_names)}",
    ]
    if mapping.block_reasons:
        lines.append("  Block reasons:")
        for reason in mapping.block_reasons:
            lines.append(f"    - {reason}")
    if mapping.warnings:
        lines.append("  Warnings:")
        for warning in mapping.warnings:
            lines.append(f"    - {warning}")
    lines.append("  Joints:")
    for joint in mapping.joints:
        flag = "OK" if joint.compatible and joint.policy_index >= 0 else "MISSING"
        lines.append(
            f"    [{flag}] {joint.body_name} <- {joint.policy_name or '(no policy match)'}"
        )
    return "\n".join(lines)


def format_mapped_action_report(mapped: MappedAction) -> dict[str, Any]:
    """Build a structured report for a mapped action."""
    return {
        "blocked": mapped.blocked,
        "compatibility": mapped.compatibility.value,
        "body_joint_names": mapped.body_joint_names,
        "body_action_values": mapped.body_action_values,
        "block_reasons": mapped.block_reasons,
        "warnings": mapped.warnings,
        "reference_frame": mapped.reference_frame,
        "chunk_size": mapped.chunk_size,
    }
