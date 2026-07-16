"""Resolve the action space of an effective body for policy mapping."""

from __future__ import annotations

from typing import Any

from rosclaw.body.action_mapping.schema import BodyActionSpace


def resolve_body_action_space(
    body: Any,
    *,
    representation: str = "joint_position",
    body_id: str = "",
) -> BodyActionSpace:
    """Extract a ``BodyActionSpace`` from an ``EffectiveBody`` or ``EurdfProfile``.

    The body is expected to expose either a ``joints`` dict (effective body) or
    a ``joints`` list of dicts with a ``name`` key (e-URDF profile).
    """
    resolved_id = body_id or getattr(body, "body_instance_id", "") or getattr(body, "profile_id", "")
    joints: dict[str, Any] = {}
    raw_joints = getattr(body, "joints", None) or {}
    if isinstance(raw_joints, dict):
        joints = raw_joints
    elif isinstance(raw_joints, list):
        joints = {j.get("name", f"joint_{i}"): j for i, j in enumerate(raw_joints)}

    joint_names: list[str] = []
    units: list[str] = []
    limits: dict[str, dict[str, float]] = {}
    for name, joint in joints.items():
        if not name:
            continue
        j_type = joint.get("type", "revolute")
        if j_type not in {"revolute", "prismatic", "continuous"}:
            continue
        joint_names.append(name)
        if representation == "joint_position":
            units.append(_default_unit_for_joint_type(j_type))
        else:
            units.append("")
        limits[name] = _extract_limits(joint)

    return BodyActionSpace(
        body_id=resolved_id,
        representation=representation,
        joint_names=joint_names,
        units=units,
        joint_limits=limits,
        reference_frame=getattr(body, "frames", {}).get("root", "") if hasattr(body, "frames") else "",
    )


def _default_unit_for_joint_type(joint_type: str) -> str:
    if joint_type == "prismatic":
        return "m"
    return "rad"


def _extract_limits(joint: dict[str, Any]) -> dict[str, float]:
    limits = joint.get("limits", {})
    if not isinstance(limits, dict):
        return {}
    result: dict[str, float] = {}
    for key in ("lower", "upper", "velocity", "effort"):
        value = limits.get(key)
        if isinstance(value, (int, float)):
            result[key] = float(value)
    return result
