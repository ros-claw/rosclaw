"""Generate and apply body action mappings."""

from __future__ import annotations

from typing import Any

from rosclaw.body.action_mapping.schema import (
    ActionMapping,
    ActionSpace,
    BodyActionSpace,
    JointMapping,
    MappedAction,
    MappingCompatibility,
)
from rosclaw.body.action_mapping.units import get_unit_conversion


def generate_action_mapping(
    policy_space: ActionSpace,
    body_space: BodyActionSpace,
    *,
    allow_partial: bool = False,
) -> ActionMapping:
    """Build a mapping between a policy action space and a body action space.

    The current implementation matches policy action names to body joint names
    directly.  Scale, offset and sign are taken from explicit metadata when
    present; otherwise a 1:1 identity mapping is assumed.
    """
    joints: list[JointMapping] = []
    warnings: list[str] = []
    block_reasons: list[str] = []

    if not body_space.joint_names:
        return ActionMapping(
            policy_space=policy_space,
            body_space=body_space,
            compatibility=MappingCompatibility.INCOMPATIBLE,
            block_reasons=["Body has no controllable joints"],
            allow_partial=allow_partial,
        )

    if not policy_space.names:
        return ActionMapping(
            policy_space=policy_space,
            body_space=body_space,
            compatibility=MappingCompatibility.UNKNOWN,
            block_reasons=["Policy action names are missing"],
            allow_partial=allow_partial,
        )

    if policy_space.representation not in ("joint_position", "joint_velocity", "joint_torque"):
        return ActionMapping(
            policy_space=policy_space,
            body_space=body_space,
            compatibility=MappingCompatibility.UNKNOWN,
            block_reasons=[
                f"Unsupported policy action representation: {policy_space.representation}"
            ],
            allow_partial=allow_partial,
        )

    if policy_space.representation != body_space.representation:
        return ActionMapping(
            policy_space=policy_space,
            body_space=body_space,
            compatibility=MappingCompatibility.INCOMPATIBLE,
            block_reasons=[
                f"Representation mismatch: policy={policy_space.representation} "
                f"body={body_space.representation}"
            ],
            allow_partial=allow_partial,
        )

    policy_name_to_index = {name: i for i, name in enumerate(policy_space.names)}
    body_name_to_index = {name: i for i, name in enumerate(body_space.joint_names)}

    for body_name in body_space.joint_names:
        body_index = body_name_to_index[body_name]
        policy_index = policy_name_to_index.get(body_name)
        body_unit = body_space.units[body_index] if body_index < len(body_space.units) else ""

        if policy_index is None:
            joints.append(
                JointMapping(
                    policy_name="",
                    body_name=body_name,
                    policy_index=-1,
                    body_index=body_index,
                    compatible=False,
                    reason="No matching policy action name",
                )
            )
            continue

        policy_name = policy_space.names[policy_index]
        policy_unit = (
            policy_space.units[policy_index] if policy_index < len(policy_space.units) else ""
        )
        converter = get_unit_conversion(policy_unit, body_unit)
        compatible = converter is not None
        unit_conversion = f"{policy_unit}->{body_unit}" if policy_unit != body_unit else None
        if not compatible:
            reason = f"Cannot convert unit '{policy_unit}' to body unit '{body_unit}'"
            warnings.append(reason)
        else:
            reason = None

        # Extract explicit calibration metadata if present.
        scale = float(policy_space.metadata.get("scale", 1.0))
        offset = float(policy_space.metadata.get("offset", 0.0))
        sign = int(policy_space.metadata.get("sign", 1))

        joints.append(
            JointMapping(
                policy_name=policy_name,
                body_name=body_name,
                policy_index=policy_index,
                body_index=body_index,
                scale=scale,
                offset=offset,
                sign=sign,
                unit_conversion=unit_conversion,
                compatible=compatible,
                reason=reason,
            )
        )

    matched = [j for j in joints if j.policy_index >= 0]
    unmatched_body = [j for j in joints if j.policy_index < 0]
    unmatched_policy = [
        name for name in policy_space.names if name not in body_space.joint_names
    ]

    if not matched:
        compatibility = MappingCompatibility.INCOMPATIBLE
        block_reasons.append("No policy action names match body joint names")
    elif unmatched_body or unmatched_policy:
        compatibility = MappingCompatibility.PARTIAL
        if unmatched_body:
            warnings.append(
                f"Body joints without policy match: {[j.body_name for j in unmatched_body]}"
            )
        if unmatched_policy:
            warnings.append(f"Policy actions without body match: {unmatched_policy}")
        if not allow_partial:
            block_reasons.append("Partial mapping is not allowed by default")
    else:
        any_conversion = any(j.unit_conversion for j in joints)
        any_incompatible = any(not j.compatible for j in joints)
        if any_incompatible:
            compatibility = MappingCompatibility.INCOMPATIBLE
            block_reasons.append("One or more matched joints have incompatible units")
        elif any_conversion:
            compatibility = MappingCompatibility.CONVERTIBLE
        else:
            compatibility = MappingCompatibility.EXACT

    return ActionMapping(
        policy_space=policy_space,
        body_space=body_space,
        joints=joints,
        compatibility=compatibility,
        block_reasons=block_reasons,
        warnings=warnings,
        allow_partial=allow_partial,
    )


def map_action_to_body(
    policy_action_values: list[float],
    mapping: ActionMapping,
    *,
    chunk_size: int | None = None,
) -> MappedAction:
    """Apply an action mapping to a flat or chunked policy action vector.

    If ``chunk_size`` is provided, ``policy_action_values`` is interpreted as
    ``chunk_size * len(policy_space.names)`` entries and the returned
    ``body_action_values`` are ordered for the body for each chunk.
    """
    if mapping.is_blocked():
        return MappedAction(
            body_action_values=[],
            body_joint_names=[],
            compatibility=mapping.compatibility,
            blocked=True,
            block_reasons=mapping.block_reasons,
            warnings=mapping.warnings,
        )

    body_dim = len(mapping.body_space.joint_names)
    total = len(policy_action_values)
    chunks = chunk_size or 1
    expected = len(mapping.policy_space.names) * chunks
    if total != expected:
        return MappedAction(
            body_action_values=[],
            body_joint_names=[],
            compatibility=mapping.compatibility,
            blocked=True,
            block_reasons=[
                f"Action length mismatch: expected {expected} values, got {total}"
            ],
            warnings=mapping.warnings,
        )

    result: list[float] = []
    for chunk_idx in range(chunks):
        chunk_offset = chunk_idx * len(mapping.policy_space.names)
        body_row = [0.0] * body_dim
        for joint in mapping.joints:
            if joint.policy_index < 0:
                continue
            value = policy_action_values[chunk_offset + joint.policy_index]
            value = joint.sign * (joint.scale * value + joint.offset)
            if joint.unit_conversion:
                converter = get_unit_conversion(
                    mapping.policy_space.units[joint.policy_index],
                    mapping.body_space.units[joint.body_index],
                )
                if converter is not None:
                    value = converter(value)
            body_row[joint.body_index] = value
        result.extend(body_row)

    return MappedAction(
        body_action_values=result,
        body_joint_names=list(mapping.body_space.joint_names),
        compatibility=mapping.compatibility,
        blocked=False,
        block_reasons=[],
        warnings=mapping.warnings,
        reference_frame=mapping.body_space.reference_frame,
        chunk_size=chunk_size,
    )


def map_action_proposal_to_body(
    proposal: dict[str, Any],
    mapping: ActionMapping,
) -> MappedAction:
    """Convenience wrapper for ``rosclaw.action_proposal.v2`` proposals."""
    action = proposal.get("action", {})
    values = action.get("values", [])
    chunk = proposal.get("chunk", {})
    chunk_size = chunk.get("size") if isinstance(chunk, dict) else None
    return map_action_to_body(values, mapping, chunk_size=chunk_size)
