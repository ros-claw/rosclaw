"""Descriptor helpers (PR-05) — thin re-export of the ToolDescriptorV2 contract
plus construction helpers, so callers never import the contracts package for
day-to-day catalog work."""

from __future__ import annotations

from rosclaw.contracts.agent.tool import (
    ExecutionClass,
    ToolDescriptorV2,
    ToolEvidenceClass,
    ToolSideEffectClass,
)


def observation_descriptor(tool_id: str, *, source: str, **overrides) -> ToolDescriptorV2:
    """Convenience constructor for a model-callable OBSERVE descriptor."""
    return ToolDescriptorV2(
        tool_id=tool_id,
        source=source,
        execution_class=ExecutionClass.OBSERVE,
        **overrides,
    )


def physical_action_descriptor(tool_id: str, *, source: str, **overrides) -> ToolDescriptorV2:
    """Convenience constructor for a PHYSICAL_ACTION descriptor (never model-callable)."""
    return ToolDescriptorV2(
        tool_id=tool_id,
        source=source,
        execution_class=ExecutionClass.PHYSICAL_ACTION,
        side_effect_class=overrides.pop("side_effect_class", ToolSideEffectClass.REVERSIBLE),
        model_callable=False,
        requires_exact_action_grant=True,
        **overrides,
    )


__all__ = [
    "ExecutionClass",
    "ToolDescriptorV2",
    "ToolEvidenceClass",
    "ToolSideEffectClass",
    "observation_descriptor",
    "physical_action_descriptor",
]
