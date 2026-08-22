"""ToolDescriptorV2 — the catalog entry for every capability the Agent may see (大纲 §7.2).

A descriptor is *metadata*, never an execution permit. Two hard invariants
are enforced at the contract layer (fail closed):

* ``PHYSICAL_ACTION`` tools are never ``model_callable`` — a physical action
  only ever flows through ``Decision REQUEST_APPROVAL → Operator → Grant →
  REQUEST_ACTION → rosclawd``.
* ``requires_exact_action_grant=True`` implies ``model_callable=False``.

No field of this contract may carry credentials; MCP server auth is always
referenced as ``env:VAR`` on the *server* config, never on the descriptor.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from rosclaw.contracts.common import ContractModel, ValidationError


class ExecutionClass(StrEnum):
    """What kind of thing happens when this tool runs (大纲 §7.1)."""

    OBSERVE = "OBSERVE"  # read-only observation; model-callable with evidence wrapping
    COMPUTE = "COMPUTE"  # pure compute, no physical side effect; model-callable
    PHYSICAL_ACTION = "PHYSICAL_ACTION"  # physical side effect; NEVER model-callable


class ToolSideEffectClass(StrEnum):
    NONE = "NONE"
    REVERSIBLE = "REVERSIBLE"
    IRREVERSIBLE = "IRREVERSIBLE"


class ToolEvidenceClass(StrEnum):
    MEASURED = "MEASURED"  # from a live sensor/ROS topic
    SIMULATED = "SIMULATED"  # from simulation; never usable as REAL proof
    CONFIGURED = "CONFIGURED"  # from static config/profile
    DERIVED = "DERIVED"  # computed from other evidence


class ToolDescriptorV2(ContractModel):
    SCHEMA = "rosclaw.tool_descriptor.v2"

    schema_version: Literal["rosclaw.tool_descriptor.v2"] = "rosclaw.tool_descriptor.v2"
    tool_id: str = Field(min_length=1)
    version: str = "1.0.0"
    #: e.g. "native:agentd", "mcp:limo-ros-mcp"
    source: str = Field(min_length=1)
    execution_class: ExecutionClass = ExecutionClass.OBSERVE
    side_effect_class: ToolSideEffectClass = ToolSideEffectClass.NONE
    #: 七审 §2.5：效果域——POLICY_AUTO 只对 SIMULATION_STATE_ONLY 生效；
    #: 空值 fail closed（不自动）。
    effect_domain: str = ""
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    supported_modes: list[Literal["SIMULATION", "SHADOW", "REAL"]] = Field(
        default_factory=lambda: ["SIMULATION"]
    )
    #: empty = body-agnostic
    required_body_types: list[str] = Field(default_factory=list)
    freshness_ms: int | None = Field(default=None, ge=0)
    timeout_ms: int = Field(default=2000, gt=0)
    #: PR-N6C：只有明确声明 cooperative cancel 的 executor 才允许
    #: deadline 杀死；未声明 → timeout_ms 不作为执行截止（不墙钟杀）。
    cooperative_cancel: bool = False
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "LOW"
    evidence_class: ToolEvidenceClass = ToolEvidenceClass.MEASURED
    #: verifier id, e.g. "schema+timestamp+frame"; empty = no verifier
    verifier: str = ""
    idempotent: bool = True
    model_callable: bool = True
    requires_exact_action_grant: bool = False
    #: reliability/latency/cost hints for resolver ranking (never for safety)
    reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    typical_latency_ms: int = Field(default=100, ge=0)
    cost_hint: float = Field(default=0.0, ge=0.0)
    #: capability names this tool requires to be online (e.g. ROS topics)
    required_capabilities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_physical_action_invariants(self) -> ToolDescriptorV2:
        if self.execution_class is ExecutionClass.PHYSICAL_ACTION and self.model_callable:
            raise ValidationError(
                f"tool {self.tool_id!r}: PHYSICAL_ACTION tools are never model_callable "
                "(fail closed — physical actions flow only through Operator grant)"
            )
        if self.requires_exact_action_grant and self.model_callable:
            raise ValidationError(
                f"tool {self.tool_id!r}: requires_exact_action_grant implies "
                "model_callable=False (fail closed)"
            )
        if self.execution_class is ExecutionClass.PHYSICAL_ACTION and not (
            self.requires_exact_action_grant
        ):
            raise ValidationError(
                f"tool {self.tool_id!r}: PHYSICAL_ACTION must declare "
                "requires_exact_action_grant=True"
            )
        if self.side_effect_class is not ToolSideEffectClass.NONE and (
            self.execution_class is ExecutionClass.OBSERVE
        ):
            raise ValidationError(
                f"tool {self.tool_id!r}: OBSERVE tools must have side_effect_class=NONE; "
                "reclassify as PHYSICAL_ACTION or COMPUTE"
            )
        return self
