"""CapabilityDescriptorV2 / ToolProjectionV1（PR-N5A，调整方案 §三.N5A）。

职责拆分：
- CapabilityDescriptorV2 描述"ROSClaw 具备什么能力"——领域事实，
  与任何 Harness 无关；
- ToolProjectionV1 描述该能力如何投影给 Harness/模型/TUI/CLI——
  一个能力可投影成多种形态（Pi Tool / Codex MCP Tool / TUI 卡片 /
  CLI 命令 / fixture）。

硬不变量（fail closed）：
- EffectClassV1 是 N5C 单一 Effect Contract 的目标词表（8 值冻结）；
- PHYSICAL_EFFECT 能力的原始 executor 永远不得 direct 投影给模型
  （在 Registry.register_projection 强制，合约层只承载数据）；
- 物理效应默认 resource_provenance_required=True；
- 合约任何字段不得携带凭据（沿用 ADR-0006/0007 边界）。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import ConfigDict, Field

from rosclaw.contracts.common import ContractModel


class EffectClassV1(StrEnum):
    """单一 Effect Contract 词表（N5C 正式启用，此处先冻结枚举）。"""

    READ_ONLY = "READ_ONLY"
    PURE_COMPUTE = "PURE_COMPUTE"
    WORKSPACE_WRITE = "WORKSPACE_WRITE"
    HOST_PROCESS = "HOST_PROCESS"
    NETWORK_EFFECT = "NETWORK_EFFECT"
    SIMULATED_EFFECT = "SIMULATED_EFFECT"
    SHADOW_PROPOSAL = "SHADOW_PROPOSAL"
    PHYSICAL_EFFECT = "PHYSICAL_EFFECT"


class CapabilityEffectV1(ContractModel):
    """能力的效应声明：执行前冻结、写事件，审批/并发/Verifier 共读。"""

    class_: EffectClassV1 = Field(alias="class")
    #: 效应域，如 simulation_state / workspace_fs / physical_body
    domain: str = ""
    reversible: bool = True
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "LOW"
    #: REAL 上下文必须新鲜（PHYSICAL_EFFECT/SHADOW_PROPOSAL 才有意义）
    requires_fresh_real_context: bool = False

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class CapabilityCompatibilityV1(ContractModel):
    modes: list[Literal["SIMULATION", "SHADOW", "REAL"]] = Field(
        default_factory=lambda: ["SIMULATION"]
    )
    body_types: list[str] = Field(default_factory=list)
    robot_ids: list[str] = Field(default_factory=list)
    resource_kinds: list[str] = Field(default_factory=list)
    runtime_requirements: list[str] = Field(default_factory=list)


class CapabilityExecutionV1(ContractModel):
    #: 如 python:rosclaw.agentd.sim_trajectory:simulate
    executor_ref: str = ""
    long_running: bool = False
    cancellation: Literal["cooperative", "immediate", "none"] = "cooperative"
    concurrency: Literal["exclusive", "shared", "serialized"] = "shared"
    idempotent: bool = True


class CapabilityEvidenceV1(ContractModel):
    #: 如 MEASURED/SIMULATED/CONFIGURED/DERIVED/SIM_DYN_ROLLOUT
    evidence_class: str = ""
    verifier_ref: str = ""
    resource_provenance_required: bool = False


class CapabilityDescriptorV2(ContractModel):
    """ROSClaw 能力描述符（canonical）——Harness 无关的领域事实。"""

    SCHEMA: ClassVar[str] = "rosclaw.capability.v2"
    HASH_PREFIX: ClassVar[str] = "capability"

    schema_version: Literal["rosclaw.capability.v2"] = "rosclaw.capability.v2"
    capability_id: str = Field(min_length=1)
    version: str = "1.0.0"
    #: 如 rosclaw-builtin / mcp:limo-ros-mcp / hub:*
    source: str = Field(min_length=1)
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    effect: CapabilityEffectV1 = Field(
        default_factory=lambda: CapabilityEffectV1(**{"class": EffectClassV1.READ_ONLY})
    )
    compatibility: CapabilityCompatibilityV1 = Field(
        default_factory=CapabilityCompatibilityV1
    )
    execution: CapabilityExecutionV1 = Field(default_factory=CapabilityExecutionV1)
    evidence: CapabilityEvidenceV1 = Field(default_factory=CapabilityEvidenceV1)
    #: WP-2：引用端口——本能力消费/产出哪些 TypedRef kind
    #: （snapshot 只暴露可实际连接的组合）。
    accepts_refs: list[dict[str, Any]] = Field(default_factory=list)
    produces_refs: list[dict[str, Any]] = Field(default_factory=list)
    #: 自由元数据（如 adapted_from 迁移标注）；不得携带凭据。
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        # 物理效应默认要求资源证明（fail closed；显式 False 是有意的声明）。
        if (
            self.effect.class_ is EffectClassV1.PHYSICAL_EFFECT
            and "resource_provenance_required" not in self.evidence.model_fields_set
        ):
            self.evidence.resource_provenance_required = True


class ProjectionExposure(StrEnum):
    DIRECT = "direct"  # 直接暴露给模型调用
    PROPOSE_ONLY = "propose_only"  # 仅生成 propose_* 工具（进 ActionAdmission）
    INTERNAL = "internal"  # 不进模型面（TUI/CLI/Code Mode 内部用）


class ToolProjectionV1(ContractModel):
    """能力 → 模型/Harness 工具的投影描述。"""

    SCHEMA: ClassVar[str] = "rosclaw.tool_projection.v1"
    HASH_PREFIX: ClassVar[str] = "tool_projection"

    schema_version: Literal["rosclaw.tool_projection.v1"] = "rosclaw.tool_projection.v1"
    tool_name: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    presentation_ref: str = ""
    exposure: ProjectionExposure = ProjectionExposure.DIRECT


__all__ = [
    "CapabilityCompatibilityV1",
    "CapabilityDescriptorV2",
    "CapabilityEffectV1",
    "CapabilityEvidenceV1",
    "CapabilityExecutionV1",
    "EffectClassV1",
    "ProjectionExposure",
    "ToolProjectionV1",
]
