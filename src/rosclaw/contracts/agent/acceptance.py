"""AcceptanceSpecV2（PR-N8，调整方案 §七）——每个 revision 冻结一份。

验收条件来源按优先级合并（AcceptanceCompilerV2）：
安全策略最低标准 + Capability 内置 acceptance template + 用户显式要求
+ 任务类型默认标准 + 模型建议的更严格标准。模型只能加严，不能放宽。
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class AcceptanceSpecV2(ContractModel):
    """一个 task revision 的冻结验收规格。"""

    SCHEMA: ClassVar[str] = "rosclaw.acceptance_spec.v2"
    HASH_PREFIX: ClassVar[str] = "acceptance_spec"

    schema_version: Literal["rosclaw.acceptance_spec.v2"] = (
        "rosclaw.acceptance_spec.v2"
    )
    spec_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    required_artifacts: list[str] = Field(default_factory=list)
    evidence_classes: list[str] = Field(default_factory=list)
    resource_provenance_required: bool = False
    numeric_thresholds: dict[str, float] = Field(default_factory=dict)
    visual_requirements: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    allowed_execution_modes: list[str] = Field(
        default_factory=lambda: ["SIMULATION"]
    )
    required_receipt: str = ""
    verifier_refs: list[str] = Field(default_factory=list)
    #: 归因：每个来源贡献了哪些约束（审计可回放）。
    sources: dict[str, Any] = Field(default_factory=dict)


__all__ = ["AcceptanceSpecV2"]
