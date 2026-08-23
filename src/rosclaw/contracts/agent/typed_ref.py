"""TypedRefV1（WP-2，0823 审计 §四.WP-2）——能力间统一引用。

模型看到的必须是一张可组合能力图：plan/trace/render/verification
引用带完整身份与血缘——producer_capability + digest + body/world +
task/revision + storage_backend。Capability Descriptor 声明
accepts_refs/produces_refs；Snapshot 只暴露能实际连接的工具组合。
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class TypedRefV1(ContractModel):
    """能力产物的类型化引用（plan/trace/render/verification…）。"""

    SCHEMA: ClassVar[str] = "rosclaw.typed_ref.v1"
    HASH_PREFIX: ClassVar[str] = "typed_ref"

    schema_version: Literal["rosclaw.typed_ref.v1"] = "rosclaw.typed_ref.v1"
    #: plan / trace / render / verification（可扩展）
    kind: str = Field(min_length=1)
    #: rosclaw://plan/plan_abc123 等可解析 URI
    uri: str = Field(min_length=1)
    producer_capability: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    body_id: str = ""
    world_id: str = ""
    task_id: str = ""
    revision: int = 0
    #: disk:sim/plans / mcp:inprocess / ...
    storage_backend: str = ""


__all__ = ["TypedRefV1"]
