"""CapabilitySnapshotV1（PR-N5D，调整方案 §三.N5D）——当前回合的精确工具面。

Capability Registry 按当前 body/mode/health 过滤后生成的快照：
模型只见 active 里的精确强类型工具；excluded 附机器原因码（只能经
rosclaw inspect capability 查看）。执行携带 snapshot digest——
registry 变化不静默换工具（CAPABILITY_SNAPSHOT_CHANGED）。
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class SnapshotActiveToolV1(ContractModel):
    """一个可物化的模型工具（由能力投影）。"""

    tool_name: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    exposure: Literal["direct", "propose_only", "internal"] = "direct"
    effect_class: str = ""
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class SnapshotExcludedV1(ContractModel):
    capability_id: str = Field(min_length=1)
    #: 机器原因码，如 CAPABILITY_QUARANTINED / EFFECT_NOT_EXPOSABLE /
    #: MODE_FORBIDDEN / BODY_INCOMPATIBLE
    reason: str = Field(min_length=1)


class CapabilitySnapshotV1(ContractModel):
    SCHEMA: ClassVar[str] = "rosclaw.capability_snapshot.v1"
    HASH_PREFIX: ClassVar[str] = "capability_snapshot"
    HASH_EXCLUDE_FIELD: ClassVar[str] = "digest"

    schema_version: Literal["rosclaw.capability_snapshot.v1"] = (
        "rosclaw.capability_snapshot.v1"
    )
    generation: int = Field(ge=0)
    #: 内容 digest（sha256:；registry 任何可见变化都会改变它）
    digest: str = ""
    body_id: str = ""
    mode: str = "SIMULATION"
    active: list[SnapshotActiveToolV1] = Field(default_factory=list)
    excluded: list[SnapshotExcludedV1] = Field(default_factory=list)


__all__ = [
    "CapabilitySnapshotV1",
    "SnapshotActiveToolV1",
    "SnapshotExcludedV1",
]
