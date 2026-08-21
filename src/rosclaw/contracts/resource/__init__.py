"""ResourceManifestV1（PR-N4，N 总纲 §8.2）——资源解析的统一合约。

正式执行/交付只使用 canonical 生产资源；test fixture 永不进入
production 解析结果。
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class ResourceManifestV1(ContractModel):
    """资源清单：身份 + 信任 + 质量 + 兼容性 + 路径 + digest。"""

    SCHEMA: ClassVar[str] = "rosclaw.resource_manifest.v1"
    HASH_PREFIX: ClassVar[str] = "resource_manifest"

    schema_version: Literal["rosclaw.resource_manifest.v1"] = (
        "rosclaw.resource_manifest.v1"
    )
    resource_id: str  # 例：robot:ur5e
    kind: str  # robot/sensor/actuator/world/scene/model/dataset/provider/mcp/skill/capability/executor/benchmark/tutorial/example
    version: str = "1.0"
    source: str = ""  # e-urdf-zoo / rosclaw-bundled / hub / workspace
    trust: str = "ROSCLAW_OFFICIAL"
    quality: Literal["PRODUCTION", "TEST_FIXTURE", "EXPERIMENTAL"] = "PRODUCTION"
    canonical: bool = True
    compatibility: dict = Field(default_factory=dict)
    paths: dict[str, str] = Field(default_factory=dict)
    digests: dict[str, str] = Field(default_factory=dict)
