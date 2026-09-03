"""ToolAssetV1（0902 审计 R2-1，§4.3）——工具/附件资产契约。

工具（夹爪、吸盘、标记笔…）是有 manifest、多格式适配（MJCF/USD/
URDF）和挂载变换的一等资产——不是渲染脚本里硬编码的几何体。

诚实性不变式（§4.3：可视化附件不得冒充真实接触）：
- `physical` 必填无默认——visual-only 附件必须显式声明，接触/执行
  证据校验（R0-3 receipt.tool_ref 等 verifier）只认 physical=True
  的附件。
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, field_validator

from rosclaw.contracts.common import ContractModel

#: 适配格式冻结集合（当前先做 MJCF，USD/URDF 为契约预留）。
TOOL_ADAPTER_FORMATS = ("mjcf", "usd", "urdf")


class ToolMountV1(ContractModel):
    """挂载变换：父坐标系 + 可选 SE(3) 偏移（米/四元数 xyzw）。"""

    SCHEMA: ClassVar[str] = "rosclaw.tool_mount.v1"

    schema_version: Literal["rosclaw.tool_mount.v1"] = "rosclaw.tool_mount.v1"
    parent_frame: str = Field(min_length=1)
    position_m: list[float] = Field(default=[0.0, 0.0, 0.0], min_length=3, max_length=3)
    orientation_xyzw: list[float] = Field(
        default=[0.0, 0.0, 0.0, 1.0], min_length=4, max_length=4
    )


class ToolAssetV1(ContractModel):
    """工具资产 manifest。"""

    SCHEMA: ClassVar[str] = "rosclaw.tool_asset.v1"

    schema_version: Literal["rosclaw.tool_asset.v1"] = "rosclaw.tool_asset.v1"
    tool_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    #: 是否物理实体（参与接触/碰撞）——必填，无默认（visual-only
    #: 必须显式 False，杜绝可视化冒充真实接触）。
    physical: bool
    #: 格式 → 资产路径（至少一个；格式冻结）。
    adapters: dict[str, str]
    mount: ToolMountV1

    @field_validator("adapters")
    @classmethod
    def _known_adapters(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("adapters 至少一个（mjcf/usd/urdf）")
        for fmt in value:
            if fmt not in TOOL_ADAPTER_FORMATS:
                raise ValueError(
                    f"未知适配格式: {fmt}（冻结集合 {TOOL_ADAPTER_FORMATS}）"
                )
        return value


__all__ = ["TOOL_ADAPTER_FORMATS", "ToolAssetV1", "ToolMountV1"]
