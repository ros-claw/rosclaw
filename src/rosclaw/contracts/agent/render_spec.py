"""RenderSpecV1 / RenderProfileV1（0902 审计 R2-1，§4.3 通用渲染器）。

不再为五角星写专用画线逻辑：渲染输入是一等契约——任意已登记
本体（body_ref + RenderProfile）、附件（attachments + mount_frame）、
世界（world_ref）、overlay（实际/规划轨迹、waypoints、接触点、
安全区、传感器视锥）、相机预设、输出格式，全走同一路径。

诚实性不变式（§4.3/R2.5）：
- overlay kind 与 camera preset 是冻结枚举——未知值拒绝，不静默吞；
- source_ref 把 overlay 绑定到具体 trace/plan（渲染必须渲染"这次"
  的证据，不允许旧产物冒充——0902 假成功教训）；
- RenderProfile 让每个本体自带 root/qpos 映射/EEF frame/默认相机，
  渲染器不按本体名 hardcode。
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, field_validator

from rosclaw.contracts.common import ContractModel

#: overlay 冻结集合（§4.3 列举的六类）。
OVERLAY_KINDS = (
    "actual_eef_trace",
    "planned_trace",
    "waypoints",
    "contact_points",
    "safety_zone",
    "sensor_frustum",
)

#: 相机预设冻结集合（现有 sim_render 已实现的三档）。
CAMERA_PRESETS = ("follow", "free", "top")

#: 输出格式冻结集合。
RENDER_OUTPUTS = ("mp4", "gif")

#: trace 类 overlay 的 source_ref 必须指到具体证据（trace:/plan: URI）。
_TRACE_SOURCE_PREFIXES = ("trace:", "plan:")


class RenderOverlayV1(ContractModel):
    """渲染叠加层：语义 kind + 证据来源引用。"""

    SCHEMA: ClassVar[str] = "rosclaw.render_overlay.v1"

    schema_version: Literal["rosclaw.render_overlay.v1"] = "rosclaw.render_overlay.v1"
    kind: str = Field(min_length=1)
    #: trace:t1 / plan:p1 —— trace 类 overlay 必填且必须可解析形态。
    source_ref: str = ""

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in OVERLAY_KINDS:
            raise ValueError(f"未知 overlay kind: {value}（冻结集合 {OVERLAY_KINDS}）")
        return value

    def model_post_init(self, __context: object, /) -> None:
        if self.kind in ("actual_eef_trace", "planned_trace", "contact_points") and (
            not self.source_ref.startswith(_TRACE_SOURCE_PREFIXES)
        ):
            raise ValueError(
                f"overlay {self.kind} 的 source_ref 必须指向具体证据"
                f"（trace:/plan: 前缀），实际: {self.source_ref!r}"
            )


class RenderAttachmentV1(ContractModel):
    """附件挂载：tool_ref + 挂载坐标系（变换细节在 ToolAsset）。"""

    SCHEMA: ClassVar[str] = "rosclaw.render_attachment.v1"

    schema_version: Literal["rosclaw.render_attachment.v1"] = "rosclaw.render_attachment.v1"
    tool_ref: str = Field(min_length=1)
    mount_frame: str = Field(min_length=1)


class RenderCameraV1(ContractModel):
    """相机：预设 + 可选参数覆盖。"""

    SCHEMA: ClassVar[str] = "rosclaw.render_camera.v1"

    schema_version: Literal["rosclaw.render_camera.v1"] = "rosclaw.render_camera.v1"
    preset: str = Field(min_length=1)
    #: 预设之外的显式参数（lookat/distance/elevation…），键值自由但
    #: 预设名冻结。
    params: dict[str, float | str] = Field(default_factory=dict)

    @field_validator("preset")
    @classmethod
    def _known_preset(cls, value: str) -> str:
        if value not in CAMERA_PRESETS:
            raise ValueError(f"未知相机预设: {value}（冻结集合 {CAMERA_PRESETS}）")
        return value


class RenderSpecV1(ContractModel):
    """渲染任务规格（§4.3 主契约）。"""

    SCHEMA: ClassVar[str] = "rosclaw.render_spec.v1"

    schema_version: Literal["rosclaw.render_spec.v1"] = "rosclaw.render_spec.v1"
    body_ref: str = Field(min_length=1)
    world_ref: str = Field(min_length=1)
    attachments: list[RenderAttachmentV1] = Field(default_factory=list)
    overlays: list[RenderOverlayV1] = Field(default_factory=list)
    cameras: list[RenderCameraV1] = Field(default_factory=list)
    outputs: list[str] = Field(min_length=1)

    @field_validator("outputs")
    @classmethod
    def _known_outputs(cls, value: list[str]) -> list[str]:
        for out in value:
            if out not in RENDER_OUTPUTS:
                raise ValueError(f"未知输出格式: {out}（冻结集合 {RENDER_OUTPUTS}）")
        return value


class RenderProfileV1(ContractModel):
    """本体渲染档案：渲染器对任意本体的一等输入（不按本体名
    hardcode——root/qpos 映射/EEF frame/默认相机全在这里）。"""

    SCHEMA: ClassVar[str] = "rosclaw.render_profile.v1"

    schema_version: Literal["rosclaw.render_profile.v1"] = "rosclaw.render_profile.v1"
    body_id: str = Field(min_length=1)
    root_body: str = Field(min_length=1)
    #: 关节名 → qpos 索引（渲染回放 trajectory_states 的映射）。
    qpos_mapping: dict[str, int]
    eef_frame: str = Field(min_length=1)
    default_cameras: list[RenderCameraV1] = Field(default_factory=list)


__all__ = [
    "CAMERA_PRESETS",
    "OVERLAY_KINDS",
    "RENDER_OUTPUTS",
    "RenderAttachmentV1",
    "RenderCameraV1",
    "RenderOverlayV1",
    "RenderProfileV1",
    "RenderSpecV1",
]
