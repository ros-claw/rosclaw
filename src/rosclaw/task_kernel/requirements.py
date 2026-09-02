"""RequirementCompiler（0902 审计 R0-2，§3.1）——用户目标 → 材料性
可核验条款。

0902 实证（P0 事故）：用户新增"红色圆柱笔 + 3D 实际轨迹 + 不要 2D"，
自动路由只识别"机械臂+轨迹+画"关键词命中五角星 recipe——新材料性
条件被吞掉，旧视频复用，宣布 PASS/DELIVERED 假成功。

每条材料性要求编译为可核验条款（id/level/claim/verifier）：
- level=must：必须有证据才算满足；
- level=forbidden：不得出现（禁止项）；
- verifier：覆盖/验收键——recipe 声明的覆盖集合比对键，R0-3 接
  receipt 逐条核验。

硬规则：
- 快速配方只在语义覆盖率 100% 时执行（门禁在 task_router/
  auto_route）；
- 任一条款未被 recipe 声明覆盖 → 不自动路由（不猜）；
- 词类均为通用类别（工具/颜色/平面/overlay/禁止/形状）——无任务
  特例、无五角星硬编码。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    """一条材料性要求（可核验条款）。"""

    req_id: str
    level: str  # must | forbidden
    claim: str  # 用户可读的条款表述
    verifier: str  # 覆盖比对键 / 验收键（R0-3 接 receipt 核验）


def _any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(m in lowered or m in text for m in markers)


#: 词类注册表（通用类别；新增维度 = 加一行注册，不是改逻辑）。
_TOOL_MARKERS = ("笔", "持笔", "画笔", "马克笔", "夹爪", "吸盘",
                 "pen", "marker", "gripper", "suction")
_COLOR_MARKERS = ("红色", "蓝色", "绿色", "黄色", "黑色", "白色", "橙色",
                  "紫色", "red", "blue", "green", "yellow", "black",
                  "white")
_OVERLAY_TRACE_MARKERS = ("显示轨迹", "看到轨迹", "轨迹可见", "叠加轨迹",
                          "显示运动轨迹", "看到运动轨迹", "轨迹叠加",
                          "实际轨迹", "运动轨迹可见",
                          "show the trace", "show trajectory",
                          "overlay trace")
_FORBID_2D_MARKERS = ("不要 2d", "不要2d", "别给 2d", "禁止 2d",
                      "不要二维", "别来 2d", "no 2d")
_VERTICAL_MARKERS = ("垂直", "竖直", "立面", "vertical")
_HORIZONTAL_MARKERS = ("水平面", "水平桌", "水平放", "horizontal")
_SCENE_3D_MARKERS = ("3d", "三维", "场景视频", "scene video")
_VIDEO_MARKERS = ("视频", "video", "mp4", "录像")
_PREVIEW_MARKERS = ("gif", "动图", "2d 预览", "preview")
_CONTACT_MARKERS = ("接触", "贴着", "贴在", "contact")

#: 形状注册表：已知形状词 → shape key。recipe 覆盖集合之外的已知
#: 形状 = "已知但未覆盖"（门禁拦截——不允许画错形状冒充）。
_SHAPE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("star5", ("五角星", "星形", "star")),
    ("circle", ("圆形", "圆圈", "画圆", "圆环", "circle")),
    ("square", ("正方形", "方形", "square")),
    ("triangle", ("三角形", "triangle")),
    ("spiral", ("螺旋", "spiral")),
)


def compile_requirements(goal_text: str) -> list[Requirement]:
    """目标文本 → 材料性条款列表（确定性——同一文本永远同一组
    条款）。识别顺序即 req_id 顺序（deterministic）。"""
    reqs: list[Requirement] = []

    def _add(level: str, claim: str, verifier: str) -> None:
        if any(r.verifier == verifier for r in reqs):
            return
        reqs.append(Requirement(
            req_id=f"r{len(reqs) + 1}", level=level, claim=claim,
            verifier=verifier,
        ))

    for shape, markers in _SHAPE_MARKERS:
        if _any(goal_text, markers):
            _add("must", f"绘制形状：{shape}", f"shape.{shape}")
            break
    if _any(goal_text, _TOOL_MARKERS):
        _add("must", "末端挂载工具", "receipt.tool_ref")
    if _any(goal_text, _COLOR_MARKERS):
        _add("must", "指定颜色外观", "render.tool_color")
    if _any(goal_text, _OVERLAY_TRACE_MARKERS):
        _add("must", "3D 画面叠加本次实际末端运动轨迹",
             "receipt.overlays.actual_eef_trace")
    if _any(goal_text, _VERTICAL_MARKERS):
        _add("must", "竖直/垂直平面作业", "plan.plane.vertical")
    if _any(goal_text, _HORIZONTAL_MARKERS):
        _add("must", "水平面作业", "plan.plane.horizontal")
    if _any(goal_text, _SCENE_3D_MARKERS):
        _add("must", "3D 场景视频交付", "deliverable.scene_3d")
    if _any(goal_text, _VIDEO_MARKERS):
        _add("must", "视频交付", "deliverable.video")
    if _any(goal_text, _PREVIEW_MARKERS):
        _add("must", "2D 预览交付", "deliverable.preview_2d")
    if _any(goal_text, _CONTACT_MARKERS):
        _add("must", "接触验证", "verification.contact")
    if _any(goal_text, _FORBID_2D_MARKERS):
        _add("forbidden", "禁止只交付 2D", "delivery.not_2d_only")
    return reqs


__all__ = ["Requirement", "compile_requirements"]
