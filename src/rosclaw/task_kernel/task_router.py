"""TaskRouter（R0-1，0826 体验审计 §5.R0-1）——TaskSpecV2 → recipe。

唯一路由真相：frozen TaskSpecV2 的 goal.intent 决定走哪条确定性
recipe。未知 intent 返回 None（诚实"无 recipe"——调用方报错，
不猜不编）。recipe 注册表在 TaskExecutionService（handler 需要
runtime/home 等执行资源）；这里只放 intent → recipe_id 映射。

通用 typed planning / workspace loop 是后续路由分支（§4 目标架
构）——当前不在路由表里的 intent 一律诚实无 recipe。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rosclaw.task_kernel.requirements import select_shape

#: intent → recipe_id（冻结映射；新增 recipe = 新 intent 映射 +
#: TaskExecutionService 注册 handler，一一对应）。
RECIPE_BY_INTENT: dict[str, str] = {
    "manipulation.draw_path": "recipe:sim.draw_path",
}

#: 模型显式 goal → recipe_id 回落（spec intent 分类是通用动词规则，
#: 弱文本可能落 task.unknown——模型经 rosclaw_task 显式请求已知
#: 任务目标时是合法信号；路由仍集中在 TaskRouter，不分散）。
RECIPE_BY_GOAL: dict[str, str] = {
    "draw_shape": "recipe:sim.draw_path",
    "simulate_trajectory": "recipe:sim.draw_path",
}

#: 0902 R0-2（§3.1 硬规则）：recipe 语义覆盖声明——该 recipe 能
#: 可验证地满足哪些条款（verifier 键）。快速配方只在材料性要求
#: 100% 覆盖时执行；任一 must/forbidden 未覆盖 → 不自动路由
#: （交 Native Agent 编译 TaskSpec——不猜）。recipe 只能加速，
#: 不得绕过需求编译与验收。
RECIPE_COVERAGE: dict[str, frozenset[str]] = {
    "recipe:sim.draw_path": frozenset({
        # 形状：recipe 只会画注册过的形状（画错形状=冒充）。
        "shape.star5", "shape.circle",
        # 平面：xy 默认 + xz/yz 命名面（compile_recipe_inputs）。
        "plan.plane.horizontal", "plan.plane.vertical",
        # 交付：3D 场景视频 + 2D 预览 + 实际轨迹数据。
        "deliverable.scene_3d", "deliverable.video",
        "deliverable.preview_2d", "trace.actual",
        # R2-3（渲染器已支持，有实现才覆盖）：场景渲染按任务要求
        # 带 actual_eef_trace overlay（RenderSpec 证据绑定到本次
        # trace）；3D 场景视频在账即满足"不要只有 2D"禁止项。
        "receipt.overlays.actual_eef_trace", "delivery.not_2d_only",
        # 不覆盖（——出现即转 Native Agent）：receipt.tool_ref（挂载
        # 工具，无资产）、render.tool_color（颜色无证据通道）、
        # verification.contact（无接触验证器）、未知形状。
    }),
}


def route_recipe(spec: Mapping[str, Any], *, goal_hint: str = "") -> str | None:
    """frozen TaskSpecV2（dict 视图）→ recipe_id；无路由 → None。

    intent 映射优先；intent 无 recipe 时回落模型显式 goal hint。
    """
    goal = spec.get("goal")
    if not isinstance(goal, Mapping):
        return None
    intent = str(goal.get("intent", ""))
    recipe = RECIPE_BY_INTENT.get(intent)
    if recipe is not None:
        return recipe
    return RECIPE_BY_GOAL.get(goal_hint)




#: 竖直平面词类 → 命名面（xz：y=const 立面——画板/墙面的常见
#: 含义；yz 留给显式 "yz 面"）。
_VERTICAL_MARKERS = ("竖直平面", "垂直平面", "竖直面", "垂直面", "立面",
                     "vertical plane", "竖着", "竖直方向")
_YZ_MARKERS = ("yz 面", "yz面", "yz plane")

#: 疑问/讨论形式护栏——"怎么画五角星"是讨论不是指令，不自动执行。
_QUESTION_MARKERS = ("怎么", "如何", "吗", "？", "?", "what", "how",
                     "why", "解释", "介绍", "是什么")

#: 代码请求护栏（0902 盲写语料实证：「帮我写一段画五角星的 Python
#: 代码」被 intent 关键词误分类为 draw_shape → 直接画了一颗星——
#: 用户要的是代码不是执行）。写代码/写脚本请求不是物理执行指令。
_CODE_REQUEST_MARKERS = ("写一段", "写个代码", "写代码", "代码实现",
                         "python 代码", "python代码", "写个脚本",
                         "write code", "write a script")

#: 情绪投诉护栏（0903 体验实证：「你画的居然是个五角星！我要的是
#: 立方体！」含形状词，确定性链竟把投诉当新指令又画了一遍五角星
#: 并 PASS——投诉变二次假成功）。投诉交模型（道歉+诚实解释），
#: 不进确定性链。
_COMPLAINT_MARKERS = ("居然", "压根", "混蛋", "什么鬼", "什么东西",
                      "画的什么", "搞什么", "什么玩意")


def is_task_directive(text: str) -> bool:
    """指令形式判定：疑问句/讨论/代码请求形式不自动执行。"""
    lowered = text.lower()
    if any(m in lowered or m in text for m in _QUESTION_MARKERS):
        return False
    if any(m in lowered or m in text for m in _CODE_REQUEST_MARKERS):
        return False
    return not any(m in text for m in _COMPLAINT_MARKERS)


def compile_recipe_inputs(goal_text: str) -> dict[str, Any]:
    """NL → recipe 几何参数（缺的字段由 recipe 确定性缺省补齐）。

    0902 holdout 实证：尺寸词（"8 厘米"）此前被静默丢弃——recipe
    用默认 scale 交付与用户要求不符的尺寸。显式尺寸是材料性参数：
    识别为 scale_m 输入（识别不了尺寸含义时不猜——缺省）。 """
    import re as _re

    inputs: dict[str, Any] = {}
    lowered = goal_text.lower()
    # 0902 盲写语料实证：改成/改为/换成 后的形状才是当前目标
    # （"不要五角星了改成画圆"取前置 = 画错形状冒充）。
    shape = select_shape(goal_text)
    if shape:
        inputs["shape"] = shape
    if any(m in goal_text or m in lowered for m in _YZ_MARKERS):
        inputs["plane"] = "yz"
    elif any(m in goal_text or m in lowered for m in _VERTICAL_MARKERS):
        inputs["plane"] = "xz"
    size = _re.search(r"(\d+(?:\.\d+)?)\s*(厘米|cm|毫米|mm)(?![\w])", lowered)
    if size:
        value = float(size.group(1))
        unit = size.group(2)
        scale_m = value / 100.0 if unit in ("厘米", "cm") else value / 1000.0
        if 0.005 <= scale_m <= 0.5:  # 工作区物理上界之外不猜（诚实缺省）
            inputs["scale_m"] = scale_m
    return inputs


__all__ = [
    "RECIPE_BY_GOAL",
    "RECIPE_BY_INTENT",
    "RECIPE_COVERAGE",
    "compile_recipe_inputs",
    "is_task_directive",
    "route_recipe",
]
