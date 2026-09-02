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
        # 不覆盖（——出现即转 Native Agent）：receipt.tool_ref（挂载
        # 工具）、render.tool_color（颜色）、receipt.overlays.*（轨迹
        # 叠加）、delivery.not_2d_only（禁止项）、verification.contact
        # （接触）、未知形状。
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


#: R0-1.5（金丝雀实证）：NL → recipe 几何参数（通用词类标记——
#: 形状词/平面词是通用类别，不是五角星特例 prompt 硬编码）。
_SHAPE_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("star5", ("五角星", "星形", "star")),
    ("circle", ("圆", "circle")),
)

#: 竖直平面词类 → 命名面（xz：y=const 立面——画板/墙面的常见
#: 含义；yz 留给显式 "yz 面"）。
_VERTICAL_MARKERS = ("竖直平面", "垂直平面", "竖直面", "垂直面", "立面",
                     "vertical plane", "竖着", "竖直方向")
_YZ_MARKERS = ("yz 面", "yz面", "yz plane")

#: 疑问/讨论形式护栏——"怎么画五角星"是讨论不是指令，不自动执行。
_QUESTION_MARKERS = ("怎么", "如何", "吗", "？", "?", "what", "how",
                     "why", "解释", "介绍", "是什么")


def is_task_directive(text: str) -> bool:
    """指令形式判定：疑问句/讨论形式不自动执行。"""
    lowered = text.lower()
    return not any(m in lowered or m in text for m in _QUESTION_MARKERS)


def compile_recipe_inputs(goal_text: str) -> dict[str, Any]:
    """NL → recipe 几何参数（缺的字段由 recipe 确定性缺省补齐）。"""
    inputs: dict[str, Any] = {}
    lowered = goal_text.lower()
    for shape, markers in _SHAPE_WORDS:
        if any(m in goal_text or m in lowered for m in markers):
            inputs["shape"] = shape
            break
    if any(m in goal_text or m in lowered for m in _YZ_MARKERS):
        inputs["plane"] = "yz"
    elif any(m in goal_text or m in lowered for m in _VERTICAL_MARKERS):
        inputs["plane"] = "xz"
    return inputs


__all__ = [
    "RECIPE_BY_GOAL",
    "RECIPE_BY_INTENT",
    "RECIPE_COVERAGE",
    "compile_recipe_inputs",
    "is_task_directive",
    "route_recipe",
]
