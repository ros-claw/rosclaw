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


__all__ = ["RECIPE_BY_INTENT", "route_recipe"]
