"""TaskSpecV2 编译器（P1-C1，0824 总纲 §7.1）。

intent 分类用**通用动词规则**（中英文动词类标记——与 coordinator
的媒体交付标记同一既有模式）：形状/场景词不参与分类（五角星和
圆是同一种 manipulation.draw_path——形状特例是脆弱硬编码）。
body_ref 经 P0-G alias 归一为 canonical ``robot:<name>``。
"""

from __future__ import annotations

from rosclaw.contracts.agent.task_spec import (
    TaskConstraintsV2,
    TaskGoalV2,
    TaskPreferencesV2,
    TaskSpecV2,
    TaskSubjectsV2,
)
from rosclaw.contracts.common import new_id

#: intent ← 通用动词标记（中英文；首个命中的类胜出——draw/reach/
#: pick/navigate/inspect 是不同动词类，不是场景词）。
_INTENT_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("manipulation.draw_path", ("画", "draw", "轨迹绘制", "trace path")),
    ("manipulation.reach", ("伸到", "伸至", "reach", "移动到点", "触达")),
    ("manipulation.pick_place", ("抓起", "抓取", "pick", "place", "放到", "搬运")),
    ("mobile.navigate", ("导航", "navigate", "开到", "行驶到", "巡检")),
    ("perception.inspect", ("看一下", "检查", "inspect", "复核", "辨认", "靠近看")),
    ("learned_policy.execute", ("策略执行", "policy rollout", " learned policy")),
    ("compute.generic", ("计算", "compute", "分析", "误差", "统计")),
)
_CHAT_MARKERS = ("你好", "介绍", "hello", "你是谁", "谢谢")


def _classify_intent(goal_text: str) -> str:
    lowered = goal_text.lower()
    for intent, markers in _INTENT_MARKERS:
        if any(marker in lowered or marker in goal_text for marker in markers):
            return intent
    if any(marker in goal_text for marker in _CHAT_MARKERS):
        return "conversation.chat"
    return "task.unknown"


def compile_task_spec(
    *,
    task_id: str,
    revision: int,
    goal_text: str,
    body_id: str,
    mode: str,
    acceptance_spec_id: str,
    world_ref: str = "",
    frames: list[str] | None = None,
    allowed_effects: list[str] | None = None,
    language: str = "",
    verbosity: str = "",
) -> TaskSpecV2:
    """goal 文本 + 任务事实 → TaskSpecV2（一个 revision 一份）。"""
    body_ref = ""
    if body_id:
        from rosclaw.cognition.alias import canonical_resource_id

        body_ref = canonical_resource_id(body_id)
    return TaskSpecV2(
        spec_id=new_id("tspec"),
        task_id=task_id,
        revision=revision,
        goal=TaskGoalV2(
            natural_language=goal_text,
            intent=_classify_intent(goal_text),
        ),
        subjects=TaskSubjectsV2(body_ref=body_ref, world_ref=world_ref),
        constraints=TaskConstraintsV2(
            mode=mode,
            frames=list(frames or []),
            allowed_effects=list(allowed_effects or []),
        ),
        preferences=TaskPreferencesV2(language=language, verbosity=verbosity),
        acceptance_spec_id=acceptance_spec_id,
    )


__all__ = ["compile_task_spec"]
