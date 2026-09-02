"""TaskSpecV2 编译器（P1-C1，0824 总纲 §7.1）。

intent 分类用**通用动词规则**（中英文动词类标记——与 coordinator
的媒体交付标记同一既有模式）：形状/场景词不参与分类（五角星和
圆是同一种 manipulation.draw_path——形状特例是脆弱硬编码）。
body_ref 经 P0-G alias 归一为 canonical ``robot:<name>``。
"""

from __future__ import annotations

from rosclaw.contracts.agent.task_spec import (
    TaskConstraintsV2,
    TaskDeliverableV2,
    TaskGoalV2,
    TaskPreferencesV2,
    TaskRequirementV2,
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

#: R0-2 交付/场景标记（通用词类——与 intent 动词规则同一模式；
#: 形状/场景特例不进规则）。媒体：视频类 → scene_video（required），
#: 动画/GIF 类 → preview_animation；工具/场景/接触各一词类。
_SCENE_VIDEO_MARKERS = ("视频", "video", "mp4", "录像")
_PREVIEW_MARKERS = ("gif", "动图")
_TOOL_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tool:pen", ("笔", "pen", "画笔", "马克笔", "marker")),
)
_WORLD_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("world:tabletop", ("桌面", "桌子", "桌上", "tabletop", "table")),
)
_CONTACT_MARKERS = ("接触", "contact", "贴在", "贴着")


def _any_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(m in lowered or m in text for m in markers)


def _compile_deliverables(goal_text: str) -> list[TaskDeliverableV2]:
    """目标文本 → 交付物声明（只声明用户要的——不编造 required）。"""
    deliverables: list[TaskDeliverableV2] = []
    if _any_marker(goal_text, _SCENE_VIDEO_MARKERS):
        deliverables.append(TaskDeliverableV2(
            kind="scene_video", media_type="video/mp4", required=True,
            min_frames=30, min_resolution=[640, 360],
        ))
    if _any_marker(goal_text, _PREVIEW_MARKERS):
        deliverables.append(TaskDeliverableV2(
            kind="preview_animation", media_type="image/gif",
            required=True, min_frames=30,
        ))
    return deliverables


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
    # R0-2：工具/场景/接触声明（通用标记——无形状特例）。
    tool_ref = ""
    for ref, markers in _TOOL_MARKERS:
        if _any_marker(goal_text, markers):
            tool_ref = ref
            break
    if not world_ref:
        for ref, markers in _WORLD_MARKERS:
            if _any_marker(goal_text, markers):
                world_ref = ref
                break
    contact_required = _any_marker(goal_text, _CONTACT_MARKERS)
    # "在桌面/桌上画" = 工具与桌面接触（draw on table 的通用语义）。
    if world_ref == "world:tabletop" and _classify_intent(goal_text) in (
        "manipulation.draw_path",
        "manipulation.reach",
    ):
        contact_required = True
    # 0902 R0-2：材料性要求条款随 spec 冻结（RequirementCompiler——
    # 覆盖率门禁与逐条验收的单一事实源）。
    from rosclaw.task_kernel.requirements import compile_requirements

    requirements = [
        TaskRequirementV2(
            req_id=r.req_id, level=r.level, claim=r.claim,
            verifier=r.verifier,
        )
        for r in compile_requirements(goal_text)
    ]
    return TaskSpecV2(
        spec_id=new_id("tspec"),
        task_id=task_id,
        revision=revision,
        goal=TaskGoalV2(
            natural_language=goal_text,
            intent=_classify_intent(goal_text),
        ),
        subjects=TaskSubjectsV2(
            body_ref=body_ref, tool_ref=tool_ref, world_ref=world_ref
        ),
        constraints=TaskConstraintsV2(
            mode=mode,
            frames=list(frames or []),
            allowed_effects=list(allowed_effects or []),
            contact_required=contact_required,
            tool_axis_aligned_with_plane_normal_deg=(
                3.0 if tool_ref else None
            ),
        ),
        preferences=TaskPreferencesV2(language=language, verbosity=verbosity),
        deliverables=_compile_deliverables(goal_text),
        requirements=requirements,
        acceptance_spec_id=acceptance_spec_id,
    )


__all__ = ["compile_task_spec"]
