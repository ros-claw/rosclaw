"""TaskSpecV2（P1-C1，0824 总纲 §7.1）——任务标准工单。

任务不再只有 root_goal 自由文本：goal（natural_language + 冻结
intent 分类）、subjects（body_ref/world_ref）、constraints（mode/
frames/allowed_effects）、preferences 都是契约一等字段。验收标准
由 AcceptanceSpecV2 承载（acceptance_spec_id 关联——模型只能加严
不能放宽的既有不变量不变）。

intent 分类是**冻结分类法**（0824 §23 验收任务族）——通用动词类，
不含形状/场景特例。
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, field_validator

from rosclaw.contracts.common import ContractModel

#: intent 冻结分类（0824 §7.1/§23）。通用动词类——形状（五角星/
#: 圆）与场景词不进 intent（实证坑：形状特例是脆弱硬编码）。
INTENT_TAXONOMY = (
    "manipulation.draw_path",
    "manipulation.reach",
    "manipulation.pick_place",
    "mobile.navigate",
    "perception.inspect",
    "learned_policy.execute",
    "compute.generic",
    "conversation.chat",
    "task.unknown",
)


class TaskGoalV2(ContractModel):
    """任务目标：自然语言原文 + 分类 intent。"""

    SCHEMA: ClassVar[str] = "rosclaw.task_goal.v2"

    schema_version: Literal["rosclaw.task_goal.v2"] = "rosclaw.task_goal.v2"
    natural_language: str = Field(min_length=1)
    intent: str = Field(min_length=1)

    @field_validator("intent")
    @classmethod
    def _known_intent(cls, value: str) -> str:
        if value not in INTENT_TAXONOMY:
            raise ValueError(
                f"unknown intent {value!r} (frozen taxonomy: {INTENT_TAXONOMY})"
            )
        return value


class TaskSubjectsV2(ContractModel):
    """作用对象：body_ref（canonical robot:<name>）、tool_ref 与
    world_ref（R0-2：工具/场景是验收一等事实——"持笔在桌面"必须
    可声明、可核验，不允许最终回答自由夸大）。"""

    SCHEMA: ClassVar[str] = "rosclaw.task_subjects.v2"

    schema_version: Literal["rosclaw.task_subjects.v2"] = (
        "rosclaw.task_subjects.v2"
    )
    body_ref: str = ""
    tool_ref: str = ""
    world_ref: str = ""


#: 交付物 kind 冻结分类（R0-2，0826 体验审计 §5.R0-2）。
DELIVERABLE_KINDS = (
    "scene_video",
    "preview_animation",
    "robot_video",
    "data_report",
)

#: 交付物 kind → 产物 kind（lineage.kind 权威）：preview_2d /
#: scene_3d / robot_video 是不同产物 kind——2D 预览永远不能
#: 满足 scene_video（0826 审计：GIF 冒充场景视频是假绿）。
DELIVERABLE_KIND_TO_ARTIFACT_KIND: dict[str, str] = {
    "scene_video": "scene_3d",
    "preview_animation": "preview_2d",
    "robot_video": "robot_video",
}


class TaskDeliverableV2(ContractModel):
    """交付物声明：kind/media_type/required + 最低质量门槛。

    required=True 的交付物未出现在产物账本（按 kind 匹配）→
    DELIVERABLE_MISSING——任务成功 ≠ 用户请求成功。
    """

    SCHEMA: ClassVar[str] = "rosclaw.task_deliverable.v2"

    schema_version: Literal["rosclaw.task_deliverable.v2"] = (
        "rosclaw.task_deliverable.v2"
    )
    kind: str = Field(min_length=1)
    media_type: str = ""
    required: bool = False
    min_frames: int = 0
    min_resolution: list[int] = Field(default_factory=list)

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in DELIVERABLE_KINDS:
            raise ValueError(
                f"unknown deliverable kind {value!r} "
                f"(frozen taxonomy: {DELIVERABLE_KINDS})"
            )
        return value


class TaskConstraintsV2(ContractModel):
    """执行约束：mode / frames / allowed_effects + R0-2 接触与
    工具轴约束（contact_required / tool_axis_aligned_with_plane_
    normal_deg——语义验收的声明面）。"""

    SCHEMA: ClassVar[str] = "rosclaw.task_constraints.v2"

    schema_version: Literal["rosclaw.task_constraints.v2"] = (
        "rosclaw.task_constraints.v2"
    )
    mode: str = "SIMULATION"
    frames: list[str] = Field(default_factory=list)
    allowed_effects: list[str] = Field(default_factory=list)
    contact_required: bool = False
    tool_axis_aligned_with_plane_normal_deg: float | None = None


class TaskPreferencesV2(ContractModel):
    """偏好（非验收因素）。"""

    SCHEMA: ClassVar[str] = "rosclaw.task_preferences.v2"

    schema_version: Literal["rosclaw.task_preferences.v2"] = (
        "rosclaw.task_preferences.v2"
    )
    language: str = ""
    verbosity: str = ""


class TaskSpecV2(ContractModel):
    """TaskSpecV2（一个 task revision 一份——随 revision 冻结）。"""

    SCHEMA: ClassVar[str] = "rosclaw.task_spec.v2"
    HASH_PREFIX: ClassVar[str] = "task_spec"

    schema_version: Literal["rosclaw.task_spec.v2"] = "rosclaw.task_spec.v2"
    spec_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    goal: TaskGoalV2
    subjects: TaskSubjectsV2 = Field(default_factory=TaskSubjectsV2)
    constraints: TaskConstraintsV2 = Field(default_factory=TaskConstraintsV2)
    preferences: TaskPreferencesV2 = Field(default_factory=TaskPreferencesV2)
    #: 交付物声明（R0-2）：required 交付物未按 kind 出现在产物
    #: 账本 → DELIVERABLE_MISSING——任务成功 ≠ 用户请求成功。
    deliverables: list[TaskDeliverableV2] = Field(default_factory=list)
    #: 冻结验收规格关联（AcceptanceSpecV2.spec_id；未冻结为空——
    #: 诚实空，不编造）。
    acceptance_spec_id: str = ""


__all__ = [
    "DELIVERABLE_KINDS",
    "DELIVERABLE_KIND_TO_ARTIFACT_KIND",
    "INTENT_TAXONOMY",
    "TaskConstraintsV2",
    "TaskDeliverableV2",
    "TaskGoalV2",
    "TaskPreferencesV2",
    "TaskSpecV2",
    "TaskSubjectsV2",
]
