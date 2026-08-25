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
    """作用对象：body_ref（canonical robot:<name>）与 world_ref。"""

    SCHEMA: ClassVar[str] = "rosclaw.task_subjects.v2"

    schema_version: Literal["rosclaw.task_subjects.v2"] = (
        "rosclaw.task_subjects.v2"
    )
    body_ref: str = ""
    world_ref: str = ""


class TaskConstraintsV2(ContractModel):
    """执行约束：mode / frames / allowed_effects。"""

    SCHEMA: ClassVar[str] = "rosclaw.task_constraints.v2"

    schema_version: Literal["rosclaw.task_constraints.v2"] = (
        "rosclaw.task_constraints.v2"
    )
    mode: str = "SIMULATION"
    frames: list[str] = Field(default_factory=list)
    allowed_effects: list[str] = Field(default_factory=list)


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
    #: 冻结验收规格关联（AcceptanceSpecV2.spec_id；未冻结为空——
    #: 诚实空，不编造）。
    acceptance_spec_id: str = ""


__all__ = [
    "INTENT_TAXONOMY",
    "TaskConstraintsV2",
    "TaskGoalV2",
    "TaskPreferencesV2",
    "TaskSpecV2",
    "TaskSubjectsV2",
]
