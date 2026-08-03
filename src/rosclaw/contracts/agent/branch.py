"""ReasoningBranchV1（批次 F §8.14）：推理分支，不是世界回滚。

双时间线：
- 推理分支树：可以 fork、切换、重写后续推理；
- 物理事实线：只追加，永远不能回滚、删除或被旧分支遮蔽。

第一阶段只实现只读 tree 与 fork-创建新-SIMULATION-mission；
同 Mission 分支切换在不变量测试完成后才开放。
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class ReasoningBranchV1(ContractModel):
    SCHEMA = "rosclaw.reasoning_branch.v1"

    schema_version: Literal["rosclaw.reasoning_branch.v1"] = "rosclaw.reasoning_branch.v1"
    branch_id: str
    parent_branch_id: str | None = None
    #: fork 起点的 journal entry_id（append-only 历史中的确定位置）
    fork_from_entry_id: str
    base_mission_event_seq: int = 0
    base_context_revision: int = 0
    label: str = ""
    created_by: str
    created_at: str
    status: Literal["active", "archived"] = "active"
    #: fork 生成的新 mission（第一阶段：fork 永远开新 SIMULATION mission）
    forked_mission_id: str | None = None
    notes: list[str] = Field(default_factory=list)
