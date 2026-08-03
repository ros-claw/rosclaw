"""MissionSnapshotV1 (补充实施文档 §5.3)：重连校准用的权威快照。

SSE 断线或 sequence 出现缺口时，客户端停止乐观渲染并拉取快照重新对齐。
快照只含可公开状态——secret、Permit、daemon challenge 永不进入。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class MissionSnapshotV1(ContractModel):
    SCHEMA = "rosclaw.ui.mission_snapshot.v1"

    schema_version: Literal["rosclaw.ui.mission_snapshot.v1"] = (
        "rosclaw.ui.mission_snapshot.v1"
    )
    mission_id: str
    name: str = ""
    goal_text: str
    state: str
    mode: str
    body_id: str
    context_id: str
    context_revision: int
    task_graph_revision: int = 0
    #: authoritative event watermark the snapshot is consistent with
    last_event_sequence: int = 0
    turn_in_flight: bool = False
    pending_approvals: list[dict[str, Any]] = Field(default_factory=list)
    active_grants: list[dict[str, Any]] = Field(default_factory=list)
    open_work_orders: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    budgets: dict[str, Any] = Field(default_factory=dict)
    compaction_count: int = 0
    tool_count: int = 0
    degraded: str = ""
    captured_at: str
