"""Branch service（批次 F 第一阶段）：只读 /tree + /fork 创建新 SIMULATION mission。

硬不变量（§8.14/§14.9）：
- fork 永远创建新 mission——同 Mission 分支切换本阶段不存在；
- fork 不复制任何 authority：无 grant、approval、Permit、worker lease；
- 物理事实线只追加：fork 后新 mission 的第一轮编译强制注入最新
  Body/Self（ContextCompiler 本来就从权威存储重建）；
- 旧 DecisionV1 对新 mission 无效（新 context_id / revision 从 0 开始）；
- 物理动作进行中的 mission 拒绝 fork（fail closed）。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rosclaw.agentd.mission.store import _utcnow
from rosclaw.contracts.agent.branch import ReasoningBranchV1
from rosclaw.contracts.common import ValidationError, new_id

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService


class BranchService:
    def __init__(self, service: AgentService) -> None:
        self._service = service

    def tree(self, mission_id: str) -> dict:
        """只读树：推理分支 + 不可变物理事实线（append-only）。"""
        service = self._service
        mission = service.get_mission(mission_id)
        if mission is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        branches = self._branches_for(mission_id)
        physical_lane = [
            {
                "sequence": e.sequence,
                "type": e.type.value,
                "timestamp": e.timestamp,
            }
            for e in service.events_replay(mission_id, limit=10_000)
            if e.type.value.startswith(("approval.", "action.", "grant.", "receipt."))
        ]
        return {
            "mission_id": mission_id,
            "current_state": mission.state.value,
            "reasoning_branches": [b.model_dump(mode="json") for b in branches],
            "physical_lane": physical_lane,
            "physical_lane_note": "物理事实线只追加，永远不能回滚、删除或被旧分支遮蔽。",
        }

    def fork(
        self,
        mission_id: str,
        *,
        from_entry_id: str | None = None,
        label: str = "",
        principal: str = "user:local:1000",
    ) -> ReasoningBranchV1:
        """fork → 新 SIMULATION mission（不复制任何 authority）。"""
        service = self._service
        mission = service.get_mission(mission_id)
        if mission is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        # 物理动作进行中默认禁止 fork（fail closed，§8.14）。
        open_orders = [
            o
            for o in service._worker_manager.orders_for_mission(mission_id)
            if o.status not in ("ACCEPTED", "REJECTED", "EXPIRED", "CANCELLED", "FAILED")
        ]
        if open_orders:
            raise ValidationError(
                f"{len(open_orders)} 个 WorkOrder 未终态——fork 被拒绝（fail closed）"
            )
        # fork 基于 canonical journal（compaction 前的历史也不丢）。
        conversation = service.store.conversation_canonical(mission_id)
        entry = None
        if from_entry_id is not None:
            entry = next(
                (m for m in conversation if m.get("entry_id") == from_entry_id), None
            )
            if entry is None:
                raise ValidationError(f"unknown journal entry {from_entry_id!r}")
        # 新 SIMULATION mission（fork 永远不继承 REAL mode）。
        goal = f"[fork of {mission_id}] {label or mission.goal.text[:60]}"
        forked = service.create_mission(goal, mode="SIMULATION")
        # 推理上下文可带入（作为 untrusted 历史参考）；物理事实不复制——
        # 新 mission 第一轮编译从权威存储注入最新 Body/Self。
        if from_entry_id is not None and entry is not None:
            upto = [m for m in conversation if m.get("seq", 0) <= entry.get("seq", 0)]
            replay = [
                {
                    "role": m.get("role", "user"),
                    "content": str(m.get("content", "")),
                    "source": f"fork:{mission_id}",
                }
                for m in upto
            ]
            if replay:
                service.store.append_conversation(
                    forked.mission_id, replay, actor_id=service.actor_id
                )
        branch = ReasoningBranchV1(
            branch_id=new_id("br"),
            parent_branch_id=None,
            fork_from_entry_id=from_entry_id or "(head)",
            base_mission_event_seq=service._events.latest_sequence(mission_id),
            base_context_revision=mission.context_revision,
            label=label,
            created_by=principal,
            created_at=_utcnow(),
            forked_mission_id=forked.mission_id,
            notes=[
                "authority 不复制：无 grant/approval/Permit/worker lease",
                "旧 DecisionV1 对新 mission 无效（新 context，revision 从 0 开始）",
                "第一轮编译强制注入最新 Body/Self",
            ],
        )
        self._save_branch(mission_id, branch)
        return branch

    # -- 持久化（mission_meta 旁表，append-only JSON 行） -------------------------

    def _save_branch(self, mission_id: str, branch: ReasoningBranchV1) -> None:
        service = self._service
        service.store.connection.execute(
            "INSERT INTO reasoning_branches (branch_id, mission_id, created_at, "
            "branch_json) VALUES (?, ?, ?, ?)",
            (
                branch.branch_id,
                mission_id,
                branch.created_at,
                branch.model_dump_json(),
            ),
        )

    def _branches_for(self, mission_id: str) -> list[ReasoningBranchV1]:
        service = self._service
        try:
            rows = service.store.connection.execute(
                "SELECT branch_json FROM reasoning_branches WHERE mission_id = ? "
                "ORDER BY created_at",
                (mission_id,),
            ).fetchall()
        except Exception:  # noqa: BLE001 - 表可能尚未迁移（旧库）→ 空树
            return []
        return [ReasoningBranchV1(**json.loads(r["branch_json"])) for r in rows]
