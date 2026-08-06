"""批次 F 测试：ReasoningBranch 双时间线不变量。

- /tree 只读：推理分支 + 物理事实线
- /fork 开新 SIMULATION mission：不复制 authority、最新 Body/Self、
  旧 Decision 无效
- 物理动作进行中 fork 被拒（fail closed）
- 物理事实线不被 fork 遮蔽
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.contracts.common import ValidationError
from rosclaw.contracts.ui.commands import CommandRequestV1


def _answer(request) -> ModelTurnResultV1:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "d",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": "ANSWER",
        "summary": "ok",
        "evidence_refs": [],
    }
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": "x"},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


def _service(tmp_path: Path) -> AgentService:
    config = load_agent_config(tmp_path / "config.yaml")
    return AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), [_answer] * 50))


def _cmd(name: str, args: dict | None = None, mission_id: str | None = None, key: str = "k") -> CommandRequestV1:
    return CommandRequestV1(
        request_id=f"r_{key}",
        idempotency_key=key,
        command_name=name,
        arguments=args or {},
        mission_id=mission_id,
    )


class TestBranchTree:
    async def test_tree_readonly(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            mission = service.create_mission("tree 测试")
            await service.send_turn(mission.mission_id, "hi")
            result = await service.commands.execute(
                _cmd("tree", mission_id=mission.mission_id)
            )
            assert result.ok
            assert result.data["reasoning_branches"] == []
            assert "不能回滚" in result.data["physical_lane_note"]
        finally:
            await service.close()


class TestFork:
    async def test_fork_creates_sim_mission_without_authority(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            mission = service.create_mission("fork 源")
            await service.send_turn(mission.mission_id, "第一轮")
            history = service.store.conversation(mission.mission_id)
            entry_id = history[0]["entry_id"]
            result = await service.commands.execute(
                _cmd("fork", args={"from_entry_id": entry_id, "label": "分支A"},
                     mission_id=mission.mission_id)
            )
            assert result.ok, result.message
            forked_id = result.data["forked_mission_id"]
            assert forked_id != mission.mission_id
            forked = service.get_mission(forked_id)
            # fork 永远是 SIMULATION（不继承 REAL）且无 authority。
            assert forked.mode.value == "SIMULATION"
            assert forked.context_revision == 0
            assert service.list_grants() == []
            assert service.pending_approvals() == []
            # fork 点之前的推理历史带入（标记为 fork 来源）。
            forked_conv = service.store.conversation(forked_id)
            assert forked_conv
            assert any("fork:" in str(m.get("source", "")) for m in forked_conv)
            # 树里能看到分支。
            tree = await service.commands.execute(
                _cmd("tree", mission_id=mission.mission_id, key="k2")
            )
            assert len(tree.data["reasoning_branches"]) == 1
            # 物理事实线不被遮蔽：原 mission 事件原样存在。
            assert service.events_replay(mission.mission_id)
        finally:
            await service.close()

    async def test_fork_unknown_entry_rejected(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            mission = service.create_mission("fork 测试")
            result = await service.commands.execute(
                _cmd("fork", args={"from_entry_id": "conv_ghost_0"},
                     mission_id=mission.mission_id)
            )
            assert not result.ok and result.error_code == "fork_refused"
        finally:
            await service.close()

    async def test_fork_refused_with_open_orders(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            mission = service.create_mission("fork 护栏")
            # 塞一个未终态订单（直接写库模拟在途动作）。
            from rosclaw.agentd.workers.scheduler import ScoredCandidate
            from rosclaw.contracts.worker.order import WorkOrderV1

            order = WorkOrderV1(
                work_order_id="wo_open",
                mission_id=mission.mission_id,
                issued_by="test",
                capability="analysis.text",
                goal="在途任务",
                status="RUNNING",
            )
            scored = ScoredCandidate(
                worker_id="worker:native:basic", score=1.0, features={}, reasons=("test",)
            )
            service._worker_manager._insert(order, scored)
            with pytest.raises(ValidationError, match="未终态"):
                service.branches.fork(mission.mission_id)
        finally:
            await service.close()

    async def test_clone_honest_deferred(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            result = await service.commands.execute(_cmd("clone"))
            assert not result.ok and result.error_code == "not_implemented"
        finally:
            await service.close()
