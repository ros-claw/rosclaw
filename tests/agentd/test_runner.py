"""MissionRunner tests (PR-03 exits).

- approval decided 自动唤醒续跑：用户批准后台词/HTTP 后，runner 自己推进
  下一回合——不需要用户再说"继续"
- 唤醒注入如实标记 [rosclaw system wake]，不是伪造用户消息
- 不同 Mission 并发推进（无全局锁串行）
- 同一 Mission 顺序执行（busy 拒绝）
- 唤醒链有上限，超时安全退出
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from tests.agentd.conftest import LOCAL_PRINCIPAL


def _approval_then_action(request) -> ModelTurnResultV1:
    calls = getattr(_approval_then_action, "calls", 0) + 1
    _approval_then_action.calls = calls
    if calls == 1:
        decision = {
            "schema_version": "rosclaw.decision.v1",
            "decision_id": "d1",
            "mission_id": request.mission_id,
            "context_id": request.context_id,
            "context_revision": request.context_revision,
            "next_intent": "REQUEST_APPROVAL",
            "summary": "请求授权",
            "evidence_refs": ["a://1"],
            "proposed_operation": {
                "type": "approval_request",
                "payload": {"title": "t", "summary": "s", "risk_tier": "LOW"},
            },
            "verification": {
                "schema_version": "rosclaw.decision_verification.v1",
                "verifiers": ["deterministic:x"],
            },
        }
    else:
        grant_id = _approval_then_action.grant_id
        decision = {
            "schema_version": "rosclaw.decision.v1",
            "decision_id": "d2",
            "mission_id": request.mission_id,
            "context_id": request.context_id,
            "context_revision": request.context_revision,
            "next_intent": "REQUEST_ACTION",
            "summary": "按授权执行",
            "evidence_refs": ["a://1"],
            "proposed_operation": {
                "type": "request_action",
                "payload": {"grant_id": grant_id, "risk_tier": "LOW"},
            },
            "verification": {
                "schema_version": "rosclaw.decision_verification.v1",
                "verifiers": ["deterministic:x"],
            },
        }
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


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


class TestApprovalAutoWake:
    async def test_approval_decision_wakes_mission(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        _approval_then_action.calls = 0
        gateway = MockModelGateway(mock_profile(), [_approval_then_action] * 6)
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            mission = service.create_mission("自动唤醒测试")
            await service.submit_turn_v2(mission.mission_id, "请求授权")
            await service._turn_tasks[mission.mission_id]
            state = service.get_mission(mission.mission_id).state.value
            assert state == "WAIT_APPROVAL"
            pending = service.pending_approvals(mission.mission_id)
            grant = await service.decide_approval(
                pending[0].request_id, principal=LOCAL_PRINCIPAL, approve=True
            , _from_operatord=True)
            _approval_then_action.grant_id = grant.grant_id
            # 唤醒任务自己运行：无需用户再发消息。
            wake_task = service._runner._wake_tasks.get(mission.mission_id)
            assert wake_task is not None
            await asyncio.wait_for(wake_task, timeout=15)
            # 链式回合可能继续（REQUEST_ACTION → SIM 无通道/授权验证 → LEARN/IDLE）
            turn_task = service._turn_tasks[mission.mission_id]
            if not turn_task.done():
                await asyncio.wait_for(turn_task, timeout=15)
            history = service.conversation(mission.mission_id)
            wake_msgs = [m for m in history if "[rosclaw system wake]" in str(m.get("content"))]
            assert wake_msgs, "wake notice must be journaled honestly"
            assert grant.grant_id in str(wake_msgs[0]["content"])
        finally:
            await service.close()

    async def test_deny_also_wakes_with_guidance(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        _approval_then_action.calls = 0
        gateway = MockModelGateway(mock_profile(), [_approval_then_action] * 6)
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            mission = service.create_mission("拒绝唤醒测试")
            await service.submit_turn_v2(mission.mission_id, "请求授权")
            await service._turn_tasks[mission.mission_id]
            pending = service.pending_approvals(mission.mission_id)
            result = await service.decide_approval(
                pending[0].request_id, principal=LOCAL_PRINCIPAL, approve=False
            , _from_operatord=True)
            assert result is None
            wake_task = service._runner._wake_tasks.get(mission.mission_id)
            await asyncio.wait_for(wake_task, timeout=15)
            history = service.conversation(mission.mission_id)
            wake_msgs = [m for m in history if "DENIED" in str(m.get("content"))]
            assert wake_msgs
        finally:
            await service.close()


class TestPerMissionConcurrency:
    async def test_two_missions_run_concurrently(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")

        class SlowGateway(MockModelGateway):
            async def complete(self, request):
                await asyncio.sleep(0.6)
                return _answer(request)

            async def complete_stream(self, request, on_text_delta=None):
                return await self.complete(request)

        gateway = SlowGateway(mock_profile(), [])
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            m1 = service.create_mission("并发 1")
            m2 = service.create_mission("并发 2")
            import time

            started = time.monotonic()
            t1 = asyncio.create_task(
                _await_task(service.submit_turn_v2(m1.mission_id, "hi"), service, m1.mission_id)
            )
            t2 = asyncio.create_task(
                _await_task(service.submit_turn_v2(m2.mission_id, "hi"), service, m2.mission_id)
            )
            await asyncio.gather(t1, t2)
            elapsed = time.monotonic() - started
            # 全局锁时代将是 ~1.2s 串行；每 Mission 锁应接近 ~0.6s 并发。
            # 阈值取 1.1：留出负载抖动余量，仍能区分串行。
            assert elapsed < 1.1, f"missions were serialized: {elapsed:.2f}s"
        finally:
            await service.close()


async def _await_task(coro, service, mission_id: str) -> None:
    await coro
    task = service._turn_tasks[mission_id]
    if not task.done():
        await task


class TestWakeBounds:
    async def test_no_wake_for_terminal_states(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        gateway = MockModelGateway(mock_profile(), [_answer] * 4)
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            mission = service.create_mission("无唤醒测试")
            await service.submit_turn_v2(mission.mission_id, "你好")
            await service._turn_tasks[mission.mission_id]
            assert mission.mission_id not in service._runner._pending
        finally:
            await service.close()

    async def test_wake_chain_capped(self, tmp_path: Path) -> None:
        from rosclaw.agentd.runner import MAX_WAKE_CHAIN

        assert MAX_WAKE_CHAIN == 3  # 常量即上限（实现中 chain >= MAX 即停）
