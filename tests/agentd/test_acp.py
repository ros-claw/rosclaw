"""批次 G：ACP adapter 测试。

- session/prompt/cancel 映射到 Mission/turn
- 事件流映射为 session update（text delta、tool、approval 卡片）
- 边界：ACP 路径不产生任何 authority；approval 只呈现 id，不自动批准
"""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.adapters.acp.mapper import event_to_session_update
from rosclaw.adapters.acp.server import RosclawAcpAgent
from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1


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


class _FakeClient:
    def __init__(self) -> None:
        self.updates: list = []

    async def session_update(self, session_id: str, update) -> None:
        self.updates.append((session_id, update))


def _text_block(text: str):
    from acp import schema

    return schema.TextContentBlock(type="text", text=text)


class TestAcpMapping:
    async def test_session_prompt_cycle(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            agent = RosclawAcpAgent(service)
            client = _FakeClient()
            agent.on_connect(client)
            init = await agent.initialize()
            assert init.agent_info.name == "rosclaw-native-agent"
            session = await agent.new_session(cwd="/tmp/work")
            mission_id = session.session_id
            assert service.get_mission(mission_id) is not None
            response = await agent.prompt(session_id=mission_id, prompt=[_text_block("你好")])
            assert response.stop_reason == "end_turn"
            # cancel 不抛异常。
            await agent.cancel(session_id=mission_id)
            # 未知 session 报错而不是伪造。
            import pytest
            from acp.exceptions import RequestError

            with pytest.raises(RequestError):
                await agent.prompt(session_id="mis_ghost", prompt=[_text_block("hi")])
        finally:
            await service.close()

    async def test_list_sessions(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            agent = RosclawAcpAgent(service)
            await agent.new_session(cwd="/tmp/a")
            listing = await agent.list_sessions()
            assert len(listing.sessions) == 1
            assert "ACP session" in listing.sessions[0].title
        finally:
            await service.close()

    async def test_no_authority_via_acp(self, tmp_path: Path) -> None:
        """ACP 路径（含 permission 语义）永不产生 grant/Permit。"""
        service = _service(tmp_path)
        try:
            agent = RosclawAcpAgent(service)
            client = _FakeClient()
            agent.on_connect(client)
            session = await agent.new_session(cwd="/tmp/work")
            await agent.prompt(session_id=session.session_id, prompt=[_text_block("请执行动作")])
            assert service.list_grants() == []
            assert service.pending_approvals() == []
        finally:
            await service.close()


class TestEventMapper:
    def _event(self, etype: str, payload: dict):
        from rosclaw.contracts.agent.agent_event import (
            AgentEventType,
            AgentEventV2,
            Visibility,
        )

        return AgentEventV2(
            event_id="e1",
            sequence=1,
            mission_id="mis_x",
            timestamp="t",
            type=AgentEventType(etype),
            visibility=Visibility.USER,
            payload=payload,
        )

    def test_text_delta_maps_to_message_chunk(self) -> None:
        update = event_to_session_update(self._event("model.text.delta", {"text": "你好"}))
        assert update is not None
        assert update.content.text == "你好"

    def test_tool_lifecycle_maps(self) -> None:
        started = event_to_session_update(self._event("tool.started", {"name": "sim_get_state"}))
        assert started.status == "in_progress"
        done = event_to_session_update(
            self._event("tool.completed", {"name": "sim_get_state", "ok": True})
        )
        assert done.status == "completed"

    def test_approval_is_message_not_permission(self) -> None:
        update = event_to_session_update(
            self._event(
                "approval.requested",
                {"request_id": "appr_x", "title": "播放提示音", "risk_tier": "LOW"},
            )
        )
        # 只呈现卡片文本；绝不是 ACP permission 请求。
        assert "appr_x" in update.content.text
        assert "不构成物理授权" in update.content.text

    def test_unmapped_returns_none(self) -> None:
        assert event_to_session_update(self._event("context.compilation.started", {})) is None


class TestStdioRoundtrip:
    """真实 stdio JSON-RPC 全链路：官方 SDK client ↔ rosclaw ACP server。"""

    async def test_initialize_session_prompt_over_stdio(self, tmp_path: Path) -> None:
        import asyncio
        import os
        import sys

        import acp
        from acp import schema

        updates: list = []

        class Client(acp.Client):
            async def session_update(self, session_id: str, update, **kwargs) -> None:
                updates.append((session_id, update))

        env = dict(os.environ, ROSCLAW_ACP_TEST_HOME=str(tmp_path))
        server_script = Path(__file__).parent / "acp_test_server.py"
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(server_script),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        conn = acp.connect_to_agent(Client(), proc.stdin, proc.stdout)
        try:
            init = await conn.initialize(
                protocol_version=1,
                client_capabilities=schema.ClientCapabilities(),
            )
            assert init.agent_info.name == "rosclaw-native-agent"
            session = await conn.new_session(cwd="/tmp/work")
            assert session.session_id.startswith("mis_")
            response = await conn.prompt(
                session_id=session.session_id,
                prompt=[schema.TextContentBlock(type="text", text="你好")],
            )
            assert response.stop_reason == "end_turn"
        finally:
            proc.terminate()
            await proc.wait()
