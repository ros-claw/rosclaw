"""批次 G：ACP adapter 测试（+ Channel 设计 PR-RC-001/002/003）。

- session/prompt/cancel 映射到 Mission/turn
- session lifecycle：new/list/load/resume/close/cancel（设计 §20）
- 事件流映射为 session update（text delta、tool、worker、reasoning、approval 卡片）
- 事件订阅：journal replay → live bus，无丢失/无重复/sequence 单调（§21）
- stdio 黑盒：stdout 纯净（§36）、graceful EOF/SIGTERM（§37）、kill 后 resume（§49）
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

    async def test_capability_advertisement(self, tmp_path: Path) -> None:
        """PR-RC-001：capability 必须与实际实现一致（load_session=True 等）。"""
        service = _service(tmp_path)
        try:
            agent = RosclawAcpAgent(service)
            init = await agent.initialize()
            caps = init.agent_capabilities
            assert caps.load_session is True
            assert caps.session_capabilities.list is not None
            assert caps.session_capabilities.resume is not None
            assert caps.session_capabilities.close is not None
            # P0 附件策略（§38）：不声明 image/audio。
            assert caps.prompt_capabilities.image is False
            assert caps.prompt_capabilities.audio is False
        finally:
            await service.close()

    async def test_session_lifecycle_resume_close(self, tmp_path: Path) -> None:
        """设计 §20：close 不删 Mission；resume 重新绑定且不重放 transcript。"""
        import pytest
        from acp.exceptions import RequestError

        service = _service(tmp_path)
        try:
            agent = RosclawAcpAgent(service)
            client = _FakeClient()
            agent.on_connect(client)
            session = await agent.new_session(cwd="/tmp/work")
            sid = session.session_id
            await agent.prompt(session_id=sid, prompt=[_text_block("你好")])

            # close：释放绑定，Mission journal 保留，不归档。
            await agent.close_session(session_id=sid)
            assert service.get_mission(sid) is not None
            assert not service.mission_archived(sid)
            with pytest.raises(RequestError):
                await agent.prompt(session_id=sid, prompt=[_text_block("hi")])

            # resume：重新绑定，不重放 transcript。
            updates_before = len(client.updates)
            await agent.resume_session(session_id=sid, cwd="/tmp/work")
            assert len(client.updates) == updates_before
            response = await agent.prompt(session_id=sid, prompt=[_text_block("继续")])
            assert response.stop_reason == "end_turn"

            # list 仍然看得到该 Mission。
            listing = await agent.list_sessions()
            assert any(s.session_id == sid for s in listing.sessions)
        finally:
            await service.close()

    async def test_load_session_replays_transcript(self, tmp_path: Path) -> None:
        """session/load = 重新绑定 + transcript replay（§20）。"""
        service = _service(tmp_path)
        try:
            agent = RosclawAcpAgent(service)
            client = _FakeClient()
            agent.on_connect(client)
            session = await agent.new_session(cwd="/tmp/work")
            sid = session.session_id
            await agent.prompt(session_id=sid, prompt=[_text_block("你好")])

            client2 = _FakeClient()
            agent.on_connect(client2)
            await agent.load_session(session_id=sid, cwd="/tmp/work")
            # replay 产生 session updates（至少含回复文本）。
            assert client2.updates
        finally:
            await service.close()

    async def test_cancel_is_turn_only(self, tmp_path: Path) -> None:
        """session/cancel 只取消当前 turn，不归档 Mission（§20）。"""
        service = _service(tmp_path)
        try:
            agent = RosclawAcpAgent(service)
            agent.on_connect(_FakeClient())
            session = await agent.new_session(cwd="/tmp/work")
            sid = session.session_id
            await agent.cancel(session_id=sid)
            assert service.get_mission(sid) is not None
            assert not service.mission_archived(sid)
            # cancel 后仍可继续 prompt。
            response = await agent.prompt(session_id=sid, prompt=[_text_block("在吗")])
            assert response.stop_reason == "end_turn"
        finally:
            await service.close()

    async def test_unsupported_attachment_notice(self, tmp_path: Path) -> None:
        """设计 §38：不悄悄忽略附件——明确告知未接入。"""
        from types import SimpleNamespace

        service = _service(tmp_path)
        try:
            agent = RosclawAcpAgent(service)
            client = _FakeClient()
            agent.on_connect(client)
            session = await agent.new_session(cwd="/tmp/work")
            response = await agent.prompt(
                session_id=session.session_id,
                prompt=[SimpleNamespace(type="image", data="...", mime_type="image/png")],
            )
            assert response.stop_reason == "end_turn"
            texts = [
                u.content.text
                for _, u in client.updates
                if getattr(u, "session_update", "") == "agent_message_chunk"
            ]
            assert any("尚未接入该附件类型" in t for t in texts)
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

    def test_tool_call_id_prefers_call_id_over_name(self) -> None:
        """设计 §23：连续同名调用不得合并；call_id > event_id，绝不用 name。"""
        started = event_to_session_update(
            self._event("tool.started", {"name": "sim_get_state", "call_id": "call_1"})
        )
        assert started.tool_call_id == "call_1"
        done = event_to_session_update(
            self._event(
                "tool.completed", {"name": "sim_get_state", "call_id": "call_1", "ok": True}
            )
        )
        assert done.tool_call_id == "call_1"
        # 无 call_id 时回退 event_id（仍稳定唯一），不是 name。
        fallback = event_to_session_update(self._event("tool.started", {"name": "x"}))
        assert fallback.tool_call_id == "e1"

    def test_tool_progress_maps(self) -> None:
        update = event_to_session_update(
            self._event("tool.progress", {"name": "sim_get_state", "call_id": "c9", "note": "50%"})
        )
        assert update.tool_call_id == "c9"
        assert update.status == "in_progress"

    def test_worker_uses_stable_work_order_id(self) -> None:
        """设计 §23/§29：work_order_id 是稳定 ID；start 与 update 区分。"""
        started = event_to_session_update(
            self._event("worker.started", {"work_order_id": "wo_1", "worker_id": "codex"})
        )
        assert started.session_update == "tool_call"
        assert started.tool_call_id == "wo_1"
        assert started.status == "in_progress"
        accepted = event_to_session_update(
            self._event("worker.accepted", {"work_order_id": "wo_1", "worker_id": "codex"})
        )
        assert type(accepted).__name__ == "ToolCallUpdate"
        assert accepted.tool_call_id == "wo_1"
        assert accepted.status == "completed"
        failed = event_to_session_update(
            self._event("worker.failed", {"work_order_id": "wo_1", "worker_id": "codex"})
        )
        assert failed.status == "failed"

    def test_reasoning_delta_maps_to_thought_chunk(self) -> None:
        """设计 §26：reasoning.summary.delta → AgentThoughtChunk。"""
        update = event_to_session_update(
            self._event("reasoning.summary.delta", {"text": "先检查 Body，再检查日志。"})
        )
        assert update is not None
        assert update.session_update == "agent_thought_chunk"
        assert "先检查 Body" in update.content.text
        # 生命周期标记不单独呈现。
        assert event_to_session_update(self._event("reasoning.started", {})) is None
        assert event_to_session_update(self._event("reasoning.summary.ended", {})) is None

    def test_plan_updated_maps_to_plan(self) -> None:
        update = event_to_session_update(
            self._event(
                "plan.updated",
                {
                    "entries": [
                        {"content": "加载 Mission", "status": "completed"},
                        {"content": "读取导航日志", "status": "in_progress"},
                    ]
                },
            )
        )
        assert update.session_update == "plan"
        assert [e.content for e in update.entries] == ["加载 Mission", "读取导航日志"]

    def test_debug_and_audit_not_sent_by_default(self) -> None:
        """设计 §48：DEBUG/AUDIT 事件默认不离开 ACP。"""
        from rosclaw.contracts.agent.agent_event import Visibility

        debug = self._event("model.text.delta", {"text": "hidden"})
        debug.visibility = Visibility.DEBUG
        assert event_to_session_update(debug) is None
        audit = self._event("model.text.delta", {"text": "audit"})
        audit.visibility = Visibility.AUDIT
        assert event_to_session_update(audit) is None
        # 显式开启才发。
        assert event_to_session_update(debug, include_debug=True) is not None

    def test_usage_update_requires_real_size(self) -> None:
        """model.request.ended → usage（§22）；没有真实上下文窗口大小时不伪造。"""
        no_size = event_to_session_update(
            self._event("model.request.ended", {"prompt_tokens": 5, "completion_tokens": 5})
        )
        assert no_size is None
        with_size = event_to_session_update(
            self._event(
                "model.request.ended",
                {"prompt_tokens": 5, "completion_tokens": 5, "context_size": 8192},
            )
        )
        assert with_size.session_update == "usage_update"
        assert with_size.used == 10
        assert with_size.size == 8192

    def test_error_maps_to_visible_message(self) -> None:
        update = event_to_session_update(self._event("error", {"message": "boom"}))
        assert "boom" in update.content.text

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


class TestEventSubscription:
    """PR-RC-002：subscribe_events — journal replay → live bus。"""

    async def test_replay_then_live_monotonic_no_dup(self, tmp_path: Path) -> None:
        from rosclaw.contracts.agent.agent_event import AgentEventType

        service = _service(tmp_path)
        try:
            mission = service.create_mission("sub test")
            mid = mission.mission_id
            await service._events.append(mid, AgentEventType.MODEL_TEXT_DELTA, {"text": "a"})
            await service._events.append(mid, AgentEventType.MODEL_TEXT_DELTA, {"text": "b"})
            stream = service.subscribe_events(mid, after_sequence=0)
            try:
                first = await anext(stream)
                second = await anext(stream)
                assert (first.sequence, second.sequence) == (1, 2)
                # live 事件接续 replay，sequence 严格单调。
                await service._events.append(mid, AgentEventType.MODEL_TEXT_DELTA, {"text": "c"})
                third = await anext(stream)
                assert third.sequence == 3
                assert third.payload["text"] == "c"
            finally:
                await stream.aclose()
        finally:
            await service.close()

    async def test_reconnect_with_after_sequence(self, tmp_path: Path) -> None:
        """断线重连：after_sequence 之后的事件完整重放，无重复。"""
        from rosclaw.contracts.agent.agent_event import AgentEventType

        service = _service(tmp_path)
        try:
            mission = service.create_mission("reconnect")
            mid = mission.mission_id
            for i in range(5):
                await service._events.append(
                    mid, AgentEventType.MODEL_TEXT_DELTA, {"text": f"m{i}"}
                )
            stream = service.subscribe_events(mid, after_sequence=3)
            try:
                seqs = [(await anext(stream)).sequence for _ in range(2)]
                assert seqs == [4, 5]
            finally:
                await stream.aclose()
        finally:
            await service.close()

    async def test_live_queue_gap_filled_from_journal(self) -> None:
        """live queue 溢出造成的 sequence 空洞必须从 journal 补齐（不丢事件）。"""
        import asyncio as _asyncio

        from rosclaw.agentd.events import stream_events
        from rosclaw.contracts.agent.agent_event import (
            AgentEventType,
            AgentEventV2,
            Visibility,
        )

        def _ev(seq: int) -> AgentEventV2:
            return AgentEventV2(
                event_id=f"e{seq}",
                sequence=seq,
                mission_id="mis_x",
                timestamp="t",
                type=AgentEventType.MODEL_TEXT_DELTA,
                visibility=Visibility.USER,
                payload={"text": str(seq)},
            )

        journal = [_ev(i) for i in range(1, 6)]

        class _Bus:
            def __init__(self) -> None:
                self.queue: _asyncio.Queue = _asyncio.Queue()

            def subscribe(self, mission_id):
                return self.queue

            def unsubscribe(self, mission_id, queue) -> None:
                pass

        class _Store:
            def __init__(self) -> None:
                self.bus = _Bus()

            def replay(self, mission_id, *, after_sequence=0, before_sequence=None, limit=1000):
                return [
                    e
                    for e in journal
                    if e.sequence > after_sequence
                    and (before_sequence is None or e.sequence < before_sequence)
                ][:limit]

        store = _Store()
        # 水位在 2；live queue 只投递 seq=5（3、4 模拟溢出丢失）。
        store.bus.queue.put_nowait(_ev(5))
        stream = stream_events(store, "mis_x", after_sequence=2)
        try:
            seqs = [(await anext(stream)).sequence for _ in range(3)]
            assert seqs == [3, 4, 5]  # 空洞被 journal 补齐，无丢失
        finally:
            await stream.aclose()


class TestStdioProtocol:
    """PR-RC-001 stdio 黑盒：stdout 纯净（§36）、EOF/SIGTERM（§37）、resume（§49）。"""

    @staticmethod
    def _env(tmp_path: Path) -> dict:
        import os

        return dict(os.environ, ROSCLAW_ACP_TEST_HOME=str(tmp_path))

    @staticmethod
    def _server_script() -> Path:
        return Path(__file__).parent / "acp_test_server.py"

    async def _spawn(self, tmp_path: Path):
        import asyncio
        import sys

        return await asyncio.create_subprocess_exec(
            sys.executable,
            str(self._server_script()),
            env=self._env(tmp_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def test_stdout_contains_only_jsonrpc(self, tmp_path: Path) -> None:
        """§36：stdout 每一帧都必须可解析为 JSON-RPC，不允许任何 banner/log。"""
        import asyncio
        import json

        proc = await self._spawn(tmp_path)
        try:
            init = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {"protocolVersion": 1, "clientCapabilities": {}},
            }
            proc.stdin.write((json.dumps(init) + "\n").encode())
            await proc.stdin.drain()
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=15)
            frame = json.loads(line)  # 任何非 JSON 输出都会在这里炸掉
            assert frame.get("jsonrpc") == "2.0"
            assert "result" in frame
            assert frame["result"]["agentInfo"]["name"] == "rosclaw-native-agent"
        finally:
            proc.kill()
            await proc.wait()

    async def test_stdin_eof_exits_cleanly(self, tmp_path: Path) -> None:
        """§37：OpenClaw 关闭 ACP 子进程管道（stdin EOF）后进程必须退出。"""
        import asyncio

        proc = await self._spawn(tmp_path)
        proc.stdin.close()
        returncode = await asyncio.wait_for(proc.wait(), timeout=15)
        assert returncode == 0

    async def test_sigterm_exits_cleanly(self, tmp_path: Path) -> None:
        """§37：服务就绪后收到 SIGTERM → 优雅退出（returncode 0），不产生 orphan。"""
        import asyncio
        import json

        proc = await self._spawn(tmp_path)
        # 先完成 initialize 握手，确保 signal handler 已安装、服务在运行中。
        init = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": 1, "clientCapabilities": {}},
        }
        proc.stdin.write((json.dumps(init) + "\n").encode())
        await proc.stdin.drain()
        await asyncio.wait_for(proc.stdout.readline(), timeout=15)
        proc.terminate()
        returncode = await asyncio.wait_for(proc.wait(), timeout=15)
        assert returncode == 0

    async def test_kill_and_resume_same_mission(self, tmp_path: Path) -> None:
        """§49 continuity：kill ACP 进程 → 新进程 resume 同一 Mission → 继续 turn。"""
        import asyncio
        import sys

        import acp
        from acp import schema

        class Client(acp.Client):
            async def session_update(self, session_id: str, update, **kwargs) -> None:
                pass

        server_script = self._server_script()
        env = self._env(tmp_path)

        async def _start():
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(server_script),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            conn = acp.connect_to_agent(
                Client(), proc.stdin, proc.stdout, use_unstable_protocol=True
            )
            await conn.initialize(
                protocol_version=1, client_capabilities=schema.ClientCapabilities()
            )
            return proc, conn

        # 第一轮：建 session、跑一个 turn，然后直接 kill（模拟 ACP 子进程崩溃）。
        proc1, conn1 = await _start()
        session = await conn1.new_session(cwd="/tmp/work")
        sid = session.session_id
        resp1 = await conn1.prompt(
            session_id=sid,
            prompt=[schema.TextContentBlock(type="text", text="请记住随机码 RC-7F4A9")],
        )
        assert resp1.stop_reason == "end_turn"
        proc1.kill()
        await proc1.wait()

        # 第二轮：新进程 resume 同一 Mission，Mission 仍在、turn 可继续。
        proc2, conn2 = await _start()
        try:
            listing = await conn2.list_sessions()
            assert any(s.session_id == sid for s in listing.sessions)
            await conn2.resume_session(session_id=sid, cwd="/tmp/work")
            resp2 = await conn2.prompt(
                session_id=sid,
                prompt=[schema.TextContentBlock(type="text", text="继续")],
            )
            assert resp2.stop_reason == "end_turn"
            # close 后 Mission 保留、可再次 resume。
            await conn2.close_session(session_id=sid)
            await conn2.resume_session(session_id=sid, cwd="/tmp/work")
        finally:
            proc2.kill()
            await proc2.wait()


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
