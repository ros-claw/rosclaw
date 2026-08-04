"""rosclaw ACP server（批次 G §9.1）：`rosclaw acp serve`。

ACP（Agent Client Protocol）客户端（Zed 等编辑器）经 stdio JSON-RPC 与
本进程通信；每个 ACP session 映射一个 ROSClaw Mission，prompt 映射
turn，流式 AgentEventV2 映射 session update。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rosclaw.adapters.acp.mapper import event_to_session_update
from rosclaw.agentd.service import AgentService

PROTOCOL_VERSION = 1


class RosclawAcpAgent:
    """acp.Agent 实现：把 ACP 语义映射到 AgentService 控制面。"""

    def __init__(self, service: AgentService) -> None:
        self._service = service
        self._client = None
        self._sessions: dict[str, str] = {}  # acp session_id -> mission_id

    # -- 生命周期 ------------------------------------------------------------

    def on_connect(self, conn) -> None:
        self._client = conn

    async def initialize(self, **kwargs):
        from acp import schema

        return schema.InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=schema.AgentCapabilities(
                load_session=False,
                prompt_capabilities=schema.PromptCapabilities(),
            ),
            agent_info=schema.Implementation(
                name="rosclaw-native-agent",
                title="ROSClaw Native Agent (rosclaw-agentd)",
                version="1.0.0",
            ),
        )

    async def authenticate(self, **kwargs):
        from acp import schema

        return schema.AuthenticateResponse()

    async def new_session(self, cwd: str, **kwargs):
        from acp import schema

        mission = self._service.create_mission(f"ACP session ({Path(cwd).name})")
        session_id = mission.mission_id
        self._sessions[session_id] = mission.mission_id
        return schema.NewSessionResponse(session_id=session_id)

    async def list_sessions(self, **kwargs):
        from acp import schema

        missions = self._service.list_missions()
        return schema.ListSessionsResponse(
            sessions=[
                schema.SessionInfo(
                    session_id=m.mission_id,
                    cwd="/",
                    title=m.goal.text[:80],
                    updated_at=m.updated_at,
                )
                for m in missions
            ]
        )

    async def load_session(self, session_id: str, cwd: str, **kwargs):
        from acp import schema

        if self._service.get_mission(session_id) is None:
            from acp.exceptions import RequestError

            raise RequestError(-32002, f"unknown session {session_id!r}")
        self._sessions[session_id] = session_id
        return schema.LoadSessionResponse()

    async def prompt(self, session_id: str, prompt: list, **kwargs):
        from acp import schema

        mission_id = self._sessions.get(session_id, session_id)
        if self._service.get_mission(mission_id) is None:
            from acp.exceptions import RequestError

            raise RequestError(-32002, f"unknown session {session_id!r}")
        text = "\n".join(
            block.text for block in prompt if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            return schema.PromptResponse(stop_reason="end_turn")
        baseline = self._service.events_replay(mission_id)
        last_seq = baseline[-1].sequence if baseline else 0
        stream_task = asyncio.create_task(self._stream_updates(mission_id, last_seq))
        try:
            result = await self._service.send_turn(mission_id, text)
        finally:
            # turn 结束后的 settled/ended 尾巴事件先推送一轮再停（否则会丢）。
            await asyncio.sleep(0.3)
            await self._flush_once(mission_id)
            stream_task.cancel()
        stop = "end_turn"
        if result.state.value == "FAILED":
            stop = "refusal"
        elif result.state.value in ("WAIT_APPROVAL", "WAIT_INPUT", "SUSPENDED"):
            stop = "max_turn_requests"  # 需要外部输入/批准；客户端可继续 prompt
        return schema.PromptResponse(stop_reason=stop)

    async def cancel(self, session_id: str, **kwargs) -> None:
        mission_id = self._sessions.get(session_id, session_id)
        await self._service.cancel(mission_id)

    async def _flush_once(self, mission_id: str) -> None:
        """turn 结束后把剩余未推送事件补推一次。"""
        if self._client is None:
            return
        sent = getattr(self, "_last_sent_seq", 0)
        for event in self._service.events_replay(mission_id, after_sequence=sent):
            update = event_to_session_update(event)
            self._last_sent_seq = event.sequence
            if update is None:
                continue
            try:
                await self._client.session_update(mission_id, update)
            except Exception:  # noqa: BLE001
                return

    # -- 事件流 → session updates ------------------------------------------------

    async def _stream_updates(self, mission_id: str, after_seq: int) -> None:
        """turn 期间把新事件映射为 ACP session update（best effort）。"""
        sent = after_seq
        self._last_sent_seq = after_seq
        for _ in range(1200):  # 最长 ~10 分钟
            await asyncio.sleep(0.25)
            events = [
                e
                for e in self._service.events_replay(mission_id, after_sequence=sent)
                if e.sequence > sent
            ]
            if not events:
                if self._client is None:
                    return
                continue
            for event in events:
                sent = max(sent, event.sequence)
                self._last_sent_seq = sent
                update = event_to_session_update(event)
                if update is None or self._client is None:
                    continue
                try:
                    await self._client.session_update(mission_id, update)
                except Exception:  # noqa: BLE001 - 表示层失败不影响权威状态
                    return


async def serve_stdio(service: AgentService) -> None:
    """在 stdio 上运行 ACP agent（`rosclaw acp serve` 入口）。"""
    import acp
    from acp.agent.connection import AgentSideConnection

    agent = RosclawAcpAgent(service)
    reader, writer = await acp.stdio_streams()
    AgentSideConnection(
        lambda conn: (agent.on_connect(conn), agent)[1],
        writer,
        reader,
    )
    await asyncio.Event().wait()
