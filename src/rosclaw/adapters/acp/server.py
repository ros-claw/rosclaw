"""rosclaw ACP server（批次 G §9.1 + Channel 设计 PR-RC-001/002）：`rosclaw acp serve`。

ACP（Agent Client Protocol）客户端（OpenClaw acpx、Zed 等）经 stdio
JSON-RPC 与本进程通信；每个 ACP session 映射一个 ROSClaw Mission，
prompt 映射 turn，流式 AgentEventV2 映射 session update。

Session lifecycle 语义（Channel 设计 §20）：

```text
session/new     → 创建 Mission
session/list    → MissionStore.list_missions()
session/load    → 重新绑定已有 Mission + transcript replay
session/resume  → 重新绑定已有 Mission，不重放 transcript
session/close   → 释放 adapter 侧绑定；Mission journal 保留，绝不删除
session/cancel  → 只 cancel 当前 turn，不归档 Mission
```

边界：本进程不产生任何物理 authority；approval 只呈现只读卡片。
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from pathlib import Path

from rosclaw.adapters.acp.mapper import event_to_session_update
from rosclaw.agentd.service import AgentService

PROTOCOL_VERSION = 1

# load_session replay 的 transcript 上限（避免大 Mission 全量重放拖垮客户端）。
_LOAD_REPLAY_LIMIT = 200

_UNSUPPORTED_ATTACHMENT_NOTICE = "当前 ROSClaw Channel 版本尚未接入该附件类型（P0 仅支持文本）"


class RosclawAcpAgent:
    """acp.Agent 实现：把 ACP 语义映射到 AgentService 控制面。"""

    def __init__(self, service: AgentService) -> None:
        self._service = service
        self._client = None
        self._sessions: dict[str, str] = {}  # acp session_id -> mission_id
        self._closed_sessions: set[str] = set()
        self._stream_tasks: set[asyncio.Task] = set()
        self._last_sent_seq: dict[str, int] = {}  # mission_id -> last pushed seq
        self._turn_locks: dict[str, asyncio.Lock] = {}

    # -- 生命周期 ------------------------------------------------------------

    def on_connect(self, conn) -> None:
        self._client = conn

    async def initialize(self, **kwargs):
        from acp import schema

        return schema.InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=schema.AgentCapabilities(
                # PR-RC-001：load_session 已实现，capability 必须与实现一致。
                load_session=True,
                prompt_capabilities=schema.PromptCapabilities(
                    image=False, audio=False, embedded_context=False
                ),
                session_capabilities=schema.SessionCapabilities(
                    list=schema.SessionListCapabilities(),
                    resume=schema.SessionResumeCapabilities(),
                    close=schema.SessionCloseCapabilities(),
                ),
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
        self._closed_sessions.discard(session_id)
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

    def _bind_existing(self, session_id: str) -> str:
        """校验并绑定已有 Mission（load/resume 可重绑已 close 的会话）。"""
        from acp.exceptions import RequestError

        if self._service.get_mission(session_id) is None:
            raise RequestError(-32002, f"unknown session {session_id!r}")
        self._sessions[session_id] = session_id
        self._closed_sessions.discard(session_id)
        return session_id

    async def load_session(self, session_id: str, cwd: str, **kwargs):
        """重新绑定已有 Mission + transcript replay（§20）。"""
        from acp import schema

        mission_id = self._bind_existing(session_id)
        await self._replay_transcript(mission_id)
        return schema.LoadSessionResponse()

    async def resume_session(self, session_id: str, cwd: str, **kwargs):
        """重新连接已有 Mission，但不重放完整 transcript（§20）。"""
        from acp import schema

        self._bind_existing(session_id)
        return schema.ResumeSessionResponse()

    async def close_session(self, session_id: str, **kwargs):
        """关闭 ACP 侧绑定（§20）：取消未结束 turn、释放 adapter 资源。

        **不删除 Mission**——journal 完整保留，之后仍可 resume/load。
        """
        from acp import schema

        mission_id = self._sessions.get(session_id, session_id)
        if self._service.get_mission(mission_id) is not None:
            with contextlib.suppress(Exception):
                await self._service.cancel(mission_id)
        self._sessions.pop(session_id, None)
        self._closed_sessions.add(session_id)
        return schema.CloseSessionResponse()

    async def prompt(self, session_id: str, prompt: list, **kwargs):
        from acp import schema
        from acp.exceptions import RequestError

        mission_id = self._sessions.get(session_id, session_id)
        if session_id in self._closed_sessions:
            raise RequestError(
                -32002, f"session {session_id!r} is closed; resume or load to rebind"
            )
        if self._service.get_mission(mission_id) is None:
            raise RequestError(-32002, f"unknown session {session_id!r}")

        text = "\n".join(
            block.text for block in prompt if getattr(block, "type", None) == "text"
        ).strip()
        unsupported = [
            block for block in prompt if getattr(block, "type", None) not in (None, "text")
        ]
        if unsupported:
            # 设计 §38：不悄悄忽略附件，明确告知未接入。
            await self._push_text(mission_id, _UNSUPPORTED_ATTACHMENT_NOTICE)
        if not text:
            return schema.PromptResponse(stop_reason="end_turn")

        # 同一 session 串行执行 turn，避免并发 prompt 交错事件流。
        lock = self._turn_locks.setdefault(mission_id, asyncio.Lock())
        async with lock:
            baseline = self._service.events_replay(mission_id)
            last_seq = baseline[-1].sequence if baseline else 0
            self._last_sent_seq[mission_id] = last_seq
            stream_task = asyncio.create_task(self._stream_updates(mission_id, last_seq))
            self._stream_tasks.add(stream_task)
            stream_task.add_done_callback(self._stream_tasks.discard)
            try:
                result = await self._service.send_turn(mission_id, text)
            finally:
                # turn 结束后的 settled/ended 尾巴事件先推送一轮再停（否则会丢）。
                await asyncio.sleep(0.05)
                await self._flush_once(mission_id)
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
        stop = "end_turn"
        if result.state.value == "FAILED":
            stop = "refusal"
        elif result.state.value in ("WAIT_APPROVAL", "WAIT_INPUT", "SUSPENDED"):
            stop = "max_turn_requests"  # 需要外部输入/批准；客户端可继续 prompt
        return schema.PromptResponse(stop_reason=stop)

    async def cancel(self, session_id: str, **kwargs) -> None:
        """只 cancel 当前 turn（§20）；不归档、不删除 Mission。"""
        mission_id = self._sessions.get(session_id, session_id)
        await self._service.cancel(mission_id)

    # -- 关闭 -----------------------------------------------------------------

    async def shutdown(self) -> None:
        """进程退出前：停掉所有 stream 任务（graceful shutdown，§37）。"""
        for task in list(self._stream_tasks):
            task.cancel()
        if self._stream_tasks:
            await asyncio.gather(*self._stream_tasks, return_exceptions=True)
        self._stream_tasks.clear()

    # -- 内部 -----------------------------------------------------------------

    async def _push_text(self, mission_id: str, text: str) -> None:
        if self._client is None:
            return
        from acp import schema

        with contextlib.suppress(Exception):  # 表示层失败不影响权威状态
            await self._client.session_update(
                mission_id,
                schema.AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=schema.TextContentBlock(type="text", text=text),
                ),
            )

    async def _send_update(self, mission_id: str, update) -> bool:
        if self._client is None:
            return False
        try:
            await self._client.session_update(mission_id, update)
            return True
        except Exception:  # noqa: BLE001 - 表示层失败不影响权威状态
            return False

    async def _flush_once(self, mission_id: str) -> None:
        """把水位之后剩余未推送的事件补推一轮（journal 为准）。"""
        sent = self._last_sent_seq.get(mission_id, 0)
        for event in self._service.events_replay(mission_id, after_sequence=sent):
            if event.sequence <= sent:
                continue
            sent = event.sequence
            update = event_to_session_update(event)
            if update is None:
                continue
            await self._send_update(mission_id, update)
        self._last_sent_seq[mission_id] = sent

    async def _replay_transcript(self, mission_id: str) -> None:
        """session/load：把最近的 USER 可见事件重放为 session updates。"""
        events = self._service.events_replay(mission_id)
        tail = events[-_LOAD_REPLAY_LIMIT:] if events else []
        sent = self._last_sent_seq.get(mission_id, 0)
        for event in tail:
            update = event_to_session_update(event)
            if update is None:
                continue
            await self._send_update(mission_id, update)
            sent = max(sent, event.sequence)
        if events:
            self._last_sent_seq[mission_id] = max(sent, events[-1].sequence)

    # -- 事件流 → session updates ------------------------------------------------

    async def _stream_updates(self, mission_id: str, after_seq: int) -> None:
        """turn 期间把新事件映射为 ACP session update（订阅式，无轮询）。

        PR-RC-002：基于 AgentService.subscribe_events（journal replay +
        live bus），替代 250ms 轮询；无 event loss、无 duplicate、
        sequence 单调，断开可以 after_sequence 重连。
        """
        sent = after_seq
        stream = self._service.subscribe_events(mission_id, after_sequence=after_seq)
        try:
            async for event in stream:
                if event.sequence <= sent:
                    continue
                sent = event.sequence
                self._last_sent_seq[mission_id] = sent
                update = event_to_session_update(event)
                if update is None:
                    continue
                if not await self._send_update(mission_id, update):
                    return
        finally:
            await stream.aclose()


async def serve_stdio(service: AgentService) -> None:
    """在 stdio 上运行 ACP agent（`rosclaw acp serve` 入口）。

    Graceful shutdown（设计 §37）：stdin EOF（OpenClaw 关闭 ACP 子进程
    管道）或 SIGTERM/SIGINT 时，停 stream 任务、关连接后退出——不产生
    orphan process。

    **stdout 必须绝对纯净（§36）**：stdout 只承载 JSON-RPC 帧；任何日志
    一律走 stderr。
    """
    import acp
    from acp.agent.connection import AgentSideConnection

    agent = RosclawAcpAgent(service)
    reader, writer = await acp.stdio_streams()
    # listening=False：由我们显式驱动 receive loop，才能把 EOF 变成退出信号。
    conn = AgentSideConnection(
        lambda c: (agent.on_connect(c), agent)[1],
        writer,
        reader,
        listening=False,
        # session/resume 与 session/close 在 SDK 0.12 仍标记 unstable；
        # 协议本身已稳定（设计 §20），显式开启以消除 warn_unstable。
        use_unstable_protocol=True,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stop.set)

    listen_task = asyncio.create_task(conn.listen())  # stdin EOF 时正常返回
    stop_task = asyncio.create_task(stop.wait())
    try:
        await asyncio.wait({listen_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        listen_task.cancel()
        stop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await listen_task
        await agent.shutdown()
        with contextlib.suppress(Exception):
            await conn.close()
        # 日志一律 stderr，stdout 只属于 JSON-RPC。
        print("rosclaw acp serve: shutdown complete", file=sys.stderr)
