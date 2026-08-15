"""ACP Harness 驱动（十五审 PR-RF-3，ADR-0011）。

agentd 侧 ACP client：spawn 完整 Harness（claude-code-acp/pi-acp/…），
stdio ndjson JSON-RPC——initialize → session/new → session/prompt
（流式 session/update 落 WorkerEventStore）→ cancel。

红线：不解析 ANSI/PTY，不从文案猜状态；Harness 侧 permission/fs/
terminal 请求一律结构化拒绝（RF-4 sandbox 后按白名单开放）。
无 ACP 二进制 → 诚实 BLOCKED（preflight 失败不创建执行）。
"""

from __future__ import annotations

import asyncio
import json
import shutil
from typing import Any

#: ACP Harness 注册表（二进制 → runtime id）。
ACP_HARNESSES = {
    "harness:acp:claude-local": "claude-code-acp",
    "harness:acp:pi-acp": "pi-acp",
}


def acp_binary_for(runtime: str) -> str | None:
    binary = ACP_HARNESSES.get(runtime)
    if binary is None:
        return None
    return shutil.which(binary)


class AcpHarnessDriver:
    """一个 ACP session 的驱动（单 execution 单 session——Gate 3）。"""

    def __init__(self, runtime: str, *, cwd: str, event_sink) -> None:
        self._runtime = runtime
        self._cwd = cwd
        self._sink = event_sink  # async (kind, payload) -> None
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None

    async def _request(self, method: str, params: dict | None = None) -> dict:
        assert self._proc is not None and self._proc.stdin is not None
        request_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future
        line = json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "method": method,
             "params": params or {}}
        )
        self._proc.stdin.write(f"{line}\n".encode())
        await self._proc.stdin.drain()
        return await asyncio.wait_for(future, timeout=120)

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                return
            try:
                msg = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                future = self._pending.pop(msg["id"], None)
                if future is not None and not future.done():
                    if "error" in msg:
                        err = msg["error"]
                        future.set_exception(
                            RuntimeError(
                                f"ACP error {err.get('code')}: {err.get('message')}"
                            )
                        )
                    else:
                        future.set_result(msg["result"])
            elif msg.get("method") == "session/update":
                params = msg.get("params") or {}
                update = params.get("update") or {}
                await self._on_update(params.get("sessionId", ""), update)
            elif "method" in msg and "id" in msg:
                # Harness → client 请求：MVP 结构化拒绝（RF-4 后白名单）。
                await self._respond_error(msg["id"], msg["method"])

    async def _respond_error(self, request_id: Any, method: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(
            (json.dumps({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"{method} not granted by ROSClaw policy",
                },
            }) + "\n").encode()
        )
        await self._proc.stdin.drain()

    async def _on_update(self, session_id: str, update: dict) -> None:
        tag = str(update.get("sessionUpdate", ""))
        if tag == "agent_message_chunk":
            content = update.get("content") or {}
            await self._sink("message_delta", {
                "text": str(content.get("text", ""))[-200:],
            })
        elif tag == "tool_call":
            await self._sink("tool_started", {
                "tool": str(update.get("title", "?")),
                "tool_call_id": str(update.get("toolCallId", "")),
            })
        elif tag == "tool_call_update":
            await self._sink("tool_finished", {
                "tool_call_id": str(update.get("toolCallId", "")),
                "status": str(update.get("status", "")),
            })
        elif tag == "plan":
            await self._sink("plan_updated", {
                "entries": len(update.get("entries") or []),
            })

    async def run(self, goal: str) -> dict:
        """跑到终态——返回 {ok, stop_reason, detail}。"""
        binary = acp_binary_for(self._runtime)
        if binary is None:
            return {
                "ok": False,
                "stop_reason": "not_installed",
                "detail": f"ACP Harness {self._runtime} 未安装（preflight 失败）",
            }
        self._proc = await asyncio.create_subprocess_exec(
            binary,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            start_new_session=True,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        try:
            await self._request("initialize", {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {"name": "rosclaw", "version": "1.0"},
            })
            session = await self._request(
                "session/new", {"cwd": self._cwd, "mcpServers": []}
            )
            session_id = str(session["sessionId"])
            result = await self._request("session/prompt", {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": goal}],
            })
            stop = str(result.get("stopReason", "unknown"))
            return {
                "ok": stop in ("end_turn", "max_tokens", "refusal"),
                "stop_reason": stop,
                "detail": f"session {session_id} stop={stop}",
            }
        except (TimeoutError, RuntimeError) as exc:
            return {"ok": False, "stop_reason": "error", "detail": str(exc)[:300]}
        finally:
            if self._proc is not None and self._proc.returncode is None:
                self._proc.terminate()
            if self._reader_task is not None:
                self._reader_task.cancel()
                import contextlib

                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._reader_task
