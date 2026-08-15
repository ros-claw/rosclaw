"""Codex app-server 驱动（十五审 PR-RF-5，ADR-0011）。

Codex 的优化原生路径（runtime=harness:codex-app-server）：
`codex app-server`（stdio JSONL JSON-RPC）——initialize → initialized
通知 → thread/start（或 thread/resume 恢复）→ turn/start → 流式
item/* 通知 → turn/completed。approvalPolicy=never + sandbox=
workspaceWrite + cwd 隔离（RF-4 sandbox env）。

不解析 ANSI/PTY；事件只来自协议通知。codex 二进制缺失 → 诚实
not_installed（preflight 失败不创建执行）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
from pathlib import Path

from rosclaw.agentd.acp_driver import prepare_sandbox_home, sandbox_env


def codex_binary() -> str | None:
    return shutil.which("codex")


class CodexAppServerDriver:
    """一个 Codex thread 的驱动（单 execution 单 thread——Gate 3）。"""

    def __init__(
        self, *, cwd: str, event_sink, sandbox_home: Path | None = None
    ) -> None:
        self._cwd = cwd
        self._sink = event_sink
        self._sandbox = Path(sandbox_home) if sandbox_home else None
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._turn_done: asyncio.Future | None = None

    async def _send(self, msg: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(msg) + "\n").encode())
        await self._proc.stdin.drain()

    async def _request(
        self, method: str, params: dict | None = None, *,
        timeout: float | None = 120,
    ) -> dict:
        request_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future
        await self._send({
            "method": method, "id": request_id, "params": params or {},
        })
        if timeout is None:
            return await future
        return await asyncio.wait_for(future, timeout=timeout)

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
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
                                    f"app-server error {err.get('code')}: "
                                    f"{err.get('message')}"
                                )
                            )
                        else:
                            future.set_result(msg["result"])
                elif "method" in msg:
                    with contextlib.suppress(Exception):
                        await self._on_notification(msg["method"], msg.get("params") or {})
        finally:
            for _id, future in self._pending.items():
                if not future.done():
                    future.set_exception(RuntimeError("app-server reader stopped"))
            self._pending.clear()

    async def _on_notification(self, method: str, params: dict) -> None:
        if method == "turn/completed":
            # turn 终态通知——run() 等待的唯一完成信号（无墙钟超时）。
            if self._turn_done is not None and not self._turn_done.done():
                self._turn_done.set_result(params.get("turn") or {})
            return
        if method == "item/agentMessage/delta":
            await self._sink("message_delta", {
                "text": str(params.get("delta", ""))[-2000:],
            })
        elif method == "item/started":
            item = params.get("item") or {}
            if str(item.get("type", "")) not in ("agentMessage",):
                await self._sink("tool_started", {
                    "tool": str(item.get("type", "?")),
                    "tool_call_id": str(item.get("id", "")),
                })
        elif method == "item/completed":
            item = params.get("item") or {}
            if str(item.get("type", "")) not in ("agentMessage",):
                await self._sink("tool_finished", {
                    "tool_call_id": str(item.get("id", "")),
                    "status": str(item.get("status", "")),
                })

    async def run(self, goal: str) -> dict:
        """跑到 turn/completed——返回 {ok, stop_reason, detail}。"""
        binary = codex_binary()
        if binary is None:
            return {
                "ok": False,
                "stop_reason": "not_installed",
                "detail": "codex CLI 未安装（preflight 失败，未创建执行）",
            }
        if self._sandbox is not None:
            prepare_sandbox_home(self._sandbox)
            cwd = str(self._sandbox / "workspace")
            env = sandbox_env(self._sandbox)
            # Codex 自己的 home 限定在 sandbox（登录态由调用方经
            # harness_config_dirs=[".codex"] 注入）。
            env["CODEX_HOME"] = str(self._sandbox / "home" / ".codex")
        else:
            cwd = self._cwd
            env = None
        self._proc = await asyncio.create_subprocess_exec(
            binary, "app-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        try:
            await self._request("initialize", {
                "clientInfo": {"name": "rosclaw", "version": "1.0"},
            })
            await self._send({"method": "initialized"})
            thread = await self._request("thread/start", {
                "cwd": cwd,
                "approvalPolicy": "never",
                "sandbox": "workspaceWrite",
            })
            thread_obj = thread.get("thread") or {}
            thread_id = str(thread_obj.get("id", ""))
            loop = asyncio.get_running_loop()
            self._turn_done = loop.create_future()
            turn = await self._request("turn/start", {
                "threadId": thread_id,
                "input": [{"type": "text", "text": goal}],
            })
            turn_obj = turn.get("turn") or {}
            # turn/start 立即返回——真正的终态在 turn/completed 通知
            # （长任务不按墙钟杀）。
            completed = await self._turn_done
            status = str(completed.get("status", "unknown"))
            return {
                "ok": status in ("completed", "idle"),
                "stop_reason": status,
                "detail": f"thread {thread_id} turn {turn_obj.get('id', '?')} {status}",
            }
        except (RuntimeError, TimeoutError) as exc:
            return {"ok": False, "stop_reason": "error", "detail": str(exc)[:300]}
        finally:
            if self._proc is not None and self._proc.returncode is None:
                self._proc.terminate()
            if self._reader_task is not None:
                self._reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._reader_task
