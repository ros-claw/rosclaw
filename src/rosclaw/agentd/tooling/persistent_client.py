"""Persistent MCP stdio client（PR-12）。

per-call 短会话会让有状态 SIM 身体（如 limo-sim 的位姿）在调用之间
丢失——观测与 SIM 执行必须共享同一个 server 进程。

asyncio/anyio 的流是 loop-bound：HTTP server loop 与 CLI/测试 loop 可能
不同，因此会话按 running loop 分键——同一 loop 内共享一个进程，跨 loop
调用自动获得本 loop 的独立会话（状态一致性由 SIM 身体侧的同参数快照
语义保证：limo-sim 这类快照式 server 每进程独立，但观测/执行在同一
loop 内永远同进程）。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from rosclaw.contracts.common import ValidationError


def _safe_errlog():
    # MCP stdio spawn 的 errlog 必须有 fileno（捕获环境下
    # sys.stderr 是替身对象）。
    # 十审 W0：子进程 stderr 是内部诊断——默认写 ROSCLAW_HOME 日志文件，
    # 不糊 TUI 终端（--debug/ROSCLAW_DEBUG 时才回到 stderr）。
    import sys

    if not os.environ.get("ROSCLAW_DEBUG"):
        home = os.environ.get("ROSCLAW_HOME")
        if home:
            try:
                log_dir = os.path.join(home, "logs")
                os.makedirs(log_dir, exist_ok=True)
                return open(  # noqa: SIM115 - 进程级常量生命周期
                    os.path.join(log_dir, "mcp-child.log"), "a", buffering=1
                )
            except OSError:
                pass
    for stream in (sys.stderr, sys.__stderr__):
        try:
            stream.fileno()
            return stream
        except Exception:  # noqa: BLE001
            continue
    return open(os.devnull, "w")  # noqa: SIM115 - 进程级常量生命周期


class PersistentMcpClient:
    def __init__(self, *, command: str, args: tuple[str, ...], env: dict | None = None) -> None:
        self._command = command
        self._args = args
        self._env = env
        self._locks: dict[int, asyncio.Lock] = {}
        self._sessions: dict[int, tuple[Any, Any]] = {}  # loop id -> (session, exit_stack)

    def _loop_key(self) -> int:
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return 0

    def _lock_for(self, key: int) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def _ensure(self) -> Any:
        key = self._loop_key()
        entry = self._sessions.get(key)
        if entry is not None:
            return entry[0]
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._command, args=list(self._args), env=self._env or None
        )
        stack = AsyncExitStack()
        try:
            # pytest/捕获环境的 sys.stderr 无 fileno——anyio spawn 需要
            # 真 fd，否则 UnsupportedOperation: fileno（静默 quarantine
            # 的隐蔽根因）。
            read, write = await stack.enter_async_context(
                stdio_client(params, errlog=_safe_errlog())
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._sessions[key] = (session, stack)
        return session

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        key = self._loop_key()
        async with self._lock_for(key):
            try:
                session = await self._ensure()
                result = await session.call_tool(tool_name, arguments)
            except ValidationError:
                raise
            except Exception as exc:  # noqa: BLE001 - 连接断开：重置后重试一次
                await self._reset(key)
                try:
                    session = await self._ensure()
                    result = await session.call_tool(tool_name, arguments)
                except Exception as exc2:  # noqa: BLE001
                    raise ValidationError(
                        f"mcp call {tool_name} failed after reconnect: "
                        f"{type(exc2).__name__}: {exc2}"
                    ) from exc
            if result.isError:
                text = " ".join(getattr(b, "text", "") for b in result.content).strip()
                raise ValidationError(f"mcp tool {tool_name} error: {text or 'unknown'}")
            return "".join(getattr(b, "text", "") for b in result.content)

    async def list_tools(self) -> list:
        key = self._loop_key()
        async with self._lock_for(key):
            session = await self._ensure()
            listed = await session.list_tools()
            return list(listed.tools)

    async def _reset(self, key: int) -> None:
        entry = self._sessions.pop(key, None)
        if entry is not None:
            from contextlib import suppress

            with suppress(Exception):
                await entry[1].aclose()

    async def close(self) -> None:
        for key in list(self._sessions):
            await self._reset(key)
