"""Persistent MCP stdio client（PR-12）。

per-call 短会话会让有状态 SIM 身体（如 limo-sim 的位姿）在调用之间
丢失——观测与 SIM 执行必须共享同一个 server 进程。此客户端懒启动、
持锁复用、失败时诚实报错并可重连。
"""

from __future__ import annotations

import asyncio
from typing import Any

from rosclaw.contracts.common import ValidationError


class PersistentMcpClient:
    def __init__(self, *, command: str, args: tuple[str, ...], env: dict | None = None) -> None:
        self._command = command
        self._args = args
        self._env = env
        self._lock = asyncio.Lock()
        self._session = None
        self._exit_stack = None

    async def _ensure(self) -> None:
        if self._session is not None:
            return
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._command, args=list(self._args), env=self._env or None
        )
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._exit_stack = stack
        self._session = session

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        async with self._lock:
            try:
                await self._ensure()
                result = await self._session.call_tool(tool_name, arguments)
            except (ValidationError, ExceptionGroup):
                raise
            except Exception as exc:  # noqa: BLE001 - 连接断开：重置后重试一次
                await self._reset()
                try:
                    await self._ensure()
                    result = await self._session.call_tool(tool_name, arguments)
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
        async with self._lock:
            await self._ensure()
            listed = await self._session.list_tools()
            return list(listed.tools)

    async def _reset(self) -> None:
        stack, self._exit_stack, self._session = self._exit_stack, None, None
        if stack is not None:
            from contextlib import suppress

            with suppress(Exception):
                await stack.aclose()

    async def close(self) -> None:
        await self._reset()
