"""Persistent MCP stdio client（PR-12）。

per-call 短会话会让有状态 SIM 身体（如 limo-sim 的位姿）在调用之间
丢失——观测与 SIM 执行必须共享同一个 server 进程。

asyncio/anyio 的流是 loop-bound：HTTP server loop 与 CLI/测试 loop 可能
不同，因此会话按 running loop 分键——同一 loop 内共享一个进程，跨 loop
调用自动获得本 loop 的独立会话（状态一致性由 SIM 身体侧的同参数快照
语义保证：limo-sim 这类快照式 server 每进程独立，但观测/执行在同一
loop 内永远同进程）。

Only connection initialization can retry automatically, before dispatch. Once
call_tool begins, transport loss leaves the tool's effect unconfirmed: discard
the transport and require state reconciliation rather than replay. Cancellation
still propagates unchanged; it is not evidence that a remote effect was stopped.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from rosclaw.contracts.common import ValidationError


class McpToolOutcomeUnconfirmedError(ValidationError):
    """Dispatch began, but no tool result was received; reconcile before retry."""

    code = "MCP_TOOL_OUTCOME_UNCONFIRMED"

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"mcp tool {tool_name}: OUTCOME_UNCONFIRMED after dispatch; "
            "the tool may have executed. No automatic replay was attempted. "
            "Reconcile the operation/state before choosing whether to retry."
        )


class McpCloseUnresolvedError(ValidationError):
    """Another loop owns a session; no cross-loop teardown was attempted."""

    code = "MCP_CLOSE_UNRESOLVED"


class McpClientClosedError(ValidationError):
    code = "MCP_CLIENT_CLOSED"


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


class _OwnedMcpSession:
    """Enter and exit task-bound SDK contexts in one lifecycle task."""

    def __init__(self, command: str, args: tuple[str, ...], env: dict | None) -> None:
        self.loop = asyncio.get_running_loop()
        self.ready: asyncio.Future[Any] = self.loop.create_future()
        self.stop = asyncio.Event()
        self.task = asyncio.create_task(self._run(command, args, env))

    async def _run(self, command: str, args: tuple[str, ...], env: dict | None) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        try:
            async with AsyncExitStack() as stack:
                params = StdioServerParameters(command=command, args=list(args), env=env or None)
                read, write = await stack.enter_async_context(
                    stdio_client(params, errlog=_safe_errlog())
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self.ready.set_result(session)
                await self.stop.wait()
        except asyncio.CancelledError:
            if not self.ready.done():
                self.ready.cancel()
            raise
        except Exception as exc:
            if not self.ready.done():
                self.ready.set_exception(exc)
                return  # initializer receives the original failure, not an unobserved task error
            raise  # cleanup failure stays observable; cancellation is not swallowed

    async def aclose(self) -> None:
        if asyncio.get_running_loop() is not self.loop:
            raise McpCloseUnresolvedError("MCP session must close on its owning event loop")
        self.stop.set()
        await asyncio.shield(self.task)


class PersistentMcpClient:
    def __init__(self, *, command: str, args: tuple[str, ...], env: dict | None = None) -> None:
        self._command = command
        self._args = args
        self._env = env
        self._locks: dict[int, asyncio.Lock] = {}
        self._sessions: dict[int, tuple[Any, Any]] = {}  # loop id -> (session, lifecycle owner)
        self._owners: dict[int, _OwnedMcpSession] = {}
        self._closing = False

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
        if self._closing:
            raise McpClientClosedError("MCP client is closing; create a new client to reconnect")
        key = self._loop_key()
        entry = self._sessions.get(key)
        if entry is not None:
            return entry[0]
        owner = self._owners.get(key)
        if owner is None:
            owner = _OwnedMcpSession(self._command, self._args, self._env)
            self._owners[key] = owner
        session = await asyncio.shield(owner.ready)
        self._sessions[key] = (session, owner)
        return session

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        key = self._loop_key()
        async with self._lock_for(key):
            try:
                session = await self._ensure()
            except ValidationError:
                raise
            except Exception:  # noqa: BLE001 - no tool has been dispatched
                await self._reset(key)
                try:
                    session = await self._ensure()
                except ValidationError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise ValidationError(
                        f"mcp connection failed before tool dispatch after one reconnect: "
                        f"{type(exc).__name__}"
                    ) from exc
            try:
                result = await session.call_tool(tool_name, arguments)
            except ValidationError:
                raise
            except Exception as exc:  # noqa: BLE001 - dispatched outcome is unknown
                # Reset only the transport, never replay the potentially effective call.
                # Cleanup failure must not replace the original ambiguous outcome.
                from contextlib import suppress

                with suppress(Exception):
                    await self._reset(key)
                raise McpToolOutcomeUnconfirmedError(tool_name) from exc
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
        entry = self._sessions.get(key)
        owner = self._owners.get(key)
        lifecycle = owner or (entry[1] if entry is not None else None)
        if lifecycle is not None:
            await lifecycle.aclose()
            self._sessions.pop(key, None)
            self._owners.pop(key, None)

    async def close(self) -> None:
        loop = asyncio.get_running_loop()
        # Preflight before touching any Event or Task. We do not marshal live
        # SDK teardown across threads/loops or silently discard those entries.
        if (
            any(key != id(loop) for key in set(self._sessions) | set(self._owners))
            or any(owner.loop is not loop for owner in self._owners.values())
            or any(key not in self._owners for key in self._sessions)
        ):
            raise McpCloseUnresolvedError("MCP sessions on another loop require owner-loop cleanup")
        self._closing = True
        for key in set(self._sessions) | set(self._owners):
            async with self._lock_for(key):
                await self._reset(key)
