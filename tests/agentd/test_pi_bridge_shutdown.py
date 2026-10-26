"""Real UDS shutdown must cancel an owned in-flight request before waiting for EOF."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

from rosclaw.agentd.operator_socket import OperatorSocketServer
from rosclaw.agentd.pi_bridge.server import PiBridgeServer


@pytest.mark.parametrize("server_type", [PiBridgeServer, OperatorSocketServer])
@pytest.mark.parametrize("disconnect_peer", [False, True])
async def test_stop_cancels_pending_dispatch_instead_of_waiting_for_natural_result(
    tmp_path, disconnect_peer, server_type
):
    connection = sqlite3.connect(":memory:")
    server = server_type(
        SimpleNamespace(_store=SimpleNamespace(connection=connection)), tmp_path / "bridge.sock"
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def dispatch(*args):
        entered.set()
        try:
            await release.wait()
            return {"ok": True, "natural_completion": True}
        except asyncio.CancelledError:
            cancelled.set()
            raise

    server._dispatch = dispatch
    await server.start()
    _, writer = await asyncio.open_unix_connection(str(server._path))
    writer.write(json.dumps({"method": "pi.action.execute", "params": {}}).encode() + b"\n")
    await writer.drain()
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if disconnect_peer:
            writer.close()
            await writer.wait_closed()
        closing = asyncio.create_task(server.stop())
        await asyncio.sleep(0.05)
        assert cancelled.is_set(), "Service stop is waiting for a live action to finish naturally"
        await asyncio.wait_for(asyncio.shield(closing), 1)
        assert not server._path.exists()
    finally:
        release.set()
        if closing is not None:
            await asyncio.wait_for(asyncio.shield(closing), 2)
        writer.close()
        await asyncio.sleep(0.01)
        await server.stop()
        connection.close()


@pytest.mark.parametrize("server_type", [PiBridgeServer, OperatorSocketServer])
async def test_stop_closes_idle_accepted_connection_and_is_idempotent(tmp_path, server_type):
    connection = sqlite3.connect(":memory:")
    server = server_type(
        SimpleNamespace(_store=SimpleNamespace(connection=connection)), tmp_path / "bridge.sock"
    )
    await server.start()
    reader, writer = await asyncio.open_unix_connection(str(server._path))
    await asyncio.sleep(0.02)
    try:
        closing = asyncio.create_task(server.stop())
        await asyncio.sleep(0.05)
        assert closing.done(), "Idle connection prevents owned server shutdown"
        await closing
        assert await asyncio.wait_for(reader.read(), 1) == b""
        await asyncio.gather(server.stop(), server.stop())
    finally:
        writer.close()
        await writer.wait_closed()
        await server.stop()
        connection.close()


async def test_actual_service_shutdown_reaps_mcp_child_with_uncancellable_wait(tmp_path):
    import sys
    import time

    from rosclaw.agentd.tooling.mcp_adapter import McpCapabilityAdapter, McpServerConfig
    from tests.agentd.test_pi_tool_bridge import _setup

    script = tmp_path / "blocking_mcp.py"
    marker = tmp_path / "started.pid"
    script.write_text("""import asyncio,os,sys,time
from pathlib import Path
from mcp.server.fastmcp import FastMCP
server=FastMCP("shutdown-test",log_level="ERROR")
@server.tool()
async def hold()->dict:
    Path(sys.argv[1]).write_text(str(os.getpid()))
    await asyncio.to_thread(time.sleep,30)
    return {"natural_completion":True}
server.run(transport="stdio")
""")
    service, _ = await _setup(tmp_path)
    adapter = McpCapabilityAdapter(
        McpServerConfig(
            name="shutdown-test", command=sys.executable, args=(str(script), str(marker))
        ),
        None,
    )
    execute = adapter._make_executor("hold")
    await service.start_pi_bridge()

    async def dispatch(*args):
        return await execute({})

    service._pi_bridge._dispatch = dispatch
    _, writer = await asyncio.open_unix_connection(str(service._pi_bridge._path))
    writer.write(b'{"method":"pi.tools.execute","params":{}}\n')
    await writer.drain()
    child = None
    try:
        async with asyncio.timeout(8):
            while not marker.exists():
                await asyncio.sleep(0.01)
        child = int(marker.read_text())
        started = time.monotonic()
        await asyncio.wait_for(service.close(), 8)
        assert time.monotonic() - started < 8
        assert not __import__("pathlib").Path("/proc", str(child)).exists()
    finally:
        writer.close()
        if child is not None and __import__("pathlib").Path("/proc", str(child)).exists():
            import os
            import signal

            os.kill(child, signal.SIGKILL)
        await service.close()
