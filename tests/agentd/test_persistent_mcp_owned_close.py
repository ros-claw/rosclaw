"""Real connected MCP contexts must enter/exit in one dedicated task.

Discovery only: private SDK stdio connection, no model/ROS action/physical tool.
"""

from __future__ import annotations

import asyncio

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.service import AgentService


async def test_real_discovery_task_can_finish_before_concurrent_service_close(tmp_path):
    service = AgentService(load_agent_config(tmp_path / "config.yaml"), tmp_path)
    await asyncio.create_task(service._ensure_mcp_discovered())
    client = service._shared_mcp_client
    assert client._owners, "fixture must establish a real MCP SDK stdio session"
    owners = list(client._owners.values())
    assert all(
        owner.task is not asyncio.current_task() and not owner.task.done() for owner in owners
    )
    await asyncio.wait_for(asyncio.gather(service.close(), service.close()), 5)
    assert all(
        owner.task.done() and not owner.task.cancelled() and owner.task.exception() is None
        for owner in owners
    )
    assert not client._sessions and not client._owners
    await service.close()


async def test_cancelled_service_close_caller_does_not_cancel_real_mcp_cleanup(tmp_path):
    service = AgentService(load_agent_config(tmp_path / "config.yaml"), tmp_path)
    await service._ensure_mcp_discovered()
    client = service._shared_mcp_client
    owners = list(client._owners.values())
    assert owners
    entered = asyncio.Event()
    release = asyncio.Event()
    original = client.close

    async def gated_close():
        entered.set()
        await release.wait()
        await original()

    client.close = gated_close
    first = asyncio.create_task(service.close())
    await entered.wait()
    first.cancel()
    result = await asyncio.gather(first, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError)
    assert not service._close_task.done()
    service._store.connection.execute("SELECT 1")  # database stays open through actual SDK cleanup
    release.set()
    await asyncio.wait_for(service.close(), 5)
    assert all(owner.task.done() and owner.task.exception() is None for owner in owners)
    assert service._shared_mcp_client is None


async def test_close_serializes_active_dispatch_and_rejects_reconnection(tmp_path):
    from types import SimpleNamespace

    import pytest

    from rosclaw.agentd.tooling.persistent_client import McpClientClosedError

    service = AgentService(load_agent_config(tmp_path / "config.yaml"), tmp_path)
    await service._ensure_mcp_discovered()
    client = service._shared_mcp_client
    session = await client._ensure()
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_fake_effect(*args):
        entered.set()
        await release.wait()
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="mock-result")])

    session.call_tool = slow_fake_effect
    dispatch = asyncio.create_task(client.call_tool("mock-only-no-physical-tool", {}))
    await entered.wait()
    closing = asyncio.create_task(client.close())
    await asyncio.sleep(0)
    assert not closing.done(), (
        "close must wait for active call lock rather than invalidate its session"
    )
    release.set()
    assert await dispatch == "mock-result"
    await closing
    with pytest.raises(McpClientClosedError):
        await client.list_tools()
    await service.close()


def test_foreign_loop_close_rejects_without_touching_owner_event_or_registry():
    from types import SimpleNamespace

    import pytest

    from rosclaw.agentd.tooling.persistent_client import (
        McpCloseUnresolvedError,
        PersistentMcpClient,
    )

    foreign = asyncio.new_event_loop()
    client = PersistentMcpClient(command="never-spawn", args=())

    class ForbiddenEvent:
        def set(self):
            raise AssertionError("foreign Event was touched")

    owner = SimpleNamespace(loop=foreign, stop=ForbiddenEvent())
    client._owners[id(foreign)] = owner
    try:
        with pytest.raises(McpCloseUnresolvedError):
            asyncio.run(client.close())
        assert client._owners[id(foreign)] is owner
        assert not client._closing
    finally:
        foreign.close()
