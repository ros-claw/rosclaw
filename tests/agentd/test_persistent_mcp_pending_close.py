"""Real private SDK stdio blackhole initialization, no dispatched tools/DDS."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import pytest

from rosclaw.agentd.tooling.persistent_client import McpClientClosedError, PersistentMcpClient
from rosclaw.task_kernel.process_identity import ProcessIdentity


async def start_blackhole(tmp_path):
    pidfile = tmp_path / "sdk_child.pid"
    requestfile = tmp_path / "initialize_request.json"
    code = (
        "import os,time,pathlib,sys; p=pathlib.Path(" + repr(str(pidfile)) + ");"
        't=p.with_suffix(".tmp");t.write_text(str(os.getpid()));t.replace(p);'
        "q=sys.stdin.readline(); r=pathlib.Path(" + repr(str(requestfile)) + ");"
        'v=r.with_suffix(".tmp");v.write_text(q);v.replace(r);time.sleep(6)'
    )
    client = PersistentMcpClient(command=sys.executable, args=("-c", code))
    opening = asyncio.create_task(client.list_tools())
    async with asyncio.timeout(3):
        while not pidfile.exists() or not requestfile.exists():
            await asyncio.sleep(0.005)
    assert json.loads(requestfile.read_text())["method"] == "initialize"
    child = ProcessIdentity.capture(int(pidfile.read_text()))
    assert child and child.matches()
    owner = client._owners[client._loop_key()]
    assert not owner.ready.done() and not owner.initialized
    return client, opening, child, owner


@pytest.mark.parametrize("cancel_opening", [False, True])
async def test_pending_initialize_close_is_bounded_and_reaps_sdk_child(tmp_path, cancel_opening):
    client, opening, child, owner = await start_blackhole(tmp_path)
    if cancel_opening:
        opening.cancel()
        outcome = await asyncio.gather(opening, return_exceptions=True)
        assert isinstance(outcome[0], asyncio.CancelledError) and opening.cancelled()
    started = time.monotonic()
    await asyncio.wait_for(asyncio.gather(client.close(), client.close()), 1)
    assert time.monotonic() - started < 1
    assert not child.matches(), (
        "actual SDK-owned child birth must have exited, not just registry cleanup"
    )
    assert owner.task.done() and not owner.task.cancelled() and owner.task.exception() is None
    assert owner.initialization_stop_requested and not owner.initialized
    assert not client._sessions and not client._owners
    if not cancel_opening:
        outcome = await asyncio.gather(opening, return_exceptions=True)
        assert isinstance(outcome[0], McpClientClosedError)
        assert (
            not opening.cancelled()
        )  # expected initialization stop is typed, not external caller cancellation
    with pytest.raises(McpClientClosedError):
        await client.call_tool("must-not-dispatch", {})
    await client.close()


async def test_close_caller_cancel_propagates_while_owned_initialization_stop_finishes(tmp_path):
    client, opening, child, owner = await start_blackhole(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    original = owner.aclose

    async def gated_close():
        entered.set()
        await release.wait()
        await original()

    owner.aclose = gated_close
    closing = asyncio.create_task(client.close())
    await entered.wait()
    closing.cancel()
    result = await asyncio.gather(closing, return_exceptions=True)
    assert closing.cancelled() and isinstance(result[0], asyncio.CancelledError)
    await asyncio.wait_for(asyncio.shield(owner.task), 1)
    assert not child.matches()
    release.set()
    await client.close()
    result = await asyncio.gather(opening, return_exceptions=True)
    assert isinstance(result[0], McpClientClosedError)
    assert not client._owners
