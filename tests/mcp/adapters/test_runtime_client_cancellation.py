"""Cancellation of the owned await must not leave its renewal thread supervising motion."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from rosclaw.mcp.adapters.runtime_client import RuntimeClient


@pytest.mark.parametrize("owned", [True, False])
async def test_cancel_closes_only_its_owned_session_and_preserves_unknown_outcome(owned):
    entered = threading.Event()
    release = threading.Event()
    closed = []
    dispatched = []
    action = SimpleNamespace(
        action_id="exact_action",
        session_id="exact_action" if owned else "other_owner",
        actor_id="agent",
        agent_framework="mcp",
        body_id="body",
        capability_id="navigation.navigate_to_pose",
    )

    def wait(*args, **kwargs):
        entered.set()
        release.wait(10)
        return {"state": "FINISHED"}

    def close(identity, **kwargs):
        closed.append((identity, kwargs))
        release.set()

    daemon = SimpleNamespace(
        create_session=lambda **kw: None,
        request_action=lambda value: dispatched.append(value) or {"state": "RUNNING"},
        wait_for_action=wait,
        close_session=close,
    )
    client = RuntimeClient.__new__(RuntimeClient)
    client._daemon_client = daemon
    client._build_action = lambda **kwargs: (action, SimpleNamespace(value="SIMULATION"))
    task = asyncio.create_task(
        client.request_action(
            capability_id=action.capability_id,
            arguments={},
            body_snapshot_hash="hash",
            execution_mode="SIMULATION",
        )
    )
    try:
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(dispatched) == 1
        assert bool(closed) is owned
        if owned:
            assert closed[0][0] == "exact_action"
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
