"""Actual authenticated Unix daemon boundary; no ROS, model or hardware."""

import asyncio
import threading
from unittest.mock import Mock

import pytest

from rosclaw.agentd.action_channel import DaemonActionChannel

pytest_plugins = ("tests.daemon.test_server",)


@pytest.mark.parametrize("phase", ["create_session", "request_action"])
async def test_cancel_inflight_daemon_rpc_closes_its_actual_session(
    daemon_client, monkeypatch, phase
):
    client, _runtime_fixture, _permits, daemon = daemon_client
    entered, release = threading.Event(), threading.Event()
    sessions, actions = [], []
    original_create = daemon.service.create_session
    original_submit = daemon.service.request_action

    def record_create(*args, **kwargs):
        result = original_create(*args, **kwargs)
        sessions.append(result["session"]["session_id"])
        return result

    def record_submit(*args, **kwargs):
        result = original_submit(*args, **kwargs)
        actions.append(result["action_id"])
        return result

    monkeypatch.setattr(daemon.service, "create_session", record_create)
    monkeypatch.setattr(daemon.service, "request_action", record_submit)
    original = getattr(daemon.service, phase)

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(1.5), "test did not release in-flight server request"
        return original(*args, **kwargs)

    monkeypatch.setattr(daemon.service, phase, delayed)
    channel = DaemonActionChannel(
        client, actor_id="fixture-native", body_id="rh56-test", body_hash="sha256:body"
    )
    channel._verify_outcome = Mock(side_effect=AssertionError("cancel is not task success"))
    task = asyncio.create_task(
        channel.request_sim_action(
            capability_id="fixture.not_installed",
            arguments={},
            grant_id="fixture-simulation-only",
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0.02)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(sessions) == 1
    state = await asyncio.to_thread(client.get_session, sessions[0])
    assert state["session"]["state"] == "CLOSED"
    assert len(actions) == (1 if phase == "request_action" else 0)
    assert client.get_runtime_status()["hardware_actions_executed"] == 0
    channel._verify_outcome.assert_not_called()
