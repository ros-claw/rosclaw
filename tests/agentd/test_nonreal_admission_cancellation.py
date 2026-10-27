"""Caller cancellation must not race ahead of an owned daemon RPC."""

import asyncio
import threading

import pytest

from tests.agentd.test_nonreal_action_session_lifecycle import Client, channel, kwargs


class DelayedClient(Client):
    def __init__(self, phase):
        super().__init__()
        self.phase = phase
        self.entered = threading.Event()
        self.release = threading.Event()
        self.completed = threading.Event()
        self.close_before_submit = False

    def create_session(self, **kw):
        if self.phase == "create":
            self.entered.set()
            assert self.release.wait(3), "test did not release daemon creation"
        super().create_session(**kw)
        if self.phase == "create":
            self.completed.set()

    def request_action(self, envelope):
        if self.phase == "submit":
            self.entered.set()
            assert self.release.wait(3), "test did not release daemon admission"
        # Model the authoritative remote effect, including its position relative
        # to close_session. No robot or daemon is started by this fixture.
        self.actions.append(envelope)
        self.completed.set()
        return {"action_id": envelope.action_id}

    def close_session(self, session_id, **kw):
        self.close_before_submit |= self.phase == "submit" and not self.completed.is_set()
        return super().close_session(session_id, **kw)


@pytest.mark.parametrize("phase", ["create", "submit"])
async def test_cancel_waits_for_inflight_admission_before_closing_exact_session(phase):
    client = DelayedClient(phase)
    service = channel(client)
    task = asyncio.create_task(service.request_sim_action(**kwargs()))
    try:
        assert await asyncio.to_thread(client.entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.03)
    finally:
        client.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(client.completed.wait, 2)
    assert len(client.sessions) == 1
    assert client.closed == list(client.sessions)
    assert not client.close_before_submit
    assert len(client.actions) == (1 if phase == "submit" else 0)
    service._verify_outcome.assert_not_called()


async def test_invalid_envelope_does_not_leave_session_unclosed():
    client = Client()
    with pytest.raises(ValueError):
        await channel(client).request_sim_action(**kwargs(), timeout_sec=-1)
    assert not client.actions
    assert not client.sessions or client.closed == list(client.sessions)


async def test_cancel_during_close_still_finishes_exact_cleanup():
    client = Client()
    original = client.close_session
    entered, release = threading.Event(), threading.Event()

    def delayed_close(*args, **kw):
        entered.set()
        assert release.wait(3)
        return original(*args, **kw)

    client.close_session = delayed_close
    task = asyncio.create_task(channel(client).request_sim_action(**kwargs()))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(client.actions) == 1
    assert client.closed == list(client.sessions)


async def test_lost_creation_response_closes_same_identity_without_dispatch():
    from rosclaw.agentd.action_channel import ActionChannelError
    from rosclaw.daemon.client import DaemonClientError

    client = Client()
    original = client.create_session

    def create_then_lose(**kw):
        original(**kw)
        raise DaemonClientError("RESPONSE_LOST", "session created, response unavailable")

    client.create_session = create_then_lose
    with pytest.raises(ActionChannelError, match="RESPONSE_LOST"):
        await channel(client).request_sim_action(**kwargs())
    assert not client.actions
    assert client.closed == list(client.sessions)


async def test_cancel_and_rpc_error_keep_cancel_intent_and_cleanup():
    from rosclaw.daemon.client import DaemonClientError

    client = DelayedClient("create")
    original = client.create_session

    def create_then_lose(**kw):
        original(**kw)
        raise DaemonClientError("RESPONSE_LOST", "ambiguous creation response")

    client.create_session = create_then_lose
    task = asyncio.create_task(channel(client).request_sim_action(**kwargs()))
    try:
        assert await asyncio.to_thread(client.entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.02)
    finally:
        client.release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await task
    assert "DaemonClientError" in cancelled.value.__notes__[0]
    assert not client.actions
    assert client.closed == list(client.sessions)
