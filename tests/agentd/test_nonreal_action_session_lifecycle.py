"""Fresh approved actions must not reuse a LOST session or replay earlier work."""

import asyncio
from unittest.mock import Mock

import pytest

from rosclaw.agentd.action_channel import ActionChannelError, DaemonActionChannel
from rosclaw.daemon.client import DaemonClientError


class Client:
    def __init__(self):
        self.now = 0
        self.sessions = {}
        self.actions = []
        self.closed = []
        self.fail_wait = False
        self.fail_close = False

    def create_session(self, **kw):
        self.sessions[kw["session_id"]] = {**kw, "expires": self.now + 30}

    def request_action(self, envelope):
        if self.sessions[envelope.session_id]["expires"] <= self.now:
            raise DaemonClientError("SESSION_NOT_ACTIVE", "Expired session")
        self.actions.append(envelope)
        return {"action_id": envelope.action_id}

    def wait_for_action(self, action_id, **kw):
        if self.fail_wait:
            raise DaemonClientError("WAIT_TIMEOUT", "Lost supervision")
        return {"state": "FINISHED"}

    def get_execution_receipt(self, action_id):
        return {"action_id": action_id}

    def close_session(self, session_id, **kw):
        if self.fail_close:
            raise DaemonClientError("DAEMON_UNAVAILABLE", "Cleanup unavailable")
        self.closed.append(session_id)
        self.sessions[session_id]["expires"] = -1


def channel(client):
    result = DaemonActionChannel(
        client, actor_id="agent", body_id="sim-body", body_hash="immutable"
    )
    result._verify_outcome = Mock(return_value="verified")
    return result


def kwargs(grant="approved_1"):
    return {
        "capability_id": "navigation.navigate_to_pose",
        "arguments": {"proposal_id": "target"},
        "grant_id": grant,
    }


async def test_explicit_second_action_after_idle_has_fresh_session_and_grant():
    client = Client()
    service = channel(client)
    assert await service.request_sim_action(**kwargs()) == "verified"
    client.now = 120
    assert await service.request_sim_action(**kwargs("approved_2")) == "verified"
    assert len(client.actions) == 2
    a, b = client.actions
    assert a.session_id != b.session_id and a.action_id != b.action_id
    assert (
        a.authorization.approval_id == "approved_1" and b.authorization.approval_id == "approved_2"
    )
    assert client.closed == [a.session_id, b.session_id]
    for session in client.sessions.values():
        assert session["body_scope"] == ["sim-body"] and session["capability_scope"] == [
            "navigation.navigate_to_pose"
        ]
        assert session["ttl_ms"] == 30000


async def test_wait_failure_closes_session_and_never_resubmits_action():
    client = Client()
    client.fail_wait = True
    service = channel(client)
    with pytest.raises(ActionChannelError, match="WAIT_TIMEOUT"):
        await service.request_sim_action(**kwargs())
    assert len(client.actions) == 1 and client.closed == [client.actions[0].session_id]
    service._verify_outcome.assert_not_called()


async def test_cleanup_failure_does_not_hide_original_wait_failure():
    client = Client()
    client.fail_wait = True
    client.fail_close = True
    with pytest.raises(ActionChannelError, match="WAIT_TIMEOUT") as caught:
        await channel(client).request_sim_action(**kwargs())
    assert "cleanup failed" in caught.value.__notes__[0]
    assert len(client.actions) == 1


async def test_cleanup_failure_prevents_reporting_success():
    client = Client()
    client.fail_close = True
    with pytest.raises(ActionChannelError, match="session cleanup failed"):
        await channel(client).request_sim_action(**kwargs())
    assert len(client.actions) == 1


async def test_cancellation_closes_session_without_retry():
    client = Client()
    service = channel(client)

    def cancelled(*a, **kw):
        raise asyncio.CancelledError()

    client.wait_for_action = cancelled
    with pytest.raises(asyncio.CancelledError):
        await service.request_sim_action(**kwargs())
    assert len(client.actions) == 1 and client.closed == [client.actions[0].session_id]


async def test_real_is_rejected_before_session_or_dispatch():
    client = Client()
    with pytest.raises(ActionChannelError, match="non-real channel"):
        await channel(client).request_nonreal_action(**kwargs(), execution_mode="REAL")
    assert not client.sessions and not client.actions
