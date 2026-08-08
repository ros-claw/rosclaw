from __future__ import annotations

import time

from rosclaw.daemon.client import DaemonClient, DaemonRequestError


def test_track_action_lease_reads_owner_checked_status_and_schedules_renewal(
    monkeypatch,
) -> None:
    client = DaemonClient.__new__(DaemonClient)
    client._action_leases = {}
    status = {
        "action_id": "action-via-operatord",
        "session_id": "action-via-operatord",
        "state": "RUNNING",
        "action_lease": {"renew_interval_ms": 250},
    }
    monkeypatch.setattr(client, "get_action_status", lambda action_id: dict(status))

    observed = client.track_action_lease("action-via-operatord")

    assert observed == status
    assert client._action_leases["action-via-operatord"][0] == "action-via-operatord"
    assert client._action_leases["action-via-operatord"][2] == 0.25


def test_status_poll_renews_adopted_lease(monkeypatch) -> None:
    client = DaemonClient.__new__(DaemonClient)
    client._action_leases = {
        "action-via-operatord": ("action-via-operatord", time.monotonic() - 1.0, 0.25)
    }
    calls = []

    def call(method, params):
        calls.append((method, params))
        if method == "action.lease.renew":
            return {"action_id": params["action_id"], "action_lease": {"active": True}}
        return {"action_id": params["action_id"], "state": "RUNNING"}

    monkeypatch.setattr(client, "call", call)

    status = client.get_action_status("action-via-operatord")

    assert status["state"] == "RUNNING"
    assert [method for method, _params in calls] == [
        "action.lease.renew",
        "action.status",
    ]
    assert client._action_leases["action-via-operatord"][1] > time.monotonic()


def test_status_poll_reads_terminal_state_when_lease_renewal_races_completion(
    monkeypatch,
) -> None:
    action_id = "action-completed-before-renewal"
    client = DaemonClient.__new__(DaemonClient)
    client._action_leases = {
        action_id: (action_id, time.monotonic() - 1.0, 0.25)
    }
    calls = []

    def call(method, params):
        calls.append((method, params))
        if method == "action.lease.renew":
            raise DaemonRequestError(
                "ACTION_NOT_ACTIVE",
                "the action completed before its lease could be renewed",
            )
        return {
            "action_id": params["action_id"],
            "state": "FINISHED",
            "final_state": "FAILED",
        }

    monkeypatch.setattr(client, "call", call)

    status = client.get_action_status(action_id)

    assert status["state"] == "FINISHED"
    assert status["final_state"] == "FAILED"
    assert action_id not in client._action_leases
    assert [method for method, _params in calls] == [
        "action.lease.renew",
        "action.status",
    ]


def test_status_poll_preserves_unexpected_lease_renewal_failures(monkeypatch) -> None:
    action_id = "action-renewal-failed"
    client = DaemonClient.__new__(DaemonClient)
    client._action_leases = {
        action_id: (action_id, time.monotonic() - 1.0, 0.25)
    }

    def call(method, _params):
        assert method == "action.lease.renew"
        raise DaemonRequestError("SESSION_EXPIRED", "session expired")

    monkeypatch.setattr(client, "call", call)

    try:
        client.get_action_status(action_id)
    except DaemonRequestError as exc:
        assert exc.code == "SESSION_EXPIRED"
    else:
        raise AssertionError("unexpected renewal failure was suppressed")

    assert action_id in client._action_leases
