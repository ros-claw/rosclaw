"""MCP adapters must use rosclawd for every physical side effect."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rosclaw.kernel import ActionEnvelope
from rosclaw.mcp.adapters.runtime_client import RuntimeClient


class _FakeDaemonClient:
    def __init__(self) -> None:
        self.actions: list[ActionEnvelope] = []
        self.stop_requests: list[tuple[str, str]] = []

    def get_runtime_status(self) -> dict[str, Any]:
        return {
            "running": True,
            "daemon_pid": 4242,
            "southbound_owner": "rosclawd",
            "hardware_actions_executed": 0,
        }

    def request_action(self, action: ActionEnvelope) -> dict[str, Any]:
        self.actions.append(action)
        return {"action_id": action.action_id, "state": "QUEUED"}

    def wait_for_action(self, action_id: str, *, timeout_sec: float) -> dict[str, Any]:
        return {
            "action_id": action_id,
            "state": "FINISHED",
            "receipt": {
                "action_id": action_id,
                "execution_mode": "REAL",
                "final_state": "BLOCKED",
                "evidence_level": "REQUESTED",
                "trust_level": "UNVERIFIED",
                "usable_for_real_execution": False,
                "errors": [{"code": "AUTHORIZATION_REQUIRED"}],
            },
        }

    def get_action_status(self, action_id: str) -> dict[str, Any]:
        return {"action_id": action_id, "state": "QUEUED", "receipt": None}

    def cancel_action(self, action_id: str) -> dict[str, Any]:
        return {"action_id": action_id, "cancelled": True, "state": "CANCELLED"}

    def emergency_stop(self, reason: str, *, source: str) -> dict[str, Any]:
        self.stop_requests.append((reason, source))
        return {
            "reason": reason,
            "request_dispatched": True,
            "driver_acknowledged": True,
            "physical_stop_observed": False,
            "stopped": False,
            "final_status": "ACKNOWLEDGED",
            "execution_mode": "REAL",
            "trust_level": "UNVERIFIED",
            "usable_for_real_execution": False,
        }


@pytest.fixture
def client() -> tuple[RuntimeClient, _FakeDaemonClient]:
    daemon = _FakeDaemonClient()
    runtime_client = RuntimeClient(
        project_root=Path("/tmp/rosclaw-daemon-mcp"),
        robot_id="rh56-test",
        runtime_profile={},
        daemon_client=daemon,
    )
    return runtime_client, daemon


async def test_emergency_stop_never_initializes_local_runtime(
    client: tuple[RuntimeClient, _FakeDaemonClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_client, daemon = client

    def forbidden_runtime() -> None:
        raise AssertionError("physical MCP path attempted to initialize local Runtime")

    monkeypatch.setattr(runtime_client, "_ensure_runtime", forbidden_runtime)

    result = await runtime_client.emergency_stop("operator halt")

    assert result["final_status"] == "ACKNOWLEDGED"
    assert daemon.stop_requests == [("operator halt", "mcp.emergency_stop")]


async def test_request_action_builds_unapproved_envelope_for_daemon(
    client: tuple[RuntimeClient, _FakeDaemonClient],
) -> None:
    runtime_client, daemon = client

    result = await runtime_client.request_action(
        capability_id="rh56.finger.move",
        arguments={"finger": "index", "delta_raw": 20},
        execution_mode="REAL",
        body_snapshot_hash="sha256:body",
        principal_id="operator-1",
        approval_id="permit-1",
        wait_timeout_sec=2.0,
    )

    assert result["receipt"]["final_state"] == "BLOCKED"
    assert len(daemon.actions) == 1
    action = daemon.actions[0]
    assert action.body_id == "rh56-test"
    assert action.authorization.approved is False
    assert action.authorization.approval_id == "permit-1"


async def test_runtime_status_and_cancel_are_daemon_calls(
    client: tuple[RuntimeClient, _FakeDaemonClient],
) -> None:
    runtime_client, _daemon = client

    status = await runtime_client.get_runtime_status()
    cancelled = await runtime_client.cancel_action("action-1")

    assert status["southbound_owner"] == "rosclawd"
    assert cancelled["cancelled"] is True
