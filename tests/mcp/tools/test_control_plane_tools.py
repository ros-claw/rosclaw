"""MCP wrapper coverage for the rosclawd control-plane tools."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from rosclaw.kernel import ActionEnvelope, ExecutionMode
from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.mcp.tools import (
    cancel_action,
    get_action_status,
    get_runtime_status,
    request_action,
    set_client,
)


class _FakeDaemon:
    def __init__(self) -> None:
        self.action: ActionEnvelope | None = None

    def get_runtime_status(self) -> dict[str, Any]:
        return {
            "running": True,
            "daemon_pid": 4242,
            "daemon_uid": 991,
            "southbound_owner": "rosclawd",
        }

    def request_action(self, action: ActionEnvelope) -> dict[str, Any]:
        self.action = action
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
        return {"action_id": action_id, "state": "FINISHED", "receipt": None}

    def cancel_action(self, action_id: str) -> dict[str, Any]:
        return {
            "action_id": action_id,
            "state": "FINISHED",
            "cancelled": False,
        }


@pytest.fixture(autouse=True)
def _daemon_client() -> _FakeDaemon:
    daemon = _FakeDaemon()
    set_client(
        RuntimeClient(
            project_root=Path("/tmp/rosclaw-control-plane-tools"),
            robot_id="rh56-test",
            runtime_profile={},
            daemon_client=daemon,
        )
    )
    return daemon


def _payload(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    assert payload["ok"] is True
    return payload["data"]


async def test_runtime_status_is_reported_by_daemon() -> None:
    data = _payload(await get_runtime_status())

    assert data["running"] is True
    assert data["southbound_owner"] == "rosclawd"
    assert data["trust_level"] == "DAEMON_REPORTED"


async def test_real_action_requires_contextual_guarded_tool(
    _daemon_client: _FakeDaemon,
) -> None:
    schema = inspect.signature(request_action)
    assert "principal_id" not in schema.parameters
    assert "approval_id" not in schema.parameters

    payload = json.loads(
        await request_action(
            capability_id="rh56.finger.move",
            arguments={"finger": "index", "delta_raw": 20},
            execution_mode="REAL",
            body_snapshot_hash="sha256:body",
            action_id="action-tool-test",
        )
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INTERACTION_REQUIRED"
    assert _daemon_client.action is None


async def test_action_defaults_to_shadow_when_mode_is_omitted(
    _daemon_client: _FakeDaemon,
) -> None:
    _payload(
        await request_action(
            capability_id="rh56.finger.move",
            arguments={"finger": "index", "delta_raw": 0},
            body_snapshot_hash="sha256:body",
            action_id="action-tool-default-mode",
        )
    )

    assert _daemon_client.action is not None
    assert _daemon_client.action.execution_mode is ExecutionMode.SHADOW


async def test_action_status_and_cancel_are_bounded_daemon_calls() -> None:
    status = _payload(await get_action_status(action_id="action-tool-test"))
    cancelled = _payload(await cancel_action(action_id="action-tool-test"))

    assert status["state"] == "FINISHED"
    assert cancelled["cancelled"] is False
