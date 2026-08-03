"""MCP adapters must use rosclawd for every physical side effect."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rosclaw.kernel import ActionEnvelope
from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.mcp.schemas.common import MCPError


class _FakeDaemonClient:
    def __init__(self) -> None:
        self.actions: list[ActionEnvelope] = []
        self.stop_requests: list[tuple[str, str]] = []
        self.sessions: list[dict[str, Any]] = []
        self.closed_sessions: list[tuple[str, str]] = []
        self.supervision_state = "DISARMED"
        self.arm_reasons: list[str] = []
        self.disarm_reasons: list[str] = []
        self.permit_error: Exception | None = None

    def get_runtime_status(self) -> dict[str, Any]:
        return {
            "running": True,
            "daemon_pid": 4242,
            "southbound_owner": "rosclawd",
            "hardware_actions_executed": 0,
            "supervision_state": self.supervision_state,
        }

    def arm_runtime(self, reason: str) -> dict[str, Any]:
        self.arm_reasons.append(reason)
        self.supervision_state = "ARMED"
        return {"supervision_state": self.supervision_state, "reason": reason}

    def disarm_runtime(self, reason: str) -> dict[str, Any]:
        self.disarm_reasons.append(reason)
        self.supervision_state = "DISARMED"
        return {"supervision_state": self.supervision_state, "reason": reason}

    def request_action(self, action: ActionEnvelope) -> dict[str, Any]:
        if isinstance(action, dict):
            action = ActionEnvelope.from_dict(action)
        self.actions.append(action)
        return {"action_id": action.action_id, "state": "QUEUED"}

    def issue_execution_permit(
        self,
        action: ActionEnvelope,
        *,
        principal_id: str,
        target_peer_uid: int,
        expires_in_sec: float,
        reason: str,
    ) -> dict[str, Any]:
        if self.permit_error is not None:
            raise self.permit_error
        payload = action.to_dict()
        payload["authorization"] = {
            "principal_id": principal_id,
            "approved": True,
            "approval_id": "permit-secret",
            "scopes": [action.capability_id],
        }
        return {"authorized_action": payload, "permit": {"permit_id": "permit-secret"}}

    def create_session(self, **kwargs: Any) -> dict[str, Any]:
        self.sessions.append(kwargs)
        return {"session_id": kwargs["session_id"], "state": "ACTIVE"}

    def close_session(self, session_id: str, *, reason: str) -> dict[str, Any]:
        self.closed_sessions.append((session_id, reason))
        return {"session_id": session_id, "state": "CLOSED"}

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
    assert daemon.sessions[-1]["session_id"] == action.session_id
    assert action.session_id == action.action_id
    assert daemon.closed_sessions == [(action.session_id, "action_finished")]


async def test_request_action_preserves_operator_proposal_deadline(
    client: tuple[RuntimeClient, _FakeDaemonClient],
) -> None:
    runtime_client, daemon = client

    await runtime_client.request_action(
        capability_id="limo.set_initial_pose",
        arguments={"schema_version": "limo.initial-pose.v1"},
        execution_mode="REAL",
        body_snapshot_hash="sha256:body",
        deadline_at="2030-01-02T03:04:05Z",
        wait_timeout_sec=0.0,
    )

    assert daemon.actions[0].to_dict()["deadline_at"] == "2030-01-02T03:04:05Z"


async def test_default_shadow_sessions_are_unique_and_scoped_per_action(
    client: tuple[RuntimeClient, _FakeDaemonClient],
) -> None:
    runtime_client, daemon = client

    for capability_id in ("limo.play_tone", "limo.set_initial_pose"):
        await runtime_client.request_action(
            capability_id=capability_id,
            arguments={"schema_version": "test.v1"},
            execution_mode="SHADOW",
            body_snapshot_hash="sha256:body",
            wait_timeout_sec=0.0,
        )

    assert len({action.session_id for action in daemon.actions}) == 2
    assert all(action.session_id == action.action_id for action in daemon.actions)
    assert [session["capability_scope"] for session in daemon.sessions] == [
        ["limo.play_tone"],
        ["limo.set_initial_pose"],
    ]


async def test_interactive_confirmation_injects_permit_without_exposing_it(
    client: tuple[RuntimeClient, _FakeDaemonClient],
) -> None:
    runtime_client, daemon = client
    prepared = runtime_client.prepare_operator_action(
        capability_id="limo.set_initial_pose",
        arguments={"target_pose": {"frame_id": "map", "x": 0.75, "y": -1.25}},
        body_snapshot_hash="sha256:body",
        action_id="action-interactive",
        deadline_at="2030-01-02T03:04:05Z",
        display={"summary": "Set LIMO initial pose"},
    )

    result = await runtime_client.confirm_operator_action(
        prepared,
        principal_id="operator-1",
        confirmation={
            "accepted": True,
            "action_intent_hash": prepared.approval_request["action_intent_hash"],
        },
        wait_timeout_sec=0.0,
    )

    assert result["operator_confirmation"]["permit_injected"] is True
    assert result["operator_confirmation"]["permit_exposed"] is False
    assert result["operator_confirmation"]["supervision_armed"] is True
    assert result["operator_confirmation"]["separate_arm_required"] is False
    assert result["operator_confirmation"]["session_closed"] is False
    assert "permit" not in result
    assert "permit-secret" not in str(result)
    assert daemon.actions[-1].authorization.approved is True
    assert daemon.sessions[-1]["session_id"] == "action-interactive"
    assert daemon.arm_reasons == [
        "Operator confirmed exact REAL action action-interactive through MCP elicitation"
    ]
    assert daemon.closed_sessions == []


async def test_interactive_confirmation_closes_terminal_action_session(
    client: tuple[RuntimeClient, _FakeDaemonClient],
) -> None:
    runtime_client, daemon = client
    prepared = runtime_client.prepare_operator_action(
        capability_id="limo.play_tone",
        arguments={"frequency_hz": 660, "duration_sec": 0.8},
        body_snapshot_hash="sha256:body",
        action_id="action-terminal",
        deadline_at="2030-01-02T03:04:05Z",
    )

    result = await runtime_client.confirm_operator_action(
        prepared,
        principal_id="operator-1",
        confirmation={
            "accepted": True,
            "action_intent_hash": prepared.approval_request["action_intent_hash"],
        },
        wait_timeout_sec=2.0,
    )

    assert result["state"] == "FINISHED"
    assert result["operator_confirmation"]["session_closed"] is True
    assert result["operator_confirmation"]["session_cleanup_error"] is None
    assert daemon.closed_sessions == [("action-terminal", "confirmed_action_finished")]


async def test_interactive_confirmation_does_not_rearm_armed_generation(
    client: tuple[RuntimeClient, _FakeDaemonClient],
) -> None:
    runtime_client, daemon = client
    daemon.supervision_state = "ARMED"
    prepared = runtime_client.prepare_operator_action(
        capability_id="limo.play_tone",
        arguments={"frequency_hz": 660, "duration_sec": 0.6},
        body_snapshot_hash="sha256:body",
        action_id="action-already-armed",
        deadline_at="2030-01-02T03:04:05Z",
    )

    result = await runtime_client.confirm_operator_action(
        prepared,
        principal_id="operator-1",
        confirmation={
            "accepted": True,
            "action_intent_hash": prepared.approval_request["action_intent_hash"],
        },
        wait_timeout_sec=0.0,
    )

    assert result["operator_confirmation"]["supervision_armed"] is False
    assert result["operator_confirmation"]["separate_arm_required"] is False
    assert daemon.arm_reasons == []


async def test_interactive_confirmation_rolls_back_just_in_time_arm_on_failure(
    client: tuple[RuntimeClient, _FakeDaemonClient],
) -> None:
    runtime_client, daemon = client
    daemon.permit_error = RuntimeError("permit broker unavailable")
    prepared = runtime_client.prepare_operator_action(
        capability_id="limo.play_tone",
        arguments={"frequency_hz": 660, "duration_sec": 0.6},
        body_snapshot_hash="sha256:body",
        action_id="action-arm-rollback",
        deadline_at="2030-01-02T03:04:05Z",
    )

    with pytest.raises(MCPError, match="permit broker unavailable"):
        await runtime_client.confirm_operator_action(
            prepared,
            principal_id="operator-1",
            confirmation={
                "accepted": True,
                "action_intent_hash": prepared.approval_request["action_intent_hash"],
            },
            wait_timeout_sec=0.0,
        )

    assert daemon.supervision_state == "DISARMED"
    assert len(daemon.arm_reasons) == 1
    assert daemon.disarm_reasons == [
        "Automatic rollback after confirmed action action-arm-rollback failed"
    ]
    assert daemon.actions == []


async def test_interactive_confirmation_rejects_mismatched_intent(
    client: tuple[RuntimeClient, _FakeDaemonClient],
) -> None:
    runtime_client, daemon = client
    prepared = runtime_client.prepare_operator_action(
        capability_id="limo.set_initial_pose",
        arguments={"target_pose": {"frame_id": "map", "x": 0.75, "y": -1.25}},
        body_snapshot_hash="sha256:body",
        action_id="action-interactive-mismatch",
        deadline_at="2030-01-02T03:04:05Z",
    )

    with pytest.raises(MCPError, match="did not match the exact action"):
        await runtime_client.confirm_operator_action(
            prepared,
            principal_id="operator-1",
            confirmation={"accepted": True, "action_intent_hash": "sha256:wrong"},
        )

    assert daemon.actions == []
    assert daemon.arm_reasons == []


async def test_runtime_status_and_cancel_are_daemon_calls(
    client: tuple[RuntimeClient, _FakeDaemonClient],
) -> None:
    runtime_client, _daemon = client

    status = await runtime_client.get_runtime_status()
    cancelled = await runtime_client.cancel_action("action-1")

    assert status["southbound_owner"] == "rosclawd"
    assert cancelled["cancelled"] is True
