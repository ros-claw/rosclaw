"""Actual authenticated daemon Session cleanup; no robot, ROS or model."""

import pytest

from rosclaw.app.runner import AppRunner
from rosclaw.app.schema import AppManifest
from rosclaw.kernel import ExecutionMode

pytest_plugins = ("tests.daemon.test_server",)


@pytest.mark.parametrize("error_type", [TimeoutError, KeyboardInterrupt])
def test_lost_app_create_ack_closes_only_its_actual_daemon_session(
    daemon_client, monkeypatch, error_type
):
    client, _runtime_fixture, _permits, _daemon = daemon_client
    other = client.create_session(
        session_id="fixture-other-app-session",
        actor_id="fixture-other-app",
        agent_framework="rosclaw-app",
        body_scope=["rh56-test"],
        capability_scope=["rh56.finger.move"],
    )
    assert other["session"]["state"] == "ACTIVE"
    create = client.create_session
    created = []

    def lost_ack(**kwargs):
        result = create(**kwargs)
        created.append(result["session"]["session_id"])
        raise error_type("fixture lost acknowledgement after actual daemon creation")

    monkeypatch.setattr(client, "create_session", lost_ack)
    app = AppManifest.model_validate(
        {
            "apiVersion": "rosclaw.io/v1",
            "kind": "App",
            "metadata": {"name": "fixture-hand-app"},
            "requires": {"capabilities": ["rh56.finger.move"]},
            "workflow": [{"call": "rh56.finger.move"}],
        }
    )
    try:
        if error_type is KeyboardInterrupt:
            with pytest.raises(KeyboardInterrupt):
                AppRunner(client).run(
                    app,
                    body_id="rh56-test",
                    body_snapshot_hash="",
                    execution_mode=ExecutionMode.SIMULATION,
                )
        else:
            result = AppRunner(client).run(
                app,
                body_id="rh56-test",
                body_snapshot_hash="",
                execution_mode=ExecutionMode.SIMULATION,
            )
            assert result.status == "failed" and not result.receipts
        assert len(created) == 1
        assert client.get_session(created[0])["session"]["state"] == "CLOSED"
        assert client.get_session("fixture-other-app-session")["session"]["state"] == "ACTIVE"
        assert client.get_runtime_status()["hardware_actions_executed"] == 0
    finally:
        client.close_session("fixture-other-app-session", reason="fixture_teardown")
