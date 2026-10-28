"""App Session creation can have an effect even when its acknowledgement is lost."""

from typing import Any

import pytest

from rosclaw.app.runner import AppRunner
from rosclaw.app.schema import AppManifest
from rosclaw.kernel import ExecutionMode


def manifest():
    return AppManifest.model_validate(
        {
            "apiVersion": "rosclaw.io/v1",
            "kind": "App",
            "metadata": {"name": "fixture-inspection"},
            "requires": {"capabilities": ["camera.capture_rgbd"]},
            "workflow": [{"call": "camera.capture_rgbd"}],
        }
    )


class LostCreateAckClient:
    def __init__(self, error, *, cleanup_error=False):
        self.error = error
        self.cleanup_error = cleanup_error
        self.created = None
        self.live = set()
        self.closed = []
        self.actions = []

    def create_session(self, **kwargs: Any):
        self.created = kwargs
        self.live.add(kwargs["session_id"])
        raise self.error

    def request_action(self, action):
        self.actions.append(action)
        raise AssertionError("No action may follow an unacknowledged Session")

    def wait_for_action(self, action_id, *, timeout_sec):
        raise AssertionError("No action was submitted")

    def close_session(self, session_id, *, reason):
        self.closed.append(session_id)
        if self.cleanup_error:
            raise RuntimeError("fixture close transport unavailable")
        self.live.remove(session_id)
        return {"session_id": session_id, "reason": reason}


def test_lost_creation_ack_cleans_exact_dedicated_session_without_dispatch():
    client = LostCreateAckClient(TimeoutError("fixture creation acknowledgement lost"))
    result = AppRunner(client).run(
        manifest(),
        body_id="fixture-camera",
        body_snapshot_hash="",
        execution_mode=ExecutionMode.SIMULATION,
    )
    assert result.status == "failed"
    assert "TimeoutError" in result.error["message"]
    assert client.closed == [client.created["session_id"]] == [result.session_id]
    assert not client.live and not client.actions
    assert client.created["body_scope"] == ["fixture-camera"]
    assert client.created["capability_scope"] == ["camera.capture_rgbd"]


def test_creation_cleanup_error_preserves_original_failure_and_never_dispatches():
    client = LostCreateAckClient(TimeoutError("original creation ack lost"), cleanup_error=True)
    result = AppRunner(client).run(
        manifest(),
        body_id="fixture-camera",
        body_snapshot_hash="",
        execution_mode=ExecutionMode.SIMULATION,
    )
    assert result.status == "failed" and "original creation ack lost" in result.error["message"]
    assert client.closed == [client.created["session_id"]] and not client.actions
    # A close request is not a verified Session/physical stop when transport fails.
    assert client.live


def test_interrupt_during_creation_still_closes_the_dedicated_session():
    client = LostCreateAckClient(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        AppRunner(client).run(
            manifest(),
            body_id="fixture-camera",
            body_snapshot_hash="",
            execution_mode=ExecutionMode.SIMULATION,
        )
    assert client.closed == [client.created["session_id"]]
    assert not client.live and not client.actions


def test_real_snapshot_validation_still_precedes_any_session_effect():
    client = LostCreateAckClient(TimeoutError("must never reach fixture"))
    with pytest.raises(ValueError, match="body_snapshot_hash"):
        AppRunner(client).run(
            manifest(),
            body_id="fixture-camera",
            body_snapshot_hash="",
            execution_mode=ExecutionMode.REAL,
        )
    assert client.created is None and not client.actions and not client.closed
