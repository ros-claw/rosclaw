"""Dispatch, terminal result and physical stop remain separate facts."""

from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.action_client import Ros2ActionClient
from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor
from rosclaw.connectors.ros.transport.base import RosTransportResult


class Transport:
    def __init__(self, ok):
        self.ok = ok
        self.sent = []

    def send(self, message):
        self.sent.append(message)
        return RosTransportResult(ok=self.ok, error=None if self.ok else "disconnected")


def test_disconnect_cannot_silently_succeed_and_goal_remains_tracked():
    transport = Transport(False)
    client = Ros2ActionClient(transport)
    client._goals["owned"] = "/robot/navigate_to_pose"
    with pytest.raises(RuntimeError, match="dispatch failed: disconnected"):
        client.cancel_goal("owned")
    assert client._goals["owned"] == "/robot/navigate_to_pose"


def test_unknown_or_terminal_goal_does_not_send_empty_action_cancel():
    transport = Transport(True)
    client = Ros2ActionClient(transport)
    assert client.cancel_goal("already-terminal") is False
    assert not transport.sent


def test_successful_dispatch_is_not_terminal_acknowledgement():
    transport = Transport(True)
    client = Ros2ActionClient(transport)
    client._goals["owned"] = "/robot/navigate_to_pose"
    assert client.cancel_goal("owned") is True
    assert "owned" in client._goals
    assert transport.sent == [
        {"op": "cancel_action_goal", "action": "/robot/navigate_to_pose", "id": "owned"}
    ]


def executor(tmp_path):
    result = RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=None,
        output=tmp_path,
        body_id="sim",
        body_snapshot_hash="body",
        grid={},
    )
    result.goal_id = "owned"
    return result


def fail_cancel(_):
    raise RuntimeError("connection lost")


def test_emergency_transport_failure_still_attempts_both_sim_safety_services(tmp_path):
    instance = executor(tmp_path)
    instance.client = SimpleNamespace(cancel_goal=fail_cancel)
    calls = []

    def service(name, args):
        calls.append((name, args))
        return RosTransportResult(ok=True, data={"values": {"success": True}})

    instance._service = service
    result = instance.emergency_stop()
    assert calls == [
        ("/rosclaw_sim/cleaning", {"data": False}),
        ("/rosclaw_sim/lease", {"data": False}),
    ]
    assert instance.stopping.is_set()
    assert result["acknowledged"] is False
    assert result["physical_stop_verified"] is False
    assert result["cancel_error"] == "connection lost"


def test_failed_disable_response_is_not_acknowledged(tmp_path):
    instance = executor(tmp_path)
    instance.client = SimpleNamespace(cancel_goal=lambda _: True)
    instance._service = lambda *_: RosTransportResult(ok=True, data={"values": {"success": False}})
    result = instance.emergency_stop()
    assert result["acknowledged"] is False
    assert result["physical_stop_verified"] is False


def test_one_safety_service_exception_does_not_skip_other_service(tmp_path):
    instance = executor(tmp_path)
    instance.client = SimpleNamespace(cancel_goal=lambda _: True)
    calls = []

    def service(name, args):
        calls.append(name)
        if name.endswith("cleaning"):
            raise RuntimeError("service unavailable")
        return RosTransportResult(ok=True, data={"values": {"success": True}})

    instance._service = service
    result = instance.emergency_stop()
    assert calls == ["/rosclaw_sim/cleaning", "/rosclaw_sim/lease"]
    assert result["acknowledged"] is False
    assert result["physical_stop_verified"] is False


@pytest.mark.parametrize("terminal", [4, 6])
def test_repair_timeout_rejects_non_cancel_terminal_as_ack(tmp_path, terminal):
    import time

    instance = executor(tmp_path)
    callback = {}
    instance.client = SimpleNamespace(
        send_goal=lambda **args: callback.update(result=args["on_result"]),
        cancel_goal=lambda _: callback["result"](terminal, {"error_code": 0}),
    )
    instance.witness = SimpleNamespace(
        fresh=lambda: {
            "observation_complete": True,
            "cleaning_enabled": True,
            "lease_remaining_sec": 1.0,
        }
    )
    with pytest.raises(RuntimeError, match="did not return CANCELED"):
        instance._run_goal(
            "/navigate",
            "nav2_msgs/action/NavigateToPose",
            {},
            "owned",
            time.monotonic() + 10,
            goal_timeout_sec=0.01,
        )
