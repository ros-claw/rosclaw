"""Deep capture preserves the native graph and skips duplicate rosapi RPCs."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from rosclaw.connectors.ros.probe.rosbridge_probe import RosbridgeProbe
from rosclaw.connectors.ros.transport.base import RosTransportResult


class NativeTransport:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.sent = []

    def send(self, request):
        self.sent.append(request)
        return RosTransportResult(ok=True)

    def receive(self, timeout_sec):
        return RosTransportResult(
            ok=True,
            data={
                "op": "publish",
                "topic": "/rosclaw_probe/snapshot",
                "msg": {"data": json.dumps(self.snapshot)},
            },
        )

    def call_service(self, *args, **kwargs):
        raise AssertionError("deep native capture must not rediscover the graph via rosapi")


def capture():
    return {
        "schema_version": "rosclaw.ros_probe.v1",
        "captured_at": (datetime.now(UTC) - timedelta(seconds=20)).isoformat(),
        "environment": {"ros_generation": "ros2", "distro": "jazzy"},
        "graph": {
            "topics": [{"name": "/scan", "msg_type": "sensor_msgs/msg/LaserScan"}],
            "nodes": [],
            "services": [],
            "actions": [],
        },
        "completeness": {"graph": True},
        "errors": ["native lifecycle observation missing"],
    }


def test_native_capture_is_authoritative_without_relabeling_old_measurements():
    snapshot = capture()
    transport = NativeTransport(snapshot)
    model = RosbridgeProbe(transport).inspect(deep=True)
    assert model.graph == snapshot["graph"]
    assert model.captured_at.isoformat() == snapshot["captured_at"]
    assert model.age_ms(datetime.now(UTC)) >= 20000
    assert model.errors == snapshot["errors"]
    assert model.environment["distro"] == "jazzy"
    assert [r["op"] for r in transport.sent] == ["subscribe", "unsubscribe"]


def test_missing_native_graph_fails_closed():
    snapshot = capture()
    del snapshot["graph"]
    with pytest.raises(ValueError, match="missing its graph"):
        RosbridgeProbe(NativeTransport(snapshot)).inspect(deep=True)
