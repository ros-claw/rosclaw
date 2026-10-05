"""Existing discovery plus optional read-only native snapshot subscription."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from rosclaw.connectors.ros.discovery import RosGraphDiscovery
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.intelligence import RosSystemModel, build_system_model


class RosbridgeProbe:
    def __init__(self, transport: Any) -> None:
        self.transport = transport

    def inspect(
        self, *, robot_id: str = "unknown", body: dict | None = None, deep: bool = False
    ) -> RosSystemModel:
        native = self.read_native_snapshot() if deep else None
        if native is not None:
            if not isinstance(native.get("graph"), dict):
                raise ValueError("native probe capture is missing its graph")
            environment = native.get("environment", {})
            endpoint = getattr(self.transport, "endpoint", None)
            graph = RosGraphSnapshot.from_dict(
                {
                    **native["graph"],
                    "ros_version": environment.get("ros_generation", "unknown"),
                    "distro": environment.get("distro", "unknown"),
                    "endpoint": endpoint.url if endpoint else "unknown",
                    "captured_at": native["captured_at"],
                }
            )
        else:
            graph = RosGraphDiscovery(self.transport).discover()
        return build_system_model(graph, robot_id=robot_id, body=body, native=native)

    def read_native_snapshot(self, timeout_sec: float = 3.0) -> dict:
        topic = "/rosclaw_probe/snapshot"
        request_id = "rosprobe_" + uuid.uuid4().hex
        result = self.transport.send(
            {"op": "subscribe", "id": request_id, "topic": topic, "type": "std_msgs/msg/String"}
        )
        if not result.ok:
            raise ConnectionError(result.error)
        try:
            deadline = time.monotonic() + timeout_sec
            while time.monotonic() < deadline:
                result = self.transport.receive(timeout_sec=max(0.01, deadline - time.monotonic()))
                if not result.ok:
                    raise ConnectionError(result.error)
                message = result.data or {}
                if message.get("op") == "publish" and message.get("topic") == topic:
                    snapshot = json.loads(message["msg"]["data"])
                    if snapshot.get("schema_version") != "rosclaw.ros_probe.v1":
                        raise ValueError("unsupported probe snapshot version")
                    return snapshot
            raise TimeoutError("native probe snapshot unavailable; deep inspection is incomplete")
        finally:
            self.transport.send({"op": "unsubscribe", "id": request_id, "topic": topic})
