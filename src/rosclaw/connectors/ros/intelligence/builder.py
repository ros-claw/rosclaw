"""Combine graph and optional native measurements without inventing Body truth."""

from __future__ import annotations

from typing import Any

from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot

from .system_model import RosSystemModel


def build_system_model(
    graph: RosGraphSnapshot,
    *,
    robot_id: str = "unknown",
    body: dict[str, Any] | None = None,
    native: dict[str, Any] | None = None,
) -> RosSystemModel:
    native = native or {}
    if native and native.get("schema_version") != "rosclaw.ros_probe.v1":
        raise ValueError("unsupported native probe schema")
    environment = {"ros_generation": graph.ros_version, "distro": graph.distro}
    environment.update(native.get("environment", {}))
    # Native graph and measurements belong to the same capture; never combine
    # fresh rosapi graph with an older native capture and label it fresh.
    captured_at = native.get("captured_at", graph.captured_at)
    graph_data = native.get("graph", graph.to_dict())
    roles = {
        "sensor_msgs/msg/LaserScan": "sensing.lidar",
        "sensor_msgs/msg/Image": "sensing.rgb_or_depth",
        "nav_msgs/msg/Odometry": "state.odometry",
        "nav_msgs/msg/OccupancyGrid": "mapping.occupancy_map",
    }
    inferred = [
        {"topic": t["name"], "semantic_role": roles[t["msg_type"]], "evidence_class": "inferred"}
        for t in graph_data.get("topics", [])
        if t.get("msg_type") in roles
    ]
    return RosSystemModel(
        robot_id=robot_id,
        snapshot_id="",
        captured_at=captured_at,
        environment=environment,
        body=body or {},
        graph=graph_data,
        signals=native.get("signals", []),
        transforms=native.get("transforms", []),
        lifecycle=native.get("lifecycle", []),
        qos=native.get("qos", {"supported": graph.ros_version == "ros2", "endpoints": []}),
        navigation=native.get("navigation", {}),
        observations=native.get("observations", {}),
        completeness=native.get("completeness", {"graph": True}),
        errors=native.get("errors", []),
        inference={"topic_roles": inferred},
    ).seal()
