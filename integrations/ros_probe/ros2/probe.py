#!/usr/bin/env python3
"""Optional ROS-host process; only graph reads, subscriptions and whitelisted RPCs.

Run with the ROS host's Python after sourcing ROS. No ROSClaw dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import rclpy
from ament_index_python.packages import get_packages_with_prefixes
from lifecycle_msgs.srv import GetState
from rcl_interfaces.srv import GetParameters, ListParameters
from rclpy.action.graph import get_action_server_names_and_types_by_node
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_check_compatible
from rclpy.utilities import get_rmw_implementation_identifier
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String
from std_srvs.srv import Trigger

PROBE_TOPIC = "/rosclaw_probe/snapshot"
READ_TYPES = {
    "std_msgs/msg/Bool",
    "tf2_msgs/msg/TFMessage",
    "sensor_msgs/msg/LaserScan",
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/PointCloud2",
    "nav_msgs/msg/Odometry",
    "nav_msgs/msg/OccupancyGrid",
    "geometry_msgs/msg/PoseWithCovarianceStamped",
    "rosgraph_msgs/msg/Clock",
}


def utc_now():
    return datetime.now(UTC).isoformat()


class ReadOnlyProbe(Node):
    def __init__(self):
        super().__init__("rosclaw_readonly_probe")
        self.samples = {}
        self.subscriptions_by_topic = {}
        self.edges = {}
        self.lifecycle_states = {}
        self.parameters = {}
        self.parameter_received = {}
        self.parameter_captured_at = {}
        self.clients_by_name = {}
        self.pending = {}
        self.errors = []
        self.clock_values = deque(maxlen=20)
        self.clock_readings = deque(maxlen=200)
        self.packages = sorted(get_packages_with_prefixes())
        self.localization_observations = {}
        self.publisher = self.create_publisher(
            String, PROBE_TOPIC, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )
        self.status_publisher = self.create_publisher(String, "/rosclaw_probe/status", 1)
        self.refresh_service = self.create_service(Trigger, "/rosclaw_probe/refresh", self.refresh)
        # A paused simulation clock must not freeze its own diagnostic probe.
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(1.0, self.publish_snapshot, clock=self.wall_clock)

    def refresh(self, _request, response):
        self.discover_reads()
        response.success = True
        response.message = "Read-only refresh scheduled; no robot state mutated."
        return response

    def discover_reads(self):
        for topic, types in self.get_topic_names_and_types():
            if (
                topic in self.subscriptions_by_topic
                or len(types) != 1
                or types[0] not in READ_TYPES
            ):
                continue
            qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
            endpoints = self.get_publishers_info_by_topic(topic)
            latched = topic.endswith("/tf_static") or (
                topic == "/map"
                and endpoints
                and all(
                    p.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for p in endpoints
                )
            )
            if latched:
                qos = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            try:
                self.subscriptions_by_topic[topic] = self.create_subscription(
                    get_message(types[0]), topic, lambda msg, t=topic: self.observe(t, msg), qos
                )
            except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
                self.errors.append(f"subscribe {topic}: {exc}")
        for name, types in self.get_service_names_and_types():
            if name.endswith("/get_state") and types == ["lifecycle_msgs/srv/GetState"]:
                self.read_rpc(name, GetState, GetState.Request())
            elif name.endswith("/list_parameters") and types == [
                "rcl_interfaces/srv/ListParameters"
            ]:
                self.read_rpc(name, ListParameters, ListParameters.Request())

    def read_rpc(self, name, srv_type, request):
        if name in self.pending:
            future, started = self.pending[name]
            if not future.done() and time.monotonic() - started > 2:
                future.cancel()
                self.errors.append(f"read timeout: {name}")
                self.pending.pop(name, None)
            else:
                return
        client = self.clients_by_name.get(name)
        if client is None:
            client = self.create_client(srv_type, name)
            self.clients_by_name[name] = client
        if not client.service_is_ready():
            return
        future = client.call_async(request)
        self.pending[name] = (future, time.monotonic())

        def completed(result):
            current = self.pending.get(name)
            if current is None or current[0] is not result:
                return
            self.pending.pop(name, None)
            if result.cancelled():
                return
            error = result.exception()
            if error:
                self.errors.append(f"read {name}: {error}")
                return
            response = result.result()
            node_name = name.rsplit("/", 1)[0]
            if srv_type is GetState:
                labels = {1: "UNCONFIGURED", 2: "INACTIVE", 3: "ACTIVE", 4: "FINALIZED"}
                self.lifecycle_states[node_name] = {
                    "name": node_name,
                    "state": labels.get(response.current_state.id, "UNKNOWN"),
                    "source": name,
                    "captured_at": utc_now(),
                }
            elif srv_type is ListParameters:
                parameter_request = GetParameters.Request()
                parameter_request.names = [
                    key
                    for key in response.result.names
                    if key
                    in {
                        "use_sim_time",
                        "plugins",
                        "obstacle_layer.observation_sources",
                        "voxel_layer.observation_sources",
                        "global_frame",
                        "robot_base_frame",
                        "transform_tolerance",
                        "cmd_vel_in_topic",
                        "cmd_vel_out_topic",
                        "polygons",
                        "observation_sources",
                        "scan.topic",
                    }
                    or key.endswith(
                        (
                            ".plugin",
                            ".enabled",
                            ".use_maximum",
                            ".combination_method",
                            ".observation_sources",
                        )
                    )
                ]
                if parameter_request.names:
                    self.read_rpc(node_name + "/get_parameters", GetParameters, parameter_request)
            else:
                if len(request.names) != len(response.values):
                    self.errors.append(f"incomplete parameter response: {name}")
                    return
                values = {}
                for key, value in zip(request.names, response.values, strict=True):
                    if value.type == 1:
                        values[key] = value.bool_value
                    elif value.type == 2:
                        values[key] = value.integer_value
                    elif value.type == 3:
                        values[key] = value.double_value
                    elif value.type == 4:
                        values[key] = value.string_value
                    elif value.type == 9:
                        values[key] = list(value.string_array_value)
                self.parameters[node_name] = values
                self.parameter_received[node_name] = time.monotonic()
                self.parameter_captured_at[node_name] = utc_now()

        future.add_done_callback(completed)

    def observe(self, topic, message):
        self.samples.setdefault(topic, deque(maxlen=200)).append(time.monotonic())
        if hasattr(message, "pose") and hasattr(message.pose, "covariance"):
            covariance = message.pose.covariance
            self.localization_observations[topic] = {
                "source": topic,
                "captured_at": utc_now(),
                "position_variance": max(covariance[0], covariance[7]),
                "yaw_variance": covariance[35],
            }
        if hasattr(message, "clock"):
            value = message.clock.sec + message.clock.nanosec / 1e9
            self.clock_values.append(value)
            self.clock_readings.append((time.monotonic(), value))
        if hasattr(message, "transforms"):
            for transform in message.transforms:
                self.edges[(transform.header.frame_id, transform.child_frame_id)] = {
                    "parent": transform.header.frame_id,
                    "child": transform.child_frame_id,
                    "static": topic.endswith("/tf_static"),
                    "source": topic,
                    "captured_at": utc_now(),
                    "stamp_sec": transform.header.stamp.sec + transform.header.stamp.nanosec / 1e9,
                }

    def snapshot(self):
        stamp, wall = utc_now(), time.monotonic()
        # Retain original read times while excluding old cached configuration
        # from current readiness. A new envelope never refreshes an old RPC.
        parameters = {
            n: values
            for n, values in self.parameters.items()
            if 0 <= wall - self.parameter_received.get(n, 0) <= 5
        }
        lifecycle = []
        for stored in self.lifecycle_states.values():
            row = dict(stored)
            age = (
                datetime.fromisoformat(stamp) - datetime.fromisoformat(row["captured_at"])
            ).total_seconds()
            row["last_read_age_ms"] = age * 1000
            if not -0.1 <= age <= 5:
                row["last_observed_state"] = row["state"]
                row["state"] = "UNKNOWN"
            lifecycle.append(row)
        recent_clock_values = {
            v for received, v in self.clock_readings if 0 <= wall - received <= 1
        }
        clock_advancing = len(recent_clock_values) > 1 if self.clock_readings else None
        topics, qos_endpoints, incompatible, signals = [], [], [], []
        for topic, types in sorted(self.get_topic_names_and_types()):
            pubs = self.get_publishers_info_by_topic(topic)
            subs = self.get_subscriptions_info_by_topic(topic)

            def endpoint_name(endpoint):
                return endpoint.node_namespace.rstrip("/") + "/" + endpoint.node_name

            topics.append(
                {
                    "name": topic,
                    "msg_type": types[0] if len(types) == 1 else "",
                    "types": types,
                    "publishers": [endpoint_name(p) for p in pubs],
                    "subscribers": [endpoint_name(s) for s in subs],
                }
            )
            for kind, endpoints in [("publisher", pubs), ("subscriber", subs)]:
                for endpoint in endpoints:
                    qos = endpoint.qos_profile
                    qos_endpoints.append(
                        {
                            "topic": topic,
                            "node": endpoint_name(endpoint),
                            "kind": kind,
                            "reliability": qos.reliability.name,
                            "durability": qos.durability.name,
                            "history": qos.history.name,
                            "depth": qos.depth,
                            "deadline_ns": qos.deadline.nanoseconds,
                            "lifespan_ns": qos.lifespan.nanoseconds,
                            "liveliness": qos.liveliness.name,
                        }
                    )
            for pub in pubs:
                for sub in subs:
                    compatibility, reason = qos_check_compatible(pub.qos_profile, sub.qos_profile)
                    if compatibility.name == "ERROR":
                        incompatible.append(
                            {
                                "topic": topic,
                                "publisher": endpoint_name(pub),
                                "subscriber": endpoint_name(sub),
                                "reason": reason,
                            }
                        )
            if topic in self.subscriptions_by_topic:
                samples = list(self.samples.get(topic, ()))
                intervals = [b - a for a, b in zip(samples[:-1], samples[1:], strict=True)]
                signals.append(
                    {
                        "topic": topic,
                        "source": "native:monotonic_receive",
                        "max_age_ms": 2000 if topic.endswith("/costmap") else 1000,
                        "freshness_policy": "latched"
                        if topic.endswith("/tf_static")
                        or (
                            topic == "/map"
                            and pubs
                            and all(
                                p.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
                                for p in pubs
                            )
                        )
                        else "stream",
                        "captured_at": stamp,
                        "rate_hz": 1 / statistics.mean(intervals) if intervals else None,
                        "jitter_ms": statistics.pstdev(intervals) * 1000 if intervals else None,
                        "last_message_age_ms": (wall - samples[-1]) * 1000 if samples else None,
                        "publisher_count": len(pubs),
                        "subscriber_count": len(subs),
                    }
                )
        nodes = [
            {"name": ns.rstrip("/") + "/" + n, "namespace": ns}
            for n, ns in sorted(self.get_node_names_and_namespaces())
        ]
        actions = {}
        for name, namespace in self.get_node_names_and_namespaces():
            for action, types in get_action_server_names_and_types_by_node(self, name, namespace):
                actions[action] = {
                    "name": action,
                    "action_type": types[0] if len(types) == 1 else "",
                }
        transforms = []
        observed_times = {
            n: p["use_sim_time"]
            for n, p in parameters.items()
            if "use_sim_time" in p and n != self.get_fully_qualified_name()
        }
        runtime_sim_time = (
            next(iter(set(observed_times.values())))
            if len(set(observed_times.values())) == 1
            else None
        )
        ros_now = self.get_clock().now().nanoseconds / 1e9
        if runtime_sim_time is True:
            ros_now = self.clock_values[-1] if self.clock_values else None
        elif runtime_sim_time is None:
            ros_now = None
        for stored in self.edges.values():
            edge = dict(stored)
            edge["age_ms"] = (
                None if edge["static"] or ros_now is None else (ros_now - edge["stamp_sec"]) * 1000
            )
            edge.pop("stamp_sec", None)
            if edge["parent"] == "map" and edge["child"] == "odom":
                tolerances = [
                    p["transform_tolerance"]
                    for n, p in parameters.items()
                    if n.endswith("/amcl")
                    and isinstance(p.get("transform_tolerance"), (int, float))
                ]
                if len(tolerances) == 1 and 0 <= tolerances[0] <= 5:
                    edge["future_tolerance_ms"] = 100 + tolerances[0] * 1000
            transforms.append(edge)
        nav = {}
        localization = list(self.localization_observations.values())
        if localization:
            nav["localization_ready"] = any(
                row["position_variance"] <= 0.25
                and row["yaw_variance"] <= 0.25
                and signals
                and any(
                    s["topic"] == row["source"]
                    and s["last_message_age_ms"] is not None
                    and s["last_message_age_ms"] <= 2000
                    for s in signals
                )
                for row in localization
            )
        costmaps = [s for s in signals if s["topic"].endswith("/costmap")]
        if costmaps:
            nav["costmaps_fresh"] = all(
                s["last_message_age_ms"] is not None and s["last_message_age_ms"] <= 2000
                for s in costmaps
            )
        sources = [p for n, p in parameters.items() if "costmap" in n]
        if sources:
            nav["obstacle_source_configured"] = all(
                any(
                    p.get(k)
                    for k in (
                        "obstacle_layer.observation_sources",
                        "voxel_layer.observation_sources",
                    )
                )
                for p in sources
            )
        return {
            "schema_version": "rosclaw.ros_probe.v1",
            "captured_at": stamp,
            "environment": {
                "ros_generation": "ros2",
                "distro": os.getenv("ROS_DISTRO", "unknown"),
                "rmw": get_rmw_implementation_identifier(),
                "domain_id": os.getenv("ROS_DOMAIN_ID", "0"),
                "overlays": os.getenv("AMENT_PREFIX_PATH", "").split(":"),
                "use_sim_time": runtime_sim_time,
                "probe_use_sim_time": self.get_parameter("use_sim_time").value,
            },
            "graph": {
                "topics": topics,
                "nodes": nodes,
                "actions": list(actions.values()),
                "services": [
                    {"name": n, "srv_type": t[0] if len(t) == 1 else ""}
                    for n, t in sorted(self.get_service_names_and_types())
                ],
            },
            "signals": signals,
            "transforms": sorted(transforms, key=lambda x: (x["parent"], x["child"])),
            "lifecycle": sorted(lifecycle, key=lambda x: x["name"]),
            "qos": {
                "supported": True,
                "endpoints": qos_endpoints,
                "incompatible_pairs": incompatible,
            },
            "navigation": nav,
            "observations": {
                "node_use_sim_time": observed_times,
                "package_inventory": self.packages,
                "node_parameters": parameters,
                "parameter_captured_at": self.parameter_captured_at,
                "clock_advancing": clock_advancing,
                "clock_last_receive_age_ms": (wall - self.clock_readings[-1][0]) * 1000
                if self.clock_readings
                else None,
                "localization_quality": localization,
            },
            "completeness": {
                "graph": True,
                "qos": True,
                "signals": True,
                "tf": any(t.endswith("/tf") for t in self.subscriptions_by_topic),
                "lifecycle": bool(lifecycle)
                and all(row["state"] != "UNKNOWN" for row in lifecycle),
                "time": bool(observed_times)
                and (not any(observed_times.values()) or clock_advancing is not None),
            },
            "errors": self.errors[-50:],
        }

    def publish_snapshot(self):
        self.discover_reads()
        self.publisher.publish(String(data=json.dumps(self.snapshot(), allow_nan=False)))
        self.status_publisher.publish(
            String(data=json.dumps({"read_only": True, "captured_at": utc_now()}))
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = ReadOnlyProbe()
    try:
        if args.once:
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                node.discover_reads()
                rclpy.spin_once(node, timeout_sec=0.1)
            result = json.dumps(node.snapshot(), indent=2, allow_nan=False)
            if args.output:
                args.output.write_text(result + "\n", encoding="utf-8")
            else:
                print(result)
        else:
            rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
