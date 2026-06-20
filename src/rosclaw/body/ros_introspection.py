"""Bridge ROS graph discovery to the body runtime state.

This module lives in the body package but consumes the existing
connector-level ROS discovery code. It intentionally does not import
ROS Python client libraries.
"""

from __future__ import annotations

from typing import Any

from rosclaw.connectors.ros.discovery.graph import RosGraphDiscovery, RosGraphSnapshot
from rosclaw.connectors.ros.discovery.rosapi_resolver import RosApiResolver
from rosclaw.connectors.ros.transport.base import RosbridgeEndpoint
from rosclaw.connectors.ros.transport.rosbridge import RosbridgeTransport


class RosIntrospectionError(Exception):
    """Raised when live ROS introspection cannot complete."""


def introspect_ros(
    endpoint: RosbridgeEndpoint | None = None,
    timeout_sec: float = 5.0,
) -> tuple[RosGraphSnapshot, dict[str, Any]]:
    """Connect to a live rosbridge, capture the ROS graph, and return a
    conservative runtime-state patch for ``body.yaml``.
    """
    transport = RosbridgeTransport(endpoint=endpoint, max_retries=1, dry_run=False)
    transport.endpoint.timeout_sec = timeout_sec
    connect_result = transport.connect()
    if not connect_result.ok:
        raise RosIntrospectionError(
            f"Failed to connect to rosbridge ({transport.endpoint.url}): "
            f"{connect_result.error or 'unknown error'}"
        )
    try:
        profile = RosApiResolver(transport).resolve()
        snapshot = RosGraphDiscovery(transport, profile).discover()
        return snapshot, snapshot_to_runtime_state(snapshot)
    except RosIntrospectionError:
        raise
    except Exception as exc:
        raise RosIntrospectionError(f"ROS introspection failed: {exc}") from exc
    finally:
        transport.close()


def snapshot_to_runtime_state(snapshot: RosGraphSnapshot) -> dict[str, Any]:
    """Convert a graph snapshot to a runtime_state patch.

    The patch is intentionally conservative: it records observable facts
    (topics, nodes, parameters) and avoids inferring unavailable components.
    """
    sensor_topics = [t.name for t in snapshot.topics if t.is_sensor]
    command_topics = [t.name for t in snapshot.topics if t.is_command]

    # Per-sensor categories used by downstream skill manifests.
    camera_topics = [t.name for t in snapshot.topics if t.is_sensor and "image" in t.msg_type.lower()]
    joint_state_topics = [
        t.name for t in snapshot.topics
        if t.is_sensor and t.msg_type.lower() in {"sensor_msgs/jointstate", "sensor_msgs/joint_state"}
    ]
    point_cloud_topics = [
        t.name for t in snapshot.topics
        if t.is_sensor and "pointcloud" in t.msg_type.lower().replace("_", "")
    ]
    lidar_topics = [t.name for t in snapshot.topics if t.is_sensor and "laserscan" in t.msg_type.lower()]
    odometry_topics = [t.name for t in snapshot.topics if t.is_sensor and "odom" in t.msg_type.lower()]

    motion_services = [
        s.name for s in snapshot.services
        if s.risk_hint in {"medium", "high"}
    ]

    return {
        "online": True,
        "ros_version": snapshot.ros_version,
        "ros_distro": snapshot.distro,
        "endpoint": snapshot.endpoint,
        "sensor_topics": sorted(set(sensor_topics)),
        "command_topics": sorted(set(command_topics)),
        "active_camera_topics": sorted(set(camera_topics)),
        "active_joint_state_topics": sorted(set(joint_state_topics)),
        "active_point_cloud_topics": sorted(set(point_cloud_topics)),
        "active_lidar_topics": sorted(set(lidar_topics)),
        "active_odometry_topics": sorted(set(odometry_topics)),
        "motion_services": sorted(set(motion_services)),
        "node_count": len(snapshot.nodes),
        "param_count": len(snapshot.params),
        "snapshot_captured_at": snapshot.captured_at,
    }
