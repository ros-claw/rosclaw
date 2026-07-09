"""Tests for rosapi resolver."""

from __future__ import annotations

import pytest

from rosclaw.connectors.ros.discovery import (
    RosApiDetectionError,
    RosApiProfile,
    RosApiResolver,
    RosVersion,
)
from rosclaw.connectors.ros.transport import MockTransport, RosTransportResult


def _make_service_response(values: dict, request_id: str | None = None) -> RosTransportResult:
    return RosTransportResult(
        ok=True,
        data={"op": "service_response", "values": values},
        request_id=request_id,
    )


def test_ros2_humble_detection():
    transport = MockTransport()
    transport.queue_json({"op": "service_response", "values": {"version": 2, "distro": "humble"}})
    resolver = RosApiResolver(transport)
    profile = resolver.resolve()
    assert profile.version == RosVersion.ROS2
    assert profile.distro == "humble"
    assert profile.service_prefix == "/rosapi"
    assert profile.type_prefix == "rosapi_msgs/srv"
    assert profile.service("topics") == "/rosapi/topics"
    assert profile.service_type("topics") == "rosapi_msgs/srv/topics"


def test_ros2_jazzy_with_rosapi_node_prefix():
    transport = MockTransport()
    # First prefix /rosapi fails.
    transport.queue_response(RosTransportResult(ok=False, error="service not found"))
    # Second prefix /rosapi_node succeeds.
    transport.queue_json({"op": "service_response", "values": {"version": 2, "distro": "jazzy"}})
    resolver = RosApiResolver(transport)
    profile = resolver.resolve()
    assert profile.version == RosVersion.ROS2
    assert profile.distro == "jazzy"
    assert profile.service_prefix == "/rosapi_node"


def test_ros1_fallback():
    transport = MockTransport()
    # get_ros_version fails on both prefixes.
    transport.queue_response(RosTransportResult(ok=False, error="not found"))
    transport.queue_response(RosTransportResult(ok=False, error="not found"))
    # ROS1 distro lookup succeeds.
    transport.queue_json({"op": "service_response", "values": {"value": "noetic"}})
    resolver = RosApiResolver(transport)
    profile = resolver.resolve()
    assert profile.version == RosVersion.ROS1
    assert profile.distro == "noetic"
    assert profile.service_prefix == "/rosapi"
    assert profile.type_prefix == "rosapi"


def test_ros1_distro_from_path_value():
    transport = MockTransport()
    transport.queue_response(RosTransportResult(ok=False, error="not found"))
    transport.queue_response(RosTransportResult(ok=False, error="not found"))
    transport.queue_json({"op": "service_response", "values": {"value": "/opt/ros/noetic"}})
    resolver = RosApiResolver(transport)
    profile = resolver.resolve()
    assert profile.distro == "noetic"


def test_ros1_distro_decodes_json_encoded_param_value():
    transport = MockTransport()
    transport.queue_response(RosTransportResult(ok=False, error="not found"))
    transport.queue_response(RosTransportResult(ok=False, error="not found"))
    transport.queue_json({"op": "service_response", "values": {"value": '"noetic\\n"'}})
    resolver = RosApiResolver(transport)
    profile = resolver.resolve()
    assert profile.distro == "noetic"


def test_cache_used():
    transport = MockTransport()
    transport.queue_json({"op": "service_response", "values": {"version": 2, "distro": "humble"}})
    resolver = RosApiResolver(transport)
    p1 = resolver.resolve()
    p2 = resolver.resolve()
    assert p1 is p2
    # Only one transport request was made.
    assert len([r for r in transport.requests if "/get_ros_version" in r.get("service", "")]) == 1


def test_reset_for_test():
    transport = MockTransport()
    transport.queue_json({"op": "service_response", "values": {"version": 2, "distro": "humble"}})
    resolver = RosApiResolver(transport)
    p1 = resolver.resolve()
    resolver.reset_for_test()
    transport.queue_json({"op": "service_response", "values": {"version": 2, "distro": "jazzy"}})
    p2 = resolver.resolve()
    assert p1.distro == "humble"
    assert p2.distro == "jazzy"


def test_network_error_raises():
    transport = MockTransport()
    transport.queue_response(RosTransportResult(ok=False, error="timeout"))
    transport.queue_response(RosTransportResult(ok=False, error="timeout"))
    transport.queue_response(RosTransportResult(ok=False, error="timeout"))
    resolver = RosApiResolver(transport)
    with pytest.raises(RosApiDetectionError):
        resolver.resolve()


def test_profile_helpers():
    p = RosApiProfile(
        version=RosVersion.ROS2,
        distro="humble",
        service_prefix="/rosapi",
        type_prefix="rosapi_msgs/srv",
    )
    assert p.service("nodes") == "/rosapi/nodes"
    assert p.service_type("nodes") == "rosapi_msgs/srv/nodes"
