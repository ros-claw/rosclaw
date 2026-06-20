"""Tests for rosclaw body update-state --from-ros."""

import sys
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.body.ros_introspection import RosIntrospectionError
from rosclaw.cli import main as rosclaw_main
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot, RosTopicInfo


@pytest.fixture
def linked_body(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0
    yield tmp_path


def _make_snapshot():
    return RosGraphSnapshot(
        ros_version="ros2",
        distro="humble",
        endpoint="ws://127.0.0.1:9090",
        topics=[
            RosTopicInfo(name="/camera/image_raw", msg_type="sensor_msgs/Image", is_sensor=True),
            RosTopicInfo(name="/joint_states", msg_type="sensor_msgs/JointState", is_sensor=True),
            RosTopicInfo(name="/cmd_vel", msg_type="geometry_msgs/Twist", is_command=True),
        ],
        services=[],
        actions=[],
        nodes=[{"name": "/robot_state_publisher"}],
        params=["/robot_description"],
        captured_at=datetime.now(UTC).isoformat(),
    )


def test_update_state_from_ros(linked_body):
    snapshot = _make_snapshot()
    runtime_state = {
        "online": True,
        "ros_version": "ros2",
        "ros_distro": "humble",
        "sensor_topics": ["/camera/image_raw", "/joint_states"],
        "command_topics": ["/cmd_vel"],
        "active_camera_topics": ["/camera/image_raw"],
        "active_joint_state_topics": ["/joint_states"],
        "node_count": 1,
    }

    with patch("rosclaw.body.cli.introspect_ros", return_value=(snapshot, runtime_state)):
        with patch.object(sys, "argv", [
            "rosclaw", "body", "update-state",
            "--from-ros",
            "--reason", "live ROS 2 introspection",
        ]):
            assert rosclaw_main() == 0

    resolver = BodyResolver()
    body_yaml = resolver.get_current_body_yaml()
    assert body_yaml.runtime_state["online"] is True
    assert body_yaml.runtime_state["ros_version"] == "ros2"
    assert "/camera/image_raw" in body_yaml.runtime_state["sensor_topics"]


def test_update_state_from_ros_failure(linked_body):
    with patch("rosclaw.body.cli.introspect_ros", side_effect=RosIntrospectionError("no bridge")):
        with patch.object(sys, "argv", [
            "rosclaw", "body", "update-state",
            "--from-ros",
            "--reason", "live ROS 2 introspection",
        ]):
            assert rosclaw_main() == 1


def test_update_state_from_ros_changes_hash(linked_body):
    snapshot = _make_snapshot()
    runtime_state = {"online": True, "ros_version": "ros2"}

    resolver = BodyResolver()
    old_hash = resolver.get_effective_body_hash()

    with patch("rosclaw.body.cli.introspect_ros", return_value=(snapshot, runtime_state)):
        with patch.object(sys, "argv", [
            "rosclaw", "body", "update-state",
            "--from-ros",
            "--reason", "live ROS 2 introspection",
        ]):
            assert rosclaw_main() == 0

    assert resolver.get_effective_body_hash() != old_hash
