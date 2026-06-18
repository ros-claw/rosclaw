"""Tests for capability manifest compiler."""

from __future__ import annotations

from rosclaw.connectors.ros.compiler import CapabilityManifestCompiler
from rosclaw.connectors.ros.discovery.graph import (
    RosActionInfo,
    RosGraphSnapshot,
    RosServiceInfo,
    RosTopicInfo,
)


def _snapshot(topics=None, services=None, actions=None) -> RosGraphSnapshot:
    return RosGraphSnapshot(
        ros_version="ros2",
        distro="humble",
        endpoint="ws://127.0.0.1:9090",
        topics=topics or [],
        services=services or [],
        actions=actions or [],
        nodes=[],
        params=[],
        captured_at="2026-06-17T00:00:00Z",
    )


def test_camera_topic_becomes_observation_capability():
    snapshot = _snapshot(topics=[
        RosTopicInfo(name="/camera/image_raw", msg_type="sensor_msgs/msg/Image", is_sensor=True, risk_hint="low"),
    ])
    compiler = CapabilityManifestCompiler(robot_id="turtlebot")
    manifest = compiler.compile(snapshot)
    cap = manifest.get_capability("turtlebot.observe.camera.rgb")
    assert cap is not None
    assert cap.kind == "observation"
    assert cap.risk.level == "low"
    assert cap.risk.read_only


def test_cmd_vel_becomes_high_risk_actuation():
    snapshot = _snapshot(topics=[
        RosTopicInfo(name="/turtle1/cmd_vel", msg_type="geometry_msgs/msg/Twist", is_command=True, risk_hint="high"),
    ])
    compiler = CapabilityManifestCompiler(robot_id="turtlesim")
    manifest = compiler.compile(snapshot)
    cap = manifest.get_capability("turtlesim.base.velocity_command")
    assert cap is not None
    assert cap.kind == "actuation"
    assert cap.risk.level == "high"
    assert cap.risk.destructive
    assert cap.risk.requires_stop_guard
    assert cap.risk.max_duration_sec == 1.0


def test_preferred_service_suppresses_command_topic():
    snapshot = _snapshot(
        services=[
            RosServiceInfo(name="/go2/move", srv_type="go2_interfaces/srv/Move", risk_hint="medium"),
        ],
        topics=[
            RosTopicInfo(name="/cmd_vel", msg_type="geometry_msgs/msg/Twist", is_command=True, risk_hint="high"),
        ],
    )
    spec = {
        "preferred_interfaces": [{"ros_kind": "service", "ros_name": "/go2/move"}],
        "discouraged_interfaces": [{"ros_kind": "topic", "ros_name": "/cmd_vel"}],
    }
    compiler = CapabilityManifestCompiler(robot_id="go2", robot_spec=spec)
    manifest = compiler.compile(snapshot)
    assert manifest.get_capability("go2.go2.move") is not None
    assert manifest.get_capability("go2.base.velocity_command") is None


def test_robot_spec_safety_defaults_override():
    snapshot = _snapshot(topics=[
        RosTopicInfo(name="/cmd_vel", msg_type="geometry_msgs/msg/Twist", is_command=True, risk_hint="high"),
    ])
    spec = {
        "safety_defaults": {
            "max_linear_velocity": 0.1,
            "max_angular_velocity": 0.3,
            "max_motion_duration_sec": 0.5,
        },
    }
    compiler = CapabilityManifestCompiler(robot_id="go2", robot_spec=spec)
    manifest = compiler.compile(snapshot)
    cap = manifest.get_capability("go2.base.velocity_command")
    assert cap.safety["constraints"]["linear.x"] == [-0.1, 0.1]
    assert cap.safety["constraints"]["angular.z"] == [-0.3, 0.3]
    assert cap.risk.max_duration_sec == 0.5


def test_service_compiled_with_medium_risk():
    snapshot = _snapshot(services=[
        RosServiceInfo(name="/go2/stand_up", srv_type="std_srvs/srv/Trigger", risk_hint="medium"),
    ])
    compiler = CapabilityManifestCompiler(robot_id="go2")
    manifest = compiler.compile(snapshot)
    cap = manifest.get_capability("go2.go2.stand_up")
    assert cap is not None
    assert cap.risk.level == "medium"
    assert cap.risk.requires_sandbox


def test_action_compiled():
    snapshot = _snapshot(actions=[
        RosActionInfo(name="/navigate_to_pose", action_type="nav2_msgs/action/NavigateToPose", risk_hint="medium"),
    ])
    compiler = CapabilityManifestCompiler(robot_id="turtlebot")
    manifest = compiler.compile(snapshot)
    cap = manifest.get_capability("turtlebot.action.navigate_to_pose")
    assert cap is not None
    assert cap.interface.ros_kind == "action"


def test_manifest_to_dict_roundtrip():
    snapshot = _snapshot(topics=[
        RosTopicInfo(name="/turtle1/pose", msg_type="turtlesim/msg/Pose"),
    ])
    compiler = CapabilityManifestCompiler(robot_id="turtlesim")
    manifest = compiler.compile(snapshot)
    data = manifest.to_dict()
    assert data["schema_version"] == "rosclaw.capability_manifest.v1"
    assert data["robot_id"] == "turtlesim"
    assert len(data["capabilities"]) == 1
    assert data["capabilities"][0]["id"] == "turtlesim.observe.pose"
