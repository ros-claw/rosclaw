"""Tests for ROS CLI commands."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.cli.ros_cli import (
    cmd_doctor_ros,
    cmd_ros_compile,
    cmd_ros_emergency_stop,
    cmd_ros_inspect_capability,
    cmd_ros_list_capabilities,
    cmd_ros_ping,
    cmd_ros_validate_capability,
)
from rosclaw.connectors.ros.compiler import (
    CapabilityManifest,
    RosCapability,
    RosCapabilityRisk,
    RosInterface,
)
from rosclaw.connectors.ros.discovery.graph import (
    RosGraphSnapshot,
    RosTopicInfo,
)


class FakeArgs(SimpleNamespace):
    """ argparse.Namespace replacement with safe defaults. """

    def __getattr__(self, name: str):
        return None


def test_ros_ping_returns_structured_error_without_server():
    args = FakeArgs(endpoint="ws://127.0.0.1:9998", json=True)
    rc = cmd_ros_ping(args)
    assert rc == 1


def test_ros_emergency_stop_returns_error_without_server():
    args = FakeArgs(endpoint="ws://127.0.0.1:9998", robot_id="turtlesim", json=True)
    rc = cmd_ros_emergency_stop(args)
    assert rc == 1


def test_ros_list_capabilities_with_preloaded_manifest():
    cap = RosCapability(
        id="turtlesim.observe.pose",
        kind="observation",
        interface=RosInterface(ros_kind="topic", name="/turtle1/pose", msg_type="turtlesim/msg/Pose"),
        risk=RosCapabilityRisk(level="low", read_only=True),
    )
    manifest = CapabilityManifest(robot_id="turtlesim", capabilities=[cap])
    provider = SimpleNamespace(_manifest=manifest)
    args = FakeArgs(_ros_provider=provider, json=True)
    rc = cmd_ros_list_capabilities(args)
    assert rc == 0


def test_ros_inspect_capability_with_preloaded_manifest():
    cap = RosCapability(
        id="turtlesim.base.velocity_command",
        kind="actuation",
        interface=RosInterface(ros_kind="topic", name="/turtle1/cmd_vel", msg_type="geometry_msgs/msg/Twist"),
        risk=RosCapabilityRisk(
            level="high", read_only=False, destructive=True,
            requires_sandbox=True, requires_runtime_guard=True, requires_stop_guard=True, max_duration_sec=1.0,
        ),
        safety={"constraints": {"linear.x": [-0.2, 0.2]}},
    )
    manifest = CapabilityManifest(robot_id="turtlesim", capabilities=[cap])
    provider = SimpleNamespace(_manifest=manifest)
    args = FakeArgs(
        capability_id="turtlesim.base.velocity_command",
        _ros_provider=provider,
        json=True,
    )
    rc = cmd_ros_inspect_capability(args)
    assert rc == 0


def test_ros_validate_capability_blocks_out_of_bounds():
    cap = RosCapability(
        id="turtlesim.base.velocity_command",
        kind="actuation",
        interface=RosInterface(ros_kind="topic", name="/turtle1/cmd_vel", msg_type="geometry_msgs/msg/Twist"),
        risk=RosCapabilityRisk(
            level="high", read_only=False, destructive=True,
            requires_sandbox=True, requires_runtime_guard=True, requires_stop_guard=True, max_duration_sec=1.0,
        ),
        safety={"constraints": {"linear.x": [-0.2, 0.2]}},
    )
    manifest = CapabilityManifest(robot_id="turtlesim", capabilities=[cap])
    provider = SimpleNamespace(_manifest=manifest)
    args = FakeArgs(
        capability_id="turtlesim.base.velocity_command",
        args='{"linear": {"x": 1.0}, "duration": 0.5}',
        _ros_provider=provider,
        json=True,
    )
    rc = cmd_ros_validate_capability(args)
    assert rc == 1


def test_ros_validate_capability_allows_in_bounds():
    cap = RosCapability(
        id="turtlesim.base.velocity_command",
        kind="actuation",
        interface=RosInterface(ros_kind="topic", name="/turtle1/cmd_vel", msg_type="geometry_msgs/msg/Twist"),
        risk=RosCapabilityRisk(
            level="high", read_only=False, destructive=True,
            requires_sandbox=True, requires_runtime_guard=True, requires_stop_guard=True, max_duration_sec=1.0,
        ),
        safety={"constraints": {"linear.x": [-0.2, 0.2]}},
    )
    manifest = CapabilityManifest(robot_id="turtlesim", capabilities=[cap])
    provider = SimpleNamespace(_manifest=manifest)
    args = FakeArgs(
        capability_id="turtlesim.base.velocity_command",
        args='{"linear": {"x": 0.1}, "duration": 0.5}',
        _ros_provider=provider,
        json=True,
    )
    rc = cmd_ros_validate_capability(args)
    assert rc == 0


def test_doctor_ros_without_server_reports_not_ready():
    args = FakeArgs(endpoint="ws://127.0.0.1:9998")
    rc = cmd_doctor_ros(args)
    assert rc == 1


def test_doctor_ros_no_args_reports_not_ready():
    # Use an unused port so the test is independent of any running rosbridge.
    import argparse
    args = argparse.Namespace(endpoint="ws://127.0.0.1:59999")
    rc = cmd_doctor_ros(args)
    assert rc == 1


@pytest.mark.parametrize("cmd,expected", [
    (["-m", "rosclaw.cli", "ros", "--help"], "ros"),
    (["-m", "rosclaw.cli", "doctor", "--help"], "doctor"),
])
def test_ros_cli_help_renders(cmd, expected):
    result = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
    assert result.returncode == 0
    assert expected in result.stdout.lower()


def test_ros_list_capabilities_loads_manifest_from_file():
    cap = RosCapability(
        id="turtlesim.observe.pose",
        kind="observation",
        interface=RosInterface(ros_kind="topic", name="/turtle1/pose", msg_type="turtlesim/msg/Pose"),
        risk=RosCapabilityRisk(level="low", read_only=True),
    )
    manifest = CapabilityManifest(robot_id="turtlesim", capabilities=[cap])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(manifest.to_dict(), f)
        path = f.name
    try:
        args = FakeArgs(manifest=path, json=True)
        rc = cmd_ros_list_capabilities(args)
        assert rc == 0
    finally:
        Path(path).unlink()


def test_ros_compile_from_saved_graph():
    snapshot = RosGraphSnapshot(
        ros_version="ros2",
        distro="humble",
        endpoint="ws://127.0.0.1:9090",
        topics=[
            RosTopicInfo(name="/turtle1/cmd_vel", msg_type="geometry_msgs/msg/Twist", is_command=True, risk_hint="high"),
            RosTopicInfo(name="/turtle1/pose", msg_type="turtlesim/msg/Pose", is_sensor=True, risk_hint="low"),
        ],
        services=[],
        actions=[],
        nodes=[],
        params=[],
        captured_at="2026-06-17T00:00:00Z",
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(snapshot.to_dict(), f)
        graph_path = f.name
    try:
        args = FakeArgs(
            graph=graph_path,
            robot_id="turtlesim",
            output=None,
            json=True,
            endpoint="ws://127.0.0.1:9090",
        )
        rc = cmd_ros_compile(args)
        assert rc == 0
    finally:
        Path(graph_path).unlink()
