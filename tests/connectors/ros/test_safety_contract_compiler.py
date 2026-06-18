"""Tests for safety contract compiler."""

from __future__ import annotations

from rosclaw.connectors.ros.compiler import (
    CapabilityManifest,
    CapabilityManifestCompiler,
    RosCapability,
    RosCapabilityRisk,
    RosInterface,
    SafetyContractCompiler,
    SafetyLevel,
)
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot, RosTopicInfo


def _make_manifest(topics=None, services=None) -> CapabilityManifest:
    snapshot = RosGraphSnapshot(
        ros_version="ros2",
        distro="humble",
        endpoint="ws://127.0.0.1:9090",
        topics=topics or [],
        services=services or [],
        actions=[],
        nodes=[],
        params=[],
        captured_at="2026-06-17T00:00:00Z",
    )
    return CapabilityManifestCompiler(robot_id="turtlesim").compile(snapshot)


def test_read_only_observation_allowed():
    manifest = _make_manifest(topics=[
        RosTopicInfo(name="/camera/image_raw", msg_type="sensor_msgs/msg/Image", is_sensor=True, risk_hint="low"),
    ])
    compiler = SafetyContractCompiler()
    contract = compiler.compile(manifest)
    decision = compiler.evaluate(contract, "turtlesim.observe.camera.rgb", {})
    assert decision.decision == "ALLOW"


def test_high_risk_velocity_without_duration_blocked():
    manifest = _make_manifest(topics=[
        RosTopicInfo(name="/turtle1/cmd_vel", msg_type="geometry_msgs/msg/Twist", is_command=True, risk_hint="high"),
    ])
    compiler = SafetyContractCompiler()
    contract = compiler.compile(manifest)
    decision = compiler.evaluate(contract, "turtlesim.base.velocity_command", {"linear": {"x": 0.1}})
    assert decision.decision == "BLOCK"
    assert "velocity_command_requires_duration" in decision.violated_constraints


def test_high_risk_velocity_too_fast_blocked():
    manifest = _make_manifest(topics=[
        RosTopicInfo(name="/turtle1/cmd_vel", msg_type="geometry_msgs/msg/Twist", is_command=True, risk_hint="high"),
    ])
    compiler = SafetyContractCompiler()
    contract = compiler.compile(manifest)
    args = {
        "linear": {"x": 1.0, "y": 0.0},
        "angular": {"z": 0.0},
        "duration": 0.5,
    }
    decision = compiler.evaluate(contract, "turtlesim.base.velocity_command", args)
    assert decision.decision == "BLOCK"
    assert any("linear.x" in v for v in decision.violated_constraints)


def test_excessive_duration_blocked():
    manifest = _make_manifest(topics=[
        RosTopicInfo(name="/turtle1/cmd_vel", msg_type="geometry_msgs/msg/Twist", is_command=True, risk_hint="high"),
    ])
    compiler = SafetyContractCompiler()
    contract = compiler.compile(manifest)
    args = {
        "linear": {"x": 0.1, "y": 0.0},
        "angular": {"z": 0.0},
        "duration": 5.0,
    }
    decision = compiler.evaluate(contract, "turtlesim.base.velocity_command", args)
    assert decision.decision == "BLOCK"
    assert any("duration" in v for v in decision.violated_constraints)


def test_safe_velocity_allowed():
    manifest = _make_manifest(topics=[
        RosTopicInfo(name="/turtle1/cmd_vel", msg_type="geometry_msgs/msg/Twist", is_command=True, risk_hint="high"),
    ])
    compiler = SafetyContractCompiler()
    contract = compiler.compile(manifest)
    args = {
        "linear": {"x": 0.1, "y": 0.0},
        "angular": {"z": 0.0},
        "duration": 0.5,
    }
    decision = compiler.evaluate(contract, "turtlesim.base.velocity_command", args)
    assert decision.decision == "ALLOW"


def test_forbidden_pattern_blocked():
    cap = RosCapability(
        id="robot.raw_torque_command",
        kind="actuation",
        interface=RosInterface(ros_kind="topic", name="/raw_torque_command", msg_type="std_msgs/Float64"),
        risk=RosCapabilityRisk(level="high", read_only=False, destructive=True),
    )
    manifest = CapabilityManifest(robot_id="robot", capabilities=[cap])
    compiler = SafetyContractCompiler()
    contract = compiler.compile(manifest)
    rule = contract.get_rule("robot.raw_torque_command")
    assert rule.level == SafetyLevel.FORBIDDEN_BY_DEFAULT
    decision = compiler.evaluate(contract, "robot.raw_torque_command", {})
    assert decision.decision == "BLOCK"


def test_missing_rule_blocked():
    manifest = CapabilityManifest(robot_id="robot", capabilities=[])
    compiler = SafetyContractCompiler()
    contract = compiler.compile(manifest)
    decision = compiler.evaluate(contract, "robot.unknown", {})
    assert decision.decision == "BLOCK"
    assert "missing_safety_rule" in decision.violated_constraints
