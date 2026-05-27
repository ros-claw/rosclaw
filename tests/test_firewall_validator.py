"""Tests for FirewallValidator (Sprint 3)."""

import pytest
import numpy as np

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.e_urdf.parser import RobotModel, JointSpec
from rosclaw.firewall.validator import (
    FirewallValidator,
    SafetyEnvelope,
    ValidationRequest,
    ValidationLayer,
)


def _make_test_robot():
    """Create a simple 2-DOF robot model for testing."""
    model = RobotModel(name="test_robot")
    model.joints["j1"] = JointSpec(
        name="j1", joint_type="revolute", parent="base", child="link1",
        limits={"lower": -1.0, "upper": 1.0, "velocity": 2.0, "effort": 10.0}
    )
    model.joints["j2"] = JointSpec(
        name="j2", joint_type="revolute", parent="link1", child="link2",
        limits={"lower": -0.5, "upper": 0.5, "velocity": 1.0, "effort": 5.0}
    )
    return model


def test_safety_envelope_from_robot_model():
    """Verify soft limits are 95% of hard limits for MODERATE."""
    model = _make_test_robot()
    envelope = SafetyEnvelope.from_robot_model(model, safety_level="MODERATE")

    assert len(envelope.joint_soft_limits) == 2
    # j1: [-1.0, 1.0] * 0.95 = [-0.95, 0.95]
    assert envelope.joint_soft_limits[0] == pytest.approx((-0.95, 0.95), abs=0.01)
    # j2: [-0.5, 0.5] * 0.95 = [-0.475, 0.475]
    assert envelope.joint_soft_limits[1] == pytest.approx((-0.475, 0.475), abs=0.01)


def test_safety_envelope_strict():
    """Verify STRICT uses 90% factor."""
    model = _make_test_robot()
    envelope = SafetyEnvelope.from_robot_model(model, safety_level="STRICT")
    assert envelope.joint_soft_limits[0] == pytest.approx((-0.90, 0.90), abs=0.01)


def test_eurdf_limit_violation():
    """Trajectory exceeding joint soft limits -> critical violation."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(
        robot_model=model,
        event_bus=bus,
        safety_level="MODERATE",
    )
    validator.initialize()

    request = ValidationRequest(
        request_id="req1",
        robot_id="test",
        trajectory=[[0.0, 0.0], [0.99, 0.0]],  # j1=0.99 > 0.95 soft limit
    )
    response = validator.validate(request)

    assert not response.is_safe
    assert any(v.layer == ValidationLayer.EURDF_SOFT_LIMITS for v in response.violations)
    assert any("outside soft limit" in v.description for v in response.violations)
    validator.stop()


def test_safe_trajectory():
    """Valid trajectory within limits -> safe."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(
        robot_model=model,
        event_bus=bus,
        safety_level="MODERATE",
    )
    validator.initialize()

    request = ValidationRequest(
        request_id="req2",
        robot_id="test",
        trajectory=[[0.0, 0.0], [0.5, 0.3]],  # Within [-0.95, 0.95] and [-0.475, 0.475]
    )
    response = validator.validate(request)

    assert response.is_safe
    assert response.violation_count == 0
    validator.stop()


def test_eventbus_safe_command():
    """Valid command via EventBus -> agent.response with is_safe=True."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(
        robot_model=model,
        event_bus=bus,
        safety_level="MODERATE",
    )
    validator.initialize()

    responses = []
    bus.subscribe("agent.response", lambda e: responses.append(e.payload))

    bus.publish(Event(
        topic="agent.command",
        payload={
            "action": "move_joints",
            "trajectory": [[0.0, 0.0], [0.5, 0.3]],
            "robot_id": "test",
        },
        source="test",
        metadata={"request_id": "req3"},
    ))

    assert len(responses) == 1
    assert responses[0]["is_safe"] is True
    assert responses[0]["request_id"] == "req3"
    validator.stop()


def test_eventbus_unsafe_command():
    """Out-of-limits command via EventBus -> safety.violation published."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(
        robot_model=model,
        event_bus=bus,
        safety_level="MODERATE",
    )
    validator.initialize()

    responses = []
    violations = []
    bus.subscribe("agent.response", lambda e: responses.append(e.payload))
    bus.subscribe("safety.violation", lambda e: violations.append(e.payload))

    bus.publish(Event(
        topic="agent.command",
        payload={
            "action": "move_joints",
            "trajectory": [[0.0, 0.0], [0.99, 0.0]],  # Violates j1 limit
            "robot_id": "test",
        },
        source="test",
        metadata={"request_id": "req4"},
    ))

    assert len(responses) == 1
    assert responses[0]["is_safe"] is False
    assert len(violations) == 1
    assert violations[0]["action"] == "BLOCKED"
    validator.stop()


def test_non_movement_command_ignored():
    """Non-movement commands (grasp) are not validated."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(
        robot_model=model,
        event_bus=bus,
        safety_level="MODERATE",
    )
    validator.initialize()

    responses = []
    bus.subscribe("agent.response", lambda e: responses.append(e.payload))

    bus.publish(Event(
        topic="agent.command",
        payload={"action": "grasp", "grasp_action": "close"},
        source="test",
        metadata={"request_id": "req5"},
    ))

    assert len(responses) == 0  # Not validated, no response
    validator.stop()


def test_velocity_limit_violation():
    """High velocity between waypoints -> semantic safety error."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(
        robot_model=model,
        event_bus=bus,
        safety_level="MODERATE",
    )
    validator.initialize()

    request = ValidationRequest(
        request_id="req6",
        robot_id="test",
        trajectory=[[0.0, 0.0], [1.0, 0.0]],
        duration_per_waypoint=[0.1, 0.1],  # 1.0 rad in 0.1s = 10 rad/s > 2.0*0.95
    )
    response = validator.validate(request)

    assert not response.is_safe
    assert any(v.layer == ValidationLayer.SEMANTIC_SAFETY for v in response.violations)
    validator.stop()
