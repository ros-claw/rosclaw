"""Tests for FirewallValidator (Sprint 3)."""

import pytest

from rosclaw.core.event_bus import EventBus, Event
from rosclaw.e_urdf.parser import RobotModel, JointSpec
from rosclaw.firewall.validator import (
    FirewallValidator,
    SafetyEnvelope,
    ValidationRequest,
    ValidationLayer,
    ValidationResponse,
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


# --- Additional coverage for remaining branches ---


def test_safety_envelope_lenient():
    """LENIENT uses 99% factor — wider than MODERATE."""
    model = _make_test_robot()
    env = SafetyEnvelope.from_robot_model(model, safety_level="LENIENT")
    len_lo, len_hi = env.joint_soft_limits[0]
    mod_env = SafetyEnvelope.from_robot_model(model, safety_level="MODERATE")
    mod_lo, mod_hi = mod_env.joint_soft_limits[0]
    assert abs(len_lo) > abs(mod_lo)
    assert abs(len_hi) > abs(mod_hi)


def test_safety_envelope_unknown_level_defaults():
    """Unknown safety level uses default factor 0.95."""
    model = _make_test_robot()
    env = SafetyEnvelope.from_robot_model(model, safety_level="UNKNOWN")
    assert env.safety_level == "UNKNOWN"
    mod_env = SafetyEnvelope.from_robot_model(model, safety_level="MODERATE")
    assert env.joint_soft_limits[0] == pytest.approx(mod_env.joint_soft_limits[0])


def test_safety_envelope_positive_only_limits():
    """Joint with only positive limits."""
    from rosclaw.e_urdf.parser import RobotModel, JointSpec
    model = RobotModel(name="pos_only")
    model.joints["j1"] = JointSpec(
        name="j1", joint_type="prismatic", parent="base", child="link1",
        limits={"lower": 0.0, "upper": 1.0, "velocity": 2.0, "effort": 10.0}
    )
    env = SafetyEnvelope.from_robot_model(model)
    lo, hi = env.joint_soft_limits[0]
    assert lo >= 0.0
    assert hi <= 1.0


def test_validation_response_violation_count():
    """ValidationResponse.violation_count property."""
    resp = ValidationResponse(
        request_id="r1",
        is_safe=False,
        layers_checked=[ValidationLayer.EURDF_SOFT_LIMITS],
        violations=[],
    )
    assert resp.violation_count == 0
    from rosclaw.firewall.validator import ViolationDetail
    resp.violations.append(ViolationDetail(
        layer=ValidationLayer.EURDF_SOFT_LIMITS,
        severity="critical",
        joint_index=0,
        description="test",
    ))
    assert resp.violation_count == 1


def test_initialize_without_mujoco():
    """Initialize with no MuJoCo model path."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(model, bus, mujoco_model_path=None)
    validator.initialize()
    assert validator._envelope is not None
    assert validator._mj_model is None
    validator.stop()


def test_initialize_with_bad_mujoco_path(caplog):
    """Bad MuJoCo path logs warning and continues."""
    import logging
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(model, bus, mujoco_model_path="/nonexistent.xml")
    with caplog.at_level(logging.WARNING, logger="rosclaw.firewall.validator"):
        validator.initialize()
    assert "MuJoCo unavailable" in caplog.text
    assert validator._mj_model is None
    validator.stop()


def test_start_publishes_status():
    """start() publishes firewall.status event."""
    model = _make_test_robot()
    bus = EventBus()
    received = []
    bus.subscribe("firewall.status", lambda e: received.append(e.payload))
    validator = FirewallValidator(model, bus)
    validator.initialize()
    validator.start()
    assert len(received) == 1
    assert received[0]["state"] == "running"
    assert received[0]["safety_level"] == "MODERATE"
    validator.stop()


def test_stop_unsubscribes():
    """stop() removes agent.command subscription."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(model, bus)
    validator.initialize()
    # Topic is normalized to rosclaw.* namespace
    assert bus.subscriber_count("rosclaw.agent.command") == 1
    validator.stop()
    assert bus.subscriber_count("rosclaw.agent.command") == 0


def test_check_eurdf_limits_no_envelope():
    """_check_eurdf_limits returns empty when envelope is None."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(model, bus)
    validator.initialize()
    validator._envelope = None
    req = ValidationRequest(request_id="r1", robot_id="bot", trajectory=[[0.0, 0.0]])
    response = validator.validate(req)
    assert response.is_safe is True
    validator.stop()


def test_validate_without_mujoco_layers():
    """validate() without MuJoCo does not include MUJOCO_COLLISION layer."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(model, bus, mujoco_model_path=None)
    validator.initialize()
    req = ValidationRequest(request_id="r1", robot_id="bot", trajectory=[[0.0, 0.0]])
    response = validator.validate(req)
    assert ValidationLayer.MUJOCO_COLLISION not in response.layers_checked
    assert response.simulation_duration_ms == 0.0
    validator.stop()


def test_semantic_safety_keepout_warning():
    """keepout_zones produce warnings."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(model, bus)
    validator.initialize()
    validator._envelope.keepout_zones = [{"name": "table_edge"}]
    req = ValidationRequest(request_id="r1", robot_id="bot", trajectory=[[0.0, 0.0]])
    response = validator.validate(req)
    assert len(response.warnings) >= 1
    assert "Keepout zone" in response.warnings[0]
    validator.stop()


def test_non_critical_violation_safe():
    """Semantic 'error' severity (not 'critical') still passes is_safe."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(model, bus)
    validator.initialize()
    # Force a velocity-only violation: severity="error" not "critical"
    # trajectory within e-URDF limits ([-0.95, 0.95]) but velocity exceeds limit
    req = ValidationRequest(
        request_id="r1",
        robot_id="bot",
        trajectory=[[0.0, 0.0], [0.5, 0.0]],  # within soft limits
        duration_per_waypoint=[0.0, 0.01],  # 0.5 rad in 0.01s = 50 rad/s > 1.9
    )
    response = validator.validate(req)
    # The velocity violation has severity "error", not "critical"
    # so is_safe should be True
    assert response.is_safe is True
    assert any(v.layer == ValidationLayer.SEMANTIC_SAFETY for v in response.violations)
    validator.stop()


def test_execute_trajectory_command():
    """execute_trajectory action is validated."""
    model = _make_test_robot()
    bus = EventBus()
    validator = FirewallValidator(model, bus)
    validator.initialize()
    responses = []
    bus.subscribe("agent.response", lambda e: responses.append(e.payload))
    bus.publish(Event(
        topic="agent.command",
        payload={
            "action": "execute_trajectory",
            "robot_id": "test",
            "trajectory": [[0.0, 0.0], [0.5, 0.3]],
        },
        metadata={"request_id": "req_exec"},
    ))
    assert len(responses) == 1
    assert responses[0]["request_id"] == "req_exec"
    validator.stop()
