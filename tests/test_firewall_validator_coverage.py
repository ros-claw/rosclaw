"""FirewallValidator coverage tests — fills gaps not covered by test_firewall_validator.py."""

import pytest

from rosclaw.core.event_bus import EventBus, Event
from rosclaw.e_urdf.parser import RobotModel, JointSpec
from rosclaw.firewall.validator import (
    FirewallValidator,
    SafetyEnvelope,
    ValidationRequest,
    ValidationLayer,
    ValidationResponse,
    ViolationDetail,
)


def _make_test_robot():
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


class TestSafetyEnvelopeCoverage:
    def test_lenient_wider_than_moderate(self):
        model = _make_test_robot()
        env = SafetyEnvelope.from_robot_model(model, safety_level="LENIENT")
        len_lo, len_hi = env.joint_soft_limits[0]
        mod_env = SafetyEnvelope.from_robot_model(model, safety_level="MODERATE")
        mod_lo, mod_hi = mod_env.joint_soft_limits[0]
        assert abs(len_lo) > abs(mod_lo)
        assert abs(len_hi) > abs(mod_hi)

    def test_unknown_level_defaults(self):
        model = _make_test_robot()
        env = SafetyEnvelope.from_robot_model(model, safety_level="UNKNOWN")
        assert env.safety_level == "UNKNOWN"
        mod_env = SafetyEnvelope.from_robot_model(model, safety_level="MODERATE")
        assert env.joint_soft_limits[0] == pytest.approx(mod_env.joint_soft_limits[0])

    def test_positive_only_limits(self):
        model = RobotModel(name="pos_only")
        model.joints["j1"] = JointSpec(
            name="j1", joint_type="prismatic", parent="base", child="link1",
            limits={"lower": 0.0, "upper": 1.0, "velocity": 2.0, "effort": 10.0}
        )
        env = SafetyEnvelope.from_robot_model(model)
        lo, hi = env.joint_soft_limits[0]
        assert lo >= 0.0
        assert hi <= 1.0

    def test_negative_only_limits(self):
        model = RobotModel(name="neg_only")
        model.joints["j1"] = JointSpec(
            name="j1", joint_type="revolute", parent="base", child="link1",
            limits={"lower": -1.0, "upper": 0.0, "velocity": 2.0, "effort": 10.0}
        )
        env = SafetyEnvelope.from_robot_model(model)
        lo, hi = env.joint_soft_limits[0]
        assert lo <= 0.0
        assert hi <= 0.0


class TestValidationResponse:
    def test_violation_count_property(self):
        resp = ValidationResponse(
            request_id="r1",
            is_safe=False,
            layers_checked=[ValidationLayer.EURDF_SOFT_LIMITS],
            violations=[],
        )
        assert resp.violation_count == 0
        resp.violations.append(ViolationDetail(
            layer=ValidationLayer.EURDF_SOFT_LIMITS,
            severity="critical",
            joint_index=0,
            description="test",
        ))
        assert resp.violation_count == 1


class TestFirewallValidatorLifecycle:
    def test_initialize_without_mujoco(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus, mujoco_model_path=None)
        validator.initialize()
        assert validator._envelope is not None
        assert validator._mj_model is None
        validator.stop()

    def test_initialize_bad_mujoco_path(self, caplog):
        import logging
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus, mujoco_model_path="/nonexistent.xml")
        with caplog.at_level(logging.WARNING, logger="rosclaw.firewall.validator"):
            validator.initialize()
        assert "MuJoCo unavailable" in caplog.text
        assert validator._mj_model is None
        validator.stop()

    def test_start_publishes_status(self):
        bus = EventBus()
        received = []
        bus.subscribe("firewall.status", lambda e: received.append(e.payload))
        model = _make_test_robot()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        validator.start()
        assert len(received) == 1
        assert received[0]["state"] == "running"
        validator.stop()

    def test_stop_unsubscribes(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        assert bus.subscriber_count("rosclaw.agent.command") == 1
        validator.stop()
        assert bus.subscriber_count("rosclaw.agent.command") == 0


class TestFirewallValidatorCheckLayers:
    def test_eurdf_limits_no_envelope(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        validator._envelope = None
        req = ValidationRequest(request_id="r1", robot_id="bot", trajectory=[[0.0, 0.0]])
        response = validator.validate(req)
        assert response.is_safe is True
        validator.stop()

    def test_validate_without_mujoco_layers(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus, mujoco_model_path=None)
        validator.initialize()
        req = ValidationRequest(request_id="r1", robot_id="bot", trajectory=[[0.0, 0.0]])
        response = validator.validate(req)
        assert ValidationLayer.MUJOCO_COLLISION not in response.layers_checked
        assert response.simulation_duration_ms == 0.0
        validator.stop()

    def test_semantic_keepout_warning(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        validator._envelope.keepout_zones = [{"name": "table_edge"}]
        req = ValidationRequest(request_id="r1", robot_id="bot", trajectory=[[0.0, 0.0]])
        response = validator.validate(req)
        assert len(response.warnings) >= 1
        assert "Keepout zone" in response.warnings[0]
        validator.stop()

    def test_non_critical_violation_safe(self):
        model = _make_test_robot()
        bus = EventBus()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        req = ValidationRequest(
            request_id="r1",
            robot_id="bot",
            trajectory=[[0.0, 0.0], [0.5, 0.0]],
            duration_per_waypoint=[0.0, 0.01],
        )
        response = validator.validate(req)
        # velocity violation has severity "error", not "critical"
        assert response.is_safe is True
        assert any(v.layer == ValidationLayer.SEMANTIC_SAFETY for v in response.violations)
        validator.stop()

    def test_execute_trajectory_command(self):
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

    def test_unknown_action_ignored(self):
        model = _make_test_robot()
        bus = EventBus()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        responses = []
        bus.subscribe("agent.response", lambda e: responses.append(e))
        bus.publish(Event(
            topic="agent.command",
            payload={"action": "grasp_object", "robot_id": "test"},
        ))
        assert len(responses) == 0
        validator.stop()

    def test_blocked_publishes_violation_and_action_blocked(self):
        model = _make_test_robot()
        bus = EventBus()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        responses = []
        violations = []
        blocked_events = []
        bus.subscribe("agent.response", lambda e: responses.append(e))
        bus.subscribe("safety.violation", lambda e: violations.append(e))
        bus.subscribe("firewall.action_blocked", lambda e: blocked_events.append(e))
        bus.publish(Event(
            topic="agent.command",
            payload={
                "action": "move_joints",
                "robot_id": "test",
                "trajectory": [[10.0, 10.0]],
            },
            metadata={"request_id": "req_block"},
        ))
        assert len(responses) == 1
        assert responses[0].payload["is_safe"] is False
        assert len(violations) == 1
        assert violations[0].payload["action"] == "BLOCKED"
        assert len(blocked_events) == 1
        assert blocked_events[0].payload["request_id"] == "req_block"
        assert blocked_events[0].priority.name == "CRITICAL"
        validator.stop()


class TestCheckEURDFLimitsEdgeCases:
    def test_joint_exceeds_upper(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        req = ValidationRequest(request_id="r1", robot_id="bot", trajectory=[[2.0, 0.0]])
        v = validator._check_eurdf_limits(req)
        assert len(v) == 1
        assert v[0].joint_index == 0
        assert v[0].actual_value == 2.0
        validator.stop()

    def test_joint_below_lower(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        req = ValidationRequest(request_id="r1", robot_id="bot", trajectory=[[-2.0, 0.0]])
        v = validator._check_eurdf_limits(req)
        assert len(v) == 1
        assert v[0].limit_value < 0
        validator.stop()

    def test_multiple_violations_multiple_waypoints(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        req = ValidationRequest(
            request_id="r1", robot_id="bot",
            trajectory=[[0.0, 0.0], [2.0, -1.0]],
        )
        v = validator._check_eurdf_limits(req)
        assert len(v) == 2
        assert "waypoint 1" in v[0].description
        validator.stop()

    def test_extra_joints_ignored(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        req = ValidationRequest(
            request_id="r1", robot_id="bot",
            trajectory=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        )
        v = validator._check_eurdf_limits(req)
        assert len(v) == 0
        validator.stop()


class TestCheckSemanticSafetyVelocity:
    def test_velocity_within_limit(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        req = ValidationRequest(
            request_id="r1", robot_id="bot",
            trajectory=[[0.0, 0.0], [0.1, 0.1]],
            duration_per_waypoint=[1.0, 1.0],
        )
        v, w = validator._check_semantic_safety(req)
        assert len(v) == 0
        validator.stop()

    def test_velocity_exceeds_limit(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        req = ValidationRequest(
            request_id="r1", robot_id="bot",
            trajectory=[[0.0, 0.0], [10.0, 0.0]],
            duration_per_waypoint=[1.0, 0.1],
        )
        v, w = validator._check_semantic_safety(req)
        assert len(v) >= 1
        assert v[0].layer == ValidationLayer.SEMANTIC_SAFETY
        assert v[0].severity == "error"
        validator.stop()

    def test_no_duration_per_waypoint(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        req = ValidationRequest(
            request_id="r1", robot_id="bot",
            trajectory=[[0.0, 0.0]],
        )
        v, w = validator._check_semantic_safety(req)
        assert len(v) == 0
        validator.stop()

    def test_zero_duration_skipped(self):
        bus = EventBus()
        model = _make_test_robot()
        validator = FirewallValidator(model, bus)
        validator.initialize()
        req = ValidationRequest(
            request_id="r1", robot_id="bot",
            trajectory=[[0.0, 0.0], [10.0, 0.0]],
            duration_per_waypoint=[0.0, 0.0],
        )
        v, w = validator._check_semantic_safety(req)
        # duration is 0, skipped (avoids division by zero)
        assert len(v) == 0
        validator.stop()
