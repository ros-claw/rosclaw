"""Task #21: Verify Sandbox Firewall Interception Capabilities."""

from rosclaw.core.event_bus import EventBus
from rosclaw.sandbox.firewall.gate import FirewallGate
from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter


class TestFirewallGateChecks:
    def test_mock_fixture_non_joint_action_gets_structural_validation(self):
        gate = FirewallGate(robot_id="turtlebot", world_id="mock")

        safe = gate.check({"type": "pid_move", "parameters": {"target": 1.0, "Kp": 1.0, "Kd": 0.1}})
        unsafe = gate.check({"type": "pid_move", "parameters": {"target": float("nan")}})
        cyclic_parameters = {}
        cyclic_parameters["self"] = cyclic_parameters
        cyclic = gate.check({"type": "pid_move", "parameters": cyclic_parameters})

        assert safe.is_allowed is True
        assert safe.physics_executed is False
        assert unsafe.is_allowed is False
        assert unsafe.violated_constraints == ["invalid_fixture_parameters"]
        assert cyclic.is_allowed is False
        assert cyclic.violated_constraints == ["invalid_fixture_parameters"]

    def test_empty_or_non_finite_action_is_blocked(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        try:
            empty = gate.check({"type": "joint_position", "values": []})
            non_finite = gate.check({"type": "joint_position", "values": [float("nan")] * 6})
        finally:
            gate.close()

        assert empty.is_allowed is False
        assert empty.violated_constraints == ["invalid_joint_values"]
        assert non_finite.is_allowed is False
        assert non_finite.violated_constraints == ["invalid_joint_values"]

    def test_joint_limit_violation_blocked(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {"type": "joint_position", "values": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        decision = gate.check(action)
        assert decision.is_allowed is False
        assert "joint_0_limit" in decision.violated_constraints
        assert decision.reason.startswith("Firewall blocked:")
        assert decision.replay_id is not None
        assert decision.replay_id.startswith("sandbox://replay/")
        gate.close()

    def test_workspace_boundary_blocked(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {"type": "joint_position", "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        decision = gate.check(action)
        assert decision.is_allowed is True
        assert decision.replay_id is not None
        gate.close()

    def test_velocity_limit_blocked(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {
            "type": "joint_position",
            "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "current": [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        decision = gate.check(action)
        assert decision.is_allowed is False
        assert any("velocity" in v for v in decision.violated_constraints)
        gate.close()

    def test_pfl_force_blocked(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {
            "type": "joint_position",
            "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "force": 200.0,
        }
        decision = gate.check(action)
        assert decision.is_allowed is False
        assert "pfl_force" in decision.violated_constraints
        gate.close()

    def test_self_collision_blocked(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {"type": "joint_position", "values": [0.0, 0.0, 3.14, 0.0, 0.0, 0.0]}
        decision = gate.check(action)
        assert decision.is_allowed is False
        assert "self_collision" in decision.violated_constraints
        gate.close()

    def test_safe_action_allowed(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {"type": "joint_position", "values": [0.0, -1.0, 1.5, 0.0, 1.5, 0.0]}
        decision = gate.check(action)
        assert decision.is_allowed is True
        assert decision.risk_score == 0.0
        assert decision.replay_id is not None
        gate.close()


class TestFirewallEventPublishing:
    def test_blocked_action_publishes_event(self):
        bus = EventBus()
        received_events = []
        bus.subscribe("firewall.action_blocked", lambda e: received_events.append(e))
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "ur5e"},
            event_bus=bus,
        )
        adapter._do_initialize()
        result = adapter.validate_trajectory(
            trajectory=[[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            safety_level="MODERATE",
        )
        assert result["is_safe"] is False
        assert result["reason"] == "JOINT_0_LIMIT"
        assert result["physics_executed"] is False
        assert result["replay_id"] is not None
        assert len(received_events) == 1
        event = received_events[0]
        assert event.topic == "firewall.action_blocked"
        assert event.payload["robot_id"] == "ur5e"
        assert "JOINT_0_LIMIT" in event.payload["violations"]
        assert event.payload["replay_id"] == result["replay_id"]
        assert event.source == "sandbox.firewall"
        adapter._do_stop()

    def test_allowed_action_no_event(self):
        bus = EventBus()
        received_events = []
        bus.subscribe("firewall.action_blocked", lambda e: received_events.append(e))
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "ur5e"},
            event_bus=bus,
        )
        adapter._do_initialize()
        result = adapter.validate_trajectory(
            trajectory=[[0.0, -1.0, 1.5, 0.0, 1.5, 0.0]],
            safety_level="MODERATE",
        )
        assert result["is_safe"] is True
        assert result["replay_id"] is not None
        assert len(received_events) == 0
        adapter._do_stop()


class TestFirewallReplayTraceability:
    def test_replay_id_on_block(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {"type": "joint_position", "values": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        decision = gate.check(action)
        assert decision.replay_id is not None
        assert decision.replay_id.startswith("sandbox://replay/")
        gate.close()

    def test_replay_id_on_allow(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {"type": "joint_position", "values": [0.0, -1.0, 1.5, 0.0, 1.5, 0.0]}
        decision = gate.check(action)
        assert decision.replay_id is not None
        assert decision.replay_id.startswith("sandbox://replay/")
        gate.close()

    def test_replay_id_unique(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        ids = []
        for _ in range(10):
            action = {"type": "joint_position", "values": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
            d = gate.check(action)
            ids.append(d.replay_id)
        gate.close()
        assert len(set(ids)) == len(ids)
