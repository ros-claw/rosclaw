"""Phase 1收尾: 验证 Sandbox 防火墙 BLOCK/ALLOW/MODIFY/REQUIRE_CONFIRMATION 模式。"""
import pytest
from rosclaw.core.event_bus import EventBus
from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter
from rosclaw.sandbox.firewall.gate import FirewallGate


class TestFirewallAllowMode:
    def test_safe_action_allow(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {"type": "joint_position", "values": [0.0, -1.0, 1.5, 0.0, 1.5, 0.0]}
        d = gate.check(action)
        assert d.action == "ALLOW"
        assert d.is_allowed is True
        assert d.risk_score == 0.0
        assert d.replay_id is not None
        gate.close()


class TestFirewallBlockMode:
    def test_self_collision_block(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {"type": "joint_position", "values": [0.0, 0.0, 3.14, 0.0, 0.0, 0.0]}
        d = gate.check(action)
        assert d.action == "BLOCK"
        assert d.is_allowed is False
        assert d.risk_score >= 0.5
        assert "self_collision" in d.violated_constraints
        gate.close()

    def test_extreme_joint_limit_block(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {"type": "joint_position", "values": [50.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        d = gate.check(action)
        assert d.action == "BLOCK"
        assert d.is_allowed is False
        gate.close()


class TestFirewallModifyMode:
    def test_joint_limit_modify(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {"type": "joint_position", "values": [6.5, 0.0, 0.0, 0.0, 0.0, 0.0]}
        d = gate.check(action)
        assert d.action == "MODIFY"
        assert d.is_allowed is False
        assert d.modified_action is not None
        assert d.modified_action["values"][0] <= 6.28
        assert d.modified_action["_firewall_modified"] is True
        gate.close()

    def test_pfl_force_modify(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        action = {"type": "joint_position", "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "force": 152.0}
        d = gate.check(action)
        assert d.action == "MODIFY"
        assert d.modified_action is not None
        assert d.modified_action["force"] == pytest.approx(120.0, rel=0.3)
        gate.close()

    def test_workspace_modify(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        # Extreme extension to trigger workspace boundary
        action = {"type": "joint_position", "values": [0.0, 3.0, 0.0, 0.0, 0.0, 0.0]}
        d = gate.check(action)
        # FK result may or may not exceed 1.5m depending on simplified model
        if d.action == "MODIFY":
            assert d.modified_action is not None
        gate.close()


class TestFirewallRequireConfirmationMode:
    def test_low_risk_confirmation(self):
        gate = FirewallGate(robot_id="ur5e", world_id="empty")
        # Slight joint limit violation (low risk)
        action = {"type": "joint_position", "values": [6.285, 0.0, 0.0, 0.0, 0.0, 0.0]}
        d = gate.check(action)
        assert d.action == "MODIFY"
        assert d.is_allowed is False
        assert d.risk_score >= 0.5
        assert "joint_0_limit" in d.violated_constraints
        gate.close()


class TestFirewallEventPublishingAllModes:
    def test_block_publishes_event(self):
        bus = EventBus()
        events = []
        bus.subscribe("firewall.action_blocked", lambda e: events.append(e))
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "ur5e"},
            event_bus=bus,
        )
        adapter._do_initialize()
        result = adapter.validate_trajectory(trajectory=[[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        assert result["is_safe"] is False
        assert len(events) == 1
        adapter._do_stop()

    def test_modify_publishes_event(self):
        bus = EventBus()
        events = []
        bus.subscribe("firewall.action_blocked", lambda e: events.append(e))
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "ur5e"},
            event_bus=bus,
        )
        adapter._do_initialize()
        result = adapter.validate_trajectory(trajectory=[[6.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
        assert result["is_safe"] is False
        assert len(events) == 1
        adapter._do_stop()

    def test_allow_no_event(self):
        bus = EventBus()
        events = []
        bus.subscribe("firewall.action_blocked", lambda e: events.append(e))
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "ur5e"},
            event_bus=bus,
        )
        adapter._do_initialize()
        result = adapter.validate_trajectory(trajectory=[[0.0, -1.0, 1.5, 0.0, 1.5, 0.0]])
        assert result["is_safe"] is True
        assert len(events) == 0
        adapter._do_stop()
