"""Sandbox + e-URDF Integration Tests.

Tests that SandboxRuntimeAdapter can:
1. Load UR5e MJCF from e-URDF
2. Read safety_limits from RobotSafetyProfile
3. Apply firewall checks based on robot constraints
"""

import pytest

from rosclaw.core.event_bus import EventBus
from rosclaw.runtime import EURDFLoader, RobotRegistry
from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter


class TestSandboxEURDFIntegration:
    """Test Sandbox integration with e-URDF Physical DNA."""

    def test_sandbox_adapter_initializes_with_robot(self):
        """SandboxRuntimeAdapter should initialize with robot config from e-URDF."""
        event_bus = EventBus()
        adapter = SandboxRuntimeAdapter(
            config={
                "engine": "mujoco",
                "world_id": "tabletop",
                "robot_id": "ur5e",
            },
            event_bus=event_bus,
        )
        assert adapter._robot_id == "ur5e"
        assert adapter._engine_name == "mujoco"
        assert adapter._world_id == "tabletop"
        assert adapter._event_bus is event_bus

    def test_sandbox_adapter_lifecycle(self):
        """Sandbox should follow lifecycle: initialize -> start -> stop."""
        event_bus = EventBus()
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "ur5e"},
            event_bus=event_bus,
        )
        adapter.initialize()
        assert adapter.state.name == "READY"

        adapter.start()
        assert adapter.state.name == "RUNNING"

        adapter.stop()
        assert adapter.state.name == "STOPPED"

    def test_sandbox_health_report(self):
        """Sandbox health should report engine and world info."""
        event_bus = EventBus()
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "tabletop", "robot_id": "ur5e"},
            event_bus=event_bus,
        )
        adapter.initialize()
        health = adapter.health()

        assert health["engine"] == "mujoco"
        assert health["world"] == "tabletop"
        assert health["status"] in ("healthy", "unavailable")

    def test_eurdf_loader_finds_mjcf(self):
        """e-URDF loader should find MJCF file for UR5e."""
        loader = EURDFLoader()
        profile = loader.load("ur5e")
        sim = profile.simulation

        assert "mujoco" in sim.backends
        assert sim.backends["mujoco"]["model_file"] == "robot.mjcf.xml"

    def test_robot_safety_limits_for_sandbox(self):
        """Sandbox should be able to read robot safety limits."""
        reg = RobotRegistry()
        profile = reg.install("ur5e")
        safety = profile.safety

        # Joint soft limits
        assert "shoulder_pan_joint" in safety.joint_soft_limits
        pan = safety.joint_soft_limits["shoulder_pan_joint"]
        assert pan["lower"] == pytest.approx(-6.10, abs=0.01)
        assert pan["upper"] == pytest.approx(6.10, abs=0.01)

        # Force limits
        assert safety.pfl["max_tcp_force"] == 150.0
        assert safety.pfl["max_tcp_torque"] == 8.0

    def test_trajectory_validation_stub(self):
        """Trajectory validation should work even with stub sandbox."""
        event_bus = EventBus()
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "tabletop", "robot_id": "ur5e"},
            event_bus=event_bus,
        )
        adapter.initialize()

        # Empty trajectory should be safe
        result = adapter.validate_trajectory([], safety_level="MODERATE")
        assert result["is_safe"] is True
        assert result["risk_score"] == 0.0

    def test_robot_workspace_spherical(self):
        """UR5e workspace should be spherical for sandbox collision checks."""
        reg = RobotRegistry()
        profile = reg.install("ur5e")
        limits = profile.safety.safety_limits

        workspace = limits.get("workspace", {})
        assert workspace.get("type") == "sphere"
        assert workspace.get("radius") == pytest.approx(0.85, abs=0.01)
        assert len(workspace.get("center", [])) == 3

    def test_collision_pairs_defined(self):
        """UR5e should define collision pairs for sandbox checking."""
        reg = RobotRegistry()
        profile = reg.install("ur5e")
        limits = profile.safety.safety_limits

        pairs = limits.get("collision_pairs_to_check", [])
        assert len(pairs) > 0
        # Each pair should be a list of two link names
        for pair in pairs:
            assert len(pair) == 2
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)

    def test_power_limits_for_sandbox(self):
        """UR5e power limits should be available for sandbox."""
        reg = RobotRegistry()
        profile = reg.install("ur5e")
        limits = profile.safety.safety_limits

        power = limits.get("power_limits", {})
        assert power.get("max_power") == 350.0
        assert power.get("max_current_total") == 40.0

    def test_sandbox_adapter_with_eurdf_model(self):
        """Sandbox adapter can receive e-URDF model reference."""
        reg = RobotRegistry()
        profile = reg.install("ur5e")

        event_bus = EventBus()
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "tabletop", "robot_id": "ur5e"},
            event_bus=event_bus,
            e_urdf_model=profile.embodiment,
        )
        assert adapter._e_urdf_model is not None
        assert adapter._e_urdf_model.dof == 6
