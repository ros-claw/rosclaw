"""Integration tests for rosclaw-sandbox in v1.0 Runtime."""

import pytest


class TestSandboxRuntimeAdapter:
    def test_adapter_imports(self):
        """Verify SandboxRuntimeAdapter can be imported."""
        from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter
        assert SandboxRuntimeAdapter is not None

    def test_adapter_lifecycle(self):
        """Test SandboxRuntimeAdapter initialize/start/stop lifecycle."""
        from rosclaw.core.event_bus import EventBus
        from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter

        bus = EventBus()
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "test"},
            event_bus=bus,
        )
        adapter._do_initialize()
        assert adapter._sandbox_service is not None

        adapter._do_start()
        adapter._do_stop()

    def test_adapter_health(self):
        """Test health report."""
        from rosclaw.core.event_bus import EventBus
        from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter

        bus = EventBus()
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "test"},
            event_bus=bus,
        )
        adapter._do_initialize()

        health = adapter.health()
        assert health["status"] == "healthy"
        assert health["engine"] == "mujoco"
        assert health["world"] == "empty"
        assert "session_id" in health

        adapter._do_stop()

    def test_trajectory_validation(self):
        """Test dynamic trajectory validation."""
        from rosclaw.core.event_bus import EventBus
        from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter

        bus = EventBus()
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "universal_robots_ur5e"},
            event_bus=bus,
        )
        adapter._do_initialize()

        result = adapter.validate_trajectory(
            trajectory=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            safety_level="MODERATE",
        )

        assert "is_safe" in result
        assert "risk_score" in result
        adapter._do_stop()


class TestFirewallDynamicCollision:
    def test_mj_step_replaces_mj_forward(self):
        """Verify firewall validator uses mj_step for dynamic simulation."""
        from rosclaw.firewall.validator import FirewallValidator
        import inspect

        source = inspect.getsource(FirewallValidator._check_mujoco_collision)
        assert "mj_step" in source
        assert "mj_forward" not in source or "mj_forward(self._mj_model" not in source


class TestMCPUsesEventBus:
    def test_ur5_server_event_bus_fallback(self):
        """Verify UR5 MCP server has EventBus fallback code."""
        from rosclaw.mcp.ur5_server import UR5MCPServer
        import inspect

        source = inspect.getsource(UR5MCPServer._handle_move_joints)
        assert "event_bus" in source or "firewall" in source
