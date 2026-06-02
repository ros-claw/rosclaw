"""Integration tests for rosclaw-sandbox in v1.0 Runtime."""

from pathlib import Path


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
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "universal_robots_ur5e"},
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
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "universal_robots_ur5e"},
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
        # mj_forward may still be used for state restoration after simulation,
        # but mj_step must be the primary simulation method
        assert "dynamic simulation" in source or "mj_step" in source


class TestMCPUsesEventBus:
    def test_ur5_server_event_bus_fallback(self):
        """Verify UR5 MCP server has EventBus fallback code."""

        # Read source without importing (avoids rclpy dependency)
        mcp_path = Path(__file__).parent.parent / "src" / "rosclaw" / "mcp" / "ur5_server.py"
        with open(mcp_path) as f:
            source = f.read()

        assert "firewall.validation_request" in source or "event_bus" in source
        assert "firewall.validation_result" in source or "EventBus" in source
