"""Tests for SandboxRuntimeAdapter."""

from unittest.mock import MagicMock, patch

from rosclaw.core.event_bus import EventBus
from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter


class TestSandboxRuntimeAdapterLifecycle:
    def test_initialize_import_error_creates_stub(self, caplog):
        import logging
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        # Patch Sandbox.create to raise ImportError (caught by except ImportError)
        with patch("rosclaw.sandbox.sandbox_api.Sandbox.create", side_effect=ImportError("no module")):
            with caplog.at_level(logging.WARNING, logger="rosclaw.sandbox.runtime_adapter"):
                adapter.initialize()
        assert "Sandbox not available" in caplog.text
        assert adapter._sandbox_service is not None
        assert hasattr(adapter._sandbox_service, "session")
        adapter.stop()

    def test_initialize_exception_creates_stub(self, caplog):
        import logging
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        # Patch Sandbox.create to raise RuntimeError (caught by except Exception)
        with patch("rosclaw.sandbox.sandbox_api.Sandbox.create", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.WARNING, logger="rosclaw.sandbox.runtime_adapter"):
                adapter.initialize()
        assert "Failed to create sandbox" in caplog.text
        assert adapter._sandbox_service is not None
        adapter.stop()

    def test_stub_sandbox_has_methods(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        stub = adapter._create_stub_sandbox()
        assert hasattr(stub, "session")
        assert hasattr(stub, "reset")
        assert hasattr(stub, "close")
        # Should not raise
        stub.reset()
        stub.close()

    def test_start_calls_reset(self, caplog):
        import logging
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        mock_service = MagicMock()
        with patch("rosclaw.sandbox.sandbox_api.Sandbox.create", return_value=mock_service):
            adapter.initialize()
        with caplog.at_level(logging.INFO, logger="rosclaw.sandbox.runtime_adapter"):
            adapter.start()
        mock_service.reset.assert_called_once()
        assert "Sandbox reset and running" in caplog.text

    def test_stop_calls_close(self, caplog):
        import logging
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        mock_service = MagicMock()
        with patch("rosclaw.sandbox.sandbox_api.Sandbox.create", return_value=mock_service):
            adapter.initialize()
        with caplog.at_level(logging.INFO, logger="rosclaw.sandbox.runtime_adapter"):
            adapter.stop()
        mock_service.close.assert_called_once()
        assert "Sandbox closed" in caplog.text

    def test_stop_no_sandbox(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = None
        adapter.stop()  # Should not raise


class TestSandboxRuntimeAdapterValidateTrajectory:
    def test_validate_no_sandbox(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = None
        result = adapter.validate_trajectory([[0.0, 0.0, 0.0]])
        assert result["is_safe"] is False
        assert "not initialized" in result["reason"]

    def test_validate_empty_trajectory(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = MagicMock()
        result = adapter.validate_trajectory([])
        assert result["is_safe"] is True
        assert result["risk_score"] == 0.0

    def test_validate_exception_fallback(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = MagicMock()
        with patch("rosclaw.sandbox.firewall.gate.FirewallGate", side_effect=RuntimeError("gate error")):
            result = adapter.validate_trajectory([[0.0, 0.0, 0.0]])
        assert result["is_safe"] is False
        assert "Validation error" in result["reason"]

    def test_validate_publishes_blocked_event(self):
        bus = EventBus()
        received = []
        bus.subscribe("firewall.action_blocked", lambda e: received.append(e.payload))
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = MagicMock()

        mock_decision = MagicMock()
        mock_decision.is_allowed = False
        mock_decision.risk_score = 0.8
        mock_decision.reason = "collision"
        mock_decision.predicted_collision = True
        mock_decision.violated_constraints = ["joint_limit"]
        mock_decision.replay_id = "rep_1"

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate") as MockGate:
            instance = MockGate.return_value
            instance.check.return_value = mock_decision
            result = adapter.validate_trajectory([[0.0, 0.0, 0.0]])

        assert result["is_safe"] is False
        assert len(received) == 1
        assert received[0]["decision"] == "BLOCK"
        assert received[0]["robot_id"] == "test_bot"

    def test_validate_publishes_allowed_event(self):
        bus = EventBus()
        received = []
        bus.subscribe("firewall.action_allowed", lambda e: received.append(e.payload))
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = MagicMock()

        mock_decision = MagicMock()
        mock_decision.is_allowed = True
        mock_decision.risk_score = 0.1
        mock_decision.reason = "safe"
        mock_decision.predicted_collision = False
        mock_decision.violated_constraints = []
        mock_decision.replay_id = None

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate") as MockGate:
            instance = MockGate.return_value
            instance.check.return_value = mock_decision
            result = adapter.validate_trajectory([[0.0, 0.0, 0.0]])

        assert result["is_safe"] is True
        assert len(received) == 1
        assert received[0]["decision"] == "ALLOW"


class TestSandboxRuntimeAdapterSimulateStep:
    def test_simulate_step_no_sandbox(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = None
        result = adapter.simulate_step([0.0, 0.0, 0.0])
        assert result == {}

    def test_simulate_step_no_step_method(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = MagicMock()
        del adapter._sandbox_service.step
        result = adapter.simulate_step([0.0, 0.0, 0.0])
        assert result == {}

    def test_simulate_step_with_step(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = MagicMock()
        adapter._sandbox_service.step.return_value = {"qpos": [1.0, 2.0]}
        result = adapter.simulate_step([0.0, 0.0, 0.0])
        assert result == {"qpos": [1.0, 2.0]}

    def test_simulate_step_returns_none(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = MagicMock()
        adapter._sandbox_service.step.return_value = None
        result = adapter.simulate_step([0.0, 0.0, 0.0])
        assert result == {}


class TestSandboxRuntimeAdapterProperties:
    def test_has_physics_no_sandbox(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = None
        assert adapter.has_physics is False

    def test_has_physics_true(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = MagicMock()
        adapter._sandbox_service.has_physics = True
        assert adapter.has_physics is True

    def test_has_physics_false(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = MagicMock()
        adapter._sandbox_service.has_physics = False
        assert adapter.has_physics is False

    def test_health_with_sandbox(self):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "test_world", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = MagicMock()
        adapter._sandbox_service.session.session_id = "sess_123"
        health = adapter.health()
        assert health["status"] == "healthy"
        assert health["engine"] == "mujoco"
        assert health["world"] == "test_world"
        assert health["session_id"] == "sess_123"

    def test_health_without_sandbox(self):
        bus = EventBus()
        config = {"engine": "mock", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter._sandbox_service = None
        health = adapter.health()
        assert health["status"] == "unavailable"
        assert health["session_id"] is None
