"""Tests for SandboxRuntimeAdapter."""

from unittest.mock import MagicMock, patch

import pytest

from rosclaw.core.event_bus import EventBus
from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter
from rosclaw.sense.config import SenseConfig
from rosclaw.sense.runtime import SenseRuntime


class TestSandboxRuntimeAdapterLifecycle:
    def test_initialize_import_error_reports_unavailable(self, caplog):
        import logging

        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        with (
            patch(
                "rosclaw.sandbox.sandbox_api.Sandbox.create", side_effect=ImportError("no module")
            ),
            caplog.at_level(logging.WARNING, logger="rosclaw.sandbox.runtime_adapter"),
        ):
            adapter.initialize()
        assert "Failed to create sandbox" in caplog.text
        assert adapter._sandbox_service is None
        assert adapter.health()["status"] == "unavailable"
        adapter.stop()

    def test_initialize_exception_reports_unavailable(self, caplog):
        import logging

        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        with (
            patch("rosclaw.sandbox.sandbox_api.Sandbox.create", side_effect=RuntimeError("boom")),
            caplog.at_level(logging.WARNING, logger="rosclaw.sandbox.runtime_adapter"),
        ):
            adapter.initialize()
        assert "Failed to create sandbox" in caplog.text
        assert adapter._sandbox_service is None
        assert adapter.health()["error"] == "boom"
        adapter.stop()

    def test_fixture_sandbox_requires_explicit_engine(self):
        bus = EventBus()
        config = {"engine": "fixture", "world_id": "empty", "robot_id": "test_bot"}
        adapter = SandboxRuntimeAdapter(config, event_bus=bus)
        adapter.initialize()

        assert adapter._sandbox_service is not None
        assert adapter.health()["status"] == "fixture"
        assert adapter.health()["trust_level"] == "SYNTHETIC"
        adapter.stop()

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
        assert "Sandbox reset" in caplog.text

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
        with patch(
            "rosclaw.sandbox.firewall.gate.FirewallGate", side_effect=RuntimeError("gate error")
        ):
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

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate") as mock_gate:
            instance = mock_gate.return_value
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

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate") as mock_gate:
            instance = mock_gate.return_value
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


class TestSandboxRuntimeAdapterSenseAware:
    @pytest.fixture
    def sense_runtime(self):
        bus = EventBus()
        cfg = SenseConfig(
            robot_id="g1_lab_01",
            collector="mock",
            update_hz=0.0,
            extra={"scenario": "kick_not_ready"},
        )
        runtime = SenseRuntime(cfg, event_bus=bus, robot_id="g1_lab_01")
        runtime.initialize()
        runtime.tick()
        yield runtime
        runtime.stop()

    def _make_adapter(self, sense_runtime=None):
        bus = EventBus()
        config = {"engine": "mujoco", "world_id": "empty", "robot_id": "test_bot"}
        runtime_wrapper = (
            type("FakeRuntime", (), {"sense": sense_runtime})() if sense_runtime else None
        )
        adapter = SandboxRuntimeAdapter(config, event_bus=bus, runtime=runtime_wrapper)
        adapter._sandbox_service = MagicMock()
        return adapter

    def test_validate_injects_body_sense_snapshot(self, sense_runtime):
        adapter = self._make_adapter(sense_runtime)
        captured_action = {}

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate") as mock_gate:
            instance = mock_gate.return_value
            decision = MagicMock()
            decision.is_allowed = True
            decision.risk_score = 0.1
            decision.reason = "safe"
            decision.predicted_collision = False
            decision.violated_constraints = []
            decision.replay_id = None
            instance.check.return_value = decision

            def capture_check(action):
                captured_action.update(action)
                return decision

            instance.check.side_effect = capture_check
            adapter.validate_trajectory([[0.0, 0.0, 0.0]])

        assert "body_sense_snapshot" in captured_action
        assert captured_action["body_sense_snapshot"]["overall_status"] == "not_ready"

    def test_validate_without_runtime_does_not_inject_snapshot(self):
        adapter = self._make_adapter(None)
        captured_action = {}

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate") as mock_gate:
            instance = mock_gate.return_value
            decision = MagicMock()
            decision.is_allowed = True
            decision.risk_score = 0.1
            decision.reason = "safe"
            decision.predicted_collision = False
            decision.violated_constraints = []
            decision.replay_id = None

            def capture_check(action):
                captured_action.update(action)
                return decision

            instance.check.side_effect = capture_check
            adapter.validate_trajectory([[0.0, 0.0, 0.0]])

        assert "body_sense_snapshot" not in captured_action

    def test_validate_adapter_failure_falls_back(self, sense_runtime, caplog):
        import logging

        adapter = self._make_adapter(sense_runtime)
        adapter._sandbox_context_adapter = MagicMock()
        adapter._sandbox_context_adapter.apply.side_effect = RuntimeError("adapter broken")

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate") as mock_gate:
            instance = mock_gate.return_value
            decision = MagicMock()
            decision.is_allowed = True
            decision.risk_score = 0.0
            decision.reason = "safe"
            decision.predicted_collision = False
            decision.violated_constraints = []
            decision.replay_id = None
            instance.check.return_value = decision
            with caplog.at_level(logging.WARNING, logger="rosclaw.sandbox.runtime_adapter"):
                result = adapter.validate_trajectory([[0.0, 0.0, 0.0]])

        assert result["is_safe"] is True
        assert "without body sense" in caplog.text or "adapter broken" in caplog.text
