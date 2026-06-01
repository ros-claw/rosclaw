"""Comprehensive coverage tests for rosclaw.core.runtime.

Targets:
- RuntimeConfig dataclass defaults and custom values
- Runtime.__init__ with external event_bus, with config.event_bus
- _do_initialize paths when modules are disabled
- _do_initialize ImportError fallback paths
- _do_start / _do_stop lifecycle
- _setup_internal_subscriptions and event handlers
- get_status with different module configurations
- _load_e_urdf success and failure paths
- Module property getters (_firewall, _memory, etc.)
- RobotRegistry methods (install/list_available/get/validate)
- _on_safety_violation, _on_agent_command, _on_emergency_stop event handlers
"""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin, LifecycleState
from rosclaw.core.runtime import Runtime, RuntimeConfig, _HowProxy, _MemoryProxy


async def _async_result(result):
    """Helper coroutine that returns a result for use in mocks."""
    return result


def _make_future(result):
    """Create a coroutine that returns the given result for mocking async methods."""
    return _async_result(result)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_runtime():
    """Return a fresh Runtime instance with all modules disabled."""
    cfg = RuntimeConfig(
        robot_id="test_bot",
        robot_zoo_path="/nonexistent/zoo/path",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=False,
        enable_swarm=False,
        enable_skill_manager=False,
        enable_knowledge=False,
        enable_how=False,
        enable_provider=False,
    )
    rt = Runtime(config=cfg)
    return rt


@pytest.fixture
def mock_event_bus():
    """Return a real EventBus for integration-style tests."""
    return EventBus()


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------

class TestRuntimeConfig:
    def test_default_values(self):
        cfg = RuntimeConfig()
        assert cfg.robot_id == "rosclaw_default"
        assert cfg.robot_model_path is None
        assert cfg.robot_zoo_path is None
        assert cfg.default_eurdf_robot == "ur5e"
        assert cfg.enable_firewall is True
        assert cfg.enable_memory is True
        assert cfg.enable_practice is True
        assert cfg.enable_swarm is False
        assert cfg.enable_skill_manager is True
        assert cfg.enable_knowledge is True
        assert cfg.enable_how is True
        assert cfg.enable_provider is True
        assert cfg.joint_dof == 6
        assert cfg.sampling_rate_hz == 1000
        assert cfg.safety_level == "MODERATE"
        assert cfg.timeline_output_dir == "./practice_data"
        assert cfg.enable_mcap is False
        assert cfg.seekdb_backend == "memory"
        assert cfg.seekdb_path == "./seekdb.sqlite"
        assert cfg.embodied_memory is None
        assert cfg.providers_dir is None
        assert cfg.gpu_sam3_endpoint is None
        assert cfg.gpu_vggt_endpoint is None
        assert cfg.gpu_minicpm_endpoint is None
        assert cfg.gpu_cosmos_endpoint is None
        assert cfg.event_bus is None

    def test_custom_values(self):
        bus = EventBus()
        cfg = RuntimeConfig(
            robot_id="custom_bot",
            robot_model_path="/path/to/model.urdf",
            robot_zoo_path="/path/to/zoo",
            default_eurdf_robot="panda",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_swarm=True,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_provider=False,
            joint_dof=7,
            sampling_rate_hz=500,
            safety_level="STRICT",
            timeline_output_dir="/tmp/practice",
            enable_mcap=True,
            seekdb_backend="sqlite",
            seekdb_path="/tmp/seek.db",
            embodied_memory=MagicMock(),
            providers_dir="/tmp/providers",
            gpu_sam3_endpoint="http://localhost:8001",
            gpu_vggt_endpoint="http://localhost:8002",
            gpu_minicpm_endpoint="http://localhost:8003",
            gpu_cosmos_endpoint="http://localhost:8004",
            event_bus=bus,
        )
        assert cfg.robot_id == "custom_bot"
        assert cfg.robot_model_path == "/path/to/model.urdf"
        assert cfg.robot_zoo_path == "/path/to/zoo"
        assert cfg.default_eurdf_robot == "panda"
        assert cfg.enable_firewall is False
        assert cfg.enable_memory is False
        assert cfg.enable_practice is False
        assert cfg.enable_swarm is True
        assert cfg.enable_skill_manager is False
        assert cfg.enable_knowledge is False
        assert cfg.enable_how is False
        assert cfg.enable_provider is False
        assert cfg.joint_dof == 7
        assert cfg.sampling_rate_hz == 500
        assert cfg.safety_level == "STRICT"
        assert cfg.timeline_output_dir == "/tmp/practice"
        assert cfg.enable_mcap is True
        assert cfg.seekdb_backend == "sqlite"
        assert cfg.seekdb_path == "/tmp/seek.db"
        assert cfg.embodied_memory is not None
        assert cfg.providers_dir == "/tmp/providers"
        assert cfg.gpu_sam3_endpoint == "http://localhost:8001"
        assert cfg.gpu_vggt_endpoint == "http://localhost:8002"
        assert cfg.gpu_minicpm_endpoint == "http://localhost:8003"
        assert cfg.gpu_cosmos_endpoint == "http://localhost:8004"
        assert cfg.event_bus is bus


# ---------------------------------------------------------------------------
# Runtime.__init__
# ---------------------------------------------------------------------------

class TestRuntimeInit:
    def test_init_no_args(self):
        rt = Runtime()
        assert rt.config.robot_id == "rosclaw_default"
        assert isinstance(rt.event_bus, EventBus)
        assert rt._firewall is None
        assert rt._memory is None
        assert rt._practice is None
        assert rt._swarm is None
        assert rt._skill_manager is None
        assert rt._knowledge is None
        assert rt._how is None
        assert rt._e_urdf is None
        assert rt._robot_profile is None
        assert rt._sandbox is None
        assert rt._episode_recorder is None
        assert rt._mcp_drivers == {}
        assert rt._provider_registry is None
        assert rt._capability_router is None
        assert rt._guard_pipeline is None
        assert rt._agent_runtime is None
        assert rt._modules == []

    def test_init_with_external_event_bus(self, mock_event_bus):
        rt = Runtime(event_bus=mock_event_bus)
        assert rt.event_bus is mock_event_bus

    def test_init_with_config_event_bus(self, mock_event_bus):
        cfg = RuntimeConfig(event_bus=mock_event_bus)
        rt = Runtime(config=cfg)
        assert rt.event_bus is mock_event_bus

    def test_init_external_event_bus_takes_precedence_over_config(self, mock_event_bus):
        cfg_bus = EventBus()
        cfg = RuntimeConfig(event_bus=cfg_bus)
        rt = Runtime(config=cfg, event_bus=mock_event_bus)
        assert rt.event_bus is mock_event_bus

    def test_init_creates_executor(self):
        rt = Runtime()
        assert rt._async_executor is not None
        assert rt._executor_shutdown is False
        assert isinstance(rt._module_lock, type(threading.RLock()))


# ---------------------------------------------------------------------------
# _do_initialize — disabled module paths
# ---------------------------------------------------------------------------

class TestRuntimeInitializeDisabledModules:
    def test_initialize_all_disabled(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        assert rt.state == LifecycleState.READY
        assert rt._firewall is None
        assert rt._memory is None
        assert rt._practice is None
        assert rt._swarm is None
        assert rt._skill_manager is None
        assert rt._knowledge is None
        assert rt._how is None
        assert rt._provider_registry is None
        assert rt._capability_router is None
        assert rt._guard_pipeline is None

    def test_initialize_firewall_disabled_no_e_urdf(self):
        cfg = RuntimeConfig(
            robot_zoo_path="/nonexistent/zoo",
            enable_firewall=True, enable_memory=False,
            enable_practice=False, enable_swarm=False,
            enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False)
        rt = Runtime(config=cfg)
        # _e_urdf is None so firewall should NOT be initialized
        rt._load_e_urdf = lambda: None  # prevent actual file loading
        rt.initialize()
        assert rt._firewall is None

    def test_initialize_memory_disabled(self):
        cfg = RuntimeConfig(
            robot_zoo_path="/nonexistent/zoo",
            enable_firewall=False, enable_memory=False,
            enable_practice=False, enable_swarm=False,
            enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False)
        rt = Runtime(config=cfg)
        rt.initialize()
        assert rt._memory is None

    def test_initialize_practice_disabled(self):
        cfg = RuntimeConfig(
            robot_zoo_path="/nonexistent/zoo",
            enable_firewall=False, enable_memory=False,
            enable_practice=False, enable_swarm=False,
            enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False)
        rt = Runtime(config=cfg)
        rt.initialize()
        assert rt._practice is None
        assert rt._episode_recorder is None

    def test_initialize_swarm_disabled(self):
        cfg = RuntimeConfig(
            robot_zoo_path="/nonexistent/zoo",
            enable_firewall=False, enable_memory=False,
            enable_practice=False, enable_swarm=False,
            enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False)
        rt = Runtime(config=cfg)
        rt.initialize()
        assert rt._swarm is None

    def test_initialize_skill_manager_disabled(self):
        cfg = RuntimeConfig(
            robot_zoo_path="/nonexistent/zoo",
            enable_firewall=False, enable_memory=False,
            enable_practice=False, enable_swarm=False,
            enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False)
        rt = Runtime(config=cfg)
        rt.initialize()
        assert rt._skill_manager is None

    def test_initialize_knowledge_disabled(self):
        cfg = RuntimeConfig(
            robot_zoo_path="/nonexistent/zoo",
            enable_firewall=False, enable_memory=False,
            enable_practice=False, enable_swarm=False,
            enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False)
        rt = Runtime(config=cfg)
        rt.initialize()
        assert rt._knowledge is None

    def test_initialize_how_disabled(self):
        cfg = RuntimeConfig(
            robot_zoo_path="/nonexistent/zoo",
            enable_firewall=False, enable_memory=False,
            enable_practice=False, enable_swarm=False,
            enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False)
        rt = Runtime(config=cfg)
        rt.initialize()
        assert rt._how is None

    def test_initialize_provider_disabled(self):
        cfg = RuntimeConfig(
            robot_zoo_path="/nonexistent/zoo",
            enable_firewall=False, enable_memory=False,
            enable_practice=False, enable_swarm=False,
            enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False)
        rt = Runtime(config=cfg)
        rt.initialize()
        assert rt._provider_registry is None
        assert rt._capability_router is None
        assert rt._guard_pipeline is None


# ---------------------------------------------------------------------------
# _do_initialize — ImportError fallback paths
# ---------------------------------------------------------------------------

class TestRuntimeInitializeImportErrors:
    @patch("rosclaw.core.runtime.ProviderRegistry", None)
    @patch("rosclaw.core.runtime.CapabilityRouter", None)
    @patch("rosclaw.core.runtime.GuardPipeline", None)
    def test_provider_import_error_at_module_level(self):
        """ProviderRegistry etc. are None at module level when import fails."""
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=False, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=True,
        )
        rt = Runtime(config=cfg)
        rt.initialize()
        assert rt._provider_registry is None
        assert rt._capability_router is None
        assert rt._guard_pipeline is None

    @patch("rosclaw.core.runtime.ProviderRegistry", side_effect=ImportError)
    def test_provider_registry_raises_on_instantiation(self, _mock_reg):
        """ProviderRegistry class exists but instantiation raises."""
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=False, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=True,
        )
        rt = Runtime(config=cfg)
        rt.initialize()
        # Provider layer catches broad Exception, logs, continues
        assert rt._provider_registry is None

    @patch("rosclaw.core.runtime.ProviderRegistry")
    @patch("rosclaw.core.runtime.CapabilityRouter")
    @patch("rosclaw.core.runtime.GuardPipeline")
    @patch("rosclaw.core.runtime.SchemaGuard")
    @patch("rosclaw.core.runtime.ActionGuard")
    def test_provider_builtin_registration_raises(self, _mock_ag, _mock_sg, mock_gp, mock_cr, mock_pr):
        """Built-in provider registration raises but is caught."""
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=False, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=True,
        )
        rt = Runtime(config=cfg)
        mock_registry = MagicMock()
        mock_registry.register = MagicMock()
        mock_registry.set_provider_health = MagicMock()
        mock_pr.return_value = mock_registry
        mock_gp.return_value = MagicMock()
        mock_cr.return_value = MagicMock()

        # Make _register_builtin_providers raise by patching it
        rt._register_builtin_providers
        rt._register_builtin_providers = MagicMock(side_effect=RuntimeError("boom"))
        rt.initialize()
        rt._register_builtin_providers.assert_called_once()

    @patch("rosclaw.core.runtime.ProviderRegistry")
    @patch("rosclaw.core.runtime.CapabilityRouter")
    @patch("rosclaw.core.runtime.GuardPipeline")
    @patch("rosclaw.core.runtime.SchemaGuard")
    @patch("rosclaw.core.runtime.ActionGuard")
    def test_provider_robot_capability_registration_raises(self, _mock_ag, _mock_sg, mock_gp, mock_cr, mock_pr):
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=False, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=True,
        )
        rt = Runtime(config=cfg)
        mock_registry = MagicMock()
        mock_registry.register = MagicMock()
        mock_registry.set_provider_health = MagicMock()
        mock_pr.return_value = mock_registry
        mock_gp.return_value = MagicMock()
        mock_cr.return_value = MagicMock()

        rt._register_builtin_providers = MagicMock()
        rt._register_robot_capabilities = MagicMock(side_effect=RuntimeError("robot cap boom"))
        rt._load_external_providers = MagicMock()
        rt.initialize()
        rt._register_robot_capabilities.assert_called_once()

    @patch("rosclaw.core.runtime.ProviderRegistry")
    @patch("rosclaw.core.runtime.CapabilityRouter")
    @patch("rosclaw.core.runtime.GuardPipeline")
    @patch("rosclaw.core.runtime.SchemaGuard")
    @patch("rosclaw.core.runtime.ActionGuard")
    def test_provider_external_loading_raises(self, _mock_ag, _mock_sg, mock_gp, mock_cr, mock_pr):
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=False, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=True,
        )
        rt = Runtime(config=cfg)
        mock_registry = MagicMock()
        mock_pr.return_value = mock_registry
        mock_gp.return_value = MagicMock()
        mock_cr.return_value = MagicMock()

        rt._register_builtin_providers = MagicMock()
        rt._register_robot_capabilities = MagicMock()
        rt._load_external_providers = MagicMock(side_effect=RuntimeError("ext boom"))
        rt.initialize()
        rt._load_external_providers.assert_called_once()

    @patch("rosclaw.core.runtime.ProviderRegistry")
    @patch("rosclaw.core.runtime.CapabilityRouter")
    @patch("rosclaw.core.runtime.GuardPipeline")
    @patch("rosclaw.core.runtime.SchemaGuard")
    @patch("rosclaw.core.runtime.ActionGuard")
    def test_provider_layer_setup_success(self, _mock_ag, _mock_sg, mock_gp, mock_cr, mock_pr):
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=False, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=True,
        )
        rt = Runtime(config=cfg)
        mock_registry = MagicMock()
        mock_pr.return_value = mock_registry
        mock_gp.return_value = MagicMock()
        mock_cr.return_value = MagicMock()

        rt._register_builtin_providers = MagicMock()
        rt._register_robot_capabilities = MagicMock()
        rt._load_external_providers = MagicMock()
        rt.initialize()
        assert rt._provider_registry is mock_registry
        assert rt._capability_router is mock_cr.return_value
        assert rt._guard_pipeline is mock_gp.return_value


# ---------------------------------------------------------------------------
# _do_start / _do_stop lifecycle
# ---------------------------------------------------------------------------

class TestRuntimeLifecycle:
    def test_start_stop_with_no_modules(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        assert rt.state == LifecycleState.READY
        rt.start()
        assert rt.state == LifecycleState.RUNNING
        rt.stop()
        assert rt.state == LifecycleState.STOPPED

    def test_start_skips_modules_not_ready(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        # Inject a mock module that is NOT ready
        mock_mod = MagicMock(spec=LifecycleMixin)
        mock_mod.is_ready = False
        rt._modules.append(mock_mod)
        rt.start()
        mock_mod.start.assert_not_called()

    def test_start_calls_modules_that_are_ready(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_mod = MagicMock(spec=LifecycleMixin)
        mock_mod.is_ready = True
        rt._modules.append(mock_mod)
        rt.start()
        mock_mod.start.assert_called_once()

    def test_stop_calls_modules_in_reverse_order(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        order = []

        class TrackingModule(LifecycleMixin):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def _do_stop(self):
                order.append(self.name)

        mod1 = TrackingModule("first")
        mod1._lifecycle_state = LifecycleState.RUNNING
        mod2 = TrackingModule("second")
        mod2._lifecycle_state = LifecycleState.RUNNING
        rt._modules.extend([mod1, mod2])
        rt.stop()
        assert order == ["second", "first"]

    def test_stop_shuts_down_executor(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        assert rt._executor_shutdown is False
        rt.stop()
        assert rt._executor_shutdown is True

    def test_stop_idempotent_executor_shutdown(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        rt.stop()
        # Second stop should not raise
        rt.stop()
        assert rt._executor_shutdown is True

    def test_start_publishes_runtime_status_event(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        events = []
        rt.event_bus.subscribe("runtime.status", events.append)
        rt.start()
        assert len(events) == 1
        assert events[0].payload["state"] == "running"
        assert events[0].payload["robot_id"] == "test_bot"
        assert events[0].priority == EventPriority.HIGH

    def test_stop_publishes_shutting_down_event(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        events = []
        rt.event_bus.subscribe("runtime.status", events.append)
        rt.stop()
        status_events = [e for e in events if e.payload.get("state") == "shutting_down"]
        assert len(status_events) == 1
        assert status_events[0].priority == EventPriority.CRITICAL


# ---------------------------------------------------------------------------
# _setup_internal_subscriptions
# ---------------------------------------------------------------------------

class TestInternalSubscriptions:
    def test_subscriptions_are_set_up(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        topics = rt.event_bus.topics
        # Topics are normalized by EventBus to v1.0 standard namespace
        assert "rosclaw.safety.violation" in topics
        assert "rosclaw.agent.command" in topics
        assert "rosclaw.robot.emergency_stop" in topics
        assert "rosclaw.sandbox.action.blocked" in topics  # firewall.action_blocked normalized
        assert "rosclaw.sandbox.episode.failed" in topics
        assert "rosclaw.sandbox.action.blocked" in topics
        assert "rosclaw.runtime.execution.failed" in topics
        assert "rosclaw.critic.judgment" in topics
        assert "rosclaw.agent.capability.request" in topics


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

class TestEventHandlers:
    def test_on_safety_violation_publishes_emergency_stop(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        events = []
        rt.event_bus.subscribe("robot.emergency_stop", events.append)

        evt = Event(
            topic="safety.violation",
            payload={"reason": "joint_limit_exceeded"},
            source="firewall",
        )
        rt._on_safety_violation(evt)
        assert len(events) == 1
        assert events[0].payload["reason"] == {"reason": "joint_limit_exceeded"}
        assert events[0].priority == EventPriority.CRITICAL

    def test_on_agent_command_logs(self, fresh_runtime, caplog):
        import logging
        rt = fresh_runtime
        rt.initialize()
        evt = Event(
            topic="agent.command",
            payload={"action": "move_to", "target": [0.5, 0.2, 0.1]},
            source="agent",
        )
        with caplog.at_level(logging.INFO, logger="rosclaw.core.runtime"):
            rt._on_agent_command(evt)
        assert "move_to" in caplog.text

    def test_on_emergency_stop_with_drivers(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_driver = MagicMock()
        mock_driver.emergency_stop = MagicMock()
        rt.register_driver("arm", mock_driver)

        evt = Event(topic="robot.emergency_stop", payload={}, source="runtime")
        rt._on_emergency_stop(evt)
        mock_driver.emergency_stop.assert_called_once()

    def test_on_emergency_stop_no_emergency_stop_attr(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_driver = MagicMock()  # no emergency_stop attr
        rt.register_driver("arm", mock_driver)

        evt = Event(topic="robot.emergency_stop", payload={}, source="runtime")
        # Should not raise
        rt._on_emergency_stop(evt)

    def test_on_emergency_stop_no_drivers(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        evt = Event(topic="robot.emergency_stop", payload={}, source="runtime")
        rt._on_emergency_stop(evt)
        # No drivers, just logs — no exception

    def test_on_firewall_action_blocked_how_none(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        evt = Event(
            topic="firewall.action_blocked",
            payload={"request_id": "r1", "violations": []},
            source="firewall",
        )
        # _how is None, should return early
        rt._on_firewall_action_blocked(evt)

    def test_on_critic_judgment_memory_none(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        evt = Event(
            topic="rosclaw.critic.judgment",
            payload={"episode_id": "ep1", "status": "SUCCESS", "reward": 1.0},
            source="critic",
        )
        # _memory is None, returns early
        rt._on_critic_judgment(evt)

    def test_on_critic_judgment_with_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_memory = MagicMock()
        mock_memory.store_experience = MagicMock()
        rt._memory = mock_memory

        evt = Event(
            topic="rosclaw.critic.judgment",
            payload={
                "episode_id": "ep1",
                "status": "SUCCESS",
                "reward": 1.0,
                "reason": "good_job",
                "context": {
                    "outcome": {"skill_name": "pick"},
                    "instruction": "pick the red cup",
                },
            },
            source="critic",
        )
        rt._on_critic_judgment(evt)
        mock_memory.store_experience.assert_called_once()
        call_kwargs = mock_memory.store_experience.call_args.kwargs
        assert call_kwargs["event_id"] == "ep1"
        assert call_kwargs["event_type"] == "praxis"
        assert call_kwargs["outcome"] == "success"
        assert "pick" in call_kwargs["tags"]
        assert call_kwargs["metadata"]["auto_synced"] is True

    def test_on_capability_request_router_none(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        evt = Event(
            topic="agent.capability.request",
            payload={"request_id": "r1", "capability": "vlm.test", "inputs": {}},
            source="agent",
        )
        # _capability_router is None, returns early
        rt._on_capability_request(evt)

    def test_on_capability_request_router_raises(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_router = MagicMock()
        mock_router.invoke = MagicMock(side_effect=RuntimeError("router fail"))
        rt._capability_router = mock_router

        events = []
        rt.event_bus.subscribe("agent.capability.response", events.append)

        evt = Event(
            topic="agent.capability.request",
            payload={"request_id": "r1", "capability": "vlm.test", "inputs": {}},
            source="agent",
        )
        rt._on_capability_request(evt)
        assert len(events) == 1
        assert events[0].payload["result"]["status"] == "error"

    def test_on_sandbox_episode_failed_how_none(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        evt = Event(
            topic="rosclaw.sandbox.episode.failed",
            payload={"failure_type": "collision", "request_id": "r1"},
            source="sandbox",
        )
        rt._on_sandbox_episode_failed(evt)

    def test_on_sandbox_action_blocked_how_none(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        evt = Event(
            topic="rosclaw.sandbox.action.blocked",
            payload={"reason": "unsafe", "request_id": "r1"},
            source="sandbox",
        )
        rt._on_sandbox_action_blocked(evt)

    def test_on_runtime_execution_failed_how_none(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        evt = Event(
            topic="rosclaw.runtime.execution.failed",
            payload={"error_type": "timeout", "request_id": "r1"},
            source="runtime",
        )
        rt._on_runtime_execution_failed(evt)

    def test_on_provider_event(self, fresh_runtime, caplog):
        import logging
        rt = fresh_runtime
        rt.initialize()
        evt = Event(
            topic="provider_registered",
            payload={"name": "mock_vlm"},
            source="provider",
        )
        with caplog.at_level(logging.INFO, logger="rosclaw.core.runtime"):
            rt._on_provider_event(evt)
        assert "provider_registered" in caplog.text

    def test_on_provider_health_changed(self, fresh_runtime, caplog):
        import logging
        rt = fresh_runtime
        rt.initialize()
        evt = Event(
            topic="provider_health_changed",
            payload={"provider": "mock_vlm", "ok": True, "reason": "ping_ok"},
            source="provider",
        )
        with caplog.at_level(logging.INFO, logger="rosclaw.core.runtime"):
            rt._on_provider_health_changed(evt)
        assert "mock_vlm" in caplog.text
        assert "healthy" in caplog.text

    def test_on_provider_health_changed_unhealthy(self, fresh_runtime, caplog):
        import logging
        rt = fresh_runtime
        rt.initialize()
        evt = Event(
            topic="provider_health_changed",
            payload={"provider": "mock_vlm", "ok": False, "reason": "timeout"},
            source="provider",
        )
        with caplog.at_level(logging.INFO, logger="rosclaw.core.runtime"):
            rt._on_provider_health_changed(evt)
        assert "unhealthy" in caplog.text


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_get_status_basic(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        status = rt.get_status()
        assert status["robot_id"] == "test_bot"
        assert status["runtime_state"] == "READY"
        assert "event_bus" in status
        assert "modules" in status
        assert status["modules"]["firewall"] is False
        assert status["modules"]["memory"] is False
        assert status["modules"]["practice"] is False
        assert status["modules"]["swarm"] is False
        assert status["modules"]["skill_manager"] is False
        assert status["modules"]["e_urdf"] is False
        assert status["modules"]["provider_layer"] is False
        assert status["embodied_memory"]["attached"] is False
        assert status["embodied_memory"]["has_world_objects"] is False
        assert status["drivers"] == []

    def test_status_property(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        assert rt.status == rt.get_status()

    def test_get_status_with_modules(self):
        cfg = RuntimeConfig(
            robot_id="mod_bot",
            enable_firewall=False, enable_memory=False, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False,
        )
        rt = Runtime(config=cfg)
        rt.initialize()
        rt._firewall = MagicMock()
        rt._memory = MagicMock()
        rt._practice = MagicMock()
        status = rt.get_status()
        assert status["modules"]["firewall"] is True
        assert status["modules"]["memory"] is True
        assert status["modules"]["practice"] is True

    def test_get_status_with_driver(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.register_driver("arm", MagicMock())
        status = rt.get_status()
        assert "arm" in status["drivers"]


# ---------------------------------------------------------------------------
# _load_e_urdf
# ---------------------------------------------------------------------------

class TestLoadEURDF:
    def test_load_from_model_path(self, tmp_path):
        urdf_file = tmp_path / "robot.urdf"
        urdf_file.write_text("<robot name='test'></robot>")

        cfg = RuntimeConfig(
            robot_model_path=str(urdf_file),
            enable_firewall=False, enable_memory=False, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False,
        )
        rt = Runtime(config=cfg)
        with patch("rosclaw.e_urdf.parser.EURDFParser") as mock_parser:
            mock_instance = MagicMock()
            mock_parser.return_value = mock_instance
            rt.initialize()
            mock_parser.assert_called_once_with(str(urdf_file))
            assert rt._e_urdf is mock_instance

    def test_load_from_zoo_success(self, tmp_path):
        zoo = tmp_path / "e-urdf-zoo"
        robot_dir = zoo / "ur5e"
        robot_dir.mkdir(parents=True)
        (robot_dir / "robot.eurdf.yaml").write_text(
            "robot_id: ur5e\nname: UR5e\nvendor: Universal Robots\n"
            "version: '1.0'\ndescription: test\ndof: 6\nlinks: []\njoints: []\n"
        )
        (robot_dir / "safety.yaml").write_text("safety_level: MODERATE\n")
        (robot_dir / "semantic.yaml").write_text("semantic_version: '1.0'\n")
        (robot_dir / "capabilities.yaml").write_text("capabilities: []\n")
        (robot_dir / "benchmark.yaml").write_text("kinematic_benchmarks: []\n")
        (robot_dir / "robot.urdf").write_text("<robot name='ur5e'></robot>")

        cfg = RuntimeConfig(
            robot_zoo_path=str(zoo),
            default_eurdf_robot="ur5e",
            enable_firewall=False, enable_memory=False, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False,
        )
        rt = Runtime(config=cfg)
        with patch("rosclaw.e_urdf.parser.EURDFParser") as mock_parser:
            mock_instance = MagicMock()
            mock_parser.return_value = mock_instance
            rt.initialize()
            mock_parser.assert_called_once()
            assert rt._e_urdf is mock_instance
            assert rt._robot_profile is not None

    def test_load_from_zoo_mjcf_fallback(self, tmp_path):
        zoo = tmp_path / "e-urdf-zoo"
        robot_dir = zoo / "ur5e"
        robot_dir.mkdir(parents=True)
        (robot_dir / "robot.eurdf.yaml").write_text(
            "robot_id: ur5e\nname: UR5e\nvendor: Universal Robots\n"
            "version: '1.0'\ndescription: test\ndof: 6\nlinks: []\njoints: []\n"
        )
        (robot_dir / "safety.yaml").write_text("safety_level: MODERATE\n")
        (robot_dir / "semantic.yaml").write_text("semantic_version: '1.0'\n")
        (robot_dir / "capabilities.yaml").write_text("capabilities: []\n")
        (robot_dir / "benchmark.yaml").write_text("kinematic_benchmarks: []\n")
        (robot_dir / "robot.mjcf.xml").write_text("<mujoco></mujoco>")

        cfg = RuntimeConfig(
            robot_zoo_path=str(zoo),
            default_eurdf_robot="ur5e",
            enable_firewall=False, enable_memory=False, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False,
        )
        rt = Runtime(config=cfg)
        with patch("rosclaw.e_urdf.parser.EURDFParser") as mock_parser:
            mock_instance = MagicMock()
            mock_parser.return_value = mock_instance
            rt.initialize()
            mock_parser.assert_called_once()
            assert rt._e_urdf is mock_instance

    def test_load_from_zoo_failure(self, tmp_path):
        zoo = tmp_path / "e-urdf-zoo"
        # Zoo exists but robot doesn't
        zoo.mkdir(parents=True)

        cfg = RuntimeConfig(
            robot_zoo_path=str(zoo),
            default_eurdf_robot="nonexistent",
            enable_firewall=False, enable_memory=False, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False,
        )
        rt = Runtime(config=cfg)
        rt.initialize()
        assert rt._e_urdf is None
        assert rt._robot_profile is None


class TestModuleProperties:
    def test_firewall_property(self):
        rt = Runtime()
        assert rt.firewall is None
        mock_fw = MagicMock()
        rt._firewall = mock_fw
        assert rt.firewall is mock_fw

    def test_memory_property_none(self):
        rt = Runtime()
        assert rt.memory is None

    def test_memory_property_returns_proxy(self):
        rt = Runtime()
        mock_mem = MagicMock()
        rt._memory = mock_mem
        mem = rt.memory
        assert isinstance(mem, _MemoryProxy)

    def test_practice_property(self):
        rt = Runtime()
        assert rt.practice is None
        mock_practice = MagicMock()
        rt._practice = mock_practice
        assert rt.practice is mock_practice

    def test_swarm_property(self):
        rt = Runtime()
        assert rt.swarm is None
        mock_swarm = MagicMock()
        rt._swarm = mock_swarm
        assert rt.swarm is mock_swarm

    def test_skill_manager_property(self):
        rt = Runtime()
        assert rt.skill_manager is None
        mock_sm = MagicMock()
        rt._skill_manager = mock_sm
        assert rt.skill_manager is mock_sm

    def test_knowledge_property(self):
        rt = Runtime()
        assert rt.knowledge is None
        mock_know = MagicMock()
        rt._knowledge = mock_know
        assert rt.knowledge is mock_know

    def test_how_property_none(self):
        rt = Runtime()
        assert rt.how is None

    def test_how_property_returns_proxy(self):
        rt = Runtime()
        mock_how = MagicMock()
        rt._how = mock_how
        how = rt.how
        assert isinstance(how, _HowProxy)

    def test_e_urdf_property(self):
        rt = Runtime()
        assert rt.e_urdf is None
        mock_eurdf = MagicMock()
        rt._e_urdf = mock_eurdf
        assert rt.e_urdf is mock_eurdf

    def test_provider_registry_property(self):
        rt = Runtime()
        assert rt.provider_registry is None
        mock_reg = MagicMock()
        rt._provider_registry = mock_reg
        assert rt.provider_registry is mock_reg

    def test_capability_router_property(self):
        rt = Runtime()
        assert rt.capability_router is None
        mock_router = MagicMock()
        rt._capability_router = mock_router
        assert rt.capability_router is mock_router

    def test_guard_pipeline_property(self):
        rt = Runtime()
        assert rt.guard_pipeline is None
        mock_gp = MagicMock()
        rt._guard_pipeline = mock_gp
        assert rt.guard_pipeline is mock_gp


# ---------------------------------------------------------------------------
# RobotRegistry methods
# ---------------------------------------------------------------------------

class TestRobotRegistry:
    def test_install_and_get(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import RobotRegistry

        zoo = tmp_path / "zoo"
        robot_dir = zoo / "ur5e"
        robot_dir.mkdir(parents=True)
        (robot_dir / "robot.eurdf.yaml").write_text(
            "robot_id: ur5e_canonical\nname: UR5e\nvendor: UR\n"
            "version: '1.0'\ndescription: test\ndof: 6\nlinks: []\njoints: []\n"
        )
        (robot_dir / "safety.yaml").write_text("safety_level: MODERATE\n")
        (robot_dir / "semantic.yaml").write_text("semantic_version: '1.0'\n")
        (robot_dir / "capabilities.yaml").write_text("capabilities: []\n")
        (robot_dir / "benchmark.yaml").write_text("kinematic_benchmarks: []\n")

        reg = RobotRegistry(loader=MagicMock())
        # Patch the loader with a real one using our tmp zoo
        from rosclaw.runtime.eurdf_loader import EURDFLoader
        reg.loader = EURDFLoader(str(zoo))

        profile = reg.install("ur5e")
        assert profile is not None
        assert profile.robot_id == "ur5e_canonical"

        # Stored under both keys
        assert reg.get("ur5e") is not None
        assert reg.get("ur5e_canonical") is not None

    def test_get_auto_install_failure(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import RobotRegistry, EURDFLoader

        zoo = tmp_path / "zoo"
        zoo.mkdir(parents=True)
        reg = RobotRegistry(loader=EURDFLoader(str(zoo)))
        assert reg.get("nonexistent") is None

    def test_list_available(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import RobotRegistry, EURDFLoader

        zoo = tmp_path / "zoo"
        for name in ["ur5e", "panda", "xarm"]:
            d = zoo / name
            d.mkdir(parents=True)
            (d / "robot.eurdf.yaml").write_text(
                f"robot_id: {name}\nname: {name}\nvendor: test\n"
                f"version: '1.0'\ndescription: test\ndof: 6\nlinks: []\njoints: []\n"
            )
            (d / "safety.yaml").write_text("safety_level: MODERATE\n")
            (d / "semantic.yaml").write_text("semantic_version: '1.0'\n")
            (d / "capabilities.yaml").write_text("capabilities: []\n")
            (d / "benchmark.yaml").write_text("kinematic_benchmarks: []\n")

        reg = RobotRegistry(loader=EURDFLoader(str(zoo)))
        available = reg.list_available()
        assert sorted(available) == ["panda", "ur5e", "xarm"]

    def test_list_empty(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import RobotRegistry, EURDFLoader

        zoo = tmp_path / "zoo"
        zoo.mkdir(parents=True)
        reg = RobotRegistry(loader=EURDFLoader(str(zoo)))
        assert reg.list_available() == []
        assert reg.list() == []

    def test_validate(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import RobotRegistry, EURDFLoader

        zoo = tmp_path / "zoo"
        robot_dir = zoo / "ur5e"
        robot_dir.mkdir(parents=True)
        (robot_dir / "robot.eurdf.yaml").write_text(
            "robot_id: ur5e\nname: UR5e\nvendor: UR\n"
            "version: '1.0'\ndescription: test\ndof: 6\nlinks: []\njoints: []\n"
        )
        (robot_dir / "safety.yaml").write_text("safety_level: MODERATE\n")
        (robot_dir / "semantic.yaml").write_text("semantic_version: '1.0'\n")
        (robot_dir / "capabilities.yaml").write_text("capabilities: []\n")
        (robot_dir / "benchmark.yaml").write_text("kinematic_benchmarks: []\n")

        reg = RobotRegistry(loader=EURDFLoader(str(zoo)))
        result = reg.validate("ur5e")
        assert result["valid"] is True
        assert "robot.eurdf.yaml" in result["files_found"]

    def test_validate_missing_files(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import RobotRegistry, EURDFLoader

        zoo = tmp_path / "zoo"
        robot_dir = zoo / "ur5e"
        robot_dir.mkdir(parents=True)
        # Only create eurdf, missing safety.yaml etc.
        (robot_dir / "robot.eurdf.yaml").write_text(
            "robot_id: ur5e\nname: UR5e\nvendor: UR\n"
            "version: '1.0'\ndescription: test\ndof: 6\nlinks: []\njoints: []\n"
        )

        reg = RobotRegistry(loader=EURDFLoader(str(zoo)))
        result = reg.validate("ur5e")
        assert result["valid"] is False
        assert len(result["files_missing"]) > 0

    def test_validate_robot_not_found(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import RobotRegistry, EURDFLoader

        zoo = tmp_path / "zoo"
        zoo.mkdir(parents=True)
        reg = RobotRegistry(loader=EURDFLoader(str(zoo)))
        result = reg.validate("nonexistent")
        assert result["valid"] is False
        assert "not found" in result["errors"][0]

    def test_inspect(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import RobotRegistry, EURDFLoader

        zoo = tmp_path / "zoo"
        robot_dir = zoo / "ur5e"
        robot_dir.mkdir(parents=True)
        (robot_dir / "robot.eurdf.yaml").write_text(
            "robot_id: ur5e\nname: UR5e\nvendor: UR\n"
            "version: '1.0'\ndescription: test\ndof: 6\nlinks: []\njoints: []\n"
        )
        (robot_dir / "safety.yaml").write_text("safety_level: MODERATE\n")
        (robot_dir / "semantic.yaml").write_text("semantic_version: '1.0'\n")
        (robot_dir / "capabilities.yaml").write_text("capabilities: []\n")
        (robot_dir / "benchmark.yaml").write_text("kinematic_benchmarks: []\n")

        reg = RobotRegistry(loader=EURDFLoader(str(zoo)))
        reg.install("ur5e")
        d = reg.inspect("ur5e")
        assert d["robot_id"] == "ur5e"
        assert "embodiment" in d
        assert "safety" in d
        assert "capability" in d

    def test_inspect_not_found(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import RobotRegistry, EURDFLoader

        zoo = tmp_path / "zoo"
        zoo.mkdir(parents=True)
        reg = RobotRegistry(loader=EURDFLoader(str(zoo)))
        with pytest.raises(FileNotFoundError):
            reg.inspect("nonexistent")


# ---------------------------------------------------------------------------
# Driver registration
# ---------------------------------------------------------------------------

class TestDriverRegistration:
    def test_register_and_get_driver(self, fresh_runtime):
        rt = fresh_runtime
        driver = MagicMock()
        rt.register_driver("arm", driver)
        assert rt.get_driver("arm") is driver

    def test_get_driver_missing(self, fresh_runtime):
        rt = fresh_runtime
        assert rt.get_driver("nonexistent") is None


# ---------------------------------------------------------------------------
# _run_async
# ---------------------------------------------------------------------------

class TestRunAsync:
    def test_run_async_no_event_loop(self, fresh_runtime):
        rt = fresh_runtime
        async def coro():  # noqa: E306
            return 42
        result = rt._run_async(coro())
        assert result == 42

    def test_run_async_inside_event_loop(self, fresh_runtime):
        rt = fresh_runtime
        async def inner():  # noqa: E306
            async def coro():
                return 42
            return rt._run_async(coro())
        result = asyncio.run(inner())
        assert result == 42


# ---------------------------------------------------------------------------
# _HowProxy
# ---------------------------------------------------------------------------

class TestHowProxy:
    def test_proxy_non_callable(self):
        engine = MagicMock()
        engine.some_attr = 42
        proxy = _HowProxy(engine, lambda c: None, event_bus=MagicMock())
        assert proxy.some_attr == 42

    def test_proxy_callable_sync(self):
        engine = MagicMock()
        engine.do_something = MagicMock(return_value="result")
        bus = MagicMock()
        proxy = _HowProxy(engine, lambda c: None, event_bus=bus)
        result = proxy.do_something("arg1", kw="val")
        assert result == "result"
        engine.do_something.assert_called_once_with("arg1", kw="val")

    def test_proxy_callable_async(self):
        engine = MagicMock()
        async def async_fn(x):  # noqa: E306
            return x * 2
        engine.double = async_fn
        bus = MagicMock()
        proxy = _HowProxy(engine, lambda c: asyncio.run(c), event_bus=bus)
        result = proxy.double(5)
        assert result == 10

    def test_proxy_exception(self):
        engine = MagicMock()
        def fail():  # noqa: E306
            raise RuntimeError("boom")
        engine.fail = fail
        bus = MagicMock()
        proxy = _HowProxy(engine, lambda c: c, event_bus=bus)
        with pytest.raises(RuntimeError, match="boom"):
            proxy.fail()
        # Failed event should be published
        assert bus.publish.called

    def test_proxy_no_event_bus(self):
        engine = MagicMock()
        engine.do_it = MagicMock(return_value=123)
        proxy = _HowProxy(engine, lambda c: c, event_bus=None)
        assert proxy.do_it() == 123


# ---------------------------------------------------------------------------
# _MemoryProxy
# ---------------------------------------------------------------------------

class TestMemoryProxy:
    def test_proxy_non_callable(self):
        mem = MagicMock()
        mem.some_attr = 42
        proxy = _MemoryProxy(mem, event_bus=MagicMock())
        assert proxy.some_attr == 42

    def test_proxy_callable(self):
        mem = MagicMock()
        mem.store = MagicMock(return_value="stored")
        bus = MagicMock()
        proxy = _MemoryProxy(mem, event_bus=bus)
        result = proxy.store("data")
        assert result == "stored"
        mem.store.assert_called_once_with("data")

    def test_proxy_exception(self):
        mem = MagicMock()
        def fail():  # noqa: E306
            raise RuntimeError("boom")
        mem.fail = fail
        bus = MagicMock()
        proxy = _MemoryProxy(mem, event_bus=bus)
        with pytest.raises(RuntimeError, match="boom"):
            proxy.fail()
        assert bus.publish.called

    def test_proxy_no_event_bus(self):
        mem = MagicMock()
        mem.do_it = MagicMock(return_value=123)
        proxy = _MemoryProxy(mem, event_bus=None)
        assert proxy.do_it() == 123


# ---------------------------------------------------------------------------
# capability_invoke
# ---------------------------------------------------------------------------

class TestCapabilityInvoke:
    @patch("rosclaw.core.runtime.ProviderRegistry", None)
    @patch("rosclaw.core.runtime.CapabilityRouter", None)
    def test_capability_invoke_router_exception(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_router = MagicMock()
        mock_router.invoke = MagicMock(side_effect=RuntimeError("invoke fail"))
        rt._capability_router = mock_router

        result = rt.capability_invoke("vlm.test", {})
        assert result["status"] == "error"
        assert "invoke fail" in result["error"]

    @patch("rosclaw.core.runtime.ProviderRegistry", None)
    @patch("rosclaw.core.runtime.CapabilityRouter", None)
    def test_capability_invoke_know_precheck(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_know = MagicMock()
        mock_know.query_for_provider_selection = MagicMock(return_value={
            "has_capability": True,
        })
        rt._knowledge = mock_know

        result = rt.capability_invoke("vlm.object_grounding", {})
        mock_know.query_for_provider_selection.assert_called_once()
        # Still falls back since no router
        assert result["status"] == "ok"

    @patch("rosclaw.core.runtime.ProviderRegistry", None)
    @patch("rosclaw.core.runtime.CapabilityRouter", None)
    def test_capability_invoke_know_precheck_raises(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_know = MagicMock()
        mock_know.query_for_provider_selection = MagicMock(side_effect=RuntimeError("know fail"))
        rt._knowledge = mock_know

        result = rt.capability_invoke("vlm.object_grounding", {})
        # Should not raise, falls back
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# plan_action
# ---------------------------------------------------------------------------

class TestPlanAction:
    def test_plan_action_fallback(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        result = rt.plan_action("pick the cup", {"result": {"objects": [{"label": "cup"}]}})
        assert result["status"] == "ok"
        assert result["action"]["type"] == "pick_and_place"
        assert result["action"]["target"] == "cup"

    def test_plan_action_exception(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_sm = MagicMock()
        mock_sm.plan = MagicMock(side_effect=RuntimeError("plan fail"))
        rt._skill_manager = mock_sm

        result = rt.plan_action("pick", {})
        assert result["status"] == "error"
        assert "plan fail" in result["error"]


# ---------------------------------------------------------------------------
# sandbox_check
# ---------------------------------------------------------------------------

class TestSandboxCheck:
    def test_sandbox_check_firewall_disabled(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        result = rt.sandbox_check({"trajectory": []})
        assert result["decision"] == "ALLOW"
        assert result["reason"] == "firewall_disabled"

    def test_sandbox_check_firewall_safe(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_fw = MagicMock()
        mock_response = MagicMock()
        mock_response.is_safe = True
        mock_response.violations = []
        mock_fw.validate = MagicMock(return_value=mock_response)
        rt._firewall = mock_fw

        result = rt.sandbox_check({"trajectory": [[0, 0, 0, 0, 0, 0]]})
        assert result["decision"] == "ALLOW"
        assert result["reason"] == "safe"

    def test_sandbox_check_firewall_blocked(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_fw = MagicMock()
        mock_violation = MagicMock()
        mock_violation.layer = "kinematic"
        mock_violation.severity = "HIGH"
        mock_violation.description = "joint limit exceeded"
        mock_response = MagicMock()
        mock_response.is_safe = False
        mock_response.violations = [mock_violation]
        mock_fw.validate = MagicMock(return_value=mock_response)
        rt._firewall = mock_fw

        result = rt.sandbox_check({"trajectory": [[0, 0, 0, 0, 0, 0]]})
        assert result["decision"] == "BLOCK"
        assert "joint limit exceeded" in result["reason"]
        assert len(result["violations"]) == 1

    def test_sandbox_check_exception(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_fw = MagicMock()
        mock_fw.validate = MagicMock(side_effect=RuntimeError("fw crash"))
        rt._firewall = mock_fw

        result = rt.sandbox_check({})
        assert result["decision"] == "BLOCK"
        assert "fw crash" in result["reason"]


# ---------------------------------------------------------------------------
# _generate_trajectory
# ---------------------------------------------------------------------------

class TestGenerateTrajectory:
    def test_generate_from_target_pose(self, fresh_runtime):
        rt = fresh_runtime
        result = rt._generate_trajectory({"target_pose": [0.5, 0.2, 0.1]})
        assert len(result) == 11  # steps + 1
        assert result[0] == [0.0, 0.0, 0.0]
        assert result[-1] == [0.5, 0.2, 0.1]

    def test_generate_from_target_in_parameters(self, fresh_runtime):
        rt = fresh_runtime
        result = rt._generate_trajectory({"parameters": {"target_pose": [0.5, 0.2, 0.1]}})
        assert len(result) == 11
        assert result[-1] == [0.5, 0.2, 0.1]

    def test_generate_default(self, fresh_runtime):
        rt = fresh_runtime
        result = rt._generate_trajectory({})
        assert len(result) == 5
        assert len(result[0]) == 6


# ---------------------------------------------------------------------------
# Physical World APIs
# ---------------------------------------------------------------------------

class TestPhysicalWorldAPIs:
    def test_add_world_object_no_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        assert rt.add_world_object({"name": "cup"}) is None

    def test_add_world_object_with_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_mem = MagicMock()
        mock_mem.add_world_object = MagicMock(return_value="obj_123")
        rt._memory = mock_mem
        assert rt.add_world_object({"name": "cup"}) == "obj_123"

    def test_get_world_object_no_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        assert rt.get_world_object("obj_123") is None

    def test_get_world_object_with_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_mem = MagicMock()
        mock_mem.get_world_object = MagicMock(return_value={"name": "cup"})
        rt._memory = mock_mem
        assert rt.get_world_object("obj_123") == {"name": "cup"}

    def test_update_world_object_pose_no_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        assert rt.update_world_object_pose("obj_123", [0, 0, 0]) is False

    def test_update_world_object_pose_with_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_mem = MagicMock()
        mock_mem.update_world_object_pose = MagicMock(return_value=True)
        rt._memory = mock_mem
        assert rt.update_world_object_pose("obj_123", [0, 0, 0], "grasped") is True

    def test_search_world_objects_no_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        assert rt.search_world_objects([0, 0, 0], 1.0) == []

    def test_search_world_objects_with_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_mem = MagicMock()
        mock_mem.search_world_objects = MagicMock(return_value=[{"name": "cup"}])
        rt._memory = mock_mem
        assert rt.search_world_objects([0, 0, 0], 1.0, "scene1") == [{"name": "cup"}]

    def test_get_scene_graph_no_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        assert rt.get_scene_graph("scene1") == ([], [])

    def test_get_scene_graph_with_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_mem = MagicMock()
        mock_mem.get_scene_graph = MagicMock(return_value=([{"name": "cup"}], []))
        rt._memory = mock_mem
        assert rt.get_scene_graph("scene1") == ([{"name": "cup"}], [])

    def test_sync_scene_objects_no_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        assert rt.sync_scene_objects("scene1", [], 0.0) is None

    def test_sync_scene_objects_with_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_mem = MagicMock()
        mock_mem.sync_scene_objects = MagicMock(return_value={"updated": True})
        rt._memory = mock_mem
        result = rt.sync_scene_objects("scene1", [], 0.0, 0.5)
        assert result == {"updated": True}

    def test_cognitive_search_no_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        assert rt.cognitive_search("cup") == []

    def test_cognitive_search_with_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_mem = MagicMock()
        mock_mem.cognitive_search = MagicMock(return_value=[{"name": "cup"}])
        rt._memory = mock_mem
        result = rt.cognitive_search("cup", limit=5)
        assert result == [{"name": "cup"}]

    def test_record_trajectory_no_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        assert rt.record_trajectory("path", []) is None

    def test_record_trajectory_with_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_mem = MagicMock()
        mock_mem.record_trajectory = MagicMock(return_value=42)
        rt._memory = mock_mem
        assert rt.record_trajectory("path", [([0, 0], 0.0)]) == 42

    def test_search_similar_trajectories_no_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        assert rt.search_similar_trajectories([]) == []

    def test_search_similar_trajectories_with_memory(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_mem = MagicMock()
        mock_mem.search_similar_trajectories = MagicMock(return_value=[("traj1", 0.5)])
        rt._memory = mock_mem
        assert rt.search_similar_trajectories([], top_k=3) == [("traj1", 0.5)]


# ---------------------------------------------------------------------------
# _load_external_providers
# ---------------------------------------------------------------------------

class TestLoadExternalProviders:
    def test_no_providers_dir(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        # providers_dir is None, should return early
        rt._load_external_providers()

    def test_providers_dir_with_loader(self, fresh_runtime, tmp_path):
        rt = fresh_runtime
        rt.initialize()
        rt.config.providers_dir = str(tmp_path)
        with patch("rosclaw.provider.loader.ProviderLoader") as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.scan_directory = MagicMock(return_value=["p1", "p2"])
            mock_loader_cls.return_value = mock_loader
            rt._provider_registry = MagicMock()
            rt._load_external_providers()
            mock_loader.scan_directory.assert_called_once_with(str(tmp_path))

    def test_providers_dir_loader_import_error(self, fresh_runtime, tmp_path):
        rt = fresh_runtime
        rt.initialize()
        rt.config.providers_dir = str(tmp_path)
        rt._provider_registry = MagicMock()
        with patch("rosclaw.provider.loader.ProviderLoader", side_effect=ImportError):
            rt._load_external_providers()


# ---------------------------------------------------------------------------
# execute (full closed loop)
# ---------------------------------------------------------------------------

class TestExecute:
    def test_execute_blocked_by_firewall(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        mock_fw = MagicMock()
        mock_response = MagicMock()
        mock_response.is_safe = False
        mock_violation = MagicMock()
        mock_violation.layer = "kinematic"
        mock_violation.severity = "HIGH"
        mock_violation.description = "unsafe"
        mock_response.violations = [mock_violation]
        mock_fw.validate = MagicMock(return_value=mock_response)
        rt._firewall = mock_fw

        result = rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "trajectory": [[0, 0, 0, 0, 0, 0]],
        })
        assert result["status"] == "blocked"
        assert result["decision"] == "BLOCK"
        rt.stop()

    def test_execute_with_sandbox_no_physics(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        mock_sandbox = MagicMock()
        mock_sandbox.has_physics = False
        mock_sandbox.validate_trajectory = MagicMock(return_value={"is_safe": True})
        mock_sandbox._world_id = "test_world"
        rt._sandbox = mock_sandbox

        result = rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "trajectory": [[0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0]],
        })
        assert result["status"] == "ok"
        assert "trajectory_data" in result
        rt.stop()

    def test_execute_with_sandbox_physics(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        mock_sandbox = MagicMock()
        mock_sandbox.has_physics = True
        mock_sandbox.validate_trajectory = MagicMock(return_value={"is_safe": True})
        mock_sandbox.simulate_step = MagicMock(return_value={
            "qpos": [0.05, 0, 0, 0, 0, 0],
            "time": 0.01,
        })
        mock_sandbox._world_id = "test_world"
        rt._sandbox = mock_sandbox

        result = rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "trajectory": [[0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0]],
        })
        assert result["status"] == "ok"
        rt.stop()

    def test_execute_sandbox_validation_unsafe(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        mock_sandbox = MagicMock()
        mock_sandbox.validate_trajectory = MagicMock(return_value={
            "is_safe": False, "reason": "collision", "violations": [],
        })
        mock_sandbox._world_id = "test_world"
        rt._sandbox = mock_sandbox

        result = rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "trajectory": [[0, 0, 0, 0, 0, 0]],
        })
        assert result["status"] == "blocked"
        rt.stop()

    def test_execute_sandbox_exception(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        mock_sandbox = MagicMock()
        mock_sandbox.validate_trajectory = MagicMock(side_effect=RuntimeError("sim crash"))
        rt._sandbox = mock_sandbox

        result = rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "trajectory": [[0, 0, 0, 0, 0, 0]],
        })
        assert result["status"] == "error"
        rt.stop()

    def test_execute_generates_trajectory_when_empty(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        result = rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
        })
        assert result["status"] == "ok"
        assert "trajectory" in result
        rt.stop()

    def test_execute_critic_reward_success(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        events = []
        rt.event_bus.subscribe("rosclaw.critic.success.detected", events.append)

        rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "trajectory": [[0, 0, 0, 0, 0, 0], [0.01, 0, 0, 0, 0, 0]],
        })
        critic_events = [e for e in events if e.topic == "rosclaw.critic.success.detected"]
        assert len(critic_events) == 1
        assert critic_events[0].payload["reward"] == 1.0
        assert critic_events[0].payload["status"] == "SUCCESS"
        rt.stop()

    def test_execute_critic_reward_blocked(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        mock_fw = MagicMock()
        mock_response = MagicMock()
        mock_response.is_safe = False
        mock_response.violations = []
        mock_fw.validate = MagicMock(return_value=mock_response)
        rt._firewall = mock_fw

        events = []
        rt.event_bus.subscribe("rosclaw.critic.success.detected", events.append)

        rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "trajectory": [[0, 0, 0, 0, 0, 0]],
        })
        critic_events = [e for e in events if e.topic == "rosclaw.critic.success.detected"]
        assert len(critic_events) == 1
        assert critic_events[0].payload["reward"] == -1.0
        assert critic_events[0].payload["status"] == "BLOCKED"
        rt.stop()

    def test_execute_publishes_praxis_completed(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        events = []
        rt.event_bus.subscribe("praxis.completed", events.append)

        rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "trajectory": [[0, 0, 0, 0, 0, 0]],
        })
        assert len(events) == 1
        assert events[0].payload["event_type"] == "success"
        rt.stop()

    def test_execute_memory_auto_ingest(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        mock_mem = MagicMock()
        mock_mem.store_experience = MagicMock()
        rt._memory = mock_mem

        rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "trajectory": [[0, 0, 0, 0, 0, 0]],
        })
        mock_mem.store_experience.assert_called_once()
        rt.stop()

    def test_execute_knowledge_post_execution(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        mock_know = MagicMock()
        mock_know.record_knowledge_usage = MagicMock()
        rt._knowledge = mock_know

        rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "trajectory": [[0, 0, 0, 0, 0, 0]],
        })
        mock_know.record_knowledge_usage.assert_called_once()
        rt.stop()

    def test_execute_knowledge_post_execution_raises(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        mock_know = MagicMock()
        mock_know.record_knowledge_usage = MagicMock(side_effect=RuntimeError("know fail"))
        rt._knowledge = mock_know

        # Should not raise
        result = rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "trajectory": [[0, 0, 0, 0, 0, 0]],
        })
        assert result["status"] == "ok"
        rt.stop()

    def test_execute_dashboard_trace_updated(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        events = []
        rt.event_bus.subscribe("rosclaw.dashboard.trace.updated", events.append)

        rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "trajectory": [[0, 0, 0, 0, 0, 0]],
        })
        assert len(events) == 1
        assert events[0].payload["skill_name"] == "pick"
        rt.stop()

    def test_execute_provider_error(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt.start()
        # capability_invoke will fallback to mock since no router
        result = rt.execute({
            "instruction": "pick cup",
            "skill_name": "pick",
            "capability": "skill.pick",
            "trajectory": [[0, 0, 0, 0, 0, 0]],
        })
        assert result["status"] == "ok"
        rt.stop()


# ---------------------------------------------------------------------------
# _register_robot_capabilities
# ---------------------------------------------------------------------------

class TestRegisterRobotCapabilities:
    def test_no_robot_profile(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt._provider_registry = MagicMock()
        rt._register_robot_capabilities()
        rt._provider_registry.register.assert_not_called()

    def test_no_provider_registry(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt._robot_profile = MagicMock()
        rt._robot_profile.capability = None
        rt._register_robot_capabilities()

    def test_no_capabilities(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt._provider_registry = MagicMock()
        rt._robot_profile = MagicMock()
        rt._robot_profile.capability = MagicMock()
        rt._robot_profile.capability.capabilities = []
        rt._register_robot_capabilities()
        rt._provider_registry.register.assert_not_called()

    def test_with_capabilities(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        rt._provider_registry = MagicMock()
        rt._robot_profile = MagicMock()
        cap_profile = MagicMock()
        cap_profile.capabilities = [{"name": "grasp"}, {"id": "place", "name": ""}]
        rt._robot_profile.capability = cap_profile

        with patch("rosclaw.provider.core.manifest.ProviderManifest") as mock_manifest:
            mock_manifest.from_dict = MagicMock(return_value=MagicMock())
            rt._register_robot_capabilities()
            rt._provider_registry.register.assert_called_once()
            rt._provider_registry.set_provider_health.assert_called_once_with("robot_capabilities", ok=True)


# ---------------------------------------------------------------------------
# _register_builtin_providers
# ---------------------------------------------------------------------------

class TestRegisterBuiltinProviders:
    def test_register_builtins(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_reg = MagicMock()
        mock_reg.register = MagicMock()
        mock_reg.set_provider_health = MagicMock()
        rt._provider_registry = mock_reg

        with (
            patch("rosclaw.provider.builtins.MockVLMProvider"),
            patch("rosclaw.provider.builtins.MockSkillProvider"),
            patch("rosclaw.provider.builtins.MockCriticProvider"),
            patch("rosclaw.provider.builtins.DeepSeekProvider"),
            patch("rosclaw.provider.core.manifest.ProviderManifest") as mock_manifest,
        ):
            mock_manifest.from_dict = MagicMock(return_value=MagicMock())
            rt._register_builtin_providers()
            assert mock_reg.register.call_count >= 3  # mock_vlm, mock_skill, mock_critic

    def test_register_gpu_providers(self, fresh_runtime):
        rt = fresh_runtime
        rt.initialize()
        mock_reg = MagicMock()
        mock_reg.register = MagicMock()
        mock_reg.set_provider_health = MagicMock()
        rt._provider_registry = mock_reg

        with (
            patch("rosclaw.provider.adapters.generic.GenericProvider"),
            patch("rosclaw.provider.core.manifest.ProviderManifest") as mock_manifest,
        ):
            mock_manifest.from_dict = MagicMock(return_value=MagicMock())
            rt.config.gpu_sam3_endpoint = "http://localhost:8001"
            rt._register_gpu_providers()
            assert mock_reg.register.called


class TestHowInitialization:
    def test_how_skipped_when_no_seekdb(self, fresh_runtime, caplog):
        import logging
        rt = fresh_runtime
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=True, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=True, enable_provider=False,
        )
        rt = Runtime(config=cfg)
        # Memory will initialize but we'll mock it to have no seekdb_client
        with patch("rosclaw.memory.interface.MemoryInterface") as mock_mem_cls:
            mock_mem = MagicMock()
            mock_mem.seekdb_client = None
            mock_mem_cls.return_value = mock_mem
            with caplog.at_level(logging.INFO, logger="rosclaw.core.runtime"):
                rt.initialize()
            assert rt._how is None
            assert "skipped" in caplog.text or "HeuristicEngine" in caplog.text


# ---------------------------------------------------------------------------
# Knowledge initialization with seed
# ---------------------------------------------------------------------------

class TestKnowledgeInitialization:
    def test_knowledge_seed_called(self, fresh_runtime):
        rt = fresh_runtime
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=True, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=True,
            enable_how=False, enable_provider=False,
        )
        rt = Runtime(config=cfg)
        with patch("rosclaw.memory.interface.MemoryInterface") as mock_mem_cls, \
             patch("rosclaw.know.interface.KnowledgeInterface") as mock_know_cls, \
             patch("rosclaw.know.storage.seed_knowledge_graph") as mock_seed:
            mock_mem = MagicMock()
            mock_mem.seekdb_client = MagicMock()
            mock_mem_cls.return_value = mock_mem
            mock_know = MagicMock()
            mock_know_cls.return_value = mock_know
            rt.initialize()
            mock_seed.assert_called_once()

    def test_knowledge_no_seekdb(self, fresh_runtime):
        rt = fresh_runtime
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=True, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=True,
            enable_how=False, enable_provider=False,
        )
        rt = Runtime(config=cfg)
        with patch("rosclaw.memory.interface.MemoryInterface") as mock_mem_cls, \
             patch("rosclaw.know.interface.KnowledgeInterface") as mock_know_cls, \
             patch("rosclaw.know.storage.seed_knowledge_graph") as mock_seed:
            mock_mem = MagicMock()
            mock_mem.seekdb_client = None
            mock_mem_cls.return_value = mock_mem
            mock_know = MagicMock()
            mock_know_cls.return_value = mock_know
            rt.initialize()
            mock_seed.assert_not_called()


# ---------------------------------------------------------------------------
# Sandbox initialization with RobotRegistry lookup
# ---------------------------------------------------------------------------

class TestSandboxInitialization:
    def test_sandbox_canonical_lookup(self, fresh_runtime):
        rt = fresh_runtime
        rt.config.robot_id = "my_robot"
        with patch("rosclaw.runtime.eurdf_loader.RobotRegistry") as mock_reg_cls:
            mock_profile = MagicMock()
            mock_profile.robot_id = "canonical_robot_id"
            mock_reg = MagicMock()
            mock_reg.get = MagicMock(return_value=mock_profile)
            mock_reg_cls.return_value = mock_reg
            with patch("rosclaw.sandbox.runtime_adapter.SandboxRuntimeAdapter") as mock_adapter:
                mock_adapter.return_value = MagicMock()
                rt.initialize()
                mock_reg.get.assert_called_once_with("my_robot")

    def test_sandbox_registry_lookup_exception(self, fresh_runtime):
        rt = fresh_runtime
        with patch("rosclaw.runtime.eurdf_loader.RobotRegistry") as mock_reg_cls:
            mock_reg_cls.side_effect = RuntimeError("registry fail")
            with patch("rosclaw.sandbox.runtime_adapter.SandboxRuntimeAdapter") as mock_adapter:
                mock_adapter.return_value = MagicMock()
                rt.initialize()
                # Should still initialize sandbox with original robot_id

    def test_sandbox_import_error(self, fresh_runtime):
        rt = fresh_runtime
        with patch("rosclaw.sandbox.runtime_adapter.SandboxRuntimeAdapter", side_effect=ImportError):
            rt.initialize()
            assert rt._sandbox is None


# ---------------------------------------------------------------------------
# Memory seekdb backend selection
# ---------------------------------------------------------------------------

class TestMemorySeekDB:
    def test_sqlite_backend(self, fresh_runtime):
        rt = fresh_runtime
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=True, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False,
            seekdb_backend="sqlite",
            seekdb_path=":memory:",
        )
        rt = Runtime(config=cfg)
        with patch("rosclaw.memory.interface.MemoryInterface") as mock_mem_cls, \
             patch("rosclaw.memory.seekdb_client.SeekDBSQLiteClient") as mock_sqlite:
            mock_mem = MagicMock()
            mock_mem_cls.return_value = mock_mem
            mock_sqlite.return_value = MagicMock()
            rt.initialize()
            mock_sqlite.assert_called_once_with(":memory:")

    def test_memory_backend(self, fresh_runtime):
        rt = fresh_runtime
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=True, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False,
            seekdb_backend="memory",
        )
        rt = Runtime(config=cfg)
        with patch("rosclaw.memory.interface.MemoryInterface") as mock_mem_cls, \
             patch("rosclaw.memory.seekdb_client.SeekDBMemoryClient") as mock_mem_client:
            mock_mem = MagicMock()
            mock_mem_cls.return_value = mock_mem
            mock_mem_client.return_value = MagicMock()
            rt.initialize()
            mock_mem_client.assert_called_once()


# ---------------------------------------------------------------------------
# EpisodeRecorder and Critic initialization
# ---------------------------------------------------------------------------

class TestPracticeSubmodules:
    def test_episode_recorder_import_error(self, fresh_runtime):
        rt = fresh_runtime
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=False, enable_practice=True,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False,
        )
        rt = Runtime(config=cfg)
        with patch("rosclaw.practice.timeline.UnifiedTimeline") as mock_tl, \
             patch("rosclaw.practice.episode_recorder.EpisodeRecorder", side_effect=ImportError):
            mock_tl.return_value = MagicMock()
            rt.initialize()
            assert rt._episode_recorder is None

    def test_critic_import_error(self, fresh_runtime):
        rt = fresh_runtime
        cfg = RuntimeConfig(
            enable_firewall=False, enable_memory=False, enable_practice=True,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False,
        )
        rt = Runtime(config=cfg)
        with patch("rosclaw.practice.timeline.UnifiedTimeline") as mock_tl, \
             patch("rosclaw.practice.episode_recorder.EpisodeRecorder") as mock_er, \
             patch("rosclaw.critic.basic_critic.BasicCritic", side_effect=ImportError):
            mock_tl.return_value = MagicMock()
            mock_er.return_value = MagicMock()
            rt.initialize()
            assert not hasattr(rt, "_critic") or rt._critic is None


# ---------------------------------------------------------------------------
# Firewall validator import error
# ---------------------------------------------------------------------------

class TestFirewallImportError:
    def test_firewall_import_error(self, tmp_path):
        urdf_file = tmp_path / "robot.urdf"
        urdf_file.write_text("<robot></robot>")
        cfg = RuntimeConfig(
            robot_model_path=str(urdf_file),
            enable_firewall=True, enable_memory=False, enable_practice=False,
            enable_swarm=False, enable_skill_manager=False, enable_knowledge=False,
            enable_how=False, enable_provider=False,
        )
        rt = Runtime(config=cfg)
        with patch("rosclaw.firewall.validator.FirewallValidator", side_effect=ImportError):
            rt.initialize()
            assert rt._firewall is None


# ---------------------------------------------------------------------------
# EURDFLoader edge cases
# ---------------------------------------------------------------------------

class TestEURDFLoaderEdgeCases:
    def test_loader_default_zoo_path(self):
        from rosclaw.runtime.eurdf_loader import EURDFLoader
        loader = EURDFLoader()
        # Default path should be relative to project
        assert loader.zoo_path is not None

    def test_loader_list_robots_empty_zoo(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import EURDFLoader
        loader = EURDFLoader(str(tmp_path))
        assert loader.list_robots() == []

    def test_loader_list_robots_with_robots(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import EURDFLoader
        r1 = tmp_path / "ur5e"
        r1.mkdir()
        (r1 / "robot.eurdf.yaml").write_text("robot_id: r1\nname: R1\n")
        r2 = tmp_path / "panda"
        r2.mkdir()
        (r2 / "robot.eurdf.yaml").write_text("robot_id: r2\nname: R2\n")
        loader = EURDFLoader(str(tmp_path))
        assert loader.list_robots() == ["panda", "ur5e"]

    def test_loader_load_missing_robot(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import EURDFLoader
        loader = EURDFLoader(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent")

    def test_loader_load_yaml_optional_missing(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import EURDFLoader
        robot_dir = tmp_path / "ur5e"
        robot_dir.mkdir()
        (robot_dir / "robot.eurdf.yaml").write_text(
            "robot_id: ur5e\nname: UR5e\nvendor: UR\n"
            "version: '1.0'\ndescription: test\ndof: 6\nlinks: []\njoints: []\n"
        )
        # Missing safety.yaml, semantic.yaml, etc.
        loader = EURDFLoader(str(tmp_path))
        profile = loader.load("ur5e")
        assert profile.robot_id == "ur5e"

    def test_validate_bad_eurdf_yaml(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import EURDFLoader
        robot_dir = tmp_path / "ur5e"
        robot_dir.mkdir()
        (robot_dir / "robot.eurdf.yaml").write_text("not: valid: yaml: [")
        (robot_dir / "safety.yaml").write_text("safety_level: MODERATE\n")
        (robot_dir / "semantic.yaml").write_text("semantic_version: '1.0'\n")
        (robot_dir / "capabilities.yaml").write_text("capabilities: []\n")
        (robot_dir / "benchmark.yaml").write_text("kinematic_benchmarks: []\n")
        loader = EURDFLoader(str(tmp_path))
        result = loader.validate("ur5e")
        assert result["valid"] is False
        assert "Failed to parse" in result["errors"][0]

    def test_validate_missing_required_fields(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import EURDFLoader
        robot_dir = tmp_path / "ur5e"
        robot_dir.mkdir()
        (robot_dir / "robot.eurdf.yaml").write_text(
            "robot_id: ur5e\nname: UR5e\n"
        )
        (robot_dir / "safety.yaml").write_text("safety_level: MODERATE\n")
        (robot_dir / "semantic.yaml").write_text("semantic_version: '1.0'\n")
        (robot_dir / "capabilities.yaml").write_text("capabilities: []\n")
        (robot_dir / "benchmark.yaml").write_text("kinematic_benchmarks: []\n")
        loader = EURDFLoader(str(tmp_path))
        result = loader.validate("ur5e")
        assert result["valid"] is False
        assert any("missing field" in e for e in result["errors"])

    def test_load_yaml_static_method(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import EURDFLoader
        yf = tmp_path / "test.yaml"
        yf.write_text("key: value\n")
        result = EURDFLoader._load_yaml(yf)
        assert result == {"key": "value"}

    def test_load_yaml_missing(self, tmp_path):
        from rosclaw.runtime.eurdf_loader import EURDFLoader
        result = EURDFLoader._load_yaml(tmp_path / "missing.yaml")
        assert result is None
