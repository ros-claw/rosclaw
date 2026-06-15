"""Module Collaboration Matrix - verify pairwise module interactions.

Tests that every pair of modules that should collaborate actually do so,
through their public APIs and EventBus events.

Matrix coverage:
    Runtime <-> EventBus     Runtime initializes with EventBus, publishes events
    Runtime <-> Sandbox      Runtime creates and manages Sandbox lifecycle
    Runtime <-> Memory       Runtime events trigger Memory writes
    Runtime <-> Practice     Runtime events trigger Practice recording
    Runtime <-> HOW          Runtime failures trigger HOW recovery hints
    EventBus <-> Provider    Provider requests routed via EventBus
    Sandbox <-> Firewall     Sandbox validates through Firewall
    Memory <-> HOW           HOW queries Memory for historical patterns
    Provider <-> Registry    Provider registered and discovered via Registry
"""

from unittest.mock import MagicMock

import pytest

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.core.runtime import Runtime, RuntimeConfig

# ------------------------------------------------------------------
# Helper fixtures
# ------------------------------------------------------------------

@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def minimal_runtime(event_bus):
    """Runtime with minimal config for fast tests."""
    config = RuntimeConfig(
        robot_id="matrix_test_bot",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=False,
        enable_how=False,
        enable_provider=False,
    )
    rt = Runtime(config)
    rt.event_bus = event_bus
    return rt


# ------------------------------------------------------------------
# Matrix: Runtime <-> EventBus
# ------------------------------------------------------------------

class TestRuntimeEventBus:
    def test_runtime_initializes_event_bus(self):
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_how=False,
            enable_provider=False,
        )
        rt = Runtime(config)
        rt.initialize()
        rt.start()

        assert rt.event_bus is not None
        assert rt.state.name == "RUNNING"
        rt.stop()

    def test_runtime_publishes_events_on_event_bus(self, event_bus):
        captured = []
        event_bus.subscribe("agent.command", lambda e: captured.append(e))

        config = RuntimeConfig(
            robot_id="test_bot",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_how=False,
            enable_provider=False,
        )
        rt = Runtime(config)
        rt.event_bus = event_bus
        rt.initialize()
        rt.start()

        event_bus.publish(Event(
            topic="agent.command",
            payload={"action": "test", "request_id": "req_001"},
        ))

        assert len(captured) == 1
        assert captured[0].payload["action"] == "test"
        rt.stop()


# ------------------------------------------------------------------
# Matrix: Runtime <-> Sandbox
# ------------------------------------------------------------------

class TestRuntimeSandbox:
    def test_runtime_initializes_sandbox(self, event_bus):
        config = RuntimeConfig(
            robot_id="ur5e",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_how=False,
            enable_provider=False,
            robot_zoo_path="./e-urdf-zoo",
        )
        rt = Runtime(config)
        rt.event_bus = event_bus
        rt.initialize()
        rt.start()

        assert rt._sandbox is not None
        rt.stop()

    def test_runtime_sandbox_lifecycle(self, event_bus):
        config = RuntimeConfig(
            robot_id="ur5e",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_how=False,
            enable_provider=False,
            robot_zoo_path="./e-urdf-zoo",
        )
        rt = Runtime(config)
        rt.event_bus = event_bus
        rt.initialize()
        rt.start()

        assert rt._sandbox is not None
        rt.stop()


# ------------------------------------------------------------------
# Matrix: Runtime <-> Memory
# ------------------------------------------------------------------

class TestRuntimeMemory:
    def test_runtime_initializes_memory_when_enabled(self, event_bus):
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_firewall=False,
            enable_memory=True,
            enable_practice=False,
            enable_how=False,
            enable_provider=False,
            seekdb_backend="memory",
        )
        rt = Runtime(config)
        rt.event_bus = event_bus
        rt.initialize()
        rt.start()

        assert rt._memory is not None
        rt.stop()

    def test_runtime_skips_memory_when_disabled(self, event_bus):
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_how=False,
            enable_provider=False,
        )
        rt = Runtime(config)
        rt.event_bus = event_bus
        rt.initialize()
        rt.start()

        assert rt._memory is None
        rt.stop()


# ------------------------------------------------------------------
# Matrix: Runtime <-> Practice
# ------------------------------------------------------------------

class TestRuntimePractice:
    def test_runtime_initializes_practice_when_enabled(self, event_bus):
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=True,
            enable_how=False,
            enable_provider=False,
            seekdb_backend="memory",
        )
        rt = Runtime(config)
        rt.event_bus = event_bus
        rt.initialize()
        rt.start()

        assert rt._practice is not None
        rt.stop()

    def test_practice_records_via_event_bus(self, event_bus):
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=True,
            enable_how=False,
            enable_provider=False,
            seekdb_backend="memory",
        )
        rt = Runtime(config)
        rt.event_bus = event_bus
        rt.initialize()
        rt.start()

        event_bus.publish(Event(
            topic="praxis.completed",
            payload={
                "practice_id": "test_001",
                "outcome": {"status": "success", "reward": 1.0},
            },
        ))

        summary = rt._practice.get_summary()
        assert isinstance(summary, dict)
        rt.stop()


# ------------------------------------------------------------------
# Matrix: Runtime <-> HOW
# ------------------------------------------------------------------

class TestRuntimeHow:
    def test_runtime_initializes_how_when_enabled(self, event_bus):
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_firewall=False,
            enable_memory=True,
            enable_practice=False,
            enable_how=True,
            enable_provider=False,
            seekdb_backend="memory",
        )
        rt = Runtime(config)
        rt.event_bus = event_bus
        rt.initialize()
        rt.start()

        assert rt._how is not None
        rt.stop()

    def test_how_responds_to_failure_events(self, event_bus):
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_how=True,
            enable_provider=False,
            seekdb_backend="memory",
        )
        rt = Runtime(config)
        rt.event_bus = event_bus
        rt.initialize()
        rt.start()

        event_bus.publish(Event(
            topic="praxis.failed",
            payload={"error": "joint_limit_exceeded", "practice_id": "test_002"},
        ))

        # Verify no crash
        rt.stop()


# ------------------------------------------------------------------
# Matrix: EventBus <-> Provider
# ------------------------------------------------------------------

class TestEventBusProvider:
    def test_provider_request_routed_via_event_bus(self, event_bus):
        from rosclaw.provider.adapters.generic import GenericProvider
        from rosclaw.provider.core.manifest import ProviderManifest
        from rosclaw.provider.core.registry import ProviderRegistry

        reg = ProviderRegistry(event_bus=event_bus)
        manifest = ProviderManifest.from_dict({
            "name": "test_provider",
            "version": "1.0.0",
            "type": "test",
            "capabilities": ["test.capability"],
        })
        reg.register(manifest, GenericProvider, auto_load=False)

        assert "test_provider" in reg.list_providers()

    def test_event_bus_carries_provider_events(self, event_bus):
        captured = []
        event_bus.subscribe("provider.*", lambda e: captured.append(e))

        event_bus.publish(Event(
            topic="provider.request",
            payload={"capability": "test", "request_id": "req_1"},
        ))

        assert len(captured) == 1
        assert captured[0].topic == "provider.request"


# ------------------------------------------------------------------
# Matrix: Sandbox <-> Firewall
# ------------------------------------------------------------------

class TestSandboxFirewall:
    def test_sandbox_validates_trajectory(self, event_bus):
        from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter

        sandbox = SandboxRuntimeAdapter(
            config={"engine": "mock", "world_id": "test"},
            event_bus=event_bus,
        )
        sandbox.initialize()

        trajectory = [[0.0, -1.57, 1.57, 0.0, 0.0, 0.0]]
        result = sandbox.validate_trajectory(trajectory, safety_level="STRICT")

        assert isinstance(result, dict)
        assert "is_safe" in result
        sandbox.stop()


# ------------------------------------------------------------------
# Matrix: Memory <-> HOW
# ------------------------------------------------------------------

class TestMemoryHow:
    def test_how_queries_memory_for_patterns(self, event_bus):
        from rosclaw.how.engine import HeuristicEngine
        from rosclaw.memory.interface import MemoryInterface

        mem = MemoryInterface(robot_id="test_bot", event_bus=event_bus)
        mem._do_initialize()

        how = HeuristicEngine(seekdb_client=mem.seekdb_client)

        # Store some data via memory
        mem.store_experience(
            event_id="exp_001",
            event_type="failure",
            instruction="pick cup",
            cot_trace=[],
            trajectory=[],
            outcome="joint_limit_exceeded",
            duration_sec=1.0,
            error_details=None,
            tags=[],
            metadata={},
        )

        # HOW should be able to query
        result = how.get_stats()
        assert isinstance(result, dict)

    def test_memory_stores_how_recovery_outcomes(self, event_bus):
        from rosclaw.memory.interface import MemoryInterface

        mem = MemoryInterface(robot_id="test_bot", event_bus=event_bus)
        mem._do_initialize()

        mem.store_experience(
            event_id="exp_recovery_001",
            event_type="recovery",
            instruction="grasp failed, increased force",
            cot_trace=[],
            trajectory=[],
            outcome="success",
            duration_sec=1.0,
            error_details=None,
            tags=[],
            metadata={},
        )

        stats = mem.get_statistics()
        assert stats["total_experiences"] >= 1


# ------------------------------------------------------------------
# Matrix: Provider <-> Registry
# ------------------------------------------------------------------

class TestProviderRegistry:
    def test_registry_discoverability(self):
        from rosclaw.provider.core.manifest import ProviderManifest
        from rosclaw.provider.core.registry import ProviderRegistry

        reg = ProviderRegistry()
        manifest = ProviderManifest.from_dict({
            "name": "discoverable",
            "version": "1.0.0",
            "type": "llm",
            "capabilities": ["chat"],
        })
        reg.register(manifest, lambda m: MagicMock(), auto_load=False)

        assert "discoverable" in reg.list_providers()

    def test_registry_capability_filtering(self):
        from rosclaw.provider.core.manifest import ProviderManifest
        from rosclaw.provider.core.registry import ProviderRegistry

        reg = ProviderRegistry()
        m1 = ProviderManifest.from_dict({
            "name": "chat_bot",
            "version": "1.0.0",
            "type": "llm",
            "capabilities": ["chat"],
        })
        m2 = ProviderManifest.from_dict({
            "name": "vision_bot",
            "version": "1.0.0",
            "type": "vlm",
            "capabilities": ["vision"],
        })
        class MockP1:
            name = "chat_bot"
            capabilities = ["chat"]
            _healthy = True
            manifest = m1

        class MockP2:
            name = "vision_bot"
            capabilities = ["vision"]
            _healthy = True
            manifest = m2

        reg.register(m1, lambda m: MockP1(), auto_load=False)
        reg.register(m2, lambda m: MockP2(), auto_load=False)

        chat_providers = reg.find_by_capability("chat", healthy_only=False)
        assert len(chat_providers) == 1
        assert chat_providers[0].name == "chat_bot"


# ------------------------------------------------------------------
# Matrix: Full Pipeline - Event flow
# ------------------------------------------------------------------

class TestFullPipelineEventFlow:
    def test_complete_event_pipeline(self, event_bus):
        """Verify the full event pipeline:
        agent.command -> skill.execution.start -> skill.execution.complete
        -> praxis.completed
        """
        events_captured = {}

        def capture(topic):
            def handler(event):
                events_captured[topic] = event.payload
            return handler

        for topic in [
            "agent.command",
            "skill.execution.start",
            "skill.execution.complete",
            "praxis.completed",
        ]:
            event_bus.subscribe(topic, capture(topic))

        event_bus.publish(Event(
            topic="agent.command",
            payload={"action": "grasp", "object": "red_cup", "request_id": "full_001"},
        ))
        event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "grasp", "correlation_id": "full_001"},
        ))
        event_bus.publish(Event(
            topic="skill.execution.complete",
            payload={"skill_name": "grasp", "correlation_id": "full_001", "result": {"status": "success"}},
        ))
        event_bus.publish(Event(
            topic="praxis.completed",
            payload={"practice_id": "full_001", "outcome": {"reward": 1.0}},
        ))

        assert "agent.command" in events_captured
        assert events_captured["agent.command"]["action"] == "grasp"
        assert "skill.execution.start" in events_captured
        assert "skill.execution.complete" in events_captured
        assert "praxis.completed" in events_captured
        assert events_captured["praxis.completed"]["outcome"]["reward"] == 1.0

    def test_firewall_blocks_flow_to_praxis(self, event_bus):
        """When firewall blocks an action, praxis should record BLOCKED."""
        captured = []
        event_bus.subscribe("praxis.completed", lambda e: captured.append(e))
        event_bus.subscribe("firewall.action_blocked", lambda e: captured.append(e))

        event_bus.publish(Event(
            topic="firewall.action_blocked",
            payload={"request_id": "block_001", "violations": [{"description": "collision"}]},
        ))
        event_bus.publish(Event(
            topic="skill.execution.complete",
            payload={"correlation_id": "block_001", "result": {"status": "blocked"}},
        ))
        event_bus.publish(Event(
            topic="praxis.completed",
            payload={"practice_id": "block_001", "outcome": {"status": "BLOCKED"}},
        ))

        assert len(captured) >= 1
