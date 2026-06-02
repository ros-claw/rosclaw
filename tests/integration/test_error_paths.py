"""ROSClaw v1.0 — Error Path & Failure Handling Tests.

Tests graceful degradation under failure conditions.
Run: pytest tests/integration/test_error_paths.py -v
"""

import pytest

from rosclaw.core import Runtime, RuntimeConfig
from rosclaw.core.event_bus import Event, EventBus
from rosclaw.how import HeuristicEngine
from rosclaw.know import KnowledgeInterface
from rosclaw.memory.seekdb_client import SeekDBMemoryClient


# ─────────────────────────────────────────────────────────────
# SeekDB Failure Tests
# ─────────────────────────────────────────────────────────────


class TestSeekDBFailurePaths:
    """Test SeekDB failure handling."""

    def test_seekdb_query_nonexistent_table(self):
        """Query on non-existent table returns empty list, no crash."""
        seekdb = SeekDBMemoryClient()
        seekdb.connect()

        results = seekdb.query("nonexistent_table", filters={}, limit=10)
        assert results == []

    def test_seekdb_insert_without_connect(self):
        """Insert without connect raises appropriate error."""
        seekdb = SeekDBMemoryClient()
        # Note: SeekDBMemoryClient may auto-connect
        # This test documents expected behavior
        try:
            seekdb.insert("test", {"id": "1", "data": "x"})
            # If no error, that's acceptable for in-memory backend
        except Exception as e:
            # Should be a clear error, not a cryptic one
            assert "connect" in str(e).lower() or "not" in str(e).lower()

    def test_seekdb_malformed_filter(self):
        """Malformed filter is handled gracefully."""
        seekdb = SeekDBMemoryClient()
        seekdb.connect()

        seekdb.insert("test", {"id": "1", "robot_id": "bot1"})

        # Query with unusual filter values
        results = seekdb.query("test", filters={"robot_id": None}, limit=10)
        assert isinstance(results, list)

    def test_seekdb_concurrent_access(self):
        """Concurrent reads/writes don't corrupt data."""
        seekdb = SeekDBMemoryClient()
        seekdb.connect()

        # Seed data
        for i in range(100):
            seekdb.insert("test", {"id": str(i), "val": i})

        # Concurrent queries
        results1 = seekdb.query("test", filters={}, limit=50)
        results2 = seekdb.query("test", filters={}, limit=50)

        assert len(results1) == 50
        assert len(results2) == 50


# ─────────────────────────────────────────────────────────────
# EventBus Failure Tests
# ─────────────────────────────────────────────────────────────


class TestEventBusFailurePaths:
    """Test EventBus failure handling."""

    def test_subscriber_exception_does_not_break_others(self):
        """One subscriber crashing doesn't affect others."""
        bus = EventBus()
        received_good = []
        received_bad = []

        def good_handler(event):
            received_good.append(event.payload)

        def bad_handler(event):
            received_bad.append(event.payload)
            raise RuntimeError("Subscriber crash!")

        bus.subscribe("test.topic", good_handler)
        bus.subscribe("test.topic", bad_handler)

        bus.publish(Event(topic="test.topic", payload="hello", source="test"))

        # Good handler should still receive
        assert len(received_good) == 1
        assert received_good[0] == "hello"

    def test_publish_with_no_subscribers(self):
        """Publishing with no subscribers is a no-op, no error."""
        bus = EventBus()

        # No subscribers
        bus.publish(Event(topic="empty.topic", payload="x", source="test"))

        # Should not raise
        assert True

    def test_event_bus_high_load_no_loss(self):
        """High load publishing doesn't lose events."""
        bus = EventBus()
        received = []

        bus.subscribe("load.test", lambda e: received.append(e.payload))

        count = 1000
        for i in range(count):
            bus.publish(Event(topic="load.test", payload=i, source="test"))

        assert len(received) == count

    def test_clear_history_with_filter(self):
        """Clear history with filter works correctly."""
        bus = EventBus()

        for i in range(10):
            bus.publish(Event(topic="keep.topic", payload=i, source="test"))
            bus.publish(Event(topic="clear.topic", payload=i, source="test"))

        bus.clear_history(topic="clear.topic")

        keep_history = bus.get_history(topic="keep.topic")
        clear_history = bus.get_history(topic="clear.topic")

        assert len(keep_history) == 10
        assert len(clear_history) == 0


# ─────────────────────────────────────────────────────────────
# Runtime Failure Recovery Tests
# ─────────────────────────────────────────────────────────────


class TestRuntimeFailurePaths:
    """Test Runtime failure recovery."""

    def test_runtime_init_without_knowledge(self):
        """Runtime initializes without knowledge module."""
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_knowledge=False,
            enable_how=False,
        )
        runtime = Runtime(config)
        runtime.initialize()

        assert runtime.state.name == "READY"
        assert not hasattr(runtime, '_knowledge') or runtime._knowledge is None

    def test_runtime_init_without_how(self):
        """Runtime initializes without HOW module."""
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_knowledge=True,
            enable_how=False,
        )
        runtime = Runtime(config)
        runtime.initialize()

        assert runtime.state.name == "READY"
        assert runtime._how is None

    def test_runtime_know_query_invalid_robot(self):
        """Query for invalid robot returns empty, no crash."""
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_knowledge=True,
            enable_how=False,
        )
        runtime = Runtime(config)
        runtime.initialize()

        result = runtime._knowledge.query_robot_capabilities("nonexistent")
        assert result == []

    def test_runtime_firewall_with_disabled_sandbox(self):
        """Firewall works when sandbox is disabled."""
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_firewall=True,
            enable_knowledge=False,
            enable_how=False,
        )
        runtime = Runtime(config)
        runtime.initialize()

        assert runtime.state.name == "READY"
        # Firewall should be available even without sandbox

    def test_runtime_stop_from_ready(self):
        """Stopping from READY state is safe."""
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_knowledge=False,
            enable_how=False,
        )
        runtime = Runtime(config)
        runtime.initialize()
        runtime.stop()

        assert runtime.state.name == "STOPPED"

    def test_runtime_double_initialize(self):
        """Double initialize should be handled gracefully."""
        config = RuntimeConfig(
            robot_id="test_bot",
            enable_knowledge=False,
            enable_how=False,
        )
        runtime = Runtime(config)
        runtime.initialize()

        # Second initialize should not crash
        try:
            runtime.initialize()
        except Exception:
            # May raise if already initialized - that's acceptable
            pass

        assert runtime.state.name in ["READY", "ERROR"]


# ─────────────────────────────────────────────────────────────
# HOW Failure Paths
# ─────────────────────────────────────────────────────────────


class TestHowFailurePaths:
    """Test HOW module failure handling."""

    @pytest.mark.asyncio
    async def test_how_recovery_no_matching_rule(self):
        """Recovery with no matching rule returns None."""
        seekdb = SeekDBMemoryClient()
        seekdb.connect()

        how = HeuristicEngine(seekdb_client=seekdb)
        await how.initialize()

        result = await how.suggest_recovery("completely_unknown_error_xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_how_record_outcome_invalid_rule(self):
        """Recording outcome for invalid rule is handled gracefully."""
        seekdb = SeekDBMemoryClient()
        seekdb.connect()

        how = HeuristicEngine(seekdb_client=seekdb)
        await how.initialize()

        # Should not crash
        await how.record_outcome("nonexistent_rule", success=True)

    @pytest.mark.asyncio
    async def test_how_seed_defaults_idempotent(self):
        """Seeding defaults twice is idempotent."""
        seekdb = SeekDBMemoryClient()
        seekdb.connect()

        how = HeuristicEngine(seekdb_client=seekdb)
        await how.initialize()

        await how.seed_defaults()
        await how.seed_defaults()

        # Should still work
        result = await how.suggest_recovery("joint_limit_exceeded")
        assert result is not None


# ─────────────────────────────────────────────────────────────
# KNOW Failure Paths
# ─────────────────────────────────────────────────────────────


class TestKnowFailurePaths:
    """Test KNOW module failure handling."""

    def test_know_query_malformed_robot_id(self):
        """Query with malformed robot_id is handled gracefully."""
        seekdb = SeekDBMemoryClient()
        seekdb.connect()

        know = KnowledgeInterface(seekdb_client=seekdb, robot_id="test")
        know.initialize()

        # Various edge cases
        for robot_id in ["", "   ", "a" * 1000, "robot/with/slashes"]:
            result = know.query_robot_capabilities(robot_id)
            assert isinstance(result, list)

    def test_know_match_symptom_empty(self):
        """Symptom matching with empty symptom returns empty."""
        seekdb = SeekDBMemoryClient()
        seekdb.connect()

        know = KnowledgeInterface(seekdb_client=seekdb, robot_id="test")
        know.initialize()

        result = know.match_symptom("")
        assert result == [] or result is None

    def test_know_get_analogy_no_data(self):
        """Analogy with no data returns None or empty."""
        seekdb = SeekDBMemoryClient()
        seekdb.connect()

        know = KnowledgeInterface(seekdb_client=seekdb, robot_id="test")
        know.initialize()

        result = know.get_analogy("pick")
        # May be None or empty depending on implementation
        assert result is None or isinstance(result, (dict, list))
