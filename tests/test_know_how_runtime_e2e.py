"""End-to-end test: KNOW + HOW + Runtime integration."""

import pytest
import asyncio
import time

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.core.event_bus import Event
from rosclaw.agent_runtime.mcp_hub import MCPHub


class TestKnowHowRuntimeE2E:
    """End-to-end integration of KNOW + HOW + Runtime."""

    @pytest.fixture
    def runtime(self):
        config = RuntimeConfig(
            robot_id="ur5e_test",
            enable_firewall=False,
            enable_memory=True,
            enable_practice=False,
            enable_knowledge=True,
            enable_how=True,
            enable_provider=False,
            seekdb_backend="memory",
        )
        rt = Runtime(config)
        rt._do_initialize()
        yield rt
        rt._do_stop()

    def test_runtime_has_knowledge(self, runtime):
        assert runtime.knowledge is not None
        assert hasattr(runtime.knowledge, "match_symptom")
        assert hasattr(runtime.knowledge, "query_robot_capabilities")

    def test_runtime_has_how(self, runtime):
        assert runtime.how is not None
        assert hasattr(runtime.how, "suggest_recovery")
        assert hasattr(runtime.how, "record_outcome")

    def test_knowledge_queries_work(self, runtime):
        ki = runtime.knowledge
        assert len(ki._patterns) >= 8
        match = ki.match_symptom("torque overflow on joint 2")
        assert match is not None
        assert match["pattern_id"] == "Torque_Overflow"
        rule = ki.get_safety_rule("Torque_Overflow")
        assert "SAFETY" in rule
        assert "Fix:" in rule

    def test_how_queries_work(self, runtime):
        how = runtime.how
        recovery = how.suggest_recovery("joint limit exceeded")
        assert recovery is not None
        assert recovery["source"] == "heuristic"
        assert "action" in recovery
        no_match = how.suggest_recovery("unicorn magical failure")
        assert no_match is None

    def test_know_how_symbiosis(self, runtime):
        ki = runtime.knowledge
        how = runtime.how
        # KNOW matches the error log to a symptom pattern
        # Use an error that both KNOW (keyword matching) and HOW (substring) can handle
        error_log = "velocity exceeds limit and diverging"
        match = ki.match_symptom(error_log)
        assert match is not None
        symptom = match["symptom"]
        # HOW suggests recovery based on the original error log
        recovery = how.suggest_recovery(error_log)
        assert recovery is not None
        assert recovery["action"] != ""
        # KNOW provides cross-domain analogy for the matched symptom
        analogy = ki.get_analogy(symptom)
        assert analogy is not None
        assert len(analogy["analogies"]) > 0

    def test_mcp_knowledge_tool_via_runtime(self, runtime):
        hub = MCPHub(event_bus=runtime.event_bus, robot_id="ur5e_test", runtime=runtime)
        hub._do_initialize()
        hub._register_query_knowledge_tool()
        hub._register_get_safety_heuristic_tool()
        hub._register_get_recovery_strategy_tool()

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                hub.handle_tool_call("query_knowledge", {
                    "query_type": "symptom",
                    "query": "torque overflow",
                })
            )
        finally:
            loop.close()
        assert result["status"] == "ok"
        assert result["matched"] is True
        assert result["result"]["pattern_id"] == "Torque_Overflow"
        hub._do_stop()

    def test_mcp_how_tool_via_runtime(self, runtime):
        hub = MCPHub(event_bus=runtime.event_bus, robot_id="ur5e_test", runtime=runtime)
        hub._do_initialize()
        hub._register_get_recovery_strategy_tool()

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                hub.handle_tool_call("get_recovery_strategy", {
                    "error_log": "collision detected during move",
                })
            )
        finally:
            loop.close()
        assert result["status"] == "ok"
        assert result["matched"] is True
        assert "action" in result
        hub._do_stop()

    def test_mcp_safety_heuristic_via_runtime(self, runtime):
        hub = MCPHub(event_bus=runtime.event_bus, robot_id="ur5e_test", runtime=runtime)
        hub._do_initialize()
        hub._register_get_safety_heuristic_tool()

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                hub.handle_tool_call("get_safety_heuristic", {
                    "condition": "torque_overflow",
                })
            )
        finally:
            loop.close()
        assert result["status"] == "ok"
        assert "SAFETY: Torque_Overflow" in result["safety_rule"]
        hub._do_stop()

    def test_seekdb_knowledge_graph_populated(self, runtime):
        seekdb = runtime.memory.seekdb_client
        count = seekdb.count("knowledge_graph")
        assert count > 0

    def test_seekdb_heuristic_rules_populated(self, runtime):
        seekdb = runtime.memory.seekdb_client
        count = seekdb.count("heuristic_rules")
        assert count > 0

    def test_event_bus_knowledge_topics(self, runtime):
        received = []
        def handler(event):  # noqa: E306
            received.append(event.topic)
        runtime.event_bus.subscribe("knowledge.query", handler)
        runtime.event_bus.subscribe("heuristic.recovery_suggested", handler)
        runtime.event_bus.publish(Event(
            topic="knowledge.query",
            payload={"query": "capabilities"},
            source="test",
        ))
        runtime.event_bus.publish(Event(
            topic="heuristic.recovery_suggested",
            payload={"rule_id": "r1", "action": "retry"},
            source="test",
        ))
        time.sleep(0.1)
        assert "knowledge.query" in received
        assert "heuristic.recovery_suggested" in received
