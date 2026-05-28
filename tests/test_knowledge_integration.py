"""Integration tests for Knowledge Graph (rosclaw.know) module.

Tests cover:
- KnowledgeInterface initialization and lifecycle
- Symptom matching with curated patterns
- Robot capability queries
- Cross-domain analogy retrieval
- Safety rule retrieval
- MCP tool integration (via handle_tool_call)
"""

import pytest

from rosclaw.know.interface import KnowledgeInterface
from rosclaw.know.storage import seed_knowledge_graph
from rosclaw.know.graph import count_knowledge_facts
from rosclaw.memory.seekdb_client import SeekDBMemoryClient


class TestKnowledgeInterface:
    """Unit tests for KnowledgeInterface query engine."""

    def test_initialization(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()
        assert ki._initialized is True
        assert len(ki._patterns) > 0  # Curated patterns loaded
        ki._do_stop()
        assert ki._initialized is False

    def test_curated_patterns_loaded(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()
        expected = [
            "Torque_Overflow",
            "Velocity_Divergence",
            "Memory_Exhaustion",
            "Numerical_Instability",
            "Oscillation_Divergence",
            "Communication_Timeout",
            "Gradient_Explosion",
            "Compile_Error",
        ]
        for label in expected:
            assert label in ki._patterns, f"Missing curated pattern: {label}"
        ki._do_stop()

    def test_match_symptom_torque_overflow(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()
        match = ki.match_symptom("torque overflow on joint 2")
        assert match is not None
        assert match["pattern_id"] == "Torque_Overflow"
        assert "fix" in match
        ki._do_stop()

    def test_match_symptom_velocity_divergence(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()
        match = ki.match_symptom("velocity diverging to infinity")
        assert match is not None
        assert match["pattern_id"] == "Velocity_Divergence"
        ki._do_stop()

    def test_match_symptom_memory_exhaustion(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()
        match = ki.match_symptom("cuda out of memory error")
        assert match is not None
        assert match["pattern_id"] == "Memory_Exhaustion"
        ki._do_stop()

    def test_match_symptom_numerical_instability(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()
        match = ki.match_symptom("nan in loss gradient")
        assert match is not None
        assert match["pattern_id"] == "Gradient_Explosion"
        ki._do_stop()

    def test_match_symptom_no_match(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()
        match = ki.match_symptom("completely unrelated error about unicorns")
        assert match is None
        ki._do_stop()

    def test_get_analogy(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()
        analogy = ki.get_analogy("torque overflow situation")
        assert analogy is not None
        assert analogy["pattern_id"] == "Torque_Overflow"
        assert len(analogy["analogies"]) > 0
        ki._do_stop()

    def test_get_safety_rule(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()
        rule = ki.get_safety_rule("Torque_Overflow")
        assert "SAFETY: Torque_Overflow" in rule
        assert "Fix:" in rule
        ki._do_stop()

    def test_get_safety_rule_unknown(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()
        rule = ki.get_safety_rule("Unknown_Label")
        assert rule == ""
        ki._do_stop()


class TestKnowledgeStorage:
    """Tests for SeekDB knowledge_graph seeding."""

    def test_seed_knowledge_graph(self):
        client = SeekDBMemoryClient()
        client.connect()
        counts = seed_knowledge_graph(client)
        assert counts["total"] > 0
        assert counts["capabilities"] > 0
        assert counts["symptoms"] > 0

        facts = count_knowledge_facts(client)
        assert facts["total"] == counts["total"]
        assert facts["capabilities"] == counts["capabilities"]
        assert facts["symptoms"] == counts["symptoms"]

    def test_seed_idempotent(self):
        client = SeekDBMemoryClient()
        client.connect()
        counts1 = seed_knowledge_graph(client)
        counts2 = seed_knowledge_graph(client)
        assert counts1["total"] == counts2["total"]

    def test_seed_with_none_client(self):
        counts = seed_knowledge_graph(None)
        assert counts["total"] == 0


class TestKnowledgeInterfaceWithSeekDB:
    """Tests for KnowledgeInterface backed by SeekDB."""

    def test_load_from_seekdb(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()

        caps = ki.query_robot_capabilities("ur5e")
        assert len(caps) > 0
        assert "grasp" in caps
        assert "6dof_arm" in caps

        ki._do_stop()

    def test_load_symptoms_from_seekdb(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="panda", seekdb_client=client)
        ki._do_initialize()

        symptoms = [s["symptom"] for s in ki._symptoms if s["subject"] == "panda"]
        assert "Collision_Detected" in symptoms

        ki._do_stop()


class TestMCPKnowledgeTool:
    """Tests for MCP query_knowledge tool integration."""

    def test_mcp_query_capability(self):
        from rosclaw.agent_runtime.mcp_hub import MCPHub
        from rosclaw.core.event_bus import EventBus

        event_bus = EventBus()
        hub = MCPHub(event_bus=event_bus, robot_id="ur5e")
        hub._do_initialize()

        hub._register_query_knowledge_tool()
        hub._register_get_safety_heuristic_tool()

        tool_names = [t["name"] for t in hub.tools]
        assert "query_knowledge" in tool_names
        assert "get_safety_heuristic" in tool_names

        hub._do_stop()

    def test_mcp_handle_query_knowledge_capability(self):
        from rosclaw.agent_runtime.mcp_hub import MCPHub
        from rosclaw.core.event_bus import EventBus
        import asyncio

        event_bus = EventBus()
        hub = MCPHub(event_bus=event_bus, robot_id="ur5e")
        hub._register_query_knowledge_tool()
        hub._register_get_safety_heuristic_tool()

        class MockRuntime:
            def __init__(self):
                self.knowledge = KnowledgeInterface(robot_id="ur5e")
                self.knowledge._do_initialize()

        hub.runtime = MockRuntime()

        result = asyncio.get_event_loop().run_until_complete(
            hub.handle_tool_call("query_knowledge", {
                "query_type": "capability",
                "query": "ur5e",
            })
        )

        assert result["status"] == "ok"
        assert result["query_type"] == "capability"
        assert result["robot_id"] == "ur5e"

    def test_mcp_handle_query_knowledge_symptom(self):
        from rosclaw.agent_runtime.mcp_hub import MCPHub
        from rosclaw.core.event_bus import EventBus
        import asyncio

        event_bus = EventBus()
        hub = MCPHub(event_bus=event_bus, robot_id="ur5e")
        hub._register_query_knowledge_tool()

        class MockRuntime:
            def __init__(self):
                self.knowledge = KnowledgeInterface(robot_id="ur5e")
                self.knowledge._do_initialize()

        hub.runtime = MockRuntime()

        result = asyncio.get_event_loop().run_until_complete(
            hub.handle_tool_call("query_knowledge", {
                "query_type": "symptom",
                "query": "torque overflow on joint 2",
            })
        )

        assert result["status"] == "ok"
        assert result["query_type"] == "symptom"
        assert result["matched"] is True
        assert result["result"]["pattern_id"] == "Torque_Overflow"

    def test_mcp_handle_get_safety_heuristic(self):
        from rosclaw.agent_runtime.mcp_hub import MCPHub
        from rosclaw.core.event_bus import EventBus
        import asyncio

        event_bus = EventBus()
        hub = MCPHub(event_bus=event_bus, robot_id="ur5e")
        hub._register_get_safety_heuristic_tool()

        class MockRuntime:
            def __init__(self):
                self.knowledge = KnowledgeInterface(robot_id="ur5e")
                self.knowledge._do_initialize()

        hub.runtime = MockRuntime()

        result = asyncio.get_event_loop().run_until_complete(
            hub.handle_tool_call("get_safety_heuristic", {
                "condition": "torque_overflow",
            })
        )

        assert result["status"] == "ok"
        assert result["condition"] == "torque_overflow"
        assert "SAFETY: Torque_Overflow" in result["safety_rule"]
