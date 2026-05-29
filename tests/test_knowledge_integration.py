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


class TestRobotCapabilityQuery:
    """Tests for robot_capability_query method."""

    def test_capability_query_with_seekdb(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()

        # Query which robots can grasp
        robots = ki.robot_capability_query("grasp")
        assert "ur5e" in robots
        assert "panda" in robots
        assert "unitree_g1" in robots

        # Query pick_and_place
        robots = ki.robot_capability_query("pick_and_place")
        assert "ur5e" in robots
        assert "panda" in robots

        ki._do_stop()

    def test_capability_query_no_match(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()

        robots = ki.robot_capability_query("nonexistent_skill")
        assert robots == []

        ki._do_stop()

    def test_capability_query_empty_skill(self):
        ki = KnowledgeInterface(robot_id="ur5e")
        ki._do_initialize()
        assert ki.robot_capability_query("") == []
        ki._do_stop()


class TestTaskDecompositionHint:
    """Tests for task_decomposition_hint method."""

    def test_decompose_pick_and_place(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()

        result = ki.task_decomposition_hint("pick and place")
        assert result is not None
        assert result["matched_pattern"] == "pick and place"
        assert "navigate_to_object" in result["steps"]
        assert "grasp" in result["steps"]
        assert "release" in result["steps"]
        assert result["step_count"] == 6
        assert result["confidence"] > 0.8
        ki._do_stop()

    def test_decompose_walk_to_point(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()

        result = ki.task_decomposition_hint("walk to point")
        assert result is not None
        assert result["matched_pattern"] == "walk to point"
        assert "balance" in result["steps"]
        assert "plan_footsteps" in result["steps"]
        ki._do_stop()

    def test_decompose_assembly(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()

        result = ki.task_decomposition_hint("assembly task")
        assert result is not None
        assert result["matched_pattern"] == "assembly"
        assert "grasp_part" in result["steps"]
        assert "verify_fit" in result["steps"]
        ki._do_stop()

    def test_decompose_no_match(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()

        result = ki.task_decomposition_hint("completely unrelated nonsense task")
        assert result is None
        ki._do_stop()

    def test_decompose_empty(self):
        ki = KnowledgeInterface(robot_id="test_robot")
        ki._do_initialize()

        assert ki.task_decomposition_hint("") is None
        ki._do_stop()


class TestCompositionalReasoning:
    """Tests for can_perform_task and recommend_robot_for_task."""

    def test_can_perform_task_yes(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()

        result = ki.can_perform_task("ur5e", "pick and place")
        assert result is not None
        assert result["robot_id"] == "ur5e"
        assert result["can_perform"] is True
        assert "grasp" in result["matched_capabilities"]
        assert result["missing_capabilities"] == []

        ki._do_stop()

    def test_can_perform_task_no(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="spot", seekdb_client=client)
        ki._do_initialize()

        # Spot can inspect_surface but not grasp, so can't do pick_and_place
        result = ki.can_perform_task("spot", "pick and place")
        assert result is not None
        assert result["can_perform"] is False
        assert "grasp" in result["missing_capabilities"]

        ki._do_stop()

    def test_can_perform_task_unknown(self):
        ki = KnowledgeInterface(robot_id="test")
        ki._do_initialize()
        assert ki.can_perform_task("test", "nonsense task xyz") is None
        ki._do_stop()

    def test_recommend_robot_for_task(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()

        recs = ki.recommend_robot_for_task("pick and place")
        assert len(recs) > 0
        # UR5e and Panda should be top candidates (both have grasp + pick_and_place)
        top = recs[0]
        assert top["score"] == 1.0  # Perfect match
        assert "grasp" in top["matched_capabilities"]
        assert "pick_and_place" in top["matched_capabilities"]

        ki._do_stop()

    def test_recommend_robot_sort_objects(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()

        recs = ki.recommend_robot_for_task("sort objects")
        assert len(recs) > 0
        # ur5e and agilex_piper have sort_objects
        robot_ids = [r["robot_id"] for r in recs if r["score"] == 1.0]
        assert "ur5e" in robot_ids

        ki._do_stop()

    def test_recommend_robot_empty_task(self):
        ki = KnowledgeInterface(robot_id="test")
        ki._do_initialize()
        assert ki.recommend_robot_for_task("") == []
        ki._do_stop()
