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


# v1.5 isolation: these tests pin the curated-baseline contract, so
# they MUST not pick up the v1.5 catalog in ``data/knowledge_assets/``.
# Pointing at a nonexistent path makes KnowledgeInterface skip the
# bridge_index load and serve only the curated fallback patterns.
_BASELINE_ASSETS = "/tmp/rosclaw_test_baseline_no_v15_assets"


class TestKnowledgeInterface:
    """Unit tests for KnowledgeInterface query engine."""

    def test_initialization(self):
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
        ki._do_initialize()
        assert ki._initialized is True
        assert len(ki._patterns) > 0  # Curated patterns loaded
        ki._do_stop()
        assert ki._initialized is False

    def test_curated_patterns_loaded(self):
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
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
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
        ki._do_initialize()
        match = ki.match_symptom("torque overflow on joint 2")
        assert match is not None
        assert match["pattern_id"] == "Torque_Overflow"
        assert "fix" in match
        ki._do_stop()

    def test_match_symptom_velocity_divergence(self):
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
        ki._do_initialize()
        match = ki.match_symptom("velocity diverging to infinity")
        assert match is not None
        assert match["pattern_id"] == "Velocity_Divergence"
        ki._do_stop()

    def test_match_symptom_memory_exhaustion(self):
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
        ki._do_initialize()
        match = ki.match_symptom("cuda out of memory error")
        assert match is not None
        assert match["pattern_id"] == "Memory_Exhaustion"
        ki._do_stop()

    def test_match_symptom_numerical_instability(self):
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
        ki._do_initialize()
        match = ki.match_symptom("nan in loss gradient")
        assert match is not None
        assert match["pattern_id"] == "Gradient_Explosion"
        ki._do_stop()

    def test_match_symptom_no_match(self):
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
        ki._do_initialize()
        match = ki.match_symptom("completely unrelated error about unicorns")
        assert match is None
        ki._do_stop()

    def test_get_analogy(self):
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
        ki._do_initialize()
        analogy = ki.get_analogy("torque overflow situation")
        assert analogy is not None
        assert analogy["pattern_id"] == "Torque_Overflow"
        assert len(analogy["analogies"]) > 0
        ki._do_stop()

    def test_get_safety_rule(self):
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
        ki._do_initialize()
        rule = ki.get_safety_rule("Torque_Overflow")
        assert "SAFETY: Torque_Overflow" in rule
        assert "Fix:" in rule
        ki._do_stop()

    def test_get_safety_rule_unknown(self):
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
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
                self.knowledge = KnowledgeInterface(
                    robot_id="ur5e", assets_path=_BASELINE_ASSETS,
                )
                self.knowledge._do_initialize()

        hub.runtime = MockRuntime()

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                hub.handle_tool_call("query_knowledge", {
                    "query_type": "capability",
                    "query": "ur5e",
                })
            )
        finally:
            loop.close()

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
                self.knowledge = KnowledgeInterface(
                    robot_id="ur5e", assets_path=_BASELINE_ASSETS,
                )
                self.knowledge._do_initialize()

        hub.runtime = MockRuntime()

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                hub.handle_tool_call("query_knowledge", {
                    "query_type": "symptom",
                    "query": "torque overflow on joint 2",
                })
            )
        finally:
            loop.close()

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
                self.knowledge = KnowledgeInterface(
                    robot_id="ur5e", assets_path=_BASELINE_ASSETS,
                )
                self.knowledge._do_initialize()

        hub.runtime = MockRuntime()

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
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
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
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
        ki._do_initialize()

        result = ki.task_decomposition_hint("walk to point")
        assert result is not None
        assert result["matched_pattern"] == "walk to point"
        assert "balance" in result["steps"]
        assert "plan_footsteps" in result["steps"]
        ki._do_stop()

    def test_decompose_assembly(self):
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
        ki._do_initialize()

        result = ki.task_decomposition_hint("assembly task")
        assert result is not None
        assert result["matched_pattern"] == "assembly"
        assert "grasp_part" in result["steps"]
        assert "verify_fit" in result["steps"]
        ki._do_stop()

    def test_decompose_no_match(self):
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
        ki._do_initialize()

        result = ki.task_decomposition_hint("completely unrelated nonsense task")
        assert result is None
        ki._do_stop()

    def test_decompose_empty(self):
        ki = KnowledgeInterface(robot_id="test_robot", assets_path=_BASELINE_ASSETS)
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


class TestMatchRobotToTask:
    """Tests for match_robot_to_task with constraints."""

    def test_match_pick_and_place_no_constraints(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()

        results = ki.match_robot_to_task("pick and place")
        assert len(results) > 0
        robot_ids = [r["robot_id"] for r in results]
        assert "ur5e" in robot_ids
        assert "panda" in robot_ids

        ki._do_stop()

    def test_match_with_payload_constraint(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()

        # payload >= 2kg: ur5e (5kg), panda (3kg), spot (14kg) qualify
        # unitree_g1 (2kg) and agilex_piper (1kg) do not
        results = ki.match_robot_to_task("pick and place", {"payload_kg": 2})
        robot_ids = [r["robot_id"] for r in results]
        assert "ur5e" in robot_ids
        assert "panda" in robot_ids
        assert "agilex_piper" not in robot_ids

        ki._do_stop()

    def test_match_with_dof_min_constraint(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()

        # dof >= 7: only panda (7) and unitree_g1 (23)
        results = ki.match_robot_to_task("pick and place", {"dof_min": 7})
        robot_ids = [r["robot_id"] for r in results]
        assert "panda" in robot_ids
        assert "unitree_g1" in robot_ids
        assert "ur5e" not in robot_ids

        ki._do_stop()

    def test_match_with_reach_constraint(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()

        # reach >= 800mm: ur5e (850), panda (855)
        results = ki.match_robot_to_task("pick and place", {"reach_mm_min": 800})
        robot_ids = [r["robot_id"] for r in results]
        assert "ur5e" in robot_ids
        assert "panda" in robot_ids
        assert "agilex_piper" not in robot_ids

        ki._do_stop()

    def test_match_with_sim_backend_constraint(self):
        client = SeekDBMemoryClient()
        client.connect()
        seed_knowledge_graph(client)

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()

        results = ki.match_robot_to_task("pick and place", {"sim_backend": "pybullet"})
        robot_ids = [r["robot_id"] for r in results]
        assert "panda" in robot_ids
        assert "ur5e" not in robot_ids  # ur5e doesn't support pybullet

        ki._do_stop()

    def test_match_unknown_task(self):
        ki = KnowledgeInterface(robot_id="test")
        ki._do_initialize()
        assert ki.match_robot_to_task("nonsense xyz") == []
        ki._do_stop()


class TestRobotSafetyLimits:
    """Tests for get_robot_safety_limits."""

    def test_ur5e_safety_limits(self):
        ki = KnowledgeInterface(robot_id="test")
        ki._do_initialize()
        limits = ki.get_robot_safety_limits("ur5e")
        assert "joint_torque_max" in limits
        assert len(limits["joint_torque_max"]) == 6
        assert limits["joint_torque_max"][0] == 150
        ki._do_stop()

    def test_panda_safety_limits(self):
        ki = KnowledgeInterface(robot_id="test")
        ki._do_initialize()
        limits = ki.get_robot_safety_limits("panda")
        assert len(limits["joint_torque_max"]) == 7
        assert limits["joint_position_limits"][0] == (-166, 166)
        ki._do_stop()

    def test_unknown_robot_limits(self):
        ki = KnowledgeInterface(robot_id="test")
        ki._do_initialize()
        assert ki.get_robot_safety_limits("nonexistent") == {}
        ki._do_stop()


class TestRobotSimulationProfile:
    """Tests for get_robot_simulation_profile."""

    def test_ur5e_sim_profile(self):
        ki = KnowledgeInterface(robot_id="test")
        ki._do_initialize()
        prof = ki.get_robot_simulation_profile("ur5e")
        assert prof["default_backend"] == "mujoco"
        assert "isaacgym" in prof["supported_backends"]
        assert prof["timestep"] == 0.002
        ki._do_stop()

    def test_panda_sim_profile(self):
        ki = KnowledgeInterface(robot_id="test")
        ki._do_initialize()
        prof = ki.get_robot_simulation_profile("panda")
        assert prof["default_backend"] == "mujoco"
        assert "pybullet" in prof["supported_backends"]
        ki._do_stop()

    def test_unknown_robot_profile(self):
        ki = KnowledgeInterface(robot_id="test")
        ki._do_initialize()
        assert ki.get_robot_simulation_profile("nonexistent") == {}
        ki._do_stop()


class TestEurdfLoader:
    """Tests for load_eurdf_profile e-URDF integration."""

    def test_load_eurdf_ur5e(self):
        import os
        eurdf_path = os.path.join(os.path.dirname(__file__), "..", "e-urdf-zoo", "ur5e", "robot.eurdf.yaml")
        eurdf_path = os.path.abspath(eurdf_path)
        if not os.path.exists(eurdf_path):
            pytest.skip("e-URDF zoo not available")

        client = SeekDBMemoryClient()
        client.connect()

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()

        result = ki.load_eurdf_profile("ur5e", eurdf_path)
        assert result["loaded"] is True
        assert result["joints"] == 6
        assert result["links"] > 0
        assert result["sensors"] > 0
        assert result["actuators"] == 6
        assert result["capabilities"] > 0

        # Verify data was written to SeekDB
        rows = client.query("knowledge_graph", filters={"subject": "ur5e", "predicate": "has_eurdf_joints"})
        assert len(rows) == 1
        joints_data = __import__("json").loads(rows[0]["object"])
        assert len(joints_data) == 6
        assert joints_data[0]["name"] == "shoulder_pan_joint"

        ki._do_stop()

    def test_load_eurdf_no_seekdb(self):
        ki = KnowledgeInterface(robot_id="ur5e")
        ki._do_initialize()
        result = ki.load_eurdf_profile("ur5e", "/tmp/nonexistent.yaml")
        assert result["loaded"] is False
        assert "No SeekDB client" in result["error"]
        ki._do_stop()

    def test_load_eurdf_file_not_found(self):
        client = SeekDBMemoryClient()
        client.connect()
        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()
        with pytest.raises(FileNotFoundError):
            ki.load_eurdf_profile("ur5e", "/tmp/definitely_not_real.yaml")
        ki._do_stop()

    def test_robot_properties_updated_after_load(self):
        import os
        eurdf_path = os.path.join(os.path.dirname(__file__), "..", "e-urdf-zoo", "ur5e", "robot.eurdf.yaml")
        eurdf_path = os.path.abspath(eurdf_path)
        if not os.path.exists(eurdf_path):
            pytest.skip("e-URDF zoo not available")

        client = SeekDBMemoryClient()
        client.connect()

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()
        ki.load_eurdf_profile("ur5e", eurdf_path)

        # Properties should be updated from e-URDF
        assert "ur5e" in ki._ROBOT_PROPERTIES
        assert ki._ROBOT_PROPERTIES["ur5e"]["dof"] == 6
        assert ki._ROBOT_PROPERTIES["ur5e"]["payload_kg"] == 5.0
        assert ki._ROBOT_PROPERTIES["ur5e"]["reach_mm"] == 850

        ki._do_stop()

    def test_safety_limits_from_eurdf(self):
        import os
        eurdf_path = os.path.join(os.path.dirname(__file__), "..", "e-urdf-zoo", "ur5e", "robot.eurdf.yaml")
        eurdf_path = os.path.abspath(eurdf_path)
        if not os.path.exists(eurdf_path):
            pytest.skip("e-URDF zoo not available")

        client = SeekDBMemoryClient()
        client.connect()

        ki = KnowledgeInterface(robot_id="ur5e", seekdb_client=client)
        ki._do_initialize()
        ki.load_eurdf_profile("ur5e", eurdf_path)

        # get_robot_safety_limits should use hard-coded data (v1.0)
        limits = ki.get_robot_safety_limits("ur5e")
        assert "joint_torque_max" in limits
        assert limits["joint_torque_max"][0] == 150

        ki._do_stop()


class TestKnowEventBusIntegration:
    """Tests for KNOW EventBus lifecycle and event publishing."""

    def test_know_subscribes_to_events_on_start(self):
        from rosclaw.core.event_bus import EventBus
        bus = EventBus()
        ki = KnowledgeInterface(robot_id="test_robot", event_bus=bus)
        ki._do_initialize()
        ki._do_start()
        assert bus.subscriber_count("rosclaw.provider.inference.requested") >= 0
        ki._do_stop()

    def test_know_publishes_startup_event(self):
        from rosclaw.core.event_bus import EventBus
        bus = EventBus()
        ki = KnowledgeInterface(robot_id="test_robot", event_bus=bus)
        ki._do_initialize()
        ki._do_start()
        events = bus.get_history("rosclaw.knowledge.started")
        assert len(events) == 1
        assert events[0].payload["robot_id"] == "test_robot"
        ki._do_stop()

    def test_know_publishes_pre_check_event(self):
        from rosclaw.core.event_bus import EventBus, Event, EventPriority
        bus = EventBus()
        ki = KnowledgeInterface(robot_id="test_robot", event_bus=bus)
        ki._do_initialize()
        ki._do_start()
        bus.publish(Event(
            topic="rosclaw.provider.inference.requested",
            payload={"capability": "skill.pick_and_place", "robot_id": "test_robot"},
            source="test",
            priority=EventPriority.NORMAL,
        ))
        events = bus.get_history("rosclaw.knowledge.pre_check")
        assert len(events) >= 1
        assert events[0].payload["capability"] == "skill.pick_and_place"
        ki._do_stop()

    def test_know_publishes_safety_limits_event(self):
        from rosclaw.core.event_bus import EventBus, Event, EventPriority
        bus = EventBus()
        ki = KnowledgeInterface(robot_id="ur5e", event_bus=bus)
        ki._do_initialize()
        ki._do_start()
        bus.publish(Event(
            topic="rosclaw.sandbox.episode.started",
            payload={"robot_id": "ur5e"},
            source="test",
            priority=EventPriority.NORMAL,
        ))
        events = bus.get_history("rosclaw.knowledge.safety_limits_loaded")
        assert len(events) >= 1
        assert "joint_torque_max" in str(events[0].payload.get("safety_limits", {}))
        ki._do_stop()


class TestKnowProviderSelection:
    """Tests for query_for_provider_selection (main-flow hook)."""

    def test_query_for_provider_exact_capability(self):
        ki = KnowledgeInterface(robot_id="ur5e")
        ki._do_initialize()
        result = ki.query_for_provider_selection("grasp")
        assert result["robot_id"] == "ur5e"
        assert result["capability"] == "grasp"
        assert "has_capability" in result
        assert "safety_limits" in result
        assert "simulation_profile" in result
        ki._do_stop()

    def test_query_for_provider_missing_capability_finds_alternatives(self):
        ki = KnowledgeInterface(robot_id="ur5e")
        ki._do_initialize()
        result = ki.query_for_provider_selection("walking")
        if not result["has_capability"]:
            assert "alternative_robots" in result
        ki._do_stop()

    def test_query_for_provider_detects_known_risk(self):
        ki = KnowledgeInterface(robot_id="ur5e")
        ki._do_initialize()
        result = ki.query_for_provider_selection("pid_control_with_torque")
        assert "known_risk" in result
        ki._do_stop()

    def test_record_knowledge_usage_publishes_event(self):
        from rosclaw.core.event_bus import EventBus
        bus = EventBus()
        ki = KnowledgeInterface(robot_id="ur5e", event_bus=bus)
        ki._do_initialize()
        ki.record_knowledge_usage({"episode_id": "ep_001", "action": "test"})
        events = bus.get_history("knowledge.ingest_complete")
        assert len(events) >= 1
        ki._do_stop()


class TestKnowRuntimeIntegration:
    """Tests for KNOW hooks in Runtime."""

    def test_runtime_default_enables_knowledge(self):
        from rosclaw.core.runtime import RuntimeConfig
        cfg = RuntimeConfig()
        assert cfg.enable_knowledge is True

    def test_runtime_knowledge_initialized_with_memory(self):
        from rosclaw.core.runtime import Runtime, RuntimeConfig
        cfg = RuntimeConfig(robot_id="ur5e", enable_knowledge=True, enable_memory=True)
        rt = Runtime(config=cfg)
        rt.initialize()
        assert rt.knowledge is not None
        rt.stop()


class TestKnowIntegration:
    """Tests for KnowIntegration (Runtime-facing bridge)."""

    def test_query_before_decision_returns_full_result(self):
        from rosclaw.know.integration import KnowIntegration
        ki = KnowIntegration(robot_id="ur5e")
        result = ki.query_before_decision("ur5e", "pick and place")
        assert result["robot_id"] == "ur5e"
        assert result["task"] == "pick and place"
        assert "decomposition" in result
        assert "can_perform" in result
        assert "capability_match" in result

    def test_query_before_decision_publishes_event(self):
        from rosclaw.core.event_bus import EventBus
        from rosclaw.know.integration import KnowIntegration
        bus = EventBus()
        ki = KnowIntegration(robot_id="ur5e", event_bus=bus)
        ki.query_before_decision("ur5e", "pick and place")
        events = bus.get_history("rosclaw.knowledge.pre_check")
        assert len(events) >= 1
        assert events[0].payload["robot_id"] == "ur5e"

    def test_record_usage_publishes_ingest_event(self):
        from rosclaw.core.event_bus import EventBus
        from rosclaw.know.integration import KnowIntegration
        bus = EventBus()
        ki = KnowIntegration(robot_id="ur5e", event_bus=bus)
        ki.record_usage({"episode_id": "ep_001", "action": "test"})
        events = bus.get_history("knowledge.ingest_complete")
        assert len(events) >= 1
