"""Coverage tests for MCPHub edge cases and uncovered branches."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from rosclaw.agent_runtime.mcp_hub import AgentContext, MCPHub
from rosclaw.core.event_bus import Event, EventBus

# ------------------------------------------------------------------
# AgentContext
# ------------------------------------------------------------------

class TestAgentContext:
    def test_to_mcp_context(self):
        ctx = AgentContext(
            session_id="s1",
            robot_id="bot1",
            current_task="pick",
            task_history=[{"step": 1}],
            robot_model_description="UR5e",
            current_joint_positions=[0.1, 0.2],
            current_end_effector_pose=[1.0, 2.0, 3.0],
            active_skills=["grasp"],
            safety_level="strict",
        )
        d = ctx.to_mcp_context()
        assert d["session_id"] == "s1"
        assert d["robot"]["id"] == "bot1"
        assert d["robot"]["current_state"]["joint_positions"] == [0.1, 0.2]
        assert d["current_task"] == "pick"
        assert d["active_skills"] == ["grasp"]
        assert d["safety_level"] == "strict"


# ------------------------------------------------------------------
# Lifecycle & initialization
# ------------------------------------------------------------------

class TestMCPHubLifecycle:
    def test_initialize_import_error_falls_back(self, caplog):
        import logging
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        with patch("mcp.server.Server", side_effect=ImportError("no mcp")), caplog.at_level(logging.WARNING, logger="rosclaw.agent_runtime.mcp_hub"):
            hub.initialize()
        assert "mock mode" in caplog.text
        assert hub._server is None
        hub.stop()

    def test_stop_cancels_pending_futures(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()

        loop = asyncio.new_event_loop()
        fut = loop.create_future()
        hub._pending_requests["req_1"] = fut
        hub.stop()
        assert fut.cancelled()
        assert len(hub._pending_requests) == 0
        loop.close()

    def test_tools_property(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        tools = hub.tools
        assert isinstance(tools, list)
        assert len(tools) > 0
        hub.stop()

    def test_update_robot_description(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.update_robot_description("UR5e arm")
        assert hub.context.robot_model_description == "UR5e arm"

    def test_do_start_logs(self, caplog):
        import logging
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        with caplog.at_level(logging.INFO, logger="rosclaw.agent_runtime.mcp_hub"):
            hub.start()
        assert "MCP Hub started" in caplog.text
        hub.stop()


# ------------------------------------------------------------------
# Event handlers
# ------------------------------------------------------------------

class TestMCPHubEventHandlers:
    def test_on_joint_states(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        bus.publish(Event(topic="robot.joint_states", payload={"positions": [0.1, 0.2, 0.3]}))
        assert hub.context.current_joint_positions == [0.1, 0.2, 0.3]
        hub.stop()

    def test_on_joint_states_non_dict_payload(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        bus.publish(Event(topic="robot.joint_states", payload=[0.1, 0.2]))
        # Should not crash, positions unchanged
        hub.stop()

    def test_on_end_effector_pose(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        bus.publish(Event(topic="robot.end_effector_pose", payload={"x": 1.0, "y": 2.0}))
        assert hub.context.current_end_effector_pose == {"x": 1.0, "y": 2.0}
        hub.stop()

    def test_on_agent_response(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()

        loop = asyncio.new_event_loop()
        fut = loop.create_future()
        hub._pending_requests["abc123"] = fut
        bus.publish(Event(
            topic="agent.response",
            payload={"result": {"status": "ok"}},
            metadata={"request_id": "abc123"},
        ))
        assert fut.done()
        assert fut.result() == {"status": "ok"}
        loop.close()
        hub.stop()


# ------------------------------------------------------------------
# handle_tool_call — low-level tools without runtime
# ------------------------------------------------------------------

class TestHandleToolCallNoRuntime:
    @pytest.mark.asyncio
    async def test_get_robot_state(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        hub.context.current_joint_positions = [0.1, 0.2]
        result = await hub.handle_tool_call("get_robot_state", {})
        assert result["status"] == "ok"
        assert result["robot_state"]["joint_positions"] == [0.1, 0.2]
        hub.stop()

    @pytest.mark.asyncio
    async def test_emergency_stop(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        result = await hub.handle_tool_call("emergency_stop", {})
        assert result["status"] == "emergency_stop_triggered"
        hub.stop()

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        result = await hub.handle_tool_call("nonexistent", {})
        assert "error" in result
        hub.stop()

    @pytest.mark.asyncio
    async def test_move_joints_timeout(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        # No subscriber to agent.command — will timeout immediately
        hub._default_timeout = 0.01
        result = await hub.handle_tool_call("move_joints", {"joint_positions": [0.1, 0.2]})
        assert result["status"] == "command_issued"
        hub.stop()

    @pytest.mark.asyncio
    async def test_grasp_timeout(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        hub._default_timeout = 0.01
        result = await hub.handle_tool_call("grasp", {"action": "close"})
        assert result["status"] == "command_issued"
        hub.stop()

    @pytest.mark.asyncio
    async def test_validate_trajectory_timeout(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        hub._default_timeout = 0.01
        result = await hub.handle_tool_call("validate_trajectory", {"waypoints": [[0.0] * 6]})
        assert result["status"] == "validation_requested"
        hub.stop()

    @pytest.mark.asyncio
    async def test_move_joints_with_response(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()

        def responder(event):
            req_id = event.metadata.get("request_id")
            bus.publish(Event(
                topic="agent.response",
                payload={"status": "ok", "moved": True},
                metadata={"request_id": req_id},
            ))

        bus.subscribe("agent.command", responder)
        result = await hub.handle_tool_call("move_joints", {"joint_positions": [0.1, 0.2]})
        assert result["status"] == "ok"
        hub.stop()

    @pytest.mark.asyncio
    async def test_grasp_with_response(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()

        def responder(event):
            req_id = event.metadata.get("request_id")
            bus.publish(Event(
                topic="agent.response",
                payload={"status": "ok", "closed": True},
                metadata={"request_id": req_id},
            ))

        bus.subscribe("agent.command", responder)
        result = await hub.handle_tool_call("grasp", {"action": "close"})
        assert result["status"] == "ok"
        hub.stop()

    @pytest.mark.asyncio
    async def test_validate_trajectory_with_response(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()

        def responder(event):
            req_id = event.metadata.get("request_id")
            bus.publish(Event(
                topic="agent.response",
                payload={"status": "ok", "valid": True},
                metadata={"request_id": req_id},
            ))

        bus.subscribe("agent.command", responder)
        result = await hub.handle_tool_call("validate_trajectory", {"waypoints": [[0.0] * 6]})
        assert result["status"] == "ok"
        hub.stop()

    @pytest.mark.asyncio
    async def test_query_world_objects_via_tool_call(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        result = await hub.handle_tool_call("query_world_objects", {"scene_id": "s1"})
        assert result["status"] == "error"
        hub.stop()

    @pytest.mark.asyncio
    async def test_get_scene_graph_via_tool_call(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        result = await hub.handle_tool_call("get_scene_graph", {"scene_id": "s1"})
        assert result["status"] == "error"
        hub.stop()

    @pytest.mark.asyncio
    async def test_cognitive_search_via_tool_call(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        result = await hub.handle_tool_call("cognitive_search", {"query": "cup"})
        assert result["status"] == "error"
        hub.stop()

    @pytest.mark.asyncio
    async def test_query_knowledge_via_tool_call(self):
        mock_rt = MagicMock()
        mock_rt.knowledge.query_robot_capabilities.return_value = ["grasp"]
        bus = EventBus()
        hub = MCPHub(event_bus=bus, runtime=mock_rt)
        hub.initialize()
        result = await hub.handle_tool_call("query_knowledge", {"query_type": "capability", "query": "ur5e"})
        assert result["status"] == "ok"
        hub.stop()


# ------------------------------------------------------------------
# _route_capability edge cases
# ------------------------------------------------------------------

class TestRouteCapability:
    @pytest.mark.asyncio
    async def test_route_capability_no_event_bus(self):
        hub = MCPHub(event_bus=None)
        # Skip initialization to avoid None event_bus crash
        hub._lifecycle_state = hub.state.READY
        result = await hub._route_capability("vlm.test", {})
        assert result["status"] == "failed"
        assert "EventBus not available" in result["error"]

    @pytest.mark.asyncio
    async def test_route_capability_timeout(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        hub._default_timeout = 0.01
        result = await hub._route_capability("vlm.test", {})
        assert result["status"] == "failed"
        assert "timeout" in result["error"]
        hub.stop()

    @pytest.mark.asyncio
    async def test_route_capability_success_via_event_bus(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        hub._default_timeout = 1.0

        # Subscribe to capability request and respond
        def responder(event):
            req_id = event.payload.get("request_id")
            bus.publish(Event(
                topic="agent.capability.response",
                payload={"request_id": req_id, "result": {"found": True}},
            ))

        bus.subscribe("agent.capability.request", responder)
        result = await hub._route_capability("vlm.scene_understanding", {"query": "cup"})
        assert result["found"] is True
        hub.stop()


# ------------------------------------------------------------------
# _route_capability_direct (with mocked Runtime)
# ------------------------------------------------------------------

class TestRouteCapabilityDirect:
    @pytest.mark.asyncio
    async def test_direct_no_runtime(self):
        hub = MCPHub(event_bus=EventBus(), runtime=None)
        hub.initialize()
        result = await hub._route_capability_direct("r1", "vlm.test", {}, {})
        assert result["status"] == "failed"
        assert "Runtime not available" in result["error"]
        hub.stop()

    @pytest.mark.asyncio
    async def test_direct_no_router(self):
        mock_rt = MagicMock()
        mock_rt.capability_router = None
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = await hub._route_capability_direct("r1", "vlm.test", {}, {})
        assert result["status"] == "failed"
        assert "CapabilityRouter not available" in result["error"]
        hub.stop()

    @pytest.mark.asyncio
    async def test_direct_router_success(self):
        mock_resp = MagicMock()
        mock_resp.is_ok = True
        mock_resp.provider = "test_provider"
        mock_resp.result = {"boxes": []}
        mock_resp.confidence = 0.9
        mock_resp.latency_ms = 15
        mock_resp.warnings = []
        mock_resp.errors = []

        async def _invoke_ok(req):
            return mock_resp
        mock_router = MagicMock()
        mock_router.invoke = _invoke_ok

        mock_rt = MagicMock()
        mock_rt.capability_router = mock_router
        mock_rt.guard_pipeline = None

        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = await hub._route_capability_direct("r1", "vlm.test", {}, {})
        assert result["status"] == "ok"
        assert result["provider"] == "test_provider"
        hub.stop()

    @pytest.mark.asyncio
    async def test_direct_router_exception(self):
        async def _invoke_err(req):
            raise RuntimeError("boom")
        mock_router = MagicMock()
        mock_router.invoke = _invoke_err

        mock_rt = MagicMock()
        mock_rt.capability_router = mock_router
        mock_rt.guard_pipeline = None

        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = await hub._route_capability_direct("r1", "vlm.test", {}, {})
        assert result["status"] == "failed"
        assert "boom" in result["error"]
        hub.stop()

    @pytest.mark.asyncio
    async def test_direct_guard_blocks(self):
        mock_resp = MagicMock()
        mock_resp.is_ok = True
        mock_resp.result = {"action": "dangerous"}

        async def _invoke_guard(req):
            return mock_resp
        mock_router = MagicMock()
        mock_router.invoke = _invoke_guard

        mock_guard = MagicMock()
        mock_guard.check = MagicMock(side_effect=RuntimeError("unsafe"))

        mock_rt = MagicMock()
        mock_rt.capability_router = mock_router
        mock_rt.guard_pipeline = mock_guard

        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = await hub._route_capability_direct("r1", "vlm.test", {}, {})
        assert result["status"] == "blocked"
        assert "Guard blocked" in result["error"]
        hub.stop()


# ------------------------------------------------------------------
# Semantic handlers with mocked Runtime
# ------------------------------------------------------------------

class TestSemanticHandlers:
    @pytest.mark.asyncio
    async def test_observe_scene_routes_capability(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        hub._default_timeout = 0.01
        result = await hub._handle_observe_scene({"image_topic": "/cam", "query": "cup"})
        assert result["status"] == "failed"
        hub.stop()

    @pytest.mark.asyncio
    async def test_locate_object_routes_capability(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        hub._default_timeout = 0.01
        result = await hub._handle_locate_object({"object_name": "cup"})
        assert result["status"] == "failed"
        hub.stop()

    @pytest.mark.asyncio
    async def test_delegate_skill_routes_capability(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        hub._default_timeout = 0.01
        result = await hub._handle_delegate_skill({"skill": "grasp", "target": {}})
        assert result["status"] == "failed"
        hub.stop()

    @pytest.mark.asyncio
    async def test_verify_task_success_routes_capability(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        hub._default_timeout = 0.01
        result = await hub._handle_verify_task_success({"task_description": "pick cup"})
        assert result["status"] == "failed"
        hub.stop()


# ------------------------------------------------------------------
# _handle_query_knowledge
# ------------------------------------------------------------------

class TestHandleQueryKnowledge:
    def test_no_runtime(self):
        hub = MCPHub(event_bus=EventBus(), runtime=None)
        hub.initialize()
        result = hub._handle_query_knowledge({"query_type": "capability", "query": "ur5e"})
        assert result["status"] == "error"
        assert "Runtime not available" in result["error"]
        hub.stop()

    def test_no_knowledge_module(self):
        mock_rt = MagicMock()
        mock_rt.knowledge = None
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = hub._handle_query_knowledge({"query_type": "capability", "query": "ur5e"})
        assert result["status"] == "error"
        assert "Knowledge module not available" in result["error"]
        hub.stop()

    def test_capability_query(self):
        mock_rt = MagicMock()
        mock_rt.knowledge.query_robot_capabilities.return_value = ["grasp", "place"]
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = hub._handle_query_knowledge({"query_type": "capability", "query": "ur5e"})
        assert result["status"] == "ok"
        assert result["query_type"] == "capability"
        assert result["count"] == 2
        hub.stop()

    def test_symptom_query(self):
        mock_rt = MagicMock()
        mock_rt.knowledge.match_symptom.return_value = {"symptom": "overload"}
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = hub._handle_query_knowledge({"query_type": "symptom", "query": "torque high"})
        assert result["status"] == "ok"
        assert result["matched"] is True
        hub.stop()

    def test_analogy_query(self):
        mock_rt = MagicMock()
        mock_rt.knowledge.get_analogy.return_value = {"analogy": "spring"}
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = hub._handle_query_knowledge({"query_type": "analogy", "query": "force"})
        assert result["status"] == "ok"
        assert result["matched"] is True
        hub.stop()

    def test_unknown_query_type(self):
        mock_rt = MagicMock()
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = hub._handle_query_knowledge({"query_type": "bogus", "query": "x"})
        assert result["status"] == "error"
        assert "Unknown query_type" in result["error"]
        hub.stop()


# ------------------------------------------------------------------
# _handle_get_recovery_strategy
# ------------------------------------------------------------------

class TestHandleGetRecoveryStrategy:
    @pytest.mark.asyncio
    async def test_no_runtime(self):
        hub = MCPHub(event_bus=EventBus(), runtime=None)
        hub.initialize()
        result = await hub._handle_get_recovery_strategy({"error_log": "boom"})
        assert result["status"] == "error"
        assert "Runtime not available" in result["error"]
        hub.stop()

    @pytest.mark.asyncio
    async def test_no_how_engine(self):
        mock_rt = MagicMock()
        mock_rt.how = None
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = await hub._handle_get_recovery_strategy({"error_log": "boom"})
        assert result["status"] == "error"
        assert "HeuristicEngine not available" in result["error"]
        hub.stop()

    @pytest.mark.asyncio
    async def test_empty_error_log(self):
        mock_rt = MagicMock()
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = await hub._handle_get_recovery_strategy({"error_log": ""})
        assert result["status"] == "error"
        assert "error_log is required" in result["error"]
        hub.stop()

    @pytest.mark.asyncio
    async def test_exception_from_suggest_recovery(self):
        mock_rt = MagicMock()
        mock_rt.how.suggest_recovery.side_effect = RuntimeError("lookup fail")
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = await hub._handle_get_recovery_strategy({"error_log": "error"})
        assert result["status"] == "error"
        assert "Recovery lookup failed" in result["error"]
        hub.stop()

    @pytest.mark.asyncio
    async def test_no_recovery_found(self):
        mock_rt = MagicMock()
        mock_rt.how.suggest_recovery.return_value = None
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = await hub._handle_get_recovery_strategy({"error_log": "weird error"})
        assert result["status"] == "ok"
        assert result["matched"] is False
        hub.stop()

    @pytest.mark.asyncio
    async def test_recovery_found(self):
        mock_rt = MagicMock()
        mock_rt.how.suggest_recovery.return_value = {
            "rule_id": "r1",
            "condition": "overload",
            "action": "reduce speed",
            "priority": 1,
            "source": "heuristic",
        }
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = await hub._handle_get_recovery_strategy({"error_log": "torque high"})
        assert result["status"] == "ok"
        assert result["matched"] is True
        assert result["rule_id"] == "r1"
        hub.stop()

    @pytest.mark.asyncio
    async def test_recovery_coroutine_awaited(self):
        async def async_recovery(error_log):
            return {"rule_id": "async_r1", "condition": "async", "action": "wait", "priority": 0}

        mock_rt = MagicMock()
        mock_rt.how.suggest_recovery = async_recovery
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = await hub._handle_get_recovery_strategy({"error_log": "async error"})
        assert result["status"] == "ok"
        assert result["rule_id"] == "async_r1"
        hub.stop()


# ------------------------------------------------------------------
# _handle_get_safety_heuristic
# ------------------------------------------------------------------

class TestHandleGetSafetyHeuristic:
    def test_no_runtime(self):
        hub = MCPHub(event_bus=EventBus(), runtime=None)
        hub.initialize()
        result = hub._handle_get_safety_heuristic({"condition": "torque_overflow"})
        assert result["status"] == "error"
        assert "Runtime not available" in result["error"]
        hub.stop()

    def test_no_knowledge(self):
        mock_rt = MagicMock()
        mock_rt.knowledge = None
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = hub._handle_get_safety_heuristic({"condition": "torque_overflow"})
        assert result["status"] == "error"
        assert "Knowledge module not available" in result["error"]
        hub.stop()

    def test_unknown_condition(self):
        mock_rt = MagicMock()
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = hub._handle_get_safety_heuristic({"condition": "unknown_thing"})
        assert result["status"] == "error"
        assert "Unknown condition" in result["error"]
        assert "known_conditions" in result
        hub.stop()

    def test_valid_condition(self):
        mock_rt = MagicMock()
        mock_rt.knowledge.get_safety_rule.return_value = {"threshold": 100}
        hub = MCPHub(event_bus=EventBus(), runtime=mock_rt)
        hub.initialize()
        result = hub._handle_get_safety_heuristic({"condition": "torque_overflow"})
        assert result["status"] == "ok"
        assert result["condition"] == "torque_overflow"
        assert result["safety_rule"]["threshold"] == 100
        hub.stop()


# ------------------------------------------------------------------
# Physical world handlers (with mocked Runtime)
# ------------------------------------------------------------------

class TestPhysicalWorldHandlers:
    def test_query_world_objects_event_bus_path(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        result = hub._handle_query_world_objects({"scene_id": "scene_1"})
        # No runtime, returns error
        assert result["status"] == "error"
        hub.stop()

    def test_query_world_objects_runtime_fallback(self):
        bus = EventBus()
        mock_rt = MagicMock()
        mock_obj = MagicMock()
        mock_obj.to_dict.return_value = {"obj_id": "obj_1"}
        mock_rt.search_world_objects.return_value = [mock_obj]
        hub = MCPHub(event_bus=bus, runtime=mock_rt)
        hub.initialize()
        result = hub._handle_query_world_objects({"scene_id": "scene_1"})
        assert result["status"] == "ok"
        assert result["count"] == 1
        assert result["objects"][0]["obj_id"] == "obj_1"
        hub.stop()

    def test_query_world_objects_vec3_import_error(self):
        bus = EventBus()
        mock_rt = MagicMock()
        mock_rt.search_world_objects.return_value = []
        hub = MCPHub(event_bus=bus, runtime=mock_rt)
        hub.initialize()
        with patch.dict("sys.modules", {"rosclaw.e_urdf.parser": None}):
            result = hub._handle_query_world_objects({"scene_id": "scene_1"})
        assert result["status"] == "ok"
        hub.stop()

    def test_get_scene_graph_event_bus_path(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        result = hub._handle_get_scene_graph({"scene_id": "scene_1"})
        assert result["status"] == "error"
        hub.stop()

    def test_get_scene_graph_runtime_fallback(self):
        bus = EventBus()
        mock_rt = MagicMock()
        mock_obj = MagicMock()
        mock_obj.to_dict.return_value = {"obj_id": "o1"}
        mock_rel = MagicMock()
        mock_rel.to_dict.return_value = {"subject_id": "o1", "object_id": "o2", "relation": "left"}
        mock_rt.get_scene_graph.return_value = ([mock_obj], [mock_rel])
        hub = MCPHub(event_bus=bus, runtime=mock_rt)
        hub.initialize()
        result = hub._handle_get_scene_graph({"scene_id": "scene_1"})
        assert result["status"] == "ok"
        assert result["object_count"] == 1
        assert result["relation_count"] == 1
        hub.stop()

    def test_cognitive_search_event_bus_path(self):
        bus = EventBus()
        hub = MCPHub(event_bus=bus)
        hub.initialize()
        result = hub._handle_cognitive_search({"query": "red cup"})
        assert result["status"] == "error"
        hub.stop()

    def test_cognitive_search_runtime_fallback(self):
        bus = EventBus()
        mock_rt = MagicMock()
        mock_atom = MagicMock()
        mock_atom.to_dict.return_value = {"content": "found cup"}
        mock_rt.cognitive_search.return_value = [mock_atom]
        hub = MCPHub(event_bus=bus, runtime=mock_rt)
        hub.initialize()
        result = hub._handle_cognitive_search({"query": "red cup"})
        assert result["status"] == "ok"
        assert result["count"] == 1
        hub.stop()


# ------------------------------------------------------------------
# Static helper methods
# ------------------------------------------------------------------

class TestStaticHelpers:
    def test_world_object_to_dict_with_to_dict(self):
        obj = MagicMock()
        obj.to_dict.return_value = {"obj_id": "o1", "label": "cup"}
        result = MCPHub._world_object_to_dict(obj)
        assert result["label"] == "cup"

    def test_world_object_to_dict_fallback(self):
        obj = MagicMock()
        del obj.to_dict
        obj.obj_id = "o2"
        result = MCPHub._world_object_to_dict(obj)
        assert result["obj_id"] == "o2"

    def test_relation_to_dict_with_to_dict(self):
        rel = MagicMock()
        rel.to_dict.return_value = {"subject_id": "s1"}
        result = MCPHub._relation_to_dict(rel)
        assert result["subject_id"] == "s1"

    def test_relation_to_dict_fallback(self):
        rel = MagicMock()
        del rel.to_dict
        rel.subject_id = "s1"
        rel.object_id = "o1"
        rel.relation = "on"
        result = MCPHub._relation_to_dict(rel)
        assert result["relation"] == "on"

    def test_memory_atom_to_dict_with_to_dict(self):
        atom = MagicMock()
        atom.to_dict.return_value = {"content": "hello"}
        result = MCPHub._memory_atom_to_dict(atom)
        assert result["content"] == "hello"

    def test_memory_atom_to_dict_fallback(self):
        atom = MagicMock()
        del atom.to_dict
        atom.content = "world"
        result = MCPHub._memory_atom_to_dict(atom)
        assert result["content"] == "world"


# ------------------------------------------------------------------
# handle_tool_call with provider layer (semantic tools)
# ------------------------------------------------------------------

class TestHandleToolCallWithProviderLayer:
    @pytest.fixture
    def provider_hub(self):
        mock_rt = MagicMock()
        mock_rt.capability_router = MagicMock()
        bus = EventBus()
        hub = MCPHub(event_bus=bus, runtime=mock_rt)
        hub.initialize()
        return hub

    @pytest.mark.asyncio
    async def test_handle_observe_scene(self, provider_hub):
        hub = provider_hub
        hub._default_timeout = 0.01
        result = await hub.handle_tool_call("observe_scene", {"image_topic": "/cam"})
        assert result["status"] == "failed"
        hub.stop()

    @pytest.mark.asyncio
    async def test_handle_locate_object(self, provider_hub):
        hub = provider_hub
        hub._default_timeout = 0.01
        result = await hub.handle_tool_call("locate_object", {"object_name": "cup"})
        assert result["status"] == "failed"
        hub.stop()

    @pytest.mark.asyncio
    async def test_handle_delegate_skill(self, provider_hub):
        hub = provider_hub
        hub._default_timeout = 0.01
        result = await hub.handle_tool_call("delegate_skill", {"skill": "grasp"})
        assert result["status"] == "failed"
        hub.stop()

    @pytest.mark.asyncio
    async def test_handle_verify_task_success(self, provider_hub):
        hub = provider_hub
        hub._default_timeout = 0.01
        result = await hub.handle_tool_call("verify_task_success", {"task_description": "pick"})
        assert result["status"] == "failed"
        hub.stop()

    @pytest.mark.asyncio
    async def test_handle_get_recovery_strategy(self, provider_hub):
        hub = provider_hub
        hub.runtime.how = None
        result = await hub.handle_tool_call("get_recovery_strategy", {"error_log": "boom"})
        assert result["status"] == "error"
        hub.stop()

    @pytest.mark.asyncio
    async def test_handle_get_safety_heuristic(self, provider_hub):
        hub = provider_hub
        hub.runtime.knowledge = None
        result = await hub.handle_tool_call("get_safety_heuristic", {"condition": "torque_overflow"})
        assert result["status"] == "error"
        hub.stop()
