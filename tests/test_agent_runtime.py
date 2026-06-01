"""Tests for Agent Runtime."""


from rosclaw.core.event_bus import EventBus, Event
from rosclaw.agent_runtime.mcp_hub import MCPHub, AgentContext


def test_agent_context():
    ctx = AgentContext(session_id="sess_1", robot_id="ur5e_001")
    ctx.current_joint_positions = [0.1] * 6
    mcp = ctx.to_mcp_context()
    assert mcp["session_id"] == "sess_1"
    assert mcp["robot"]["id"] == "ur5e_001"
    assert len(mcp["robot"]["current_state"]["joint_positions"]) == 6


def test_mcp_hub_tools():
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()
    tools = hub.tools
    assert len(tools) == 8
    names = [t["name"] for t in tools]
    assert "move_joints" in names
    assert "grasp" in names
    assert "emergency_stop" in names
    assert "query_world_objects" in names
    assert "get_scene_graph" in names
    assert "cognitive_search" in names
    hub.stop()


async def test_mcp_hub_handle_move_joints_timeout():
    """Test move_joints falls back to command_issued when no response."""
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()
    # No response handler registered, so it should timeout and fallback
    result = await hub.handle_tool_call("move_joints", {"joint_positions": [0.1] * 6, "duration": 1.0})
    assert result["status"] == "command_issued"
    assert result["action"] == "move_joints"
    hub.stop()


async def test_mcp_hub_handle_move_joints_with_response():
    """Test move_joints receives response from execution layer."""
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()

    # Register a mock execution layer that responds
    def mock_executor(event):
        request_id = event.metadata.get("request_id")
        if request_id and event.payload.get("action") == "move_joints":
            bus.publish(Event(
                topic="agent.response",
                payload={
                    "status": "success",
                    "action": "move_joints",
                    "final_positions": event.payload["joint_positions"],
                },
                source="mock_executor",
                metadata={"request_id": request_id},
            ))

    bus.subscribe("agent.command", mock_executor)

    result = await hub.handle_tool_call("move_joints", {"joint_positions": [0.1] * 6, "duration": 1.0})
    assert result["status"] == "success"
    assert result["action"] == "move_joints"
    hub.stop()


async def test_mcp_hub_handle_grasp():
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()
    result = await hub.handle_tool_call("grasp", {"action": "close", "force": 0.8})
    assert result["status"] == "command_issued"
    hub.stop()


async def test_mcp_hub_handle_emergency_stop():
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()
    result = await hub.handle_tool_call("emergency_stop", {})
    assert result["status"] == "emergency_stop_triggered"
    hub.stop()


async def test_mcp_hub_unknown_tool():
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()
    result = await hub.handle_tool_call("nonexistent", {})
    assert "error" in result
    hub.stop()


def test_mcp_hub_context_update():
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()
    hub.update_robot_description("UR5e robot arm")
    assert hub.context.robot_model_description == "UR5e robot arm"
    hub.stop()


async def test_mcp_hub_command_response_pattern():
    """Test the full command-response pattern with request_id matching."""
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()

    received_requests = []

    def mock_handler(event):
        received_requests.append(event.metadata.get("request_id"))
        request_id = event.metadata.get("request_id")
        # Simulate async processing
        bus.publish(Event(
            topic="agent.response",
            payload={"status": "completed", "request_id": request_id},
            source="mock",
            metadata={"request_id": request_id},
        ))

    bus.subscribe("agent.command", mock_handler)

    result = await hub.handle_tool_call("move_joints", {"joint_positions": [0.5] * 6})
    assert result["status"] == "completed"
    assert len(received_requests) == 1
    assert received_requests[0] is not None
    hub.stop()


# ------------------------------------------------------------------
# Provider-aware MCPHub tests
# ------------------------------------------------------------------

def test_mcp_hub_semantic_tools_with_runtime():
    """When attached to a Runtime with provider layer, MCPHub exposes semantic tools."""
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(RuntimeConfig(robot_id="test_bot", enable_provider=True))
    runtime.initialize()

    bus = runtime.event_bus
    hub = MCPHub(bus, robot_id="test_bot", runtime=runtime)
    hub.initialize()

    tools = hub.tools
    names = [t["name"] for t in tools]

    assert "observe_scene" in names
    assert "locate_object" in names
    assert "delegate_skill" in names
    assert "verify_task_success" in names
    assert "get_robot_state" in names
    assert "emergency_stop" in names

    # Low-level tools should NOT be present in semantic mode
    assert "move_joints" not in names
    assert "grasp" not in names

    hub.stop()
    runtime.stop()


async def test_mcp_hub_observe_scene_via_provider():
    """Test observe_scene routes through capability router to mock VLM."""
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(RuntimeConfig(robot_id="test_bot", enable_provider=True))
    runtime.initialize()

    bus = runtime.event_bus
    hub = MCPHub(bus, robot_id="test_bot", runtime=runtime)
    hub.initialize()

    result = await hub.handle_tool_call("observe_scene", {"query": "what do you see?"})
    assert result.get("status") == "ok", f"Expected ok, got: {result}"
    assert result["capability"] == "vlm.scene_understanding"
    assert "result" in result

    hub.stop()
    runtime.stop()


async def test_mcp_hub_locate_object_via_provider():
    """Test locate_object routes through capability router to mock VLM."""
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(RuntimeConfig(robot_id="test_bot", enable_provider=True))
    runtime.initialize()

    bus = runtime.event_bus
    hub = MCPHub(bus, robot_id="test_bot", runtime=runtime)
    hub.initialize()

    result = await hub.handle_tool_call("locate_object", {"object_name": "red cup"})
    assert result.get("status") == "ok", f"Expected ok, got: {result}"
    assert result["capability"] == "vlm.object_grounding"
    assert "result" in result
    assert result["result"]["objects"][0]["label"] == "red cup"

    hub.stop()
    runtime.stop()


async def test_mcp_hub_delegate_skill_via_provider():
    """Test delegate_skill routes through capability router to mock skill provider."""
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(RuntimeConfig(robot_id="test_bot", enable_provider=True))
    runtime.initialize()

    bus = runtime.event_bus
    hub = MCPHub(bus, robot_id="test_bot", runtime=runtime)
    hub.initialize()

    result = await hub.handle_tool_call("delegate_skill", {
        "skill": "grasp",
        "target": {"object": "red cup"},
    })
    assert result.get("status") == "ok", f"Expected ok, got: {result}"
    assert result["capability"] == "skill.grasp"
    assert result["result"]["skill"] == "grasp"
    assert result["result"]["status"] == "dispatched"

    hub.stop()
    runtime.stop()


async def test_mcp_hub_verify_task_success_via_provider():
    """Test verify_task_success routes through capability router to mock critic."""
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(RuntimeConfig(robot_id="test_bot", enable_provider=True))
    runtime.initialize()

    bus = runtime.event_bus
    hub = MCPHub(bus, robot_id="test_bot", runtime=runtime)
    hub.initialize()

    result = await hub.handle_tool_call("verify_task_success", {
        "task_description": "pick up the red cup",
    })
    assert result.get("status") == "ok", f"Expected ok, got: {result}"
    assert result["capability"] == "critic.success_detection"
    assert result["result"]["success"] is True

    hub.stop()
    runtime.stop()


async def test_mcp_hub_emergency_stop_with_runtime():
    """Emergency stop works in both semantic and low-level modes."""
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(RuntimeConfig(robot_id="test_bot", enable_provider=True))
    runtime.initialize()

    bus = runtime.event_bus
    hub = MCPHub(bus, robot_id="test_bot", runtime=runtime)
    hub.initialize()

    result = await hub.handle_tool_call("emergency_stop", {})
    assert result["status"] == "emergency_stop_triggered"

    hub.stop()
    runtime.stop()
