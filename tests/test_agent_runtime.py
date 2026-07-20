"""Tests for Agent Runtime."""

from typing import Any

from rosclaw.agent_runtime.mcp_hub import AgentContext, MCPHub
from rosclaw.core.event_bus import Event, EventBus
from rosclaw.kernel import ActionEnvelope, ExecutionMode


class _FakeDaemon:
    def __init__(self) -> None:
        self.actions: list[ActionEnvelope] = []
        self.stop_reasons: list[str] = []

    def request_action(self, action: ActionEnvelope) -> dict[str, Any]:
        self.actions.append(action)
        return {"action_id": action.action_id, "state": "QUEUED"}

    def wait_for_action(self, action_id: str, *, timeout_sec: float) -> dict[str, Any]:
        action = next(item for item in self.actions if item.action_id == action_id)
        return {
            "action_id": action_id,
            "state": "FINISHED",
            "receipt": {
                "action_id": action_id,
                "execution_mode": action.execution_mode.value,
                "final_state": "FAILED",
                "trust_level": "UNVERIFIED",
                "usable_for_real_execution": False,
                "errors": [{"code": "EXECUTOR_UNAVAILABLE"}],
            },
        }

    def emergency_stop(self, reason: str, *, source: str) -> dict[str, Any]:
        self.stop_reasons.append(reason)
        return {
            "request_dispatched": True,
            "driver_acknowledged": True,
            "physical_stop_observed": False,
            "stopped": False,
            "final_status": "ACKNOWLEDGED",
        }


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
    assert len(tools) == 13
    names = [t["name"] for t in tools]
    assert "move_joints" in names
    assert "grasp" in names
    assert "emergency_stop" in names
    assert "query_world_objects" in names
    assert "get_scene_graph" in names
    assert "cognitive_search" in names
    assert "rosclaw_task_pack" in names
    assert "rosclaw_match_symptom" in names
    assert "get_body_sense" in names
    assert "get_body_readiness" in names
    assert "explain_body_block" in names
    hub.stop()


async def test_mcp_hub_handle_move_joints_timeout():
    """Physical motion is blocked when no Runtime gateway is attached."""
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()
    result = await hub.handle_tool_call(
        "move_joints", {"joint_positions": [0.1] * 6, "duration": 1.0}
    )
    assert result["status"] == "blocked"
    assert result["error"] == "DAEMON_UNAVAILABLE"
    hub.stop()


async def test_mcp_hub_handle_move_joints_with_response():
    """An EventBus success fixture cannot bypass the action gateway."""
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()

    # Register a mock execution layer that responds
    def mock_executor(event):
        request_id = event.metadata.get("request_id")
        if request_id and event.payload.get("action") == "move_joints":
            bus.publish(
                Event(
                    topic="agent.response",
                    payload={
                        "status": "success",
                        "action": "move_joints",
                        "final_positions": event.payload["joint_positions"],
                    },
                    source="mock_executor",
                    metadata={"request_id": request_id},
                )
            )

    bus.subscribe("agent.command", mock_executor)

    result = await hub.handle_tool_call(
        "move_joints", {"joint_positions": [0.1] * 6, "duration": 1.0}
    )
    assert result["status"] == "blocked"
    assert result["error"] == "DAEMON_UNAVAILABLE"
    hub.stop()


async def test_mcp_hub_handle_grasp():
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()
    result = await hub.handle_tool_call("grasp", {"action": "close", "force": 0.8})
    assert result["status"] == "blocked"
    assert result["error"] == "DAEMON_UNAVAILABLE"
    hub.stop()


async def test_mcp_hub_handle_emergency_stop():
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()
    result = await hub.handle_tool_call("emergency_stop", {})
    assert result["status"] == "failed"
    assert result["error"] == "DAEMON_UNAVAILABLE"
    assert result["request_dispatched"] is False
    assert result["stopped"] is False
    hub.stop()


async def test_legacy_physical_tools_use_daemon_not_attached_runtime():
    class _RuntimeTrap:
        capability_router = None

        def submit_action(self, _action):
            raise AssertionError("MCPHub attempted in-process physical execution")

        def request_emergency_stop(self, _reason, **_kwargs):
            raise AssertionError("MCPHub attempted in-process emergency stop")

    bus = EventBus()
    daemon = _FakeDaemon()
    hub = MCPHub(
        bus,
        robot_id="test",
        runtime=_RuntimeTrap(),
        daemon_client=daemon,
    )
    hub.initialize()

    action = await hub.handle_tool_call(
        "move_joints",
        {
            "joint_positions": [0.1] * 6,
            "body_snapshot_hash": "sha256:test-body",
        },
    )
    stop = await hub.handle_tool_call("emergency_stop", {})

    assert action["status"] == "failed"
    assert action["receipt"]["errors"][0]["code"] == "EXECUTOR_UNAVAILABLE"
    assert daemon.actions[0].authorization.approved is False
    assert daemon.actions[0].execution_mode is ExecutionMode.SHADOW
    assert stop["status"] == "acknowledged"
    assert daemon.stop_reasons == ["LLM emergency stop command"]
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
    """Legacy EventBus responses cannot manufacture physical completion."""
    bus = EventBus()
    hub = MCPHub(bus, robot_id="test")
    hub.initialize()

    received_requests = []

    def mock_handler(event):
        received_requests.append(event.metadata.get("request_id"))
        request_id = event.metadata.get("request_id")
        # Simulate async processing
        bus.publish(
            Event(
                topic="agent.response",
                payload={"status": "completed", "request_id": request_id},
                source="mock",
                metadata={"request_id": request_id},
            )
        )

    bus.subscribe("agent.command", mock_handler)

    result = await hub.handle_tool_call("move_joints", {"joint_positions": [0.5] * 6})
    assert result["status"] == "blocked"
    assert received_requests == []
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
    """Unverified mock providers cannot dispatch physical skills."""
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(RuntimeConfig(robot_id="test_bot", enable_provider=True))
    runtime.initialize()

    bus = runtime.event_bus
    hub = MCPHub(bus, robot_id="test_bot", runtime=runtime)
    hub.initialize()

    result = await hub.handle_tool_call(
        "delegate_skill",
        {
            "skill": "grasp",
            "target": {"object": "red cup"},
        },
    )
    assert result.get("status") == "error"
    assert "No provider passed constraints" in result["error"]

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

    result = await hub.handle_tool_call(
        "verify_task_success",
        {
            "task_description": "pick up the red cup",
        },
    )
    assert result.get("status") == "ok", f"Expected ok, got: {result}"
    assert result["capability"] == "critic.success_detection"
    assert result["result"]["success"] is True

    hub.stop()
    runtime.stop()


async def test_mcp_hub_emergency_stop_with_runtime():
    """Emergency stop does not claim success without a driver acknowledgement."""
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(RuntimeConfig(robot_id="test_bot", enable_provider=True))
    runtime.initialize()

    bus = runtime.event_bus
    hub = MCPHub(bus, robot_id="test_bot", runtime=runtime)
    hub.initialize()

    result = await hub.handle_tool_call("emergency_stop", {})
    assert result["status"] == "failed"
    assert result["stopped"] is False
    assert result["final_status"] == "FAILED"

    hub.stop()
    runtime.stop()
