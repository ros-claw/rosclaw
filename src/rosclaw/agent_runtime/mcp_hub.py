"""
ROSClaw MCP Hub - LLM Interface

Provides MCP (Model Context Protocol) server that exposes
robot control tools to LLMs. This is the primary interface
between AI agents and the physical world.

The MCP Hub:
1. Registers semantic capability tools (VLM, Skill, Critic) when provider layer is available
2. Falls back to low-level control tools when provider layer is unavailable
3. Maintains AgentContext with grounding information
4. Validates all commands through the Digital Twin Firewall
5. Publishes events to the EventBus for module coordination
6. Uses command-response pattern (NOT fire-and-forget)
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin

logger = logging.getLogger("rosclaw.agent_runtime.mcp_hub")


@dataclass
class AgentContext:
    """
    Context maintained for an LLM agent session.

    This provides the "grounding" - the physical understanding
    that allows the LLM to reason about the robot and world.
    """

    session_id: str
    robot_id: str
    current_task: str | None = None
    task_history: list[dict] = field(default_factory=list)
    robot_model_description: str = ""
    current_joint_positions: list[float] = field(default_factory=list)
    current_end_effector_pose: list[float] | None = None
    active_skills: list[str] = field(default_factory=list)
    safety_level: str = "strict"

    def to_mcp_context(self) -> dict:
        """Convert to MCP context format for LLM."""
        return {
            "session_id": self.session_id,
            "robot": {
                "id": self.robot_id,
                "description": self.robot_model_description,
                "current_state": {
                    "joint_positions": self.current_joint_positions,
                    "end_effector_pose": self.current_end_effector_pose,
                },
            },
            "current_task": self.current_task,
            "active_skills": self.active_skills,
            "safety_level": self.safety_level,
        }


class MCPHub(LifecycleMixin):
    """
    MCP Server Hub for ROSClaw.

    Exposes robot control capabilities to LLMs through the
    Model Context Protocol. All tool calls are validated
    and routed through the EventBus using command-response pattern.

    When a Runtime with provider layer is attached, MCPHub exposes
    semantic capability tools (e.g., locate_object, delegate_skill)
    that route through the CapabilityRouter. Otherwise, it falls back
    to low-level control primitives.
    """

    def __init__(
        self,
        event_bus: EventBus,
        robot_id: str = "rosclaw_default",
        runtime: Any | None = None,
        tracer: Any | None = None,
    ):
        super().__init__()
        self.event_bus = event_bus
        self.robot_id = robot_id
        self.runtime = runtime
        self.context = AgentContext(
            session_id="default",
            robot_id=robot_id,
        )
        self._tools: dict[str, dict] = {}
        self._server: Any | None = None
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._default_timeout: float = 30.0
        if tracer is None:
            from rosclaw.observability.tracer import Tracer, get_tracer

            runtime_tracer = getattr(runtime, "tracer", None)
            tracer = (
                runtime_tracer
                if isinstance(runtime_tracer, Tracer)
                else get_tracer(event_bus)
            )
        self._tracer = tracer

    def _do_initialize(self) -> None:
        """Initialize MCP server and register tools."""
        try:
            from mcp.server import Server

            self._server = Server("rosclaw-mcp")
            self._register_all_tools()
            logger.info("MCP server initialized")
        except ImportError:
            logger.warning("MCP library not available, running in mock mode")
            self._server = None

        # Subscribe to robot state updates
        self.event_bus.subscribe("robot.joint_states", self._on_joint_states)
        self.event_bus.subscribe("robot.end_effector_pose", self._on_end_effector_pose)
        # Subscribe to command responses
        self.event_bus.subscribe("agent.response", self._on_agent_response)
        self.event_bus.subscribe("agent.capability.response", self._on_agent_response)

    def _do_start(self) -> None:
        """Start the MCP server."""
        logger.info("MCP Hub started")

    def _do_stop(self) -> None:
        """Stop the MCP server."""
        logger.info("MCP Hub stopped")
        # Cancel any pending futures
        for fut in self._pending_requests.values():
            if not fut.done():
                fut.cancel()
        self._pending_requests.clear()

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------
    @property
    def _has_provider_layer(self) -> bool:
        return (
            self.runtime is not None
            and getattr(self.runtime, "capability_router", None) is not None  # noqa: W503
        )

    def _register_all_tools(self) -> None:
        """Register all tools based on available runtime capabilities."""
        if self._has_provider_layer:
            self._register_semantic_tools()
        else:
            self._register_low_level_tools()

    # -- Semantic capability tools (provider-aware) --
    def _register_semantic_tools(self) -> None:
        """Register semantic tools that route through CapabilityRouter."""
        self._register_observe_scene_tool()
        self._register_locate_object_tool()
        self._register_delegate_skill_tool()
        self._register_verify_task_success_tool()
        self._register_get_state_tool()
        self._register_emergency_stop_tool()
        self._register_query_knowledge_tool()
        self._register_get_safety_heuristic_tool()
        self._register_get_recovery_strategy_tool()
        self._register_knowledge_compiler_tools()
        self._register_sense_tools()

    def _register_observe_scene_tool(self) -> None:
        self._tools["observe_scene"] = {
            "name": "observe_scene",
            "description": "Analyze the current scene using VLM. Returns detected objects, scene type, and any risks.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_topic": {
                        "type": "string",
                        "description": "ROS camera topic name (e.g., /camera/color/image_raw)",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional natural language query about the scene",
                    },
                },
            },
        }

    def _register_locate_object_tool(self) -> None:
        self._tools["locate_object"] = {
            "name": "locate_object",
            "description": "Locate a specific object in the scene using VLM object grounding. Returns bounding box and confidence.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_name": {
                        "type": "string",
                        "description": "Name of the object to locate (e.g., 'red cup', 'screwdriver')",
                    },
                    "image_topic": {
                        "type": "string",
                        "description": "ROS camera topic name",
                    },
                },
                "required": ["object_name"],
            },
        }

    def _register_delegate_skill_tool(self) -> None:
        self._tools["delegate_skill"] = {
            "name": "delegate_skill",
            "description": "Delegate a high-level skill execution (grasp, place, pick_and_place, etc.) to the skill provider.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "enum": [
                            "grasp",
                            "place",
                            "pick_and_place",
                            "push",
                            "pull",
                            "navigate",
                            "inspect",
                        ],
                        "description": "Skill to execute",
                    },
                    "target": {
                        "type": "object",
                        "description": "Target specification (object name, pose, waypoint, etc.)",
                    },
                    "constraints": {
                        "type": "object",
                        "description": "Optional constraints (force, speed, approach_direction)",
                    },
                },
                "required": ["skill"],
            },
        }

    def _register_verify_task_success_tool(self) -> None:
        self._tools["verify_task_success"] = {
            "name": "verify_task_success",
            "description": "Verify whether the current or last task was completed successfully using the critic provider.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Description of what success looks like",
                    },
                    "image_topic": {
                        "type": "string",
                        "description": "Optional camera topic for visual verification",
                    },
                },
                "required": ["task_description"],
            },
        }

    # -- Low-level control tools (fallback when no provider layer) --
    def _register_low_level_tools(self) -> None:
        """Register low-level robot control tools."""
        self._register_move_tool()
        self._register_grasp_tool()
        self._register_get_state_tool()
        self._register_validate_trajectory_tool()
        self._register_emergency_stop_tool()
        self._register_query_world_objects_tool()
        self._register_get_scene_graph_tool()
        self._register_cognitive_search_tool()
        self._register_knowledge_compiler_tools()
        self._register_sense_tools()

    def _register_move_tool(self) -> None:
        self._tools["move_joints"] = {
            "name": "move_joints",
            "description": "Move robot joints to target positions",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "joint_positions": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target joint positions in radians",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Movement duration in seconds",
                        "default": 2.0,
                    },
                },
                "required": ["joint_positions"],
            },
        }

    def _register_grasp_tool(self) -> None:
        self._tools["grasp"] = {
            "name": "grasp",
            "description": "Control the gripper to grasp or release",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["close", "open"],
                        "description": "Grasp action",
                    },
                    "force": {
                        "type": "number",
                        "description": "Grasp force (0-1)",
                        "default": 0.5,
                    },
                },
                "required": ["action"],
            },
        }

    def _register_get_state_tool(self) -> None:
        self._tools["get_robot_state"] = {
            "name": "get_robot_state",
            "description": "Get current robot joint positions and state",
            "inputSchema": {"type": "object", "properties": {}},
        }

    def _register_validate_trajectory_tool(self) -> None:
        self._tools["validate_trajectory"] = {
            "name": "validate_trajectory",
            "description": "Validate a trajectory through Digital Twin before execution",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "waypoints": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "List of joint position waypoints",
                    },
                },
                "required": ["waypoints"],
            },
        }

    def _register_emergency_stop_tool(self) -> None:
        self._tools["emergency_stop"] = {
            "name": "emergency_stop",
            "description": "EMERGENCY STOP - halt all robot motion immediately",
            "inputSchema": {"type": "object", "properties": {}},
        }

    def _register_query_knowledge_tool(self) -> None:
        self._tools["query_knowledge"] = {
            "name": "query_knowledge",
            "description": "Query the Knowledge Graph for robot capabilities, known failure symptoms, or cross-domain engineering analogies.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["capability", "symptom", "analogy"],
                        "description": "Type of knowledge to query",
                    },
                    "query": {
                        "type": "string",
                        "description": "The query text (e.g. robot ID for capabilities, error log for symptoms, situation for analogies)",
                    },
                },
                "required": ["query_type", "query"],
            },
        }

    def _register_get_safety_heuristic_tool(self) -> None:
        self._tools["get_safety_heuristic"] = {
            "name": "get_safety_heuristic",
            "description": "Get a safety heuristic rule for a known dangerous condition (torque_overflow, velocity_divergence, memory_exhaustion, numerical_instability).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "enum": [
                            "torque_overflow",
                            "velocity_divergence",
                            "memory_exhaustion",
                            "numerical_instability",
                        ],
                        "description": "The dangerous condition to get a heuristic for",
                    },
                },
                "required": ["condition"],
            },
        }

    def _register_get_recovery_strategy_tool(self) -> None:
        self._tools["get_recovery_strategy"] = {
            "name": "get_recovery_strategy",
            "description": "Get a heuristic recovery strategy for a failure condition. Queries the HeuristicEngine for known recovery patterns.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "error_log": {
                        "type": "string",
                        "description": "Error message or failure description",
                    },
                    "previous_scores": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional list of previous verifier scores",
                    },
                },
                "required": ["error_log"],
            },
        }

    # -- rosclaw-know (compiled knowledge) tools --
    def _register_knowledge_compiler_tools(self) -> None:
        """Register tools backed by the rosclaw-know compiled catalog."""
        # Idempotent: both _register_semantic_tools and _register_low_level_tools
        # call us, so guard against double registration.
        if "rosclaw_task_pack" in self._tools:
            return
        self._register_rosclaw_task_pack_tool()
        self._register_rosclaw_match_symptom_tool()

    def _register_rosclaw_task_pack_tool(self) -> None:
        self._tools["rosclaw_task_pack"] = {
            "name": "rosclaw_task_pack",
            "description": (
                "Get the pre-flight knowledge pack for a task: known failure modes, "
                "fix patterns, anti-patterns, and expected signals. Pull this BEFORE "
                "selecting a provider so the agent reasons against compiled knowledge "
                "rather than learning the same lessons twice."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task identifier from the compiled TaskCard catalog (e.g. 'task_robotics_pid_tuning').",
                    },
                    "embodiment_id": {
                        "type": "string",
                        "description": "Optional robot/embodiment identifier — reserved for embodiment-aware ranking.",
                    },
                    "top_k_patterns": {
                        "type": "integer",
                        "description": "Maximum number of fix patterns to return.",
                        "default": 5,
                    },
                },
                "required": ["task_id"],
            },
        }

    def _register_rosclaw_match_symptom_tool(self) -> None:
        self._tools["rosclaw_match_symptom"] = {
            "name": "rosclaw_match_symptom",
            "description": (
                "Match an error signature or runtime symptom against the compiled "
                "FailureMode catalog. Returns the best-fitting pattern with its fix "
                "recipe and anti-pattern, or null when no candidate clears the threshold."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "error_signature": {
                        "type": "string",
                        "description": "Free-text symptom or error log line to match (e.g. 'PID integral wind-up').",
                    },
                },
                "required": ["error_signature"],
            },
        }

    def _register_query_world_objects_tool(self) -> None:
        self._tools["query_world_objects"] = {
            "name": "query_world_objects",
            "description": "Query world objects in a scene by spatial region or scene ID. Requires EmbodiedMemory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "string", "description": "Scene identifier"},
                    "center_x": {"type": "number", "description": "Search center X coordinate"},
                    "center_y": {"type": "number", "description": "Search center Y coordinate"},
                    "center_z": {"type": "number", "description": "Search center Z coordinate"},
                    "radius": {
                        "type": "number",
                        "description": "Search radius in meters",
                        "default": 2.0,
                    },
                },
                "required": ["scene_id"],
            },
        }

    def _register_get_scene_graph_tool(self) -> None:
        self._tools["get_scene_graph"] = {
            "name": "get_scene_graph",
            "description": "Get the scene graph (objects and spatial relations) for a scene. Requires EmbodiedMemory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "string", "description": "Scene identifier"},
                },
                "required": ["scene_id"],
            },
        }

    def _register_cognitive_search_tool(self) -> None:
        self._tools["cognitive_search"] = {
            "name": "cognitive_search",
            "description": "Cognitive search across memory: semantic + spatial + temporal. Requires EmbodiedMemory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query"},
                    "scene_id": {"type": "string", "description": "Optional scene filter"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        }

    def _register_sense_tools(self) -> None:
        """Register body sense MCP tools when a SenseRuntime is available."""
        from rosclaw.sense.mcp_tools import register_sense_tools

        register_sense_tools(self._tools)

    # ------------------------------------------------------------------
    # Tool call dispatch
    # ------------------------------------------------------------------
    async def handle_tool_call(self, name: str, arguments: dict) -> dict:
        """Trace and dispatch one real MCP tool invocation."""

        trace_id = arguments.get("trace_id") or f"mcp_{uuid.uuid4().hex[:16]}"
        side_effect = name in {"move_joints", "grasp", "emergency_stop", "delegate_skill"}
        async with self._tracer.start_span(
            "mcp.call_tool",
            "MCP",
            source="mcp_hub",
            operation=name,
            trace_id=str(trace_id),
            attributes={
                "mcp.server": "rosclaw-mcp",
                "tool.name": name,
                "tool.side_effect": side_effect,
                "safety.level": self.context.safety_level,
            },
            robot_id=self.robot_id,
            session_id=self.context.session_id,
        ) as span:
            span.set_input(arguments)
            result = await self._dispatch_tool_call(name, arguments)
            span.set_output(result)
            status = str(result.get("status", "")).lower()
            if status == "blocked":
                span.set_status("BLOCKED", result.get("error") or result.get("message"))
            elif status in {"failed", "error", "timeout"} or "error" in result:
                span.set_status("ERROR", result.get("error") or result.get("message"))
            return result

    async def _dispatch_tool_call(self, name: str, arguments: dict) -> dict:
        """
        Handle an MCP tool call from an LLM.

        All tool calls are converted to EventBus events using
        command-response pattern for reliable execution feedback.
        """
        logger.info("Tool call: %s(%s)", name, arguments)

        # Semantic capability tools (provider-aware)
        if name == "observe_scene":
            return await self._handle_observe_scene(arguments)
        elif name == "locate_object":
            return await self._handle_locate_object(arguments)
        elif name == "delegate_skill":
            return await self._handle_delegate_skill(arguments)
        elif name == "verify_task_success":
            return await self._handle_verify_task_success(arguments)

        # Low-level control tools
        elif name == "move_joints":
            return await self._handle_move_joints(arguments)
        elif name == "grasp":
            return await self._handle_grasp(arguments)
        elif name == "get_robot_state":
            return self._handle_get_state()
        elif name == "validate_trajectory":
            return await self._handle_validate_trajectory(arguments)
        elif name == "emergency_stop":
            return self._handle_emergency_stop()
        elif name == "query_world_objects":
            return self._handle_query_world_objects(arguments)
        elif name == "get_scene_graph":
            return self._handle_get_scene_graph(arguments)
        elif name == "cognitive_search":
            return self._handle_cognitive_search(arguments)

        # Knowledge tools
        elif name == "query_knowledge":
            return self._handle_query_knowledge(arguments)
        elif name == "get_safety_heuristic":
            return self._handle_get_safety_heuristic(arguments)

        # Heuristic recovery tool (HOW)
        elif name == "get_recovery_strategy":
            return await self._handle_get_recovery_strategy(arguments)

        # rosclaw-know compiled-knowledge tools
        elif name == "rosclaw_task_pack":
            return self._handle_rosclaw_task_pack(arguments)
        elif name == "rosclaw_match_symptom":
            return self._handle_rosclaw_match_symptom(arguments)

        # Body sense tools
        elif name == "get_body_sense":
            return self._handle_get_body_sense(arguments)
        elif name == "get_body_readiness":
            return self._handle_get_body_readiness(arguments)
        elif name == "explain_body_block":
            return self._handle_explain_body_block(arguments)

        else:
            return {"error": f"Unknown tool: {name}"}

    # ------------------------------------------------------------------
    # Semantic capability handlers
    # ------------------------------------------------------------------
    async def _route_capability(
        self,
        capability: str,
        inputs: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Route a capability request via EventBus.

        Architecture: MCPHub publishes agent.capability.request to EventBus.
        Provider layer (via Runtime) subscribes and publishes agent.capability.response.
        Zero direct MCPHub -> ProviderRegistry coupling.
        """
        if self.event_bus is None:
            return {
                "status": "failed",
                "capability": capability,
                "error": "EventBus not available",
            }

        request_id = str(uuid.uuid4())[:8]
        ctx = {
            "robot": self.robot_id,
            "safety_level": self.context.safety_level,
            **(context or {}),
        }

        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        self.event_bus.publish(
            Event(
                topic="agent.capability.request",
                payload={
                    "request_id": request_id,
                    "capability": capability,
                    "inputs": inputs,
                    "context": ctx,
                    "constraints": {"safety_level": self.context.safety_level.upper()},
                },
                source="mcp_hub",
                priority=EventPriority.HIGH,
            )
        )

        try:
            result = await asyncio.wait_for(future, timeout=self._default_timeout)
            return result
        except TimeoutError:
            return {
                "status": "failed",
                "capability": capability,
                "error": "No provider responded via EventBus (timeout)",
            }
        finally:
            self._pending_requests.pop(request_id, None)

    async def _route_capability_direct(
        self,
        request_id: str,
        capability: str,
        inputs: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Direct Runtime fallback for capability routing.

        Kept for backward compatibility until Provider layer fully migrates
        to EventBus subscription.
        """
        if self.runtime is None:
            return {
                "status": "failed",
                "capability": capability,
                "error": "Runtime not available",
            }

        router = getattr(self.runtime, "capability_router", None)
        guard = getattr(self.runtime, "guard_pipeline", None)

        if router is None:
            return {
                "status": "failed",
                "capability": capability,
                "error": "CapabilityRouter not available",
            }

        # Lazy import to avoid hard dependency at module load time
        from rosclaw.provider.core.request import ProviderRequest

        request = ProviderRequest(
            request_id=request_id,
            capability=capability,
            inputs=inputs,
            context=context,
            constraints={"safety_level": self.context.safety_level.upper()},
        )

        try:
            response = await router.invoke(request)
        except Exception as e:
            return {
                "status": "failed",
                "capability": capability,
                "error": str(e),
            }

        # Run guard pipeline on executable outputs
        if guard and getattr(response, "result", None):
            try:
                guard.check(request, response)
            except Exception as e:
                return {
                    "status": "blocked",
                    "capability": capability,
                    "error": f"Guard blocked: {e}",
                }

        return {
            "status": "ok" if response.is_ok else "failed",
            "capability": capability,
            "provider": getattr(response, "provider", ""),
            "result": getattr(response, "result", {}),
            "confidence": getattr(response, "confidence", None),
            "latency_ms": getattr(response, "latency_ms", None),
            "warnings": getattr(response, "warnings", []),
            "errors": getattr(response, "errors", []),
        }

    async def _handle_observe_scene(self, arguments: dict) -> dict:
        """Handle observe_scene via VLM scene_understanding capability."""
        inputs = {
            "camera_topic": arguments.get("image_topic", ""),
            "text": arguments.get("query", ""),
        }
        return await self._route_capability("vlm.scene_understanding", inputs)

    async def _handle_locate_object(self, arguments: dict) -> dict:
        """Handle locate_object via VLM object_grounding capability."""
        inputs = {
            "query": arguments.get("object_name", ""),
            "camera_topic": arguments.get("image_topic", ""),
        }
        return await self._route_capability("vlm.object_grounding", inputs)

    async def _handle_delegate_skill(self, arguments: dict) -> dict:
        """Handle delegate_skill via Skill provider."""
        skill = arguments.get("skill", "")
        capability = f"skill.{skill}"
        inputs = {
            "target": arguments.get("target", {}),
            "constraints": arguments.get("constraints", {}),
        }
        return await self._route_capability(capability, inputs)

    async def _handle_verify_task_success(self, arguments: dict) -> dict:
        """Handle verify_task_success via Critic provider."""
        inputs = {
            "task": arguments.get("task_description", ""),
            "camera_topic": arguments.get("image_topic", ""),
        }
        return await self._route_capability("critic.success_detection", inputs)

    # ------------------------------------------------------------------
    # Low-level control handlers
    # ------------------------------------------------------------------
    async def _send_command_and_wait(
        self,
        topic: str,
        payload: dict,
        timeout: float | None = None,
    ) -> dict:
        """
        Send a command via EventBus and wait for response.

        Uses request-response pattern:
        1. Generate unique request_id
        2. Create asyncio.Future
        3. Publish command with request_id in metadata
        4. Await response future with timeout
        5. Return execution result
        """
        request_id = str(uuid.uuid4())[:8]
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        event = Event(
            topic=topic,
            payload=payload,
            source="mcp_hub",
            priority=EventPriority.HIGH,
            metadata={"request_id": request_id},
        )
        self.event_bus.publish(event)

        try:
            result = await asyncio.wait_for(future, timeout=timeout or self._default_timeout)
            return result
        except TimeoutError:
            return {
                "status": "timeout",
                "message": f"Command timed out after {timeout or self._default_timeout}s",
                "request_id": request_id,
            }
        finally:
            self._pending_requests.pop(request_id, None)

    def _on_agent_response(self, event: Event) -> None:
        """Handle command responses from other modules."""
        request_id = event.payload.get("request_id") or event.metadata.get("request_id")
        if request_id and request_id in self._pending_requests:
            future = self._pending_requests[request_id]
            if not future.done():
                future.set_result(event.payload.get("result", event.payload))

    async def _handle_move_joints(self, arguments: dict) -> dict:
        """Handle move_joints tool call with command-response."""
        positions = arguments.get("joint_positions", [])
        duration = arguments.get("duration", 2.0)

        result = await self._send_command_and_wait(
            topic="agent.command",
            payload={
                "action": "move_joints",
                "joint_positions": positions,
                "duration": duration,
            },
        )

        if result.get("status") == "timeout":
            return {
                "status": "command_issued",
                "action": "move_joints",
                "target_positions": positions,
                "warning": "No response received from execution layer",
            }
        return result

    async def _handle_grasp(self, arguments: dict) -> dict:
        """Handle grasp tool call with command-response."""
        action = arguments.get("action", "close")
        force = arguments.get("force", 0.5)

        result = await self._send_command_and_wait(
            topic="agent.command",
            payload={
                "action": "grasp",
                "grasp_action": action,
                "force": force,
            },
        )

        if result.get("status") == "timeout":
            return {
                "status": "command_issued",
                "action": f"grasp_{action}",
                "warning": "No response received from execution layer",
            }
        return result

    def _handle_get_state(self) -> dict:
        """Handle get_robot_state tool call."""
        return {
            "status": "ok",
            "robot_state": {
                "joint_positions": self.context.current_joint_positions,
                "end_effector_pose": self.context.current_end_effector_pose,
            },
        }

    async def _handle_validate_trajectory(self, arguments: dict) -> dict:
        """Handle validate_trajectory tool call with command-response."""
        waypoints = arguments.get("waypoints", [])

        result = await self._send_command_and_wait(
            topic="agent.command",
            payload={
                "action": "validate_trajectory",
                "waypoints": waypoints,
            },
        )

        if result.get("status") == "timeout":
            return {
                "status": "validation_requested",
                "waypoints_count": len(waypoints),
                "warning": "No response received from validation layer",
            }
        return result

    def _handle_emergency_stop(self) -> dict:
        """Handle emergency_stop tool call."""
        self.event_bus.publish(
            Event(
                topic="robot.emergency_stop",
                payload={"reason": "LLM emergency stop command"},
                source="mcp_hub",
                priority=EventPriority.CRITICAL,
            )
        )
        return {"status": "emergency_stop_triggered"}

    # ------------------------------------------------------------------
    # Physical world handlers
    # ------------------------------------------------------------------

    def _handle_query_world_objects(self, arguments: dict) -> dict:
        """Handle query_world_objects tool call via EventBus (preferred) or Runtime fallback."""
        scene_id = arguments.get("scene_id", "")
        radius = arguments.get("radius", 2.0)
        cx = arguments.get("center_x", 0.0)
        cy = arguments.get("center_y", 0.0)
        cz = arguments.get("center_z", 0.0)

        # Publish query event to EventBus for decoupled world-state access
        if self.event_bus is not None:
            self.event_bus.publish(
                Event(
                    topic="world.objects.query",
                    payload={
                        "scene_id": scene_id,
                        "radius": radius,
                        "center": {"x": cx, "y": cy, "z": cz},
                    },
                    source="mcp_hub",
                    priority=EventPriority.NORMAL,
                )
            )

        # Fallback: direct Runtime access when no subscriber responded
        if self.runtime is not None and hasattr(self.runtime, "search_world_objects"):
            try:
                from rosclaw.e_urdf.parser import Vec3

                center = Vec3(cx, cy, cz)
            except ImportError:
                center = {"x": cx, "y": cy, "z": cz}
            results = self.runtime.search_world_objects(center, radius, scene_id)
            return {
                "status": "ok",
                "scene_id": scene_id,
                "count": len(results),
                "objects": [self._world_object_to_dict(o) for o in results],
            }

        return {"status": "error", "error": "Runtime not available"}

    def _handle_get_scene_graph(self, arguments: dict) -> dict:
        """Handle get_scene_graph tool call via EventBus (preferred) or Runtime fallback."""
        scene_id = arguments.get("scene_id", "")

        if self.event_bus is not None:
            self.event_bus.publish(
                Event(
                    topic="world.scene_graph.query",
                    payload={"scene_id": scene_id},
                    source="mcp_hub",
                    priority=EventPriority.NORMAL,
                )
            )

        if self.runtime is not None and hasattr(self.runtime, "get_scene_graph"):
            objects, relations = self.runtime.get_scene_graph(scene_id)
            return {
                "status": "ok",
                "scene_id": scene_id,
                "object_count": len(objects),
                "relation_count": len(relations),
                "objects": [self._world_object_to_dict(o) for o in objects],
                "relations": [self._relation_to_dict(r) for r in relations],
            }

        return {"status": "error", "error": "Runtime not available"}

    def _handle_cognitive_search(self, arguments: dict) -> dict:
        """Handle cognitive_search tool call via EventBus (preferred) or Runtime fallback."""
        query = arguments.get("query", "")
        limit = arguments.get("limit", 10)

        if self.event_bus is not None:
            self.event_bus.publish(
                Event(
                    topic="memory.cognitive.query",
                    payload={"query": query, "limit": limit},
                    source="mcp_hub",
                    priority=EventPriority.NORMAL,
                )
            )

        if self.runtime is not None and hasattr(self.runtime, "cognitive_search"):
            results = self.runtime.cognitive_search(query, limit=limit)
            return {
                "status": "ok",
                "query": query,
                "count": len(results),
                "results": [self._memory_atom_to_dict(r) for r in results],
            }

        return {"status": "error", "error": "Runtime not available"}

    def _handle_query_knowledge(self, arguments: dict) -> dict:
        """Handle query_knowledge tool call."""
        if self.runtime is None:
            return {"status": "error", "error": "Runtime not available"}
        query_type = arguments.get("query_type", "")
        query = arguments.get("query", "")

        knowledge = getattr(self.runtime, "knowledge", None)
        if knowledge is None:
            return {"status": "error", "error": "Knowledge module not available"}

        if query_type == "capability":
            capabilities = knowledge.query_robot_capabilities(query)
            return {
                "status": "ok",
                "query_type": "capability",
                "robot_id": query,
                "count": len(capabilities),
                "capabilities": capabilities,
            }
        elif query_type == "symptom":
            match = knowledge.match_symptom(query)
            return {
                "status": "ok",
                "query_type": "symptom",
                "matched": match is not None,
                "result": match,
            }
        elif query_type == "analogy":
            analogy = knowledge.get_analogy(query)
            return {
                "status": "ok",
                "query_type": "analogy",
                "matched": analogy is not None,
                "result": analogy,
            }
        else:
            return {"status": "error", "error": f"Unknown query_type: {query_type}"}

    async def _handle_get_recovery_strategy(self, arguments: dict) -> dict:
        """Handle get_recovery_strategy tool call."""
        if self.runtime is None:
            return {"status": "error", "error": "Runtime not available"}

        how_engine = getattr(self.runtime, "how", None)
        if how_engine is None:
            return {"status": "error", "error": "HeuristicEngine not available"}

        error_log = arguments.get("error_log", "")
        if not error_log:
            return {"status": "error", "error": "error_log is required"}

        try:
            recovery = how_engine.suggest_recovery(error_log)
            if asyncio.iscoroutine(recovery):
                recovery = await recovery
        except Exception as exc:
            return {"status": "error", "error": f"Recovery lookup failed: {exc}"}

        if recovery is None:
            return {
                "status": "ok",
                "matched": False,
                "message": "No heuristic recovery strategy found for this error.",
            }

        return {
            "status": "ok",
            "matched": True,
            "rule_id": recovery.get("rule_id", ""),
            "condition": recovery.get("condition", ""),
            "action": recovery.get("action", ""),
            "priority": recovery.get("priority", 0),
            "source": recovery.get("source", "heuristic"),
        }

    def _handle_get_safety_heuristic(self, arguments: dict) -> dict:
        """Handle get_safety_heuristic tool call."""
        if self.runtime is None:
            return {"status": "error", "error": "Runtime not available"}

        knowledge = getattr(self.runtime, "knowledge", None)
        if knowledge is None:
            return {"status": "error", "error": "Knowledge module not available"}

        condition = arguments.get("condition", "")
        # Map snake_case to KnowledgeInterface's Safety_Pattern labels
        label_map = {
            "torque_overflow": "Torque_Overflow",
            "velocity_divergence": "Velocity_Divergence",
            "memory_exhaustion": "Memory_Exhaustion",
            "numerical_instability": "Numerical_Instability",
        }
        label = label_map.get(condition, "")
        if not label:
            return {
                "status": "error",
                "error": f"Unknown condition: {condition}",
                "known_conditions": list(label_map.keys()),
            }

        rule = knowledge.get_safety_rule(label)
        return {
            "status": "ok",
            "condition": condition,
            "safety_rule": rule,
        }

    def _handle_rosclaw_task_pack(self, arguments: dict) -> dict:
        """Handle rosclaw_task_pack tool call."""
        task_id = arguments.get("task_id", "")
        if not task_id:
            return {"status": "error", "error": "task_id is required"}

        embodiment_id = arguments.get("embodiment_id") or None
        top_k = int(arguments.get("top_k_patterns", 5) or 5)

        try:
            from rosclaw.know.task_pack_adapter import task_pack_for
        except ImportError as exc:
            return {
                "status": "error",
                "error": f"task_pack_adapter unavailable: {exc}",
            }

        try:
            pack = task_pack_for(
                task_id,
                embodiment_id=embodiment_id,
                top_k_patterns=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error": f"task_pack build failed: {exc}",
            }

        return {"status": "ok", "task_pack": pack}

    def _handle_rosclaw_match_symptom(self, arguments: dict) -> dict:
        """Handle rosclaw_match_symptom tool call."""
        if self.runtime is None:
            return {"status": "error", "error": "Runtime not available"}

        knowledge = getattr(self.runtime, "knowledge", None)
        if knowledge is None:
            return {"status": "error", "error": "Knowledge module not available"}

        signature = arguments.get("error_signature", "")
        if not signature:
            return {"status": "error", "error": "error_signature is required"}

        try:
            match = knowledge.match_symptom(signature)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"match_symptom failed: {exc}"}

        return {
            "status": "ok",
            "matched": match is not None,
            "result": match,
        }

    def _handle_get_body_sense(self, arguments: dict) -> dict:
        """Handle get_body_sense tool call."""
        from rosclaw.sense.mcp_tools import handle_get_body_sense

        return handle_get_body_sense(self, arguments)

    def _handle_get_body_readiness(self, arguments: dict) -> dict:
        """Handle get_body_readiness tool call."""
        from rosclaw.sense.mcp_tools import handle_get_body_readiness

        return handle_get_body_readiness(self, arguments)

    def _handle_explain_body_block(self, arguments: dict) -> dict:
        """Handle explain_body_block tool call."""
        from rosclaw.sense.mcp_tools import handle_explain_body_block

        return handle_explain_body_block(self, arguments)

    @staticmethod
    def _world_object_to_dict(obj: Any) -> dict:
        """Serialize a WorldObject-like instance to dict."""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return {"obj_id": getattr(obj, "obj_id", "")}

    @staticmethod
    def _relation_to_dict(rel: Any) -> dict:
        """Serialize a SpatialRelation-like instance to dict."""
        if hasattr(rel, "to_dict"):
            return rel.to_dict()
        return {
            "subject_id": getattr(rel, "subject_id", ""),
            "object_id": getattr(rel, "object_id", ""),
            "relation": getattr(rel, "relation", ""),
        }

    @staticmethod
    def _memory_atom_to_dict(atom: Any) -> dict:
        """Serialize a MemoryAtom-like instance to dict."""
        if hasattr(atom, "to_dict"):
            return atom.to_dict()
        return {"content": getattr(atom, "content", "")}

    def _on_joint_states(self, event: Event) -> None:
        """Update context with joint state."""
        payload = event.payload
        if isinstance(payload, dict) and "positions" in payload:
            self.context.current_joint_positions = payload["positions"]

    def _on_end_effector_pose(self, event: Event) -> None:
        """Update context with end effector pose."""
        self.context.current_end_effector_pose = event.payload

    def update_robot_description(self, description: str) -> None:
        """Update robot model description in context."""
        self.context.robot_model_description = description

    @property
    def tools(self) -> list[dict]:
        """Get list of available tools."""
        return list(self._tools.values())
