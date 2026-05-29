#!/usr/bin/env python3
"""
ROSClaw Minimal MCP Server — stdio transport for Claude Code integration.

This server provides MCP tools for ROSClaw system operations without
dependency on ROS2 or real hardware. It wraps MCPHub and exposes:
- Robot discovery and state
- Sandbox task execution
- Practice/Memory queries
- Forge asset compilation

Usage:
    PYTHONPATH=src python3 -m rosclaw.mcp.minimal_server
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from rosclaw.agent_runtime.mcp_hub import MCPHub
from rosclaw.core.event_bus import EventBus


class ROSClawMinimalMCPServer:
    """Lightweight MCP server for ROSClaw system integration."""

    def __init__(self):
        self.server = Server("rosclaw-minimal")
        self.event_bus = EventBus()
        self.hub = MCPHub(event_bus=self.event_bus, robot_id="rosclaw_default")
        # Redirect hub init prints to stderr so they don't interfere with stdio JSON-RPC
        import io
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            self.hub.initialize()
        finally:
            sys.stdout = old_stdout

        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            tools = []
            for name, spec in self.hub._tools.items():
                tools.append(
                    Tool(
                        name=spec["name"],
                        description=spec["description"],
                        inputSchema=spec["inputSchema"],
                    )
                )
            # Add system-level tools
            tools.extend(self._system_tools())
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            try:
                if name.startswith("system."):
                    result = await self._handle_system_tool(name, arguments)
                else:
                    result = await self.hub.handle_tool_call(name, arguments)
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]

    def _system_tools(self) -> list[Tool]:
        return [
            Tool(
                name="system.list_robots",
                description="List all available robots in the e-URDF-Zoo registry",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="system.list_providers",
                description="List all registered capability providers (llm, vlm, skill, critic, etc.)",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="system.run_sandbox_task",
                description="Run a robot task in sandbox: validates through firewall, executes mock/sim action, records episode",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "robot_id": {"type": "string", "description": "Robot identifier (e.g., ur5e, g1, turtlebot)"},
                        "task": {"type": "string", "description": "Task name (e.g., pid_move, reach, g1_walk)"},
                        "world": {"type": "string", "description": "Sandbox world (mock, mujoco, tabletop)", "default": "mock"},
                        "parameters": {"type": "object", "description": "Task parameters", "default": {}},
                    },
                    "required": ["robot_id", "task"],
                },
            ),
            Tool(
                name="system.query_memory",
                description="Query ROSClaw Memory for past experiences, failures, or success patterns",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query or task description"},
                        "query_type": {"type": "string", "enum": ["experience", "failure", "success_pattern", "similar"], "default": "similar"},
                        "limit": {"type": "integer", "description": "Max results", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="system.explain_failure",
                description="Explain the most recent failure and get recovery suggestions from Memory/How",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Optional task ID filter"},
                        "episode_id": {"type": "string", "description": "Optional episode ID"},
                    },
                },
            ),
            Tool(
                name="system.compile_asset_bundle",
                description="Compile an Asset Bundle (Skill Manifest + Provider Manifest + Tests) for a new capability",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sdk_doc": {"type": "string", "description": "SDK documentation or capability description"},
                        "bundle_name": {"type": "string", "description": "Name for the new bundle"},
                        "staging": {"type": "boolean", "description": "Install to staging only", "default": True},
                    },
                    "required": ["sdk_doc", "bundle_name"],
                },
            ),
            Tool(
                name="system.get_version",
                description="Get ROSClaw version and system status",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    async def _handle_system_tool(self, name: str, arguments: dict) -> dict:
        if name == "system.list_robots":
            try:
                from rosclaw.runtime import RobotRegistry
                reg = RobotRegistry()
                robots = reg.list_available()
                return {"robots": robots, "count": len(robots)}
            except Exception as e:
                return {"robots": [], "count": 0, "error": str(e)}
        elif name == "system.list_providers":
            try:
                from rosclaw.provider.core.registry import ProviderRegistry
                reg = ProviderRegistry()
                providers = reg.list_providers() if hasattr(reg, "list_providers") else []
                return {"providers": providers, "count": len(providers)}
            except Exception as e:
                return {"providers": [], "count": 0, "error": str(e)}
        elif name == "system.run_sandbox_task":
            return await self._handle_run_sandbox_task(arguments)
        elif name == "system.query_memory":
            return await self._handle_query_memory(arguments)
        elif name == "system.explain_failure":
            return await self._handle_explain_failure(arguments)
        elif name == "system.compile_asset_bundle":
            return await self._handle_compile_asset_bundle(arguments)
        elif name == "system.get_version":
            return {
                "name": "rosclaw",
                "version": "1.0.0",
                "status": "ready",
                "modules": {
                    "mcp_hub": self.hub._server is not None,
                    "event_bus": True,
                },
            }
        return {"error": f"Unknown system tool: {name}"}

    async def _handle_run_sandbox_task(self, arguments: dict) -> dict:
        """Run a sandbox task: firewall check -> execute -> record episode."""
        robot_id = arguments.get("robot_id", "rosclaw_default")
        task = arguments.get("task", "unknown")
        world = arguments.get("world", "mock")
        parameters = arguments.get("parameters", {})

        # Step 1: Firewall validation
        try:
            from rosclaw.sandbox.firewall.gate import FirewallGate
            gate = FirewallGate(robot_id=robot_id, world_id=world)
            action = {"type": task, "parameters": parameters, "robot_id": robot_id}
            decision = gate.check(action)
            if not decision.is_allowed:
                return {
                    "status": "BLOCKED",
                    "robot_id": robot_id,
                    "task": task,
                    "risk_score": decision.risk_score,
                    "violations": decision.violated_constraints,
                    "replay_id": decision.replay_id,
                }
        except Exception as e:
            return {"status": "error", "phase": "firewall", "error": str(e)}

        # Step 2: Simulate execution
        import time
        start_time = time.time()
        try:
            if task == "pid_move":
                result = {"final_position": parameters.get("target", 1.0), "error": 0.02}
            elif task == "reach":
                result = {"final_pose": parameters.get("target_pose", [0.5, 0.0, 0.3]), "success": True}
            elif task == "g1_walk":
                result = {"distance": parameters.get("distance", 3.0), "falls": 0}
            else:
                result = {"message": f"Mock execution of {task}"}
            duration = time.time() - start_time
        except Exception as e:
            return {"status": "error", "phase": "execution", "error": str(e)}

        # Step 3: Record episode
        episode_id = f"ep_{int(time.time())}"
        try:
            from rosclaw.practice.episode_recorder import EpisodeRecorder
            from rosclaw.core.event_bus import EventBus, Event
            bus = EventBus()
            recorder = EpisodeRecorder(robot_id, event_bus=bus)
            recorder._do_initialize()
            bus.publish(Event(
                topic="skill.execution.start",
                payload={"episode_id": episode_id, "skill_name": task, "parameters": parameters},
                source="mcp_sandbox",
            ))
            bus.publish(Event(
                topic="skill.execution.complete",
                payload={"episode_id": episode_id, "result": result, "duration_sec": duration},
                source="mcp_sandbox",
            ))
            bus.publish(Event(
                topic="praxis.completed",
                payload={"episode_id": episode_id, "outcome": {"reward": 1.0}},
                source="mcp_sandbox",
            ))
        except Exception as e:
            return {"status": "error", "phase": "recording", "error": str(e)}

        return {
            "status": "SUCCESS",
            "robot_id": robot_id,
            "task": task,
            "world": world,
            "duration_sec": round(duration, 3),
            "episode_id": episode_id,
            "firewall": "ALLOWED",
            "risk_score": decision.risk_score if 'decision' in dir() else 0.0,
            "result": result,
        }

    async def _handle_query_memory(self, arguments: dict) -> dict:
        """Query ROSClaw Memory."""
        query = arguments.get("query", "")
        query_type = arguments.get("query_type", "similar")
        limit = arguments.get("limit", 5)
        try:
            from rosclaw.memory.interface import MemoryInterface
            mem = MemoryInterface("mcp")
            mem._do_initialize()
            if query_type == "failure":
                result = mem.explain_last_failure()
                return {"query": query, "type": query_type, "result": result}
            elif query_type == "similar":
                results = mem.find_similar_experiences(query, limit=limit)
                return {"query": query, "type": query_type, "results": results, "count": len(results)}
            else:
                stats = mem.get_statistics()
                return {"query": query, "type": query_type, "statistics": stats}
        except Exception as e:
            return {"query": query, "type": query_type, "error": str(e)}

    async def _handle_explain_failure(self, arguments: dict) -> dict:
        """Explain the most recent failure."""
        task_id = arguments.get("task_id")
        try:
            from rosclaw.memory.interface import MemoryInterface
            mem = MemoryInterface("mcp")
            mem._do_initialize()
            failure = mem.explain_last_failure(task_id=task_id)
            if failure:
                return {
                    "status": "found",
                    "failure_id": failure.get("id"),
                    "failure_type": failure.get("failure_type"),
                    "root_cause": failure.get("root_cause"),
                    "recovery_hint": failure.get("recovery_hint"),
                    "sandbox_intervened": failure.get("sandbox_intervened"),
                }
            return {"status": "not_found", "message": "No failure records found"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _handle_compile_asset_bundle(self, arguments: dict) -> dict:
        """Compile an Asset Bundle using Forge BundleCompiler."""
        sdk_doc = arguments.get("sdk_doc", "")
        bundle_name = arguments.get("bundle_name", "")
        staging = arguments.get("staging", True)
        try:
            from rosclaw.forge.bundle_compiler import BundleCompiler
            compiler = BundleCompiler()
            bundle = compiler.compile(sdk_doc, bundle_name)
            return {
                "status": "generated",
                "bundle_name": bundle_name,
                "staging": staging,
                "staging_ready": bundle.staging_ready,
                "production_ready": bundle.production_ready,
                "files": list(bundle.files.keys()),
                "validation": bundle.validation,
                "next_step": "Run critic validation before promoting to production",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def run(self) -> None:
        async with stdio_server() as (read, write):
            init_options = InitializationOptions(
                server_name="rosclaw-minimal",
                server_version="1.0.0",
                capabilities=self.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )
            await self.server.run(read, write, init_options)


def main() -> None:
    server = ROSClawMinimalMCPServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[ROSClaw MCP] Shutdown complete", file=sys.stderr)
    finally:
        try:
            server.hub.stop()
        except ValueError:
            pass  # stdout may be closed during teardown


if __name__ == "__main__":
    main()
