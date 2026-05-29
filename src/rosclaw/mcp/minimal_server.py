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

from mcp.server import Server
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
                name="system.get_version",
                description="Get ROSClaw version and system status",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    async def _handle_system_tool(self, name: str, arguments: dict) -> dict:
        if name == "system.list_robots":
            try:
                from rosclaw.sandbox.eurdf.loader import list_robots
                robots = list_robots()
                return {"robots": robots, "count": len(robots)}
            except Exception as e:
                return {"robots": [], "count": 0, "error": str(e)}
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

    async def run(self) -> None:
        async with stdio_server(self.server) as (read, write):
            init_options = InitializationOptions(
                server_name="rosclaw-minimal",
                server_version="1.0.0",
                capabilities=self.server.get_capabilities(),
            )
            await self.server.run(read, write, init_options)


def main() -> None:
    server = ROSClawMinimalMCPServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[ROSClaw MCP] Shutdown complete", file=sys.stderr)
    finally:
        server.hub.stop()


if __name__ == "__main__":
    main()
