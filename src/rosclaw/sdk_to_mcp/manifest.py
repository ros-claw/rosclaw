"""MCPManifestBuilder — Build MCP server manifest from ROSClaw assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MCPServerManifest:
    """MCP Server manifest for ROSClaw."""
    name: str = "rosclaw-mcp"
    version: str = "1.0.0"
    description: str = "ROSClaw MCP Server for Physical Intelligence"
    tools: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tools": self.tools,
            "resources": self.resources,
            "prompts": self.prompts,
        }


class MCPManifestBuilder:
    """Build MCP server manifest from compiled ROSClaw assets.

    Usage:
        builder = MCPManifestBuilder()
        builder.add_tools_from_assets(compiled_assets)
        manifest = builder.build()
    """

    def __init__(self, name: str = "rosclaw-mcp", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self._tools: list[dict[str, Any]] = []
        self._resources: list[dict[str, Any]] = []
        self._prompts: list[dict[str, Any]] = []

    def add_tools_from_assets(self, assets: list[Any]) -> None:
        """Add tool schemas from compiled assets."""
        for asset in assets:
            if getattr(asset, "asset_type", None) == "tool":
                self._tools.append(asset.mcp_schema)

    def add_resources_from_assets(self, assets: list[Any]) -> None:
        """Add resource schemas from compiled assets."""
        for asset in assets:
            if getattr(asset, "asset_type", None) == "resource":
                self._resources.append(asset.mcp_schema)

    def add_prompts_from_assets(self, assets: list[Any]) -> None:
        """Add prompt schemas from compiled assets."""
        for asset in assets:
            if getattr(asset, "asset_type", None) == "prompt":
                self._prompts.append(asset.mcp_schema)

    def add_robot_state_tool(self, robot_id: str, robot_name: str) -> None:
        """Add a standard robot state query tool."""
        self._tools.append({
            "name": f"robot_{robot_id}_state",
            "description": f"Get current state of {robot_name}",
            "inputSchema": {"type": "object", "properties": {}},
        })

    def add_skill_tool(self, skill_name: str, description: str, parameters: dict[str, Any]) -> None:
        """Add a skill execution tool."""
        schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        for name, spec in parameters.items():
            param_schema = {"type": spec.get("type", "string"), "description": spec.get("description", "")}
            schema["properties"][name] = param_schema
            if spec.get("required", False):
                schema["required"].append(name)

        self._tools.append({
            "name": f"skill_{skill_name}",
            "description": description,
            "inputSchema": schema,
        })

    def build(self) -> MCPServerManifest:
        """Build and return the manifest."""
        return MCPServerManifest(
            name=self.name,
            version=self.version,
            tools=self._tools,
            resources=self._resources,
            prompts=self._prompts,
        )
