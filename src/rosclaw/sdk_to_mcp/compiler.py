"""AssetCompiler — Compile ROSClaw SDK assets to MCP-compatible format."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class CompiledAsset:
    """Result of compiling a ROSClaw SDK asset to MCP format."""
    name: str
    asset_type: str  # tool, resource, prompt
    mcp_schema: dict[str, Any]
    source_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AssetCompiler:
    """Compile ROSClaw SDK assets (skills, providers, robots) to MCP format.

    Usage:
        compiler = AssetCompiler()
        assets = compiler.compile_skill_file("skills/pick_and_place.yaml")
        assets = compiler.compile_robot_profile("ur5e")
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else None

    # ── Skill Compilation ──

    def compile_skill_file(self, path: str | Path) -> list[CompiledAsset]:
        """Compile a skill definition YAML to MCP tool schema."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            skill = yaml.safe_load(f)

        assets = []
        skill_name = skill.get("name", path.stem)

        # Main skill tool
        tool_schema = {
            "name": f"skill_{skill_name}",
            "description": skill.get("description", f"Execute {skill_name} skill"),
            "inputSchema": self._build_input_schema(skill.get("parameters", {})),
        }
        assets.append(CompiledAsset(
            name=skill_name,
            asset_type="tool",
            mcp_schema=tool_schema,
            source_path=str(path),
            metadata={"category": skill.get("category", "manipulation")},
        ))

        # Safety check prompt
        if skill.get("safety_checks"):
            prompt_schema = {
                "name": f"safety_{skill_name}",
                "description": f"Safety pre-checks for {skill_name}",
                "arguments": [{"name": "context", "description": "Execution context"}],
            }
            assets.append(CompiledAsset(
                name=f"safety_{skill_name}",
                asset_type="prompt",
                mcp_schema=prompt_schema,
                source_path=str(path),
            ))

        return assets

    # ── Provider Manifest Compilation ──

    def compile_provider_manifest(self, manifest: dict[str, Any]) -> list[CompiledAsset]:
        """Compile a ProviderManifest to MCP resource templates."""
        assets = []
        provider_name = manifest.get("name", "unknown")

        for cap in manifest.get("capabilities", []):
            cap_name = cap if isinstance(cap, str) else cap.get("name", "unknown")
            resource_schema = {
                "uri": f"rosclaw://providers/{provider_name}/{cap_name}",
                "name": cap_name,
                "mimeType": "application/json",
                "description": f"Capability {cap_name} from provider {provider_name}",
            }
            assets.append(CompiledAsset(
                name=cap_name,
                asset_type="resource",
                mcp_schema=resource_schema,
                metadata={"provider": provider_name},
            ))

        return assets

    # ── Robot Profile Compilation ──

    def compile_robot_profile(self, profile: Any) -> list[CompiledAsset]:
        """Compile an e-URDF robot profile to MCP tools/prompts."""
        assets = []
        robot_id = getattr(profile, "robot_id", "unknown")
        name = getattr(profile, "name", robot_id)

        # Robot state tool
        state_tool = {
            "name": f"robot_{robot_id}_state",
            "description": f"Get current state of {name}",
            "inputSchema": {"type": "object", "properties": {}},
        }
        assets.append(CompiledAsset(
            name=f"{robot_id}_state",
            asset_type="tool",
            mcp_schema=state_tool,
            metadata={"robot_id": robot_id},
        ))

        # Capability tools
        capability = getattr(profile, "capability", None)
        if capability:
            caps = getattr(capability, "capabilities", [])
            for cap in caps:
                cap_name = cap.get("name", "unknown") if isinstance(cap, dict) else str(cap)
                cap_tool = {
                    "name": f"robot_{robot_id}_{cap_name}",
                    "description": f"Execute {cap_name} on {name}",
                    "inputSchema": self._build_input_schema(cap.get("parameters", {}) if isinstance(cap, dict) else {}),
                }
                assets.append(CompiledAsset(
                    name=f"{robot_id}_{cap_name}",
                    asset_type="tool",
                    mcp_schema=cap_tool,
                    metadata={"robot_id": robot_id, "capability": cap_name},
                ))

        return assets

    # ── Batch Compilation ──

    def compile_directory(self, directory: str | Path, pattern: str = "*.yaml") -> list[CompiledAsset]:
        """Compile all matching files in a directory."""
        directory = Path(directory)
        assets = []
        for file_path in directory.glob(pattern):
            try:
                assets.extend(self.compile_skill_file(file_path))
            except Exception as exc:
                print(f"[AssetCompiler] Failed to compile {file_path}: {exc}")
        return assets

    # ── Export ──

    def export_to_json(self, assets: list[CompiledAsset], output_path: str | Path) -> None:
        """Export compiled assets to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "assets": [
                {
                    "name": a.name,
                    "type": a.asset_type,
                    "schema": a.mcp_schema,
                    "source": a.source_path,
                    "metadata": a.metadata,
                }
                for a in assets
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── Helpers ──

    @staticmethod
    def _build_input_schema(parameters: dict[str, Any]) -> dict[str, Any]:
        """Build JSON schema from parameter definitions."""
        schema = {"type": "object", "properties": {}, "required": []}
        for name, spec in parameters.items():
            if isinstance(spec, dict):
                schema["properties"][name] = {
                    "type": spec.get("type", "string"),
                    "description": spec.get("description", ""),
                }
                if spec.get("required", False):
                    schema["required"].append(name)
            else:
                schema["properties"][name] = {"type": "string", "description": str(spec)}
        return schema
