"""Tests for rosclaw.sdk_to_mcp module."""

import json
from pathlib import Path

import pytest
import yaml

from rosclaw.sdk_to_mcp import AssetCompiler, MCPManifestBuilder


class TestAssetCompiler:
    def test_compile_skill_file(self, tmp_path):
        skill = {
            "name": "pick_and_place",
            "description": "Pick an object and place it at target",
            "category": "manipulation",
            "parameters": {
                "object": {"type": "string", "description": "Object to pick", "required": True},
                "target": {"type": "string", "description": "Target location", "required": True},
            },
            "safety_checks": ["collision_check"],
        }
        skill_path = tmp_path / "pick_and_place.yaml"
        with open(skill_path, "w") as f:
            yaml.dump(skill, f)

        compiler = AssetCompiler()
        assets = compiler.compile_skill_file(skill_path)

        assert len(assets) == 2
        assert assets[0].asset_type == "tool"
        assert assets[0].name == "pick_and_place"
        assert assets[0].mcp_schema["name"] == "skill_pick_and_place"
        assert "object" in assets[0].mcp_schema["inputSchema"]["properties"]

    def test_compile_provider_manifest(self):
        manifest = {
            "name": "qwen_vl",
            "capabilities": ["vlm.object_grounding", "vlm.scene_description"],
        }
        compiler = AssetCompiler()
        assets = compiler.compile_provider_manifest(manifest)

        assert len(assets) == 2
        assert assets[0].asset_type == "resource"
        assert "rosclaw://providers/qwen_vl" in assets[0].mcp_schema["uri"]

    def test_compile_robot_profile(self):
        from unittest.mock import MagicMock
        profile = MagicMock()
        profile.robot_id = "ur5e"
        profile.name = "UR5e"
        cap = MagicMock()
        cap.capabilities = [{"name": "pick_and_place", "parameters": {}}]
        profile.capability = cap

        compiler = AssetCompiler()
        assets = compiler.compile_robot_profile(profile)
        assert any(a.name == "ur5e_state" for a in assets)

    def test_export_to_json(self, tmp_path):
        compiler = AssetCompiler()
        assets = compiler.compile_provider_manifest({
            "name": "test_provider", "capabilities": ["cap1"],
        })
        output = tmp_path / "output.json"
        compiler.export_to_json(assets, output)
        with open(output) as f:
            data = json.load(f)
        assert data["version"] == "1.0"

    def test_build_input_schema(self):
        params = {
            "x": {"type": "number", "description": "X coord", "required": True},
            "y": {"type": "number", "description": "Y coord"},
        }
        schema = AssetCompiler._build_input_schema(params)
        assert schema["properties"]["x"]["type"] == "number"
        assert "x" in schema["required"]
        assert "y" not in schema["required"]


class TestMCPManifestBuilder:
    def test_build_empty(self):
        builder = MCPManifestBuilder()
        manifest = builder.build()
        assert manifest.name == "rosclaw-mcp"
        assert manifest.tools == []

    def test_add_tools_from_assets(self):
        from unittest.mock import MagicMock
        builder = MCPManifestBuilder()
        asset = MagicMock()
        asset.asset_type = "tool"
        asset.mcp_schema = {"name": "test_tool"}
        builder.add_tools_from_assets([asset])
        assert len(builder.build().tools) == 1

    def test_add_robot_state_tool(self):
        builder = MCPManifestBuilder()
        builder.add_robot_state_tool("ur5e", "UR5e")
        manifest = builder.build()
        assert manifest.tools[0]["name"] == "robot_ur5e_state"

    def test_add_skill_tool(self):
        builder = MCPManifestBuilder()
        builder.add_skill_tool("pick", "Pick object", {
            "obj": {"type": "string", "description": "Object", "required": True},
        })
        manifest = builder.build()
        assert manifest.tools[0]["name"] == "skill_pick"
