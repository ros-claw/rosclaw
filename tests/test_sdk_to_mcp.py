"""Comprehensive tests for rosclaw.sdk_to_mcp module.

Targets:
  - AssetCompiler.compile_skill_file()
  - AssetCompiler.compile_provider_manifest()
  - AssetCompiler.compile_robot_profile()
  - AssetCompiler.compile_directory()
  - AssetCompiler.export_to_json()
  - MCPManifestBuilder.build()
  - End-to-end compilation flows

Coverage target: 70%+
"""

import json
from unittest.mock import MagicMock

import yaml

from rosclaw.sdk_to_mcp import AssetCompiler, MCPManifestBuilder
from rosclaw.sdk_to_mcp.compiler import CompiledAsset
from rosclaw.sdk_to_mcp.manifest import MCPServerManifest


# ───────────────────────────────
# AssetCompiler — Skill Compilation
# ───────────────────────────────

class TestAssetCompilerSkillFile:
    def test_compile_skill_file_basic(self, tmp_path):
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
        skill_path.write_text(yaml.dump(skill), encoding="utf-8")

        compiler = AssetCompiler()
        assets = compiler.compile_skill_file(skill_path)

        assert len(assets) == 2

        # Main tool
        tool = assets[0]
        assert tool.asset_type == "tool"
        assert tool.name == "pick_and_place"
        assert tool.mcp_schema["name"] == "skill_pick_and_place"
        assert tool.mcp_schema["description"] == "Pick an object and place it at target"
        assert tool.source_path == str(skill_path)
        assert tool.metadata["category"] == "manipulation"
        schema = tool.mcp_schema["inputSchema"]
        assert "object" in schema["properties"]
        assert "target" in schema["properties"]
        assert "object" in schema["required"]
        assert "target" in schema["required"]

        # Safety prompt
        prompt = assets[1]
        assert prompt.asset_type == "prompt"
        assert prompt.name == "safety_pick_and_place"
        assert prompt.mcp_schema["name"] == "safety_pick_and_place"
        assert prompt.source_path == str(skill_path)

    def test_compile_skill_file_no_safety_checks(self, tmp_path):
        skill = {
            "name": "simple_move",
            "description": "Move forward",
            "parameters": {},
        }
        skill_path = tmp_path / "simple_move.yaml"
        skill_path.write_text(yaml.dump(skill), encoding="utf-8")

        compiler = AssetCompiler()
        assets = compiler.compile_skill_file(skill_path)

        assert len(assets) == 1
        assert assets[0].asset_type == "tool"
        assert assets[0].name == "simple_move"

    def test_compile_skill_file_defaults(self, tmp_path):
        skill = {}
        skill_path = tmp_path / "unnamed.yaml"
        skill_path.write_text(yaml.dump(skill), encoding="utf-8")

        compiler = AssetCompiler()
        assets = compiler.compile_skill_file(skill_path)

        assert len(assets) == 1
        assert assets[0].name == "unnamed"  # falls back to path.stem
        assert assets[0].mcp_schema["description"] == "Execute unnamed skill"
        assert assets[0].metadata["category"] == "manipulation"

    def test_compile_skill_file_str_parameters(self, tmp_path):
        """Parameters that are plain strings (not dicts) should still work."""
        skill = {
            "name": "legacy",
            "parameters": {
                "command": "just a string description",
            },
        }
        skill_path = tmp_path / "legacy.yaml"
        skill_path.write_text(yaml.dump(skill), encoding="utf-8")

        compiler = AssetCompiler()
        assets = compiler.compile_skill_file(skill_path)
        schema = assets[0].mcp_schema["inputSchema"]
        assert schema["properties"]["command"]["type"] == "string"
        assert schema["properties"]["command"]["description"] == "just a string description"

    def test_compile_skill_file_path_as_string(self, tmp_path):
        skill = {"name": "test"}
        skill_path = tmp_path / "test.yaml"
        skill_path.write_text(yaml.dump(skill), encoding="utf-8")

        compiler = AssetCompiler()
        assets = compiler.compile_skill_file(str(skill_path))
        assert assets[0].name == "test"


# ───────────────────────────────
# AssetCompiler — Provider Manifest
# ───────────────────────────────

class TestAssetCompilerProviderManifest:
    def test_compile_provider_manifest_str_capabilities(self):
        manifest = {
            "name": "qwen_vl",
            "capabilities": ["vlm.object_grounding", "vlm.scene_description"],
        }
        compiler = AssetCompiler()
        assets = compiler.compile_provider_manifest(manifest)

        assert len(assets) == 2
        assert assets[0].asset_type == "resource"
        assert assets[0].name == "vlm.object_grounding"
        assert "rosclaw://providers/qwen_vl/vlm.object_grounding" in assets[0].mcp_schema["uri"]
        assert assets[0].metadata["provider"] == "qwen_vl"

        assert assets[1].asset_type == "resource"
        assert assets[1].name == "vlm.scene_description"

    def test_compile_provider_manifest_dict_capabilities(self):
        manifest = {
            "name": "gpt4o",
            "capabilities": [
                {"name": "vision", "description": "See things"},
                {"name": "reasoning"},
            ],
        }
        compiler = AssetCompiler()
        assets = compiler.compile_provider_manifest(manifest)

        assert len(assets) == 2
        assert assets[0].name == "vision"
        assert assets[1].name == "reasoning"

    def test_compile_provider_manifest_empty(self):
        manifest = {"name": "empty_provider", "capabilities": []}
        compiler = AssetCompiler()
        assets = compiler.compile_provider_manifest(manifest)
        assert assets == []

    def test_compile_provider_manifest_no_name(self):
        manifest = {"capabilities": ["cap1"]}
        compiler = AssetCompiler()
        assets = compiler.compile_provider_manifest(manifest)
        assert assets[0].mcp_schema["uri"] == "rosclaw://providers/unknown/cap1"


# ───────────────────────────────
# AssetCompiler — Robot Profile
# ───────────────────────────────

class TestAssetCompilerRobotProfile:
    def test_compile_robot_profile_basic(self):
        profile = MagicMock()
        profile.robot_id = "ur5e"
        profile.name = "UR5e"
        profile.capability = None

        compiler = AssetCompiler()
        assets = compiler.compile_robot_profile(profile)

        assert any(a.name == "ur5e_state" for a in assets)
        state_asset = [a for a in assets if a.name == "ur5e_state"][0]
        assert state_asset.asset_type == "tool"
        assert state_asset.mcp_schema["name"] == "robot_ur5e_state"
        assert state_asset.metadata["robot_id"] == "ur5e"

    def test_compile_robot_profile_with_capabilities(self):
        profile = MagicMock()
        profile.robot_id = "go2"
        profile.name = "Unitree Go2"
        cap = MagicMock()
        cap.capabilities = [
            {"name": "walk", "parameters": {"speed": {"type": "number", "required": True}}},
            {"name": "stand_up", "parameters": {}},
        ]
        profile.capability = cap

        compiler = AssetCompiler()
        assets = compiler.compile_robot_profile(profile)

        names = {a.name for a in assets}
        assert "go2_state" in names
        assert "go2_walk" in names
        assert "go2_stand_up" in names

        walk_asset = [a for a in assets if a.name == "go2_walk"][0]
        assert walk_asset.asset_type == "tool"
        assert "speed" in walk_asset.mcp_schema["inputSchema"]["properties"]

    def test_compile_robot_profile_str_capabilities(self):
        profile = MagicMock()
        profile.robot_id = "test_bot"
        profile.name = "Test"
        cap = MagicMock()
        cap.capabilities = ["jump", "spin"]
        profile.capability = cap

        compiler = AssetCompiler()
        assets = compiler.compile_robot_profile(profile)

        names = {a.name for a in assets}
        assert "test_bot_jump" in names
        assert "test_bot_spin" in names

    def test_compile_robot_profile_no_name(self):
        profile = MagicMock()
        profile.robot_id = "anon"
        del profile.name  # simulate missing name
        profile.capability = None

        compiler = AssetCompiler()
        assets = compiler.compile_robot_profile(profile)

        state_asset = [a for a in assets if a.name == "anon_state"][0]
        assert state_asset.mcp_schema["description"] == "Get current state of anon"


# ───────────────────────────────
# AssetCompiler — Directory & Export
# ───────────────────────────────

class TestAssetCompilerBatchAndExport:
    def test_compile_directory(self, tmp_path):
        (tmp_path / "skill_a.yaml").write_text(yaml.dump({"name": "skill_a"}), encoding="utf-8")
        (tmp_path / "skill_b.yaml").write_text(yaml.dump({"name": "skill_b"}), encoding="utf-8")
        (tmp_path / "readme.txt").write_text("not a skill", encoding="utf-8")

        compiler = AssetCompiler()
        assets = compiler.compile_directory(tmp_path, pattern="*.yaml")

        assert len(assets) == 2
        names = {a.name for a in assets}
        assert "skill_a" in names
        assert "skill_b" in names

    def test_compile_directory_ignores_bad_files(self, tmp_path, caplog):
        (tmp_path / "bad.yaml").write_text("not: valid: [", encoding="utf-8")

        compiler = AssetCompiler()
        assets = compiler.compile_directory(tmp_path, pattern="*.yaml")

        assert assets == []
        assert "Failed to compile" in caplog.text

    def test_compile_directory_empty(self, tmp_path):
        compiler = AssetCompiler()
        assets = compiler.compile_directory(tmp_path)
        assert assets == []

    def test_export_to_json(self, tmp_path):
        compiler = AssetCompiler()
        assets = compiler.compile_provider_manifest({
            "name": "test_provider",
            "capabilities": ["cap1", "cap2"],
        })
        output = tmp_path / "output.json"
        compiler.export_to_json(assets, output)

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["version"] == "1.0"
        assert len(data["assets"]) == 2
        assert data["assets"][0]["name"] == "cap1"
        assert data["assets"][0]["type"] == "resource"
        assert "schema" in data["assets"][0]

    def test_export_to_json_nested_dir(self, tmp_path):
        compiler = AssetCompiler()
        assets = [CompiledAsset(name="x", asset_type="tool", mcp_schema={})]
        output = tmp_path / "nested" / "deep" / "out.json"
        compiler.export_to_json(assets, output)
        assert output.exists()


# ───────────────────────────────
# AssetCompiler — Helpers
# ───────────────────────────────

class TestAssetCompilerHelpers:
    def test_build_input_schema_required(self):
        params = {
            "x": {"type": "number", "description": "X coord", "required": True},
            "y": {"type": "number", "description": "Y coord"},
        }
        schema = AssetCompiler._build_input_schema(params)
        assert schema["type"] == "object"
        assert schema["properties"]["x"]["type"] == "number"
        assert schema["properties"]["x"]["description"] == "X coord"
        assert "x" in schema["required"]
        assert "y" not in schema["required"]

    def test_build_input_schema_empty(self):
        schema = AssetCompiler._build_input_schema({})
        assert schema == {"type": "object", "properties": {}, "required": []}

    def test_build_input_schema_non_dict_spec(self):
        """When parameter spec is a plain string, treat it as description with type string."""
        params = {"command": "A raw string description"}
        schema = AssetCompiler._build_input_schema(params)
        assert schema["properties"]["command"]["type"] == "string"
        assert schema["properties"]["command"]["description"] == "A raw string description"

    def test_output_dir_set(self, tmp_path):
        compiler = AssetCompiler(output_dir=str(tmp_path))
        assert compiler.output_dir == tmp_path

    def test_output_dir_default(self):
        compiler = AssetCompiler()
        assert compiler.output_dir is None


# ───────────────────────────────
# MCPManifestBuilder
# ───────────────────────────────

class TestMCPManifestBuilder:
    def test_build_empty(self):
        builder = MCPManifestBuilder()
        manifest = builder.build()
        assert isinstance(manifest, MCPServerManifest)
        assert manifest.name == "rosclaw-mcp"
        assert manifest.version == "1.0.0"
        assert manifest.description == "ROSClaw MCP Server for Physical Intelligence"
        assert manifest.tools == []
        assert manifest.resources == []
        assert manifest.prompts == []

    def test_build_custom_name(self):
        builder = MCPManifestBuilder(name="custom-mcp", version="2.0.0")
        manifest = builder.build()
        assert manifest.name == "custom-mcp"
        assert manifest.version == "2.0.0"

    def test_to_dict(self):
        builder = MCPManifestBuilder()
        builder.add_robot_state_tool("ur5e", "UR5e")
        manifest = builder.build()
        d = manifest.to_dict()
        assert d["name"] == "rosclaw-mcp"
        assert d["version"] == "1.0.0"
        assert len(d["tools"]) == 1
        assert d["resources"] == []
        assert d["prompts"] == []

    def test_add_tools_from_assets(self):
        builder = MCPManifestBuilder()
        asset = MagicMock()
        asset.asset_type = "tool"
        asset.mcp_schema = {"name": "tool_a"}
        builder.add_tools_from_assets([asset])

        manifest = builder.build()
        assert len(manifest.tools) == 1
        assert manifest.tools[0]["name"] == "tool_a"

    def test_add_tools_skips_non_tools(self):
        builder = MCPManifestBuilder()
        tool = MagicMock()
        tool.asset_type = "tool"
        tool.mcp_schema = {"name": "t1"}
        resource = MagicMock()
        resource.asset_type = "resource"
        resource.mcp_schema = {"name": "r1"}
        builder.add_tools_from_assets([tool, resource])

        manifest = builder.build()
        assert len(manifest.tools) == 1
        assert manifest.tools[0]["name"] == "t1"

    def test_add_resources_from_assets(self):
        builder = MCPManifestBuilder()
        asset = MagicMock()
        asset.asset_type = "resource"
        asset.mcp_schema = {"uri": "rosclaw://test"}
        builder.add_resources_from_assets([asset])

        manifest = builder.build()
        assert len(manifest.resources) == 1
        assert manifest.resources[0]["uri"] == "rosclaw://test"

    def test_add_resources_skips_non_resources(self):
        builder = MCPManifestBuilder()
        tool = MagicMock()
        tool.asset_type = "tool"
        tool.mcp_schema = {"name": "t1"}
        builder.add_resources_from_assets([tool])

        manifest = builder.build()
        assert manifest.resources == []

    def test_add_prompts_from_assets(self):
        builder = MCPManifestBuilder()
        asset = MagicMock()
        asset.asset_type = "prompt"
        asset.mcp_schema = {"name": "safety_check"}
        builder.add_prompts_from_assets([asset])

        manifest = builder.build()
        assert len(manifest.prompts) == 1
        assert manifest.prompts[0]["name"] == "safety_check"

    def test_add_prompts_skips_non_prompts(self):
        builder = MCPManifestBuilder()
        tool = MagicMock()
        tool.asset_type = "tool"
        tool.mcp_schema = {"name": "t1"}
        builder.add_prompts_from_assets([tool])

        manifest = builder.build()
        assert manifest.prompts == []

    def test_add_robot_state_tool(self):
        builder = MCPManifestBuilder()
        builder.add_robot_state_tool("ur5e", "UR5e")
        manifest = builder.build()
        assert len(manifest.tools) == 1
        assert manifest.tools[0]["name"] == "robot_ur5e_state"
        assert manifest.tools[0]["description"] == "Get current state of UR5e"
        assert manifest.tools[0]["inputSchema"] == {"type": "object", "properties": {}}

    def test_add_skill_tool(self):
        builder = MCPManifestBuilder()
        builder.add_skill_tool("pick", "Pick object", {
            "obj": {"type": "string", "description": "Object", "required": True},
            "force": {"type": "number", "description": "Force"},
        })
        manifest = builder.build()

        assert len(manifest.tools) == 1
        tool = manifest.tools[0]
        assert tool["name"] == "skill_pick"
        assert tool["description"] == "Pick object"
        schema = tool["inputSchema"]
        assert schema["properties"]["obj"]["type"] == "string"
        assert "obj" in schema["required"]
        assert "force" not in schema["required"]

    def test_add_skill_tool_no_required(self):
        builder = MCPManifestBuilder()
        builder.add_skill_tool("move", "Move", {
            "x": {"type": "number", "description": "X"},
        })
        manifest = builder.build()
        assert manifest.tools[0]["inputSchema"]["required"] == []

    def test_combined_build(self):
        builder = MCPManifestBuilder(name="combined")
        builder.add_robot_state_tool("ur5e", "UR5e")
        builder.add_skill_tool("pick", "Pick", {"obj": {"type": "string"}})

        tool_asset = MagicMock()
        tool_asset.asset_type = "tool"
        tool_asset.mcp_schema = {"name": "extra_tool"}
        builder.add_tools_from_assets([tool_asset])

        resource_asset = MagicMock()
        resource_asset.asset_type = "resource"
        resource_asset.mcp_schema = {"uri": "res://x"}
        builder.add_resources_from_assets([resource_asset])

        prompt_asset = MagicMock()
        prompt_asset.asset_type = "prompt"
        prompt_asset.mcp_schema = {"name": "prompt_x"}
        builder.add_prompts_from_assets([prompt_asset])

        manifest = builder.build()
        assert len(manifest.tools) == 3
        assert len(manifest.resources) == 1
        assert len(manifest.prompts) == 1


# ───────────────────────────────
# End-to-End Compilation Flow
# ───────────────────────────────

class TestEndToEndCompilation:
    def test_full_skill_to_manifest_flow(self, tmp_path):
        """Compile a skill YAML and feed it into MCPManifestBuilder."""
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
        skill_path.write_text(yaml.dump(skill), encoding="utf-8")

        compiler = AssetCompiler()
        assets = compiler.compile_skill_file(skill_path)

        builder = MCPManifestBuilder(name="test-mcp")
        builder.add_tools_from_assets(assets)
        builder.add_prompts_from_assets(assets)
        manifest = builder.build()

        assert manifest.name == "test-mcp"
        assert len(manifest.tools) == 1
        assert manifest.tools[0]["name"] == "skill_pick_and_place"
        assert len(manifest.prompts) == 1
        assert manifest.prompts[0]["name"] == "safety_pick_and_place"

    def test_full_provider_to_manifest_flow(self, tmp_path):
        """Compile a provider manifest and feed it into MCPManifestBuilder."""
        provider = {
            "name": "vlm_provider",
            "capabilities": ["object_detection", "scene_description"],
        }
        compiler = AssetCompiler()
        assets = compiler.compile_provider_manifest(provider)

        builder = MCPManifestBuilder()
        builder.add_resources_from_assets(assets)
        manifest = builder.build()

        assert len(manifest.resources) == 2
        uris = {r["uri"] for r in manifest.resources}
        assert "rosclaw://providers/vlm_provider/object_detection" in uris

    def test_full_robot_to_manifest_flow(self):
        """Compile a robot profile and feed it into MCPManifestBuilder."""
        profile = MagicMock()
        profile.robot_id = "go2"
        profile.name = "Unitree Go2"
        profile.capability = None

        compiler = AssetCompiler()
        assets = compiler.compile_robot_profile(profile)

        builder = MCPManifestBuilder()
        builder.add_tools_from_assets(assets)
        manifest = builder.build()

        assert any(t["name"] == "robot_go2_state" for t in manifest.tools)

    def test_export_roundtrip(self, tmp_path):
        """Compile, build manifest, export to JSON, and verify roundtrip."""
        skill = {
            "name": "test_skill",
            "parameters": {"param1": {"type": "string", "required": True}},
        }
        skill_path = tmp_path / "test_skill.yaml"
        skill_path.write_text(yaml.dump(skill), encoding="utf-8")

        compiler = AssetCompiler()
        assets = compiler.compile_skill_file(skill_path)

        builder = MCPManifestBuilder()
        builder.add_tools_from_assets(assets)
        manifest = builder.build()

        # Export manifest dict as JSON
        out = tmp_path / "manifest.json"
        out.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["name"] == "rosclaw-mcp"
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "skill_test_skill"

    def test_directory_to_manifest_flow(self, tmp_path):
        """Compile a directory of skills into a single manifest."""
        (tmp_path / "skill_a.yaml").write_text(yaml.dump({"name": "skill_a"}), encoding="utf-8")
        (tmp_path / "skill_b.yaml").write_text(yaml.dump({"name": "skill_b"}), encoding="utf-8")

        compiler = AssetCompiler()
        assets = compiler.compile_directory(tmp_path, pattern="*.yaml")

        builder = MCPManifestBuilder()
        builder.add_tools_from_assets(assets)
        manifest = builder.build()

        names = {t["name"] for t in manifest.tools}
        assert "skill_skill_a" in names
        assert "skill_skill_b" in names


# ───────────────────────────────
# CompiledAsset Dataclass
# ───────────────────────────────

class TestCompiledAsset:
    def test_defaults(self):
        asset = CompiledAsset(name="x", asset_type="tool", mcp_schema={})
        assert asset.source_path is None
        assert asset.metadata == {}

    def test_full(self):
        asset = CompiledAsset(
            name="x",
            asset_type="tool",
            mcp_schema={"name": "x"},
            source_path="/tmp",
            metadata={"k": "v"},
        )
        assert asset.source_path == "/tmp"
        assert asset.metadata == {"k": "v"}
