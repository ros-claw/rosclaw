"""Scenario F: Forge / sdk_to_mcp Self-Extension Capability.

Validates ROSClaw's ability to self-extend by generating MCP bundles
from SDK descriptions, installing to staging, and blocking unsafe
code via critic validation.

Targets:
  - Forge compiles SDK → MCP Server + Skill Manifest + Provider Manifest + Tests + README
  - Critic blocks unsafe bundles (missing safety hooks)
  - Staging install succeeds for safe bundles
  - MCP end-to-end tool invocation works
"""

import json
import ast


from rosclaw.forge.bundle_compiler import BundleCompiler
from rosclaw.sdk_to_mcp import AssetCompiler, MCPManifestBuilder


# ───────────────────────────────
# Scenario F: Forge Bundle Generation
# ───────────────────────────────

class TestScenarioFForgeBundle:
    """Forge generates complete, parseable, safe bundles from SDK docs."""

    def test_scenario_f_forge_generates_all_artifacts(self, tmp_path):
        """F1: Forge reads SDK doc and generates complete bundle."""
        compiler = BundleCompiler()
        bundle = compiler.compile(
            "SDK: motor driver with current feedback, torque limit 2Nm, CAN bus interface",
            "motor_driver"
        )

        # Must generate all 5 artifact types
        assert any(f.endswith("mcp_server.py") for f in bundle.files)
        assert any(f.endswith("skill_manifest.json") for f in bundle.files)
        assert any(f.endswith("provider_manifest.json") for f in bundle.files)
        assert any(f.endswith("README.md") for f in bundle.files)
        assert any("tests/" in f and f.endswith(".py") for f in bundle.files)

    def test_scenario_f_generated_files_are_valid(self, tmp_path):
        """F2: Generated files are syntactically valid."""
        compiler = BundleCompiler()
        bundle = compiler.compile("SDK: IMU with 9-DOF, 1000Hz, SPI interface", "imu_9dof")

        staging_dir = tmp_path / "staging"
        for fname, content in bundle.files.items():
            fpath = staging_dir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

            if fname.endswith(".json"):
                data = json.loads(fpath.read_text())
                assert isinstance(data, dict)
            elif fname.endswith(".py"):
                ast.parse(fpath.read_text())  # Syntax check
            elif fname.endswith(".md"):
                assert len(fpath.read_text()) > 50

    def test_scenario_f_staging_install_records_manifest(self, tmp_path):
        """F3: Staging install creates verifiable manifest."""
        compiler = BundleCompiler()
        bundle = compiler.compile(
            "SDK: 6-axis force/torque sensor with ROS2 topic /ft_sensor",
            "fts_sensor"
        )

        install_root = tmp_path / ".rosclaw" / "staging"
        for fname, content in bundle.files.items():
            fpath = install_root / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

        manifest = json.loads((install_root / "fts_sensor" / "skill_manifest.json").read_text())
        assert manifest["skill_id"] == "fts_sensor"
        assert manifest["sandbox_safe"] is True
        assert "firewall_hooks" in manifest

    def test_scenario_f_critic_blocks_unsafe_sdk(self):
        """F4: Critic blocks bundles missing safety hooks."""
        compiler = BundleCompiler()
        bundle = compiler.compile("Minimal SDK with no safety info", "unsafe_bot")

        # Auto-injected safety hooks must be present
        assert bundle.validation["safety_hooks"] is True
        assert bundle.validation["preemption_ready"] is True
        # Unsafe bundle stays in staging, never production
        assert bundle.staging_ready is True
        assert bundle.production_ready is False

    def test_scenario_f_mcp_compile_asset_bundle_tool(self):
        """F5: MCP tool compile_asset_bundle works end-to-end."""
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer
        import asyncio

        server = ROSClawMinimalMCPServer()
        result = asyncio.run(server._handle_system_tool("system.compile_asset_bundle", {
            "sdk_doc": "Lidar 360°, 10Hz, ROS2 /scan, range 0.1-40m",
            "bundle_name": "lidar_360",
            "staging": True,
        }))
        assert result["status"] == "generated"
        assert "bundle_name" in result

    def test_scenario_f_sdk_to_mcp_asset_compiler_roundtrip(self, tmp_path):
        """F6: AssetCompiler roundtrip skill→manifest→export→import."""
        skill = {
            "name": "grasp_object",
            "description": "Grasp an object with force control",
            "category": "manipulation",
            "parameters": {
                "object_id": {"type": "string", "required": True},
                "force": {"type": "number", "default": 20.0},
            },
            "safety_checks": ["force_limit", "collision_check"],
        }
        skill_path = tmp_path / "grasp_object.yaml"
        skill_path.write_text(__import__("yaml").dump(skill))

        compiler = AssetCompiler()
        assets = compiler.compile_skill_file(skill_path)

        builder = MCPManifestBuilder("test_server")
        builder.add_tools_from_assets(assets)
        manifest = builder.build().to_dict()

        assert "tools" in manifest
        assert any(t["name"] == "skill_grasp_object" for t in manifest["tools"])

        # Export assets roundtrip
        export_path = tmp_path / "assets.json"
        compiler.export_to_json(assets, export_path)
        loaded = json.loads(export_path.read_text())
        assert isinstance(loaded, dict)
        assert "assets" in loaded
        assert any(a["name"] == "grasp_object" for a in loaded["assets"])

    def test_scenario_f_full_closed_loop(self, tmp_path):
        """F7: Full closed loop — SDK → Forge → Staging → Validate."""
        compiler = BundleCompiler()
        bundle = compiler.compile(
            "SDK: Temperature sensor, -40°C to 125°C, I2C, safety: max 150°C shutdown",
            "temp_sensor"
        )

        # Stage
        staging = tmp_path / "staging" / "temp_sensor"
        for fname, content in bundle.files.items():
            fpath = staging / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

        # Validate all artifacts
        assert (staging / "temp_sensor" / "mcp_server.py").exists()
        assert (staging / "temp_sensor" / "skill_manifest.json").exists()
        assert (staging / "temp_sensor" / "provider_manifest.json").exists()
        assert (staging / "temp_sensor" / "README.md").exists()
        assert (staging / "temp_sensor" / "tests" / "test_temp_sensor.py").exists()

        # Safety validation
        assert bundle.validation["safety_hooks"] is True
        assert bundle.staging_ready is True
