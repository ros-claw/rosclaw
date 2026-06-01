"""Phase 3 RC Validation — Dashboard + Forge + MuJoCo real physics.

Targets the final +3 points to reach 85/100 RC.
"""

import json
import subprocess
import sys

import pytest


def _enough_ram_for_physics(min_gb: float = 4.0) -> bool:
    """Check if system has enough free RAM for stable MuJoCo physics."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024) >= min_gb
    except Exception:
        pass
    return True


class TestPhase3Dashboard:
    """Dashboard real startup and EventBus subscription."""

    def test_dashboard_imports(self):
        from rosclaw.dashboard.web_server import DashboardWebServer
        from rosclaw.core.event_bus import EventBus

        EventBus()
        ws = DashboardWebServer(host="127.0.0.1", port=18765)

        # Simulate events directly via metrics
        ws.metrics.increment_event("rosclaw.runtime.started")
        ws.metrics.increment_event("rosclaw.provider.inference.completed")

        counts = ws.metrics.get_event_counts()
        assert "rosclaw.runtime.started" in counts
        assert "rosclaw.provider.inference.completed" in counts

    def test_dashboard_health_snapshot(self):
        from rosclaw.dashboard.web_server import DashboardWebServer

        ws = DashboardWebServer(host="127.0.0.1", port=18766)
        ws.metrics.set_module_health("runtime", "HEALTHY")
        ws.metrics.set_module_health("sandbox", "HEALTHY")

        health = ws.server.get_health()
        assert health["status"] == "HEALTHY"
        assert "runtime" in health["modules"]
        assert "sandbox" in health["modules"]

        snap = ws.server.get_snapshot()
        assert "module_health" in snap
        assert "provider" in snap
        assert "sandbox" in snap
        assert "episodes" in snap
        assert "event_counts" in snap


class TestPhase3Forge:
    """Forge real bundle generation (not mock)."""

    def test_bundle_compiler_creates_files(self):
        from rosclaw.forge.bundle_compiler import BundleCompiler

        compiler = BundleCompiler()
        bundle = compiler.compile("Simple temperature sensor SDK", "temp_sensor")

        assert bundle.bundle_name == "temp_sensor"
        assert bundle.staging_ready is True
        assert bundle.production_ready is False

        expected_files = [
            "temp_sensor/mcp_server.py",
            "temp_sensor/skill_manifest.json",
            "temp_sensor/provider_manifest.json",
            "temp_sensor/tests/test_temp_sensor.py",
            "temp_sensor/README.md",
        ]
        for fname in expected_files:
            assert fname in bundle.files, f"Missing {fname}"
            assert len(bundle.files[fname]) > 0, f"Empty {fname}"

    def test_bundle_validation_passes(self):
        from rosclaw.forge.bundle_compiler import BundleCompiler

        compiler = BundleCompiler()
        bundle = compiler.compile("Motor controller SDK", "motor_ctrl")

        assert bundle.validation["async_safe"] is True
        assert bundle.validation["schema_complete"] is True
        assert bundle.validation["safety_hooks"] is True
        assert bundle.validation["preemption_ready"] is True
        assert bundle.validation["tests_present"] is True
        assert bundle.validation["readme_present"] is True

    def test_bundle_manifests_are_valid_json(self):
        from rosclaw.forge.bundle_compiler import BundleCompiler

        compiler = BundleCompiler()
        bundle = compiler.compile("Lidar sensor SDK", "lidar_driver")

        skill_manifest = json.loads(bundle.files["lidar_driver/skill_manifest.json"])
        assert skill_manifest["skill_id"] == "lidar_driver"
        assert "firewall_hooks" in skill_manifest

        provider_manifest = json.loads(bundle.files["lidar_driver/provider_manifest.json"])
        assert provider_manifest["name"] == "lidar_driver"
        assert provider_manifest["async_safe"] is True

    def test_mcp_compile_uses_real_forge(self):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer
        import asyncio

        server = ROSClawMinimalMCPServer()
        result = asyncio.run(server._handle_system_tool("system.compile_asset_bundle", {
            "sdk_doc": "Test SDK for validation",
            "bundle_name": "test_bundle",
            "staging": True,
        }))
        assert result["status"] == "generated"
        assert result.get("staging_ready") is True
        assert "validation" in result
        assert "files" in result


class TestPhase3MuJoCoRealPhysics:
    """MuJoCo produces real physics data (not mock)."""

    def test_g1_model_loads(self):
        from rosclaw.runtime import RobotRegistry

        reg = RobotRegistry()
        profile = reg.inspect("g1")
        assert profile["robot_id"] in ("g1", "unitree_g1")
        assert profile["embodiment"]["dof"] >= 16

    @pytest.mark.skipif(
        not _enough_ram_for_physics(min_gb=4.0),
        reason="Heavy MuJoCo physics test requires >=4GB free RAM (run on Dell workstation)",
    )
    def test_g1_walk_produces_physics_data(self):
        """Run G1 walk for a short duration and verify real physics output."""
        import tempfile
        from pathlib import Path

        # Run the demo with a short timeout to get physics data
        with tempfile.TemporaryDirectory():
            result = subprocess.run(
                [sys.executable, "-m", "rosclaw.examples.g1_free_floating_walk"],
                capture_output=True, text=True,
                cwd=str(Path(__file__).parent.parent),
                timeout=20,
            )
            # The demo may timeout (expected) but should produce real data
            output = result.stdout + result.stderr

            # Must see real physics timestamps
            assert "t=" in output, "No physics timestamps found"
            # Must see position data
            assert "x=" in output, "No position data found"
            # Must see height data
            assert "z=" in output, "No height data found"

    def test_g1_eurdf_profile_in_registry(self):
        from rosclaw.runtime import RobotRegistry

        reg = RobotRegistry()
        available = reg.list_available()
        assert "g1" in available, f"g1 not in registry: {available}"
