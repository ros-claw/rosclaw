"""Tests that builtin RealSense skill manifests are discoverable and valid."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.compatibility import SkillCompatibilityChecker
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import SkillManifest
from rosclaw.runtime.eurdf_loader import RobotCompleteProfile
from rosclaw.skill.builtins import get_builtin_skill, load_builtins
from rosclaw.skill_manager.registry import SkillRegistry


class TestRealsenseSkillManifest:
    """Cover the RealSense builtin skill manifests and compatibility."""

    def test_manifest_loads_directly(self):
        manifest_path = (
            Path(__file__).parent.parent
            / "src"
            / "rosclaw"
            / "skill"
            / "builtins"
            / "realsense_capture_rgbd"
            / "skill.yaml"
        )
        manifest = SkillManifest.from_yaml(manifest_path)
        assert manifest.skill_id == "realsense_capture_rgbd"
        assert "aligned_rgbd" in manifest.requires.get("capabilities", {}).get("all_of", [])
        assert "perception_only_camera" in manifest.requires.get("robot_class", [])

    def test_resolver_discovers_builtin_manifests(self, tmp_path):
        resolver = BodyResolver(workspace=tmp_path)
        manifests = resolver._discover_skill_manifests()
        ids = {m.skill_id for m in manifests}
        assert "realsense_capture_rgbd" in ids
        assert "scene_risk_scan" in ids

    def test_realsense_capture_rgbd_compatible_with_d405_profile(self):
        from rosclaw.runtime import RobotRegistry

        registry = RobotRegistry()
        profile = registry.get("realsense_d405")
        if profile is None:
            pytest.skip("realsense_d405 profile not installed")
        assert isinstance(profile, RobotCompleteProfile)

        from rosclaw.body.compiler import EffectiveBodyCompiler
        from rosclaw.body.schema import BodyYaml, CalibrationYaml, EurdfProfile

        eurdf = EurdfProfile.from_robot_complete_profile(profile)
        body = BodyYaml(
            body_instance={"id": "d405_lab_01", "robot_model": "realsense_d405"},
            model_ref={"profile_id": "realsense_d405"},
            metadata={"perception_only": True, "no_actuation": True},
        )
        effective = EffectiveBodyCompiler().compile(eurdf, body, CalibrationYaml(), [])

        manifest_path = (
            Path(__file__).parent.parent
            / "src"
            / "rosclaw"
            / "skill"
            / "builtins"
            / "realsense_capture_rgbd"
            / "skill.yaml"
        )
        manifest = SkillManifest.from_yaml(manifest_path)
        result = SkillCompatibilityChecker().check_one(manifest, effective)
        assert result.status in ("compatible", "degraded"), result.reason

    def test_builtin_registry_loads_with_handlers(self):
        registry = SkillRegistry()
        loaded = load_builtins(registry)[1]
        names = {e.name for e in loaded}
        assert "realsense_capture_rgbd" in names
        assert registry.get("realsense_capture_rgbd") is not None
        assert registry.get("realsense_capture_rgbd").handler is not None

    def test_get_builtin_skill_metadata(self):
        entry = get_builtin_skill("realsense_capture_rgbd")
        assert entry is not None
        assert entry.metadata.get("builtin") is True
        assert "requires" in entry.metadata
