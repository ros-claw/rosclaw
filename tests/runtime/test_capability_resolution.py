"""Tests for generic body capability resolution (Milestone 5)."""

from __future__ import annotations

from pathlib import Path

from rosclaw.body.compatibility import SkillCompatibilityChecker
from rosclaw.body.compiler import EffectiveBodyCompiler
from rosclaw.body.schema import BodyYaml, CalibrationYaml, EffectiveBody, EurdfProfile, SkillManifest
from rosclaw.runtime import RobotRegistry


def _realsense_effective_body(profile_id: str = "realsense_d405") -> EffectiveBody:
    registry = RobotRegistry()
    profile = registry.get(profile_id)
    assert profile is not None, f"{profile_id} profile not installed"
    eurdf = EurdfProfile.from_robot_complete_profile(profile)
    body = BodyYaml(
        body_instance={"id": f"{profile_id}_lab_01", "robot_model": profile_id},
        model_ref={"profile_id": profile_id},
    )
    return EffectiveBodyCompiler().compile(eurdf, body, CalibrationYaml(), [])


def _manifest_with_capabilities(*capabilities: str) -> SkillManifest:
    return SkillManifest(
        skill_id="test_skill",
        skill_version="1.0.0",
        requires={"capabilities": {"all_of": list(capabilities)}},
    )


def test_generic_rgb_camera_capability_enabled_for_d405() -> None:
    effective = _realsense_effective_body("realsense_d405")
    assert "rgb_camera" in effective.capabilities.get("enabled", [])
    assert "depth_camera" in effective.capabilities.get("enabled", [])


def test_realsense_capture_rgbd_compatible_with_d405() -> None:
    effective = _realsense_effective_body("realsense_d405")
    manifest = SkillManifest.from_yaml(
        Path(__file__).parent.parent.parent
        / "src"
        / "rosclaw"
        / "skill"
        / "builtins"
        / "realsense_capture_rgbd"
        / "skill.yaml"
    )
    result = SkillCompatibilityChecker().check_one(manifest, effective)
    assert result.status in ("compatible", "degraded"), result.reason


def test_scene_risk_scan_compatible_with_d405() -> None:
    effective = _realsense_effective_body("realsense_d405")
    manifest = SkillManifest.from_yaml(
        Path(__file__).parent.parent.parent
        / "src"
        / "rosclaw"
        / "skill"
        / "builtins"
        / "scene_risk_scan"
        / "skill.yaml"
    )
    result = SkillCompatibilityChecker().check_one(manifest, effective)
    assert result.status in ("compatible", "degraded"), result.reason


def test_missing_generic_capability_blocks_skill() -> None:
    effective = _realsense_effective_body("realsense_d405")
    manifest = _manifest_with_capabilities("rgb_camera", "depth_camera", "arm")
    result = SkillCompatibilityChecker().check_one(manifest, effective)
    assert result.status == "blocked"
    assert "capability:arm" in result.missing_requirements


def test_d435i_has_imu_capability() -> None:
    effective = _realsense_effective_body("realsense_d435i")
    assert "rgb_camera" in effective.capabilities.get("enabled", [])
    assert "depth_camera" in effective.capabilities.get("enabled", [])
    assert "imu" in effective.capabilities.get("enabled", [])


def test_capability_degraded_when_body_reports_blocked() -> None:
    effective = _realsense_effective_body("realsense_d405")
    effective.capabilities["blocked"].append("depth_camera")
    effective.capabilities["enabled"] = [
        c for c in effective.capabilities.get("enabled", []) if c != "depth_camera"
    ]
    manifest = _manifest_with_capabilities("rgb_camera", "depth_camera")
    result = SkillCompatibilityChecker().check_one(manifest, effective)
    assert result.status == "blocked"
    assert "capability:depth_camera" in result.missing_requirements
