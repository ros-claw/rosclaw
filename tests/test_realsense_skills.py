"""Tests for RealSense skill packages."""
from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.skill.models import SkillPackage
from rosclaw.skill.registry import SkillLocalRegistry
from rosclaw.skill.validators import validate_package


SKILLS_ROOT = Path(__file__).parent.parent / "skills"

SKILL_NAMES = [
    "realsense_capture_frame",
    "realsense_capture_rgbd",
    "realsense_depth_health_check",
    "realsense_imu_check",
    "scene_risk_scan",
    "obstacle_visibility_check",
    "realsense_camera_info_check",
    "realsense_pointcloud_check",
]


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_skill_package_validates(name: str) -> None:
    pkg = SkillPackage(SKILLS_ROOT / name).try_load()
    report = validate_package(pkg)
    assert report.ok, f"{name} validation failed: {report.errors}"


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_skill_package_installs(name: str, tmp_path: Path) -> None:
    pkg = SkillPackage(SKILLS_ROOT / name).try_load()
    registry = SkillLocalRegistry(home=tmp_path)
    registry.add(pkg)
    assert pkg.skill_id in registry._data["skills"]


def test_all_skills_are_perception_only() -> None:
    for name in SKILL_NAMES:
        pkg = SkillPackage(SKILLS_ROOT / name).try_load()
        assert pkg.skill is not None
        assert "perception-only" in pkg.skill.metadata.tags
        assert pkg.skill.status.recommended_runtime_mode == "sandbox_first"
