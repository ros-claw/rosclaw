"""Tests for skill compatibility checking."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rosclaw.body.compatibility import SkillCompatibilityChecker
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import SkillManifest
from rosclaw.cli import main as rosclaw_main

FIXTURES = Path(__file__).parent / "fixtures" / "skills"


@pytest.fixture
def linked_body(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # Copy skill manifests into workspace skills dir
    skills_dir = tmp_path / ".rosclaw" / "skills"
    skills_dir.mkdir(parents=True)
    for path in FIXTURES.glob("*.skill.yaml"):
        (skills_dir / path.name).write_text(path.read_text(), encoding="utf-8")

    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0
    yield tmp_path


def test_walk_forward_compatible(linked_body):
    resolver = BodyResolver()
    effective = resolver.get_effective_body()
    manifest = SkillManifest.from_yaml(FIXTURES / "walk_forward.skill.yaml")
    result = SkillCompatibilityChecker().check_one(manifest, effective)
    assert result.status == "compatible"


def test_camera_nav_blocked_when_camera_unavailable(linked_body):
    resolver = BodyResolver()
    with patch.object(sys, "argv", [
        "rosclaw", "body", "update-state",
        "--set", "installed_components.sensors.head_camera.status=unavailable",
        "--reason", "test",
    ]):
        assert rosclaw_main() == 0

    effective = resolver.get_effective_body()
    manifest = SkillManifest.from_yaml(FIXTURES / "camera_nav.skill.yaml")
    result = SkillCompatibilityChecker().check_one(manifest, effective)
    assert result.status == "blocked"
    assert any("head_camera" in req for req in result.missing_requirements)


def test_dual_arm_lift_blocked_when_right_arm_unavailable(linked_body):
    resolver = BodyResolver()
    with patch.object(sys, "argv", [
        "rosclaw", "body", "update-state",
        "--set", "installed_components.actuators.right_arm.status=unavailable",
        "--reason", "test",
    ]):
        assert rosclaw_main() == 0

    effective = resolver.get_effective_body()
    manifest = SkillManifest.from_yaml(FIXTURES / "dual_arm_lift.skill.yaml")
    result = SkillCompatibilityChecker().check_one(manifest, effective)
    assert result.status == "blocked"
