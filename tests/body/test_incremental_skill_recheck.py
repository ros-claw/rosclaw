"""Tests for impact-aware incremental skill recheck."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rosclaw.body.compatibility import SkillCompatibilityChecker
from rosclaw.body.resolver import BodyResolver
from rosclaw.cli import main as rosclaw_main

FIXTURES = Path(__file__).parent / "fixtures" / "skills"


@pytest.fixture
def linked_body_with_skills(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    skills_dir = tmp_path / ".rosclaw" / "skills"
    skills_dir.mkdir(parents=True)
    for path in FIXTURES.glob("*.skill.yaml"):
        (skills_dir / path.name).write_text(path.read_text(), encoding="utf-8")

    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0
    yield tmp_path


def _tracking_check(calls):
    # Stash the real implementation so the patched method can delegate without recursion.
    original_impl = SkillCompatibilityChecker.check_one
    real_checker = SkillCompatibilityChecker()

    def wrapper(skill, body):
        calls.append((skill.skill_id, body.effective_body_hash))
        return original_impl(real_checker, skill, body)

    return wrapper


def test_diff_reports_affected_ids_for_sensor_status_change(linked_body_with_skills):
    from rosclaw.body.diff import BodyDiffer

    resolver = BodyResolver()
    old = resolver.get_effective_body()

    with patch.object(sys, "argv", [
        "rosclaw", "body", "update-state",
        "--set", "installed_components.sensors.head_camera.status=unavailable",
        "--reason", "test",
    ]):
        assert rosclaw_main() == 0

    new = resolver.get_effective_body()
    diff = BodyDiffer().diff_effective_bodies(old, new)

    assert "sensor_status" in diff.affected_categories
    assert "head_camera" in diff.affected_ids


def test_incremental_recheck_only_rechecks_affected_skills(linked_body_with_skills):
    resolver = BodyResolver()
    calls = []

    with patch.object(
        SkillCompatibilityChecker,
        "check_one",
        side_effect=_tracking_check(calls),
    ), patch.object(sys, "argv", [
        "rosclaw", "body", "update-state",
        "--set", "installed_components.sensors.head_camera.status=unavailable",
        "--reason", "test",
    ]):
        assert rosclaw_main() == 0

    # Only camera_nav depends on head_camera / visual_navigation.
    # walk_forward and dual_arm_lift should be copied from the previous report.
    assert len(calls) == 1
    assert calls[0][0] == "camera_nav"

    report = resolver.get_skill_compatibility()
    assert report.skills["camera_nav@1.0.0"].status == "blocked"
    missing = report.skills["camera_nav@1.0.0"].missing_requirements
    assert any("visual_navigation" in m or "head_camera" in m for m in missing)
    assert report.skills["walk_forward@1.0.0"].status == "compatible"


def test_incremental_recheck_preserves_unaffected_results(linked_body_with_skills):
    resolver = BodyResolver()
    old_report = resolver.get_skill_compatibility()
    calls = []

    with patch.object(
        SkillCompatibilityChecker,
        "check_one",
        side_effect=_tracking_check(calls),
    ), patch.object(sys, "argv", [
        "rosclaw", "body", "update-state",
        "--set", "installed_components.actuators.right_arm.status=unavailable",
        "--reason", "test",
    ]):
        assert rosclaw_main() == 0

    assert len(calls) == 1
    assert calls[0][0] == "dual_arm_lift"

    report = resolver.get_skill_compatibility()
    assert report.skills["dual_arm_lift@1.0.0"].status == "blocked"
    # walk_forward does not depend on the right arm and keeps its previous compatible result.
    assert report.skills["walk_forward@1.0.0"].status == "compatible"
    # camera_nav depends on visual_navigation which is already degraded; the status is copied.
    assert report.skills["camera_nav@1.0.0"].status == old_report.skills["camera_nav@1.0.0"].status
    # The copied result is still stamped with the new effective body hash.
    assert report.skills["walk_forward@1.0.0"].checked_against["body_hash"] == report.effective_body_hash


def test_note_incremental_recheck_uses_affects_list(linked_body_with_skills):
    calls = []

    with patch.object(
        SkillCompatibilityChecker,
        "check_one",
        side_effect=_tracking_check(calls),
    ), patch.object(sys, "argv", [
        "rosclaw", "body", "note",
        "Camera lens replaced.",
        "--type", "maintenance",
        "--affects", "head_camera",
    ]):
        assert rosclaw_main() == 0

    # Maintenance note does not change body.yaml, but the recheck path is still triggered
    # for skills whose requirements overlap the --affects list.
    assert len(calls) == 1
    assert calls[0][0] == "camera_nav"
