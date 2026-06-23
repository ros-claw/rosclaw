"""Tests for schema validation."""

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from rosclaw.skill.cli import _copy_template, _init_context
from rosclaw.skill.models import SkillPackage
from rosclaw.skill.validators import validate_package


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


TEMPLATE_DIR = Path(__file__).parent.parent.parent / "src" / "rosclaw" / "skill" / "templates" / "default"


def _make_valid(tmp_path: Path) -> SkillPackage:
    dest = tmp_path / "g1_kick_ball"
    context = _init_context("g1_kick_ball", "unitree_g1", "manipulation", "ros-claw")
    _copy_template(TEMPLATE_DIR, dest, context)
    return SkillPackage(dest).try_load()


def test_valid_package_passes(tmp_path: Path):
    pkg = _make_valid(tmp_path)
    report = validate_package(pkg)
    # Package integrity will fail because no hashes.json yet; other checks should pass.
    assert report.checks.get("skill_schema") is True
    assert report.checks.get("behavior_tree_lint") is True
    assert report.checks.get("providers_schema") is True
    assert report.checks.get("eurdf_compat_schema") is True
    assert report.checks.get("safety_schema") is True


def test_invalid_skill_name_fails(tmp_path: Path):
    pkg = _make_valid(tmp_path)
    pkg.skill.metadata.name = "G1 Kick Ball"  # invalid uppercase and spaces
    pkg.write_skill_yaml()
    pkg = SkillPackage(tmp_path / "g1_kick_ball").try_load()
    report = validate_package(pkg)
    assert not report.ok
    assert any("skill.yaml" in e for e in report.errors)


def test_missing_safety_fails(tmp_path: Path):
    pkg = _make_valid(tmp_path)
    (pkg.root / "safety.yaml").unlink()
    report = validate_package(pkg)
    assert not report.ok
    assert any("safety" in e.lower() for e in report.errors)


def test_disable_sandbox_fails(tmp_path: Path):
    pkg = _make_valid(tmp_path)
    text = (pkg.root / "safety.yaml").read_text(encoding="utf-8")
    (pkg.root / "safety.yaml").write_text(text + "\ndisable_sandbox: true\n", encoding="utf-8")
    report = validate_package(pkg)
    assert not report.ok
    assert any("disable_sandbox" in e for e in report.errors)
