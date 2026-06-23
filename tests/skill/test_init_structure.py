"""Tests for rosclaw skill init structure."""

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from rosclaw.skill.cli import _copy_template, _init_context
from rosclaw.skill.models import SkillPackage
from rosclaw.skill.validators import validate_file_existence


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


TEMPLATE_DIR = Path(__file__).parent.parent.parent / "src" / "rosclaw" / "skill" / "templates" / "default"


def test_init_generates_required_files(tmp_path: Path):
    dest = tmp_path / "g1_kick_ball"
    context = _init_context("g1_kick_ball", "unitree_g1", "manipulation", "ros-claw")
    _copy_template(TEMPLATE_DIR, dest, context)

    required = [
        "skill.yaml",
        "README.md",
        "SKILL.md",
        "behavior_tree.xml",
        "prompts/planner.md",
        "policies/policy.yaml",
        "policies/params/default.yaml",
        "providers.yaml",
        "e-urdf-compat.yaml",
        "safety.yaml",
        "dojo.yaml",
        "darwin_eval.yaml",
        "tests/test_schema.py",
        "evidence/practice/.gitkeep",
        "lineage.yaml",
        "CHANGELOG.md",
        ".rosclaw/lock.yaml",
    ]
    for rel in required:
        assert (dest / rel).exists(), rel


def test_skill_yaml_loads_after_init(tmp_path: Path):
    dest = tmp_path / "g1_kick_ball"
    context = _init_context("g1_kick_ball", "unitree_g1", "manipulation", "ros-claw")
    _copy_template(TEMPLATE_DIR, dest, context)
    pkg = SkillPackage(dest).try_load()
    assert pkg.skill is not None
    assert pkg.skill.metadata.name == "g1_kick_ball"


def test_required_files_validation(tmp_path: Path):
    dest = tmp_path / "g1_kick_ball"
    context = _init_context("g1_kick_ball", "unitree_g1", "manipulation", "ros-claw")
    _copy_template(TEMPLATE_DIR, dest, context)
    report = validate_file_existence(dest)
    assert report.ok
