"""Tests for rollback."""

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from rosclaw.skill.cli import _copy_template, _init_context
from rosclaw.skill.eval import evaluate_skill
from rosclaw.skill.mining import mine_skill_candidate
from rosclaw.skill.models import SkillPackage
from rosclaw.skill.promote import promote_candidate
from rosclaw.skill.rollback import rollback_skill


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


TEMPLATE_DIR = Path(__file__).parent.parent.parent / "src" / "rosclaw" / "skill" / "templates" / "default"
FIXTURES = Path(__file__).parent / "fixtures" / "practice_sessions"


def _validated_pkg(tmp_path: Path):
    dest = tmp_path / "g1_kick_ball"
    context = _init_context("g1_kick_ball", "unitree_g1", "manipulation", "ros-claw")
    _copy_template(TEMPLATE_DIR, dest, context)
    pkg = SkillPackage(dest).try_load()
    mine_skill_candidate(pkg, FIXTURES, candidate_id="candidate_0001")
    pkg = SkillPackage(dest).try_load()
    evaluate_skill(pkg, candidate_id="candidate_0001", mode="replay")
    pkg = SkillPackage(dest).try_load()
    promote_candidate(pkg, "candidate_0001", "0.1.0")
    return SkillPackage(dest).try_load()


def test_rollback_restores_validated_version(tmp_path: Path):
    pkg = _validated_pkg(tmp_path)
    original_version = pkg.skill.metadata.version
    # Promote to 0.2.0 to create a newer version, then rollback.
    mine_skill_candidate(pkg, FIXTURES, candidate_id="candidate_0002")
    pkg = SkillPackage(tmp_path / "g1_kick_ball").try_load()
    evaluate_skill(pkg, candidate_id="candidate_0002", mode="replay")
    pkg = SkillPackage(tmp_path / "g1_kick_ball").try_load()
    promote_candidate(pkg, "candidate_0002", "0.2.0")

    pkg = SkillPackage(tmp_path / "g1_kick_ball").try_load()
    assert pkg.skill.metadata.version == "0.2.0"

    result = rollback_skill(pkg, "0.1.0", reason="test rollback")
    assert result["to_version"] == "0.1.0"
    pkg = SkillPackage(tmp_path / "g1_kick_ball").try_load()
    assert pkg.skill.metadata.version == original_version
    assert any(r.to_version == "0.1.0" for r in pkg.lineage.rollbacks)
