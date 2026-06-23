"""Tests for eval and promote gates."""

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from rosclaw.skill.cli import _copy_template, _init_context
from rosclaw.skill.eval import evaluate_skill
from rosclaw.skill.mining import mine_skill_candidate
from rosclaw.skill.models import SkillPackage
from rosclaw.skill.promote import promote_candidate


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


TEMPLATE_DIR = Path(__file__).parent.parent.parent / "src" / "rosclaw" / "skill" / "templates" / "default"
FIXTURES = Path(__file__).parent / "fixtures" / "practice_sessions"


def _mine_and_eval(tmp_path: Path):
    dest = tmp_path / "g1_kick_ball"
    context = _init_context("g1_kick_ball", "unitree_g1", "manipulation", "ros-claw")
    _copy_template(TEMPLATE_DIR, dest, context)
    pkg = SkillPackage(dest).try_load()
    mine_skill_candidate(pkg, FIXTURES, candidate_id="candidate_0001")
    pkg = SkillPackage(dest).try_load()
    report = evaluate_skill(pkg, candidate_id="candidate_0001", mode="replay")
    return pkg, report


def test_eval_passes_after_mining(tmp_path: Path):
    _, report = _mine_and_eval(tmp_path)
    assert report.decision == "pass"
    assert report.checks.get("promotion_gate_check") is True


def test_promote_after_eval_pass(tmp_path: Path):
    pkg, report = _mine_and_eval(tmp_path)
    assert report.decision == "pass"
    result = promote_candidate(pkg, "candidate_0001", "0.1.0")
    assert result["version"] == "0.1.0"
    assert result["stage"] == "validated"
    assert pkg.skill.metadata.stage == "validated"
    assert any(v.version == "0.1.0" for v in pkg.lineage.versions)


def test_promote_blocked_without_eval(tmp_path: Path):
    dest = tmp_path / "g1_kick_ball"
    context = _init_context("g1_kick_ball", "unitree_g1", "manipulation", "ros-claw")
    _copy_template(TEMPLATE_DIR, dest, context)
    pkg = SkillPackage(dest).try_load()
    pkg.lineage.candidates.append(type("C", (), {"id": "candidate_0001", "status": "candidate", "eval_report": None})())
    with pytest.raises(ValueError, match="No eval report"):
        promote_candidate(pkg, "candidate_0001", "0.1.0")
