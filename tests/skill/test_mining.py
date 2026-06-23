"""Tests for practice mining."""

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from rosclaw.skill.cli import _copy_template, _init_context
from rosclaw.skill.mining import load_episodes, mine_skill_candidate
from rosclaw.skill.models import SkillPackage


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


TEMPLATE_DIR = Path(__file__).parent.parent.parent / "src" / "rosclaw" / "skill" / "templates" / "default"
FIXTURES = Path(__file__).parent / "fixtures" / "practice_sessions"


def test_load_episodes_filters_by_task():
    episodes = load_episodes(FIXTURES, type("Q", (), {"task": "g1_kick_ball", "robot": None, "min_episodes": 0, "include_failures": True})())
    assert len(episodes) == 3
    assert all(e.task == "g1_kick_ball" for e in episodes)


def test_mine_generates_candidate(tmp_path: Path):
    dest = tmp_path / "g1_kick_ball"
    context = _init_context("g1_kick_ball", "unitree_g1", "manipulation", "ros-claw")
    _copy_template(TEMPLATE_DIR, dest, context)
    pkg = SkillPackage(dest).try_load()
    report = mine_skill_candidate(pkg, FIXTURES, candidate_id="candidate_0001")
    assert report.candidate_id == "candidate_0001"
    assert (dest / "policies" / "params" / "candidate_0001.yaml").exists()
    assert (dest / "behavior_tree.candidate_0001.xml").exists()
    assert (dest / "evidence" / "reports" / "candidate_0001_mining.json").exists()
    assert pkg.skill.metadata.candidate_id == "candidate_0001"
    assert any(c.id == "candidate_0001" for c in pkg.lineage.candidates)
