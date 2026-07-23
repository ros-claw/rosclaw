"""Tests for eval and promote gates."""

import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from rosclaw.skill.cli import _copy_template, _init_context
from rosclaw.skill.eval import evaluate_skill
from rosclaw.skill.mining import mine_skill_candidate
from rosclaw.skill.models import SkillPackage
from rosclaw.skill.promote import promote_candidate
from tests.skill.evidence_helpers import write_promotion_evidence


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


TEMPLATE_DIR = (
    Path(__file__).parent.parent.parent / "src" / "rosclaw" / "skill" / "templates" / "default"
)
FIXTURES = Path(__file__).parent / "fixtures" / "practice_sessions"


def _mine_and_eval(tmp_path: Path, *, with_physics_evidence: bool = False):
    dest = tmp_path / "g1_kick_ball"
    context = _init_context("g1_kick_ball", "ur5e", "manipulation", "ros-claw")
    _copy_template(TEMPLATE_DIR, dest, context)
    pkg = SkillPackage(dest).try_load()
    mine_skill_candidate(pkg, FIXTURES, candidate_id="candidate_0001")
    pkg = SkillPackage(dest).try_load()
    if with_physics_evidence:
        write_promotion_evidence(pkg, "candidate_0001")
        pkg = SkillPackage(dest).try_load()
    else:
        from rosclaw.skill.eval import refresh_hashes

        refresh_hashes(pkg)
    report = evaluate_skill(pkg, candidate_id="candidate_0001", mode="replay")
    return pkg, report


def test_eval_needs_physics_evidence_after_mining(tmp_path: Path):
    _, report = _mine_and_eval(tmp_path)
    assert report.decision == "need_more_evidence"
    assert report.checks.get("sandbox_eval") is False
    assert report.checks.get("promotion_gate_check") is False


def test_promote_after_eval_pass(tmp_path: Path):
    pkg, report = _mine_and_eval(tmp_path, with_physics_evidence=True)
    assert report.decision == "pass"
    candidate_params = (pkg.root / "policies/params/candidate_0001.yaml").read_bytes()
    candidate_tree = (pkg.root / "behavior_tree.candidate_0001.xml").read_bytes()
    result = promote_candidate(pkg, "candidate_0001", "0.1.0")
    assert result["version"] == "0.1.0"
    assert result["stage"] == "validated"
    assert pkg.skill.metadata.stage == "validated"
    assert any(v.version == "0.1.0" for v in pkg.lineage.versions)
    assert (pkg.root / "policies/params/default.yaml").read_bytes() == candidate_params
    assert (pkg.root / "behavior_tree.xml").read_bytes() == candidate_tree


def test_eval_rejects_evidence_after_candidate_changes(tmp_path: Path):
    pkg, report = _mine_and_eval(tmp_path, with_physics_evidence=True)
    assert report.decision == "pass"
    candidate = pkg.root / "policies/params/candidate_0001.yaml"
    candidate.write_text(
        candidate.read_text(encoding="utf-8") + "\nchanged: true\n", encoding="utf-8"
    )

    report = evaluate_skill(pkg, candidate_id="candidate_0001", mode="replay")
    assert report.decision != "pass"
    assert report.checks["sandbox_eval"] is False


def test_promote_blocked_without_eval(tmp_path: Path):
    dest = tmp_path / "g1_kick_ball"
    context = _init_context("g1_kick_ball", "ur5e", "manipulation", "ros-claw")
    _copy_template(TEMPLATE_DIR, dest, context)
    pkg = SkillPackage(dest).try_load()
    pkg.lineage.candidates.append(
        type("C", (), {"id": "candidate_0001", "status": "candidate", "eval_report": None})()
    )
    with pytest.raises(ValueError, match="No eval report"):
        promote_candidate(pkg, "candidate_0001", "0.1.0")


def test_forged_persisted_pass_cannot_bypass_fresh_evaluation(tmp_path: Path):
    pkg, report = _mine_and_eval(tmp_path)
    assert report.decision == "need_more_evidence"
    report_path = pkg.root / "evidence" / "reports" / "candidate_0001_eval.json"
    payload = report.to_dict()
    payload["decision"] = "pass"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Fresh eval did not pass"):
        promote_candidate(pkg, "candidate_0001", "0.1.0")


def test_simulation_evidence_cannot_claim_official_stage(tmp_path: Path):
    pkg, report = _mine_and_eval(tmp_path, with_physics_evidence=True)
    assert report.decision == "pass"
    with pytest.raises(ValueError, match="Simulation evidence may only"):
        promote_candidate(pkg, "candidate_0001", "0.1.0", stage="official_verified")
