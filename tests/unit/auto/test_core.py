"""Unit tests for rosclaw-auto core data models."""
from rosclaw.auto.core import (
    AutoTask,
    Champion,
    ChampionCard,
    DeadEnd,
    Diagnosis,
    ExperimentSpec,
    FailureCase,
    Patch,
    Proposal,
)


def test_auto_task_roundtrip():
    t = AutoTask(id="t1", name="pick_cube", task_type="skill_tuning",
                 robot_id="panda", environment_id="maniskill", target_skill_id="pick_v1")
    d = t.to_dict()
    t2 = AutoTask.from_dict(d)
    assert t2.name == "pick_cube"
    assert t2.robot_id == "panda"


def test_failure_case():
    f = FailureCase(id="f1", praxis_event_id="evt1", task_id="t1", skill_id="pick_v1",
                    failure_mode="missed_grasp", severity="high")
    assert f.failure_mode == "missed_grasp"
    assert f.to_dict()["severity"] == "high"


def test_diagnosis():
    d = Diagnosis(id="d1", failure_id="f1", task="pick_cube", skill="pick_v1",
                  root_cause_candidates=["low_height"])
    assert d.auto_repairable is True


def test_proposal():
    p = Proposal(id="p1", task="pick_cube", target_skill_id="pick_v1",
                 hypothesis_statement="increase height", search_space={"z": [0.02, 0.08]})
    assert p.patch_type == "skill_parameter_patch"
    assert p.risk_level == "low"


def test_patch():
    patch = Patch(id="patch1", proposal_id="p1", target_skill="pick_v1",
                  changes=[{"path": "/z", "old": 0.02, "new": 0.05}])
    assert patch.patch_level == 2


def test_experiment():
    e = ExperimentSpec(id="e1", proposal_id="p1", patch_id="patch1",
                       baseline_skill_id="pick_v1", candidate_skill_id="pick_v1_c")
    assert e.evaluation["metrics"] == ["success_rate", "collision_rate", "completion_time"]


def test_evaluation_decision_promote():
    from rosclaw.auto.engine import AutoEngine
    engine = AutoEngine()
    ev = engine.create_evaluation("e1", {"success_rate": 0.4}, {"success_rate": 0.6})
    assert ev.decision.startswith("promote")


def test_evaluation_decision_reject():
    from rosclaw.auto.engine import AutoEngine
    engine = AutoEngine()
    ev = engine.create_evaluation("e1", {"success_rate": 0.6}, {"success_rate": 0.3})
    assert ev.decision == "reject"


def test_champion():
    c = Champion(id="c1", skill_id="pick_v2", task_id="t1", level="sim",
                 metrics={"success_rate": 0.76})
    assert c.level == "sim"


def test_deadend():
    d = DeadEnd(id="de1", task_id="t1", direction="scale > 0.8",
                rejection_reason="collision spikes")
    assert d.direction == "scale > 0.8"


def test_champion_card_markdown():
    card = ChampionCard(skill_id="pick_v2", previous_champion="pick_v1",
                        improvement={"success_rate": "0.58 → 0.76"})
    md = card.to_markdown()
    assert "pick_v2" in md
    assert "0.58 → 0.76" in md
