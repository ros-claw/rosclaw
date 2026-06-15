"""L6: Diagnosis / Proposal / Patch tests."""
from rosclaw.auto.core.failure import FailureCase
from rosclaw.auto.diagnosis.rule_diagnoser import RuleDiagnoser


class TestRuleDiagnoser:
    """AUTO-DIAG-001/002: Rule-based diagnosis for known failure modes."""

    def test_missed_grasp_diagnosis(self):
        """AUTO-DIAG-001: missed_grasp diagnosis correct."""
        diagnoser = RuleDiagnoser()
        fc = FailureCase(
            id="f1", praxis_event_id="e1", task_id="pick_cube",
            skill_id="pick_v1", failure_mode="missed_grasp",
        )
        diag = diagnoser.diagnose(fc)
        assert "low_pregrasp_height" in diag.root_cause_candidates
        assert "pregrasp_height" in diag.recommended_search_space
        assert diag.auto_repairable is True
        assert diag.confidence > 0.5

    def test_collision_diagnosis(self):
        """AUTO-DIAG-002: collision diagnosis correct."""
        diagnoser = RuleDiagnoser()
        fc = FailureCase(
            id="f2", praxis_event_id="e2", task_id="pick_cube",
            skill_id="pick_v1", failure_mode="collision",
        )
        diag = diagnoser.diagnose(fc)
        assert "approach_speed_too_fast" in diag.root_cause_candidates
        assert diag.auto_repairable is True
        assert diag.risk_level == "medium"

    def test_perception_lost_not_auto_repairable(self):
        """AUTO-DIAG-003: perception_lost is high risk, not auto-repairable."""
        diagnoser = RuleDiagnoser()
        fc = FailureCase(
            id="f3", praxis_event_id="e3", task_id="pick_cube",
            skill_id="pick_v1", failure_mode="perception_lost",
        )
        diag = diagnoser.diagnose(fc)
        assert diag.auto_repairable is False
        assert diag.risk_level == "high"

    def test_unknown_failure_mode_fallback(self):
        """AUTO-DIAG-004: Unknown failure mode falls back gracefully."""
        diagnoser = RuleDiagnoser()
        fc = FailureCase(
            id="f4", praxis_event_id="e4", task_id="pick_cube",
            skill_id="pick_v1", failure_mode="alien_abduction",
        )
        diag = diagnoser.diagnose(fc)
        assert diag.root_cause_candidates == ["unknown"]
        assert diag.auto_repairable is False

    def test_can_diagnose_known_modes(self):
        diagnoser = RuleDiagnoser()
        assert diagnoser.can_diagnose("missed_grasp") is True
        assert diagnoser.can_diagnose("collision") is True
        assert diagnoser.can_diagnose("unknown") is False
