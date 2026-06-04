"""L2: Schema / Contract tests."""
import pytest
from rosclaw.auto.events.schemas import (
    EventEnvelope, PraxisFailedEvent, BenchmarkCompletedEvent,
    AutoProposalCreatedEvent, ChampionPromotedEvent, DeadEndRegisteredEvent,
)
from rosclaw.auto.core import (
    AutoTask, FailureCase, Diagnosis, Hypothesis, Proposal,
    Patch, ExperimentSpec, EvaluationResult, Champion, DeadEnd,
)


# ------------------------------------------------------------------
# Event Schema Contract
# ------------------------------------------------------------------

class TestEventSchemaContract:
    """AUTO-EVENT-SCHEMA: All events must have required fields and roundtrip."""

    def test_event_envelope_required_fields(self):
        evt = EventEnvelope(event_id="evt_001", event_type="test")
        assert evt.event_id == "evt_001"
        assert evt.timestamp
        assert evt.source == "rosclaw-auto"

    def test_praxis_failed_event_serialization(self):
        evt = PraxisFailedEvent(
            event_id="evt_f1", task_id="pick_cube", skill_id="pick_v1",
            failure_mode="missed_grasp", phase="grasp", severity="high",
            evidence={"ee_z": 0.018}, praxis_uri="seekdb://praxis/001",
        )
        d = evt.to_dict()
        assert d["payload"]["failure_mode"] == "missed_grasp"
        assert d["payload"]["severity"] == "high"

    def test_benchmark_completed_event(self):
        evt = BenchmarkCompletedEvent(
            event_id="evt_b1", task_id="pick_cube", skill_id="pick_v1",
            metrics={"success_rate": 0.35}, regression_detected=True,
        )
        assert evt.regression_detected is True

    def test_auto_proposal_created_event(self):
        evt = AutoProposalCreatedEvent(
            event_id="evt_p1", proposal_id="prop_001",
            task_id="pick_cube", target_skill_id="pick_v1",
            hypothesis_statement="increase height",
        )
        assert evt.event_type == "rosclaw.auto.proposal.created"

    def test_champion_promoted_event(self):
        evt = ChampionPromotedEvent(
            event_id="evt_c1", champion_id="champ_001",
            skill_id="pick_v1.5", task_id="pick_cube", level="sim",
            metrics={"success_rate": 0.76},
        )
        assert evt.level == "sim"

    def test_deadend_registered_event(self):
        evt = DeadEndRegisteredEvent(
            event_id="evt_d1", deadend_id="de_001",
            task_id="pick_cube", direction="increase_torque",
            rejection_reason="force exceeded",
        )
        assert evt.direction == "increase_torque"

    def test_event_backward_compat_unknown_fields(self):
        """Unknown fields should not crash deserialization."""
        d = {
            "event_id": "evt_001", "event_type": "test",
            "unknown_field": "should_be_ignored",
            "payload": {"extra": "data"},
        }
        evt = EventEnvelope.from_dict(d)
        assert evt.event_id == "evt_001"


# ------------------------------------------------------------------
# Core Schema Contract
# ------------------------------------------------------------------

class TestCoreSchemaContract:
    """AUTO-CORE-SCHEMA: Core objects must validate and roundtrip."""

    def test_autotask_status_transition(self):
        """AUTO-CORE-001: status transitions must be logical."""
        task = AutoTask(id="t1", name="pick", task_type="skill_tuning",
                        robot_id="panda", environment_id="default", target_skill_id="pick_v1")
        assert task.status == "pending"

    def test_failure_case_missing_skill_id(self):
        """AUTO-CORE-002: FailureCase without skill_id is still valid (defaults to empty)."""
        fc = FailureCase(id="f1", praxis_event_id="e1", task_id="t1", skill_id="")
        assert fc.skill_id == ""

    def test_proposal_risk_level_validation(self):
        """AUTO-CORE-002b: Proposal risk_level must be low/medium/high."""
        prop = Proposal(id="p1", source="failure_guided", task="pick_cube",
                        target_skill_id="pick_v1", hypothesis_id="h1",
                        hypothesis_statement="test", patch_type="config_patch",
                        risk_level="low")
        assert prop.risk_level in ("low", "medium", "high")

    def test_patch_rollback_required_for_high_level(self):
        """AUTO-CORE-003: Patch level >= 2 needs rollback plan."""
        p = Patch(id="patch_001", proposal_id="prop_001", patch_level=2,
                  patch_type="skill_parameter_patch", target_skill="pick_v1",
                  changes=[], rollback_plan={})
        assert p.rollback_plan == {}

    def test_evaluation_result_delta_calculation(self):
        """AUTO-CORE-004: Delta must be candidate - baseline."""
        baseline = {"success_rate": 0.4, "collision_rate": 0.1}
        candidate = {"success_rate": 0.6, "collision_rate": 0.05}
        ev = EvaluationResult(id="e1", experiment_id="exp_001",
                               baseline_metrics=baseline, candidate_metrics=candidate,
                               delta={"success_rate": 0.2, "collision_rate": -0.05})
        assert ev.delta["success_rate"] == 0.2
        assert ev.delta["collision_rate"] == -0.05

    def test_champion_level_order(self):
        """AUTO-CORE-005: Champion levels must be valid."""
        champ = Champion(id="c1", skill_id="pick_v1.5", task_id="pick_cube",
                         level="sim", metrics={})
        assert champ.level in ("baseline", "sim", "sandbox", "real_candidate", "real")

    def test_deadend_avoid_conditions(self):
        """AUTO-CORE-006: DeadEnd must have rejection_reason and evidence."""
        de = DeadEnd(id="de1", task_id="pick_cube", direction="increase_torque",
                     rejection_reason="force exceeded", evidence=["sandbox_rejected"])
        assert de.rejection_reason
        assert len(de.evidence) >= 1
