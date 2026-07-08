"""Unit tests for the RH56 PracticeEmitter (lives in rosclaw-rh56-runtime)."""

from __future__ import annotations

import pytest

pytest.importorskip("rosclaw_rh56", reason="requires optional rosclaw-rh56-runtime package")

from rosclaw_rh56.body.body_state import RH56BodyState  # noqa: E402
from rosclaw_rh56.body.contact_event_detector import ContactEventRecord  # noqa: E402
from rosclaw_rh56.how.force_interventions import ForceInterventionEngine  # noqa: E402
from rosclaw_rh56.practice.practice_emitter import Rh56PracticeEmitter  # noqa: E402
from rosclaw_rh56.safety.reactive_sandbox_v2 import ForceSandboxDecision  # noqa: E402


def test_emit_physical_feedback_has_payload_fields():
    em = Rh56PracticeEmitter("prac_1", "robot_1", "body_rh56_left", "ok_contact")
    state = RH56BodyState(
        ts=1.0,
        target_angle={"thumb": 100},
        actual_angle={"thumb": 90},
        force_net_g={"thumb": 120.0},
        primary_event="desired_contact",
        secondary_tags=["stable"],
    )
    ev = em.emit_physical_feedback(state)
    assert ev["event_type"] == "physical_feedback_event"
    assert ev["body_id"] == "body_rh56_left"
    payload = ev["payload"]
    assert payload["primary_event"] == "desired_contact"
    assert payload["force_net"]["thumb"] == 120.0


def test_emit_contact_event():
    em = Rh56PracticeEmitter("prac_1", "robot_1", "body_rh56_left", "ok_contact")
    rec = ContactEventRecord(
        primary_event="desired_contact",
        secondary_tags=["stable"],
        confidence=0.95,
    )
    ev = em.emit_contact_event(rec, dofs=["thumb", "index"])
    assert ev["event_type"] == "contact_event"
    assert ev["payload"]["event_type"] == "desired_contact"
    assert ev["payload"]["dofs"] == ["thumb", "index"]


def test_emit_failure_from_sandbox():
    em = Rh56PracticeEmitter("prac_1", "robot_1", "body_rh56_left", "ok_contact")
    dec = ForceSandboxDecision(phase="in_motion", decision="stop", reason="over_contact")
    ev = em.emit_failure_from_sandbox(dec)
    assert ev["event_type"] == "failure_event"
    assert ev["payload"]["failure_type"] == "over_contact"


def test_emit_how_intervention():
    em = Rh56PracticeEmitter("prac_1", "robot_1", "body_rh56_left", "ok_contact")
    engine = ForceInterventionEngine()
    intervention = engine.suggest("over_contact")
    ev = em.emit_how_intervention(intervention, failure_id="fail_1", outcome="resolved")
    assert ev["event_type"] == "how_intervention_event"
    assert ev["payload"]["outcome"] == "resolved"
    assert ev["payload"]["failure_id"] == "fail_1"


def test_emit_candidate_policy():
    em = Rh56PracticeEmitter("prac_1", "robot_1", "body_rh56_left", "ok_contact")
    candidate = {
        "candidate_id": "c1",
        "path": "thumb_first",
        "target_pose": {"thumb": 400, "index": 500},
        "force_metrics": {"thumb_mean": 120.0},
    }
    ev = em.emit_candidate_policy(candidate, skill_id="skill_ok_contact")
    assert ev["event_type"] == "candidate_policy_event"
    assert ev["payload"]["candidate_id"] == "c1"


def test_emit_promotion_result():
    em = Rh56PracticeEmitter("prac_1", "robot_1", "body_rh56_left", "ok_contact")
    ev = em.emit_promotion_result(
        "c1", True, policy_id="ok_contact_safe_v2", metrics={"thumb_mean": 120.0}
    )
    assert ev["event_type"] == "promotion_result_event"
    assert ev["payload"]["passed"] is True


def test_emitter_validates_with_rosclaw_schema():
    from rosclaw.practice.schemas import PracticeEventEnvelope

    em = Rh56PracticeEmitter("prac_1", "robot_1", "body_rh56_left", "ok_contact")
    state = RH56BodyState(ts=1.0, force_net_g={"thumb": 100.0})
    ev = em.emit_physical_feedback(state)
    # Should not raise.
    envelope = PracticeEventEnvelope(**ev)
    assert envelope.event_type == "physical_feedback_event"
