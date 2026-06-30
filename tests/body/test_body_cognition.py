"""Tests for generic BodyCognition and PromotionGateResult."""
from __future__ import annotations

from rosclaw.body.body_cognition import BodyCognition, PromotionGateResult


def test_promotion_gate_result_passed():
    gate = PromotionGateResult(
        policy_id="ok_contact_safe_v2",
        passed=True,
        rounds=10,
        contact_detected_rate=1.0,
        force_window_pass_rate=1.0,
        max_force_net=169.0,
        failures=[],
    )
    assert gate.to_dict()["passed"] is True
    assert gate.to_dict()["rounds"] == 10


def test_body_cognition_roundtrip():
    cog = BodyCognition(
        body_id="inspire_rh56_right_dev_ttyUSB0",
        schema_version="rosclaw.body.cognition.v2_1",
        known_traits=["FORCE_ACT must be baseline-subtracted"],
        promoted_policies=[{"policy_id": "ok_contact_safe_v2", "version": "0.3.0"}],
    )
    restored = BodyCognition.from_dict(cog.to_dict())
    assert restored.body_id == cog.body_id
    assert restored.get_promoted_policy("ok_contact_safe_v2")["version"] == "0.3.0"
    assert restored.get_promoted_policy("missing") is None
