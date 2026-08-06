from __future__ import annotations

from rosclaw.knowledge.feedback_adapter import build_usage_feedback


def test_feedback_contains_governance_refs_not_practice_payload():
    feedback = build_usage_feedback(
        reference_pack_id="pack_1",
        advice_id="advice_1",
        knowledge_unit_id="unit_1",
        context_hash="a" * 64,
        verdict="useful",
        used_by_agent=True,
        receipt_ref="receipt_1",
        practice_ref="episode_1",
    )
    payload = feedback.model_dump(mode="json")
    assert payload["receipt_ref"] == "receipt_1"
    assert payload["practice_ref"] == "episode_1"
    assert not {"trajectory", "video", "sensor_data", "memory_content"}.intersection(payload)
