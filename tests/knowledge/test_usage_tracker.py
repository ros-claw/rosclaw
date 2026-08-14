"""PR-DF-10: KnowledgeUsageTracker — automatic conservative feedback loop."""

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.knowledge.usage_tracker import KnowledgeUsageTracker


class _FakeFacade:
    def __init__(self):
        self.feedback_calls = []

    def feedback(self, feedback):
        self.feedback_calls.append(feedback)
        return True


def _pack_event(bus, pack_id="pack_1", units=("unit_a", "unit_b")):
    bus.publish(
        Event(
            topic="know.reference_pack.created",
            payload={
                "reference_pack_id": pack_id,
                "knowledge_unit_ids": list(units),
                "count": len(units),
            },
        )
    )


def _judgment(bus, status="SUCCESS", episode_id="ep_1"):
    bus.publish(
        Event(
            topic="rosclaw.critic.judgment",
            payload={"episode_id": episode_id, "status": status, "practice_id": "prac_1"},
        )
    )


def test_success_emits_useful_feedback_per_unit():
    bus = EventBus()
    facade = _FakeFacade()
    tracker = KnowledgeUsageTracker(bus, facade)
    tracker.subscribe()
    _pack_event(bus)
    assert tracker.tracked_count == 1
    _judgment(bus, status="SUCCESS")
    assert tracker.tracked_count == 0
    verdicts = [f.verdict for f in facade.feedback_calls]
    assert verdicts == ["useful", "useful"]
    fb = facade.feedback_calls[0]
    assert fb.reference_pack_id == "pack_1"
    assert fb.receipt_ref == "ep_1"
    assert fb.practice_ref == "prac_1"
    assert fb.used_by_agent is True
    assert fb.origin == "verifier"


def test_failure_emits_unknown_never_misleading():
    """§31: automatic feedback must not over-attribute failures."""
    bus = EventBus()
    facade = _FakeFacade()
    tracker = KnowledgeUsageTracker(bus, facade)
    tracker.subscribe()
    _pack_event(bus, units=("unit_x",))
    _judgment(bus, status="FAILURE")
    assert [f.verdict for f in facade.feedback_calls] == ["unknown"]


def test_ttl_expiry_closes_without_feedback():
    bus = EventBus()
    facade = _FakeFacade()
    tracker = KnowledgeUsageTracker(bus, facade, ttl_s=0.0)
    tracker.subscribe()
    _pack_event(bus)
    _judgment(bus, status="SUCCESS")
    assert facade.feedback_calls == []
    assert tracker.tracked_count == 0


def test_advice_linkage_attaches_advice_id():
    bus = EventBus()
    facade = _FakeFacade()
    tracker = KnowledgeUsageTracker(bus, facade)
    tracker.subscribe()
    _pack_event(bus)
    bus.publish(
        Event(
            topic="how.advice.created",
            payload={"reference_pack_id": "pack_1", "advice_id": "adv_9"},
        )
    )
    _judgment(bus, status="SUCCESS")
    assert facade.feedback_calls[0].advice_id == "adv_9"


def test_unsubscribe_stops_feedback():
    bus = EventBus()
    facade = _FakeFacade()
    tracker = KnowledgeUsageTracker(bus, facade)
    tracker.subscribe()
    tracker.unsubscribe()
    _pack_event(bus)
    _judgment(bus)
    assert facade.feedback_calls == []
