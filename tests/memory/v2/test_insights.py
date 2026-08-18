"""PR-DF-11: MemoryInsightService — the publisher half of rosclaw.memory.insight."""

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.core.event_topics import EventTopics
from rosclaw.memory.insights import MemoryInsightService
from rosclaw.memory.seekdb_client import InMemoryStructuredStore


def _collect(bus):
    received = []
    bus.subscribe(EventTopics.MEMORY_INSIGHT_CREATED, lambda e: received.append(e.payload))
    return received


def _failure(bus, skill="pick_cup", reason="slip", episode="ep_1"):
    bus.publish(
        Event(
            topic=EventTopics.CRITIC_JUDGMENT,
            payload={
                "episode_id": episode,
                "status": "FAILURE",
                "reason": reason,
                "context": {"outcome": {"skill_name": skill}, "instruction": "pick"},
            },
        )
    )


def test_repeated_failure_insight_after_threshold():
    bus = EventBus()
    insights = _collect(bus)
    svc = MemoryInsightService(bus, InMemoryStructuredStore(), robot_id="sim", failure_threshold=3)
    svc.subscribe()
    _failure(bus)
    _failure(bus)
    assert insights == []  # below threshold
    _failure(bus)
    assert [i["insight_type"] for i in insights] == ["repeated_failure"]
    ins = insights[0]
    assert ins["skill_id"] == "pick_cup"
    assert ins["failure_type"] == "slip"
    assert ins["robot_id"] == "sim"
    assert ins["evidence_refs"]


def test_similar_failure_with_patch_when_rule_proven():
    bus = EventBus()
    insights = _collect(bus)
    store = InMemoryStructuredStore()
    store.connect()
    store.insert(
        "heuristic_rules",
        {
            "id": "rule_1",
            "failure_signature": "slip",
            "success_count": 4,
            "action_template": '{"force_set": [300, 500]}',
        },
    )
    svc = MemoryInsightService(bus, store, robot_id="sim", failure_threshold=2)
    svc.subscribe()
    _failure(bus)
    _failure(bus, episode="ep_2")
    types = [i["insight_type"] for i in insights]
    assert "similar_failure_with_patch" in types
    ins = next(i for i in insights if i["insight_type"] == "similar_failure_with_patch")
    # Auto's consumed fields are present
    assert ins["search_space"] == {"force_set": [300, 500]}
    assert ins["insight_summary"]
    assert ins["failure_id"] == "ep_2"
    assert ins["memory_refs"] == ["heuristic_rules:rule_1"]


def test_no_patch_insight_without_proven_fix():
    bus = EventBus()
    insights = _collect(bus)
    store = InMemoryStructuredStore()
    store.connect()
    store.insert(
        "heuristic_rules",
        {"id": "rule_weak", "failure_signature": "slip", "success_count": 0},
    )
    svc = MemoryInsightService(bus, store, robot_id="sim", failure_threshold=2)
    svc.subscribe()
    _failure(bus)
    _failure(bus)
    assert [i["insight_type"] for i in insights] == ["repeated_failure"]


def test_cooldown_dedup():
    bus = EventBus()
    insights = _collect(bus)
    svc = MemoryInsightService(
        bus, InMemoryStructuredStore(), robot_id="sim", failure_threshold=1, cooldown_s=3600
    )
    svc.subscribe()
    for i in range(5):
        _failure(bus, episode=f"ep_{i}")
    assert len(insights) == 1


def test_success_judgments_never_generate_insights():
    bus = EventBus()
    insights = _collect(bus)
    svc = MemoryInsightService(bus, InMemoryStructuredStore(), robot_id="sim", failure_threshold=1)
    svc.subscribe()
    bus.publish(
        Event(
            topic=EventTopics.CRITIC_JUDGMENT,
            payload={"episode_id": "ep_ok", "status": "SUCCESS", "context": {}},
        )
    )
    assert insights == []
