"""PR-DF-22 (phase-II §33): Memory Insight 2.0.

known_dead_end_revisited (proposal → similar DeadEnd → guard),
skill_regression / skill_improvement (rolling outcome windows),
harmful_recovery_pattern (proven fix that isn't working).
"""

from __future__ import annotations

import time

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.core.event_topics import EventTopics
from rosclaw.evolution.orchestrator.config import AutoConfig
from rosclaw.evolution.orchestrator.engine.auto_engine import AutoEngine
from rosclaw.evolution.orchestrator.events.subscribers import AutoSubscriber
from rosclaw.memory.insights import MemoryInsightService
from rosclaw.memory.seekdb_client import InMemoryStructuredStore
from rosclaw.storage.lineage import LineageRepository

PROPOSAL_TOPIC = "rosclaw.auto.proposal.created"


def _store():
    s = InMemoryStructuredStore()
    s.connect()
    return s


def _service(bus, store, lineage=None, cooldown=0.0):
    return MemoryInsightService(
        bus, store, robot_id="rh56", cooldown_s=cooldown, lineage_repository=lineage
    )


def _judgment(bus, skill, status, failure="force overshoot", episode="ep_1"):
    bus.publish(
        Event(
            topic=EventTopics.CRITIC_JUDGMENT,
            payload={
                "status": status,
                "skill_id": skill,
                "reason": failure,
                "episode_id": episode,
            },
        )
    )


def _collect(bus):
    seen: list[dict] = []
    bus.subscribe(EventTopics.MEMORY_INSIGHT_CREATED, lambda e: seen.append(e.payload))
    return seen


# -- known_dead_end_revisited -------------------------------------------------


def test_dead_end_revisited_skip_at_high_similarity():
    store = _store()
    store.insert(
        "dead_ends",
        {
            "id": "de_1",
            "task_id": "slide puck",
            "direction": "raise force collisions",
            "rejection_reason": "",
            "registered_at": time.time(),
        },
    )
    bus = EventBus()
    svc = _service(bus, store)
    svc.subscribe()
    seen = _collect(bus)

    bus.publish(
        Event(
            topic=PROPOSAL_TOPIC,
            payload={
                "proposal_id": "prop_1",
                "task_id": "slide puck",
                "target_skill_id": "slide",
                "hypothesis_statement": "raise force collisions",
            },
        )
    )
    assert len(seen) == 1
    ins = seen[0]
    assert ins["insight_type"] == "known_dead_end_revisited"
    assert ins["recommended_action"] == "skip"
    assert ins["dead_end_refs"] == ["de_1"]
    assert ins["similarity"] >= 0.8
    assert ins["failure_id"] == "prop_1"


def test_dead_end_revisited_narrow_at_mid_similarity():
    store = _store()
    store.insert(
        "dead_ends",
        {
            "id": "de_2",
            "task_id": "slide puck",
            "direction": "increase force",
            "rejection_reason": "collisions",
            "registered_at": time.time(),
        },
    )
    bus = EventBus()
    svc = _service(bus, store)
    svc.subscribe()
    seen = _collect(bus)

    bus.publish(
        Event(
            topic=PROPOSAL_TOPIC,
            payload={
                "proposal_id": "prop_2",
                "task_id": "slide puck",
                "target_skill_id": "slide",
                "hypothesis_statement": "increase force",
            },
        )
    )
    assert len(seen) == 1
    assert seen[0]["recommended_action"] == "narrow_search"


def test_no_insight_for_dissimilar_proposal():
    store = _store()
    store.insert(
        "dead_ends",
        {
            "id": "de_3",
            "task_id": "slide puck",
            "direction": "raise force",
            "rejection_reason": "collisions",
            "registered_at": time.time(),
        },
    )
    bus = EventBus()
    svc = _service(bus, store)
    svc.subscribe()
    seen = _collect(bus)
    bus.publish(
        Event(
            topic=PROPOSAL_TOPIC,
            payload={
                "proposal_id": "prop_3",
                "task_id": "slide puck",
                "target_skill_id": "slide",
                "hypothesis_statement": "reduce inter-round cooldown interval",
            },
        )
    )
    assert seen == []


def test_dead_end_insight_lineage_link():
    store = _store()
    lineage = LineageRepository(store)
    store.insert(
        "dead_ends",
        {
            "id": "de_4",
            "task_id": "slide",
            "direction": "raise force setpoint above 400",
            "rejection_reason": "collisions",
            "registered_at": time.time(),
        },
    )
    bus = EventBus()
    svc = _service(bus, store, lineage=lineage)
    svc.subscribe()
    seen = _collect(bus)
    bus.publish(
        Event(
            topic=PROPOSAL_TOPIC,
            payload={
                "proposal_id": "prop_4",
                "task_id": "slide",
                "target_skill_id": "slide",
                "hypothesis_statement": "raise force setpoint above 400",
            },
        )
    )
    assert seen
    parents = lineage.parents("memory_insight", seen[0]["insight_id"])
    assert any(p["to_type"] == "dead_end" and p["to_id"] == "de_4" for p in parents)


# -- skill trend -----------------------------------------------------------------


def test_skill_regression_and_improvement():
    bus = EventBus()
    svc = _service(bus, _store())
    svc.subscribe()
    seen = _collect(bus)

    # regression: 6 successes then 4 failures (first-half 1.0 -> second-half 0.25)
    for _ in range(6):
        _judgment(bus, "slide", "SUCCESS")
    for _ in range(4):
        _judgment(bus, "slide", "FAILED")
    regression = [s for s in seen if s["insight_type"] == "skill_regression"]
    assert regression, "skill_regression not emitted"
    assert regression[0]["skill_id"] == "slide"

    # improvement: 6 failures then 4 successes on another skill
    for _ in range(6):
        _judgment(bus, "pour", "FAILED")
    for _ in range(4):
        _judgment(bus, "pour", "SUCCESS")
    improvement = [s for s in seen if s["insight_type"] == "skill_improvement"]
    assert improvement, "skill_improvement not emitted"


# -- harmful recovery pattern -----------------------------------------------------


def _seed_proven_rule(store):
    store.insert(
        "heuristic_rules",
        {
            "id": "rule_1",
            "failure_type": "force overshoot",
            "success_count": 5,
            "parameter_patch": {"force": 250},
        },
    )


def test_harmful_pattern_when_fix_not_working():
    store = _store()
    _seed_proven_rule(store)
    bus = EventBus()
    svc = _service(bus, store)
    svc.subscribe()
    seen = _collect(bus)
    for i in range(6):  # 2x threshold(3)
        _judgment(bus, "slide", "FAILED", episode=f"ep_{i}")
    kinds = [s["insight_type"] for s in seen]
    assert "harmful_recovery_pattern" in kinds
    # the escalation doesn't retract the patch hint emitted at threshold —
    # harmful is emitted at 2x threshold AFTER it (dedup is per type)
    assert kinds.index("similar_failure_with_patch") < kinds.index(
        "harmful_recovery_pattern"
    )


def test_patch_insight_below_harmful_threshold():
    store = _store()
    _seed_proven_rule(store)
    bus = EventBus()
    svc = _service(bus, store)
    svc.subscribe()
    seen = _collect(bus)
    for i in range(3):  # exactly threshold
        _judgment(bus, "slide", "FAILED", episode=f"ep_{i}")
    kinds = {s["insight_type"] for s in seen}
    assert "similar_failure_with_patch" in kinds
    assert "harmful_recovery_pattern" not in kinds


# -- Auto side: apply_dead_end_guard ----------------------------------------------


def _engine(tmp_path, bus=None):
    return AutoEngine(
        config=AutoConfig(local_store_path=str(tmp_path / "auto")), event_bus=bus
    )


def test_apply_dead_end_guard_actions(tmp_path):
    engine = _engine(tmp_path)
    prop = engine.create_proposal("", "slide", "slide", "raise force", {"force": [400]})
    insight = {
        "insight_id": "ins_1",
        "recommended_action": "skip",
        "dead_end_refs": ["de_1"],
        "similarity": 0.9,
    }
    assert engine.apply_dead_end_guard(prop.id, insight) is True
    data = engine._load("proposals", prop.id)
    assert data["status"] == "rejected"
    assert data["dead_end_guard"]["dead_end_refs"] == ["de_1"]

    prop2 = engine.create_proposal("", "slide", "slide", "raise force", {"force": [400]})
    engine.apply_dead_end_guard(prop2.id, {**insight, "recommended_action": "narrow_search"})
    data2 = engine._load("proposals", prop2.id)
    assert "dead_end_narrow_review" in data2["required_gates"]

    prop3 = engine.create_proposal("", "slide", "slide", "raise force", {"force": [400]})
    engine.apply_dead_end_guard(prop3.id, {**insight, "recommended_action": ""})
    data3 = engine._load("proposals", prop3.id)
    assert "dead_end_stronger_evidence" in data3["required_gates"]

    assert engine.apply_dead_end_guard("prop_missing", insight) is False


def test_full_loop_proposal_to_guard(tmp_path):
    """proposal.created -> insight -> subscriber guards the SAME proposal."""
    store = _store()
    bus = EventBus()
    svc = _service(bus, store)
    svc.subscribe()
    engine = _engine(tmp_path, bus=bus)
    subscriber = AutoSubscriber(engine=engine, event_bus=bus)
    subscriber.subscribe_all()

    store.insert(
        "dead_ends",
        {
            "id": "de_9",
            "task_id": "slide",
            "direction": "raise force setpoint above 400",
            "rejection_reason": "collisions",
            "registered_at": time.time(),
        },
    )
    prop = engine.create_proposal(
        "", "slide", "slide", "raise force setpoint above 400", {"force": [400]}
    )  # publishes rosclaw.auto.proposal.created via engine.publisher? (bus wired)

    # engine.publisher only fires when AutoEngine got an event_bus — it did.
    data = engine._load("proposals", prop.id)
    assert data.get("dead_end_guard", {}).get("action") == "skip"
    assert data["status"] == "rejected"
