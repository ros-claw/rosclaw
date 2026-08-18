"""PR-DF-21 (phase-II §32): Knowledge Usage Ledger — observe before ranking."""

from __future__ import annotations

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.knowledge.usage_ledger import KnowledgeUsageLedger
from rosclaw.knowledge.usage_tracker import KnowledgeUsageTracker
from rosclaw.memory.seekdb_client import InMemoryStructuredStore, SQLiteStructuredStore


def _store():
    s = InMemoryStructuredStore()
    s.connect()
    return s


def test_schema_in_structured_store(tmp_path):
    store = SQLiteStructuredStore(str(tmp_path / "k.sqlite"))
    store.connect()
    store.insert(
        "knowledge_usage_events",
        {"id": "use_1", "knowledge_unit_id": "ku_1", "verdict": "useful"},
    )
    assert store.count("knowledge_usage_events", {}) == 1
    store.disconnect()


def test_ledger_record_and_aggregate():
    store = _store()
    ledger = KnowledgeUsageLedger(store)
    ledger.record("presented", reference_pack_id="rp_1", knowledge_unit_id="ku_a")
    ledger.record("used", reference_pack_id="rp_1", knowledge_unit_id="ku_a", advice_id="adv_1")
    ledger.record("useful", reference_pack_id="rp_1", knowledge_unit_id="ku_a", episode_id="ep_1")
    ledger.record("unknown", reference_pack_id="rp_2", knowledge_unit_id="ku_a")
    ledger.record("misleading", reference_pack_id="rp_3", knowledge_unit_id="ku_b")
    ledger.record("incompatible", reference_pack_id="rp_4", knowledge_unit_id="ku_b")

    agg = ledger.aggregate()
    assert agg["ku_a"]["presented_count"] == 1
    assert agg["ku_a"]["used_count"] == 1
    assert agg["ku_a"]["useful_count"] == 1
    assert agg["ku_a"]["unknown_count"] == 1
    assert agg["ku_a"]["verified_success_count"] == 1
    assert agg["ku_a"]["verified_failure_count"] == 0
    assert agg["ku_b"]["misleading_count"] == 1
    assert agg["ku_b"]["incompatible_count"] == 1
    assert agg["ku_b"]["verified_failure_count"] == 2

    only_a = ledger.aggregate("ku_a")
    assert set(only_a) == {"ku_a"}


def test_ledger_write_never_breaks_on_dead_store():
    class _Dead:
        def insert(self, *a, **k):
            raise ConnectionError("down")

        def query(self, *a, **k):
            raise ConnectionError("down")

    ledger = KnowledgeUsageLedger(_Dead())
    assert ledger.record("presented", knowledge_unit_id="ku_x") is None
    assert ledger.aggregate() == {}


def test_tracker_records_all_stages():
    store = _store()
    bus = EventBus()
    tracker = KnowledgeUsageTracker(bus, facade=None, usage_ledger=KnowledgeUsageLedger(store))
    tracker.subscribe()

    bus.publish(
        Event(
            topic="know.reference_pack.created",
            payload={
                "reference_pack_id": "rp_1",
                "knowledge_unit_ids": ["ku_a", "ku_b"],
                "robot_id": "rh56",
                "task_id": "slide",
            },
        )
    )
    bus.publish(
        Event(
            topic="how.advice.created",
            payload={"reference_pack_id": "rp_1", "advice_id": "adv_1"},
        )
    )
    bus.publish(
        Event(
            topic="rosclaw.critic.judgment",
            payload={"status": "SUCCESS", "episode_id": "ep_1", "practice_id": "prac_1"},
        )
    )

    rows = store.query("knowledge_usage_events", {})
    verdicts = sorted(r["verdict"] for r in rows)
    assert verdicts == ["presented", "presented", "used", "used", "useful", "useful"]
    agg = KnowledgeUsageLedger(store).aggregate()
    assert agg["ku_a"]["useful_count"] == 1
    assert agg["ku_a"]["verified_success_count"] == 1
    row = rows[0]
    assert row["robot_id"] == "rh56"
    assert row["task_id"] == "slide"
    outcome_row = [r for r in rows if r["verdict"] == "useful"][0]
    assert outcome_row["episode_id"] == "ep_1"
    assert outcome_row["practice_id"] == "prac_1"
    assert outcome_row["advice_id"] == "adv_1"


def test_tracker_without_ledger_keeps_df10_behavior():
    bus = EventBus()
    tracker = KnowledgeUsageTracker(bus, facade=None)  # no ledger
    tracker.subscribe()
    bus.publish(
        Event(
            topic="know.reference_pack.created",
            payload={"reference_pack_id": "rp_x", "knowledge_unit_ids": ["ku_x"]},
        )
    )
    assert tracker.tracked_count == 1
