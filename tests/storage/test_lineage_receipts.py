"""PR-DF-14: execution_receipts + lineage_edges — projector and repository."""

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.core.event_topics import EventTopics
from rosclaw.memory.seekdb_client import InMemoryStructuredStore, SQLiteStructuredStore
from rosclaw.storage.lineage import LineageRepository
from rosclaw.storage.receipts import ReceiptProjector


def _store():
    s = InMemoryStructuredStore()
    s.connect()
    return s


def test_lineage_link_parents_children_trace():
    repo = LineageRepository(_store())
    repo.link("champion", "champ_1", "promoted_from", "evaluation", "eval_9")
    repo.link("evaluation", "eval_9", "evaluated_by", "experiment", "exp_3")
    repo.link("experiment", "exp_3", "derived_from", "patch", "patch_2")

    parents = repo.parents("champion", "champ_1")
    assert parents[0]["to_id"] == "eval_9"
    children = repo.children("patch", "patch_2")
    assert children[0]["from_id"] == "exp_3"

    trace = repo.trace("champion", "champ_1")
    kinds = [(c["to_type"], c["to_id"]) for c in trace["chain"]]
    assert kinds == [("evaluation", "eval_9"), ("experiment", "exp_3"), ("patch", "patch_2")]


def test_lineage_link_idempotent_and_cycle_safe():
    repo = LineageRepository(_store())
    a = repo.link("memory", "m1", "derived_from", "episode", "e1")
    b = repo.link("memory", "m1", "derived_from", "episode", "e1")
    assert a == b
    # cycle: e1 -> m1 while m1 -> e1 exists; ancestors must terminate
    repo.link("episode", "e1", "observed_in", "memory", "m1")
    chain = repo.ancestors("memory", "m1")
    assert len(chain) <= 3


def test_receipt_projector_writes_and_links():
    store = _store()
    bus = EventBus()
    lineage = LineageRepository(store)
    proj = ReceiptProjector(bus, store, lineage=lineage)
    proj.subscribe()
    bus.publish(
        Event(
            topic=EventTopics.ACTION_RECEIPT,
            payload={
                "action_id": "act_1",
                "trace_id": "trace_1",
                "mode": "REAL",
                "final_state": "SUCCEEDED",
                "body_id": "rh56_right",
                "capability_id": "rh56.single_step",
                "evidence_level": "task_verified",
                "evidence_domain": "hardware",
                "episode_id": "ep_1",
                "practice_id": "prac_1",
                "artifacts": ["file:///tmp/x.mcap"],
            },
            trace_id="trace_1",
        )
    )
    rows = store.query("execution_receipts", {"action_id": "act_1"})
    assert len(rows) == 1
    row = rows[0]
    assert row["execution_mode"] == "REAL"
    assert row["evidence_domain"] == "hardware"
    assert row["trace_id"] == "trace_1"
    # lineage edge receipt -> action
    edges = store.query("lineage_edges", {"from_id": "rcpt_act_1"})
    assert edges and edges[0]["relation"] == "generated_from"
    assert edges[0]["to_id"] == "act_1"
    # idempotent re-delivery
    proj.project({"action_id": "act_1", "final_state": "SUCCEEDED"})
    assert store.count("execution_receipts", {"action_id": "act_1"}) == 1


def test_receipt_projector_failure_never_raises():
    class _DeadStore:
        def insert(self, *a, **k):
            raise ConnectionError("down")

        def query(self, *a, **k):
            raise ConnectionError("down")

    bus = EventBus()
    proj = ReceiptProjector(bus, _DeadStore(), lineage=None)
    proj.subscribe()
    bus.publish(
        Event(topic=EventTopics.ACTION_RECEIPT, payload={"action_id": "act_x"})
    )  # must not raise


def test_schemas_in_structured_store(tmp_path):
    """execution_receipts + lineage_edges exist in the SQLite schema set."""
    store = SQLiteStructuredStore(str(tmp_path / "k.sqlite"))
    store.connect()
    store.insert(
        "execution_receipts", {"id": "r1", "action_id": "a1", "final_state": "SUCCEEDED"}
    )
    store.insert(
        "lineage_edges",
        {"id": "l1", "from_type": "receipt", "from_id": "r1", "relation": "generated_from",
         "to_type": "action", "to_id": "a1"},
    )
    assert store.count("execution_receipts", {}) == 1
    assert store.count("lineage_edges", {}) == 1
    store.disconnect()
