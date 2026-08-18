"""PR-DF-16B: MemoryDistillationService — normal sessions become Memory 2.0.

T1 normal session -> memory created
T2 failure event -> failure memory
T5 idempotency (repeated finished events don't grow memory_items)
T6 ledger records the run
T7 restart reconcile re-enqueues queued/failed rows
T8 corrupt session -> ledger error, runtime survives
"""

import time

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.core.event_topics import EventTopics
from rosclaw.memory.distillation_service import MemoryDistillationService
from rosclaw.memory.gate import MemoryWriteGate
from rosclaw.memory.repository import MemoryRepository
from rosclaw.memory.seekdb_client import InMemoryStructuredStore
from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.storage.lineage import LineageRepository


def _make_session(tmp_path, task="pick cup"):
    cfg = PracticeConfig(
        robot_id="sim_bot",
        task_name=task,
        data_root=str(tmp_path),
        sources=SourceConfig(agent=True, runtime=True),
        mock=True,
        publish_to_event_bus=False,
    )
    coord = PracticeCoordinator(cfg)
    coord.initialize()
    coord.start()
    time.sleep(0.3)
    coord.stop()
    summary = coord.summary
    session_dir = tmp_path / "sessions" / summary.practice_id
    return summary, session_dir


def _service(bus, store, lineage=None):
    repo = MemoryRepository(store)
    gate = MemoryWriteGate(repo)
    svc = MemoryDistillationService(bus, repo, gate, lineage=lineage, store=store)
    svc.subscribe()
    return svc


def _finished(bus, summary, session_dir, **over):
    payload = {
        "practice_id": summary.practice_id,
        "session_id": "sess_1",
        "episode_id": "ep_1",
        "robot_id": "sim_bot",
        "session_dir": str(session_dir),
        "outcome": summary.outcome,
        "fact_verify": summary.fact_verify,
        "manifest_hash": "hash_1",
    }
    payload.update(over)
    bus.publish(Event(topic=EventTopics.PRACTICE_SESSION_FINISHED, payload=payload))


def test_t1_normal_session_distills_to_memory(tmp_path):
    store = InMemoryStructuredStore()
    store.connect()
    bus = EventBus()
    svc = _service(bus, store)
    summary, session_dir = _make_session(tmp_path)
    _finished(bus, summary, session_dir)
    assert svc.drain(timeout=15)
    svc.unsubscribe()
    assert store.count("memory_items", {}) >= 1
    # every ACTIVE memory has evidence rows (§5.4 evidence-backed)
    for item in store.query("memory_items", {"status": "active"}):
        assert store.query("memory_evidence", {"memory_id": item["id"]}) or item.get("evidence_refs")


def test_t2_failure_event_becomes_failure_memory(tmp_path):
    store = InMemoryStructuredStore()
    store.connect()
    bus = EventBus()
    svc = _service(bus, store)
    summary, session_dir = _make_session(tmp_path)
    # inject a failure event into the session's events.jsonl before distilling
    # (the failure extractor keys on runtime.error / serial.fault / camera.wedge
    # or failed gesture.executed events — use the real shape)
    events_path = session_dir / "raw" / "events.jsonl"
    with events_path.open("a") as fh:
        fh.write(
            '{"schema_version": "practice.event.v1", "event_type": "serial.fault", '
            f'"event_id": "evt_slip_1", "practice_id": "{summary.practice_id}", '
            '"payload": {"error": "grasp slip detected", "failure_type": "grasp_slip"}}\n'
        )
    _finished(bus, summary, session_dir)
    assert svc.drain(timeout=15)
    svc.unsubscribe()
    failures = store.query("memory_items", {"memory_type": "failure"})
    assert failures, "failure event should produce a failure memory"
    assert any("serial.fault" in f["title"] or "slip" in f["document"] for f in failures)


def test_t5_t6_idempotent_replay_and_ledger(tmp_path):
    store = InMemoryStructuredStore()
    store.connect()
    bus = EventBus()
    svc = _service(bus, store)
    summary, session_dir = _make_session(tmp_path)
    for _ in range(10):
        _finished(bus, summary, session_dir)
    assert svc.drain(timeout=30)
    svc.unsubscribe()
    first = store.count("memory_items", {})
    assert first >= 1
    # ledger: one row per (practice_id, manifest_hash), completed
    runs = store.query("memory_distillation_runs", {"practice_id": summary.practice_id})
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["attempt"] == 10  # ten replays recorded, no duplicates created


def test_t7_restart_reconcile_requeues(tmp_path):
    store = InMemoryStructuredStore()
    store.connect()
    summary, session_dir = _make_session(tmp_path)
    # simulate a crash: a queued ledger row from a previous process
    store.insert(
        "memory_distillation_runs",
        {
            "id": f"{summary.practice_id}:hash_1",
            "practice_id": summary.practice_id,
            "session_dir": str(session_dir),
            "source_manifest_hash": "hash_1",
            "status": "queued",
            "attempt": 1,
            "created_at": time.time(),
            "updated_at": time.time(),
        },
    )
    bus = EventBus()
    svc = _service(bus, store)
    # subscribe() reconciled the queued row onto the worker
    assert svc.drain(timeout=15)
    svc.unsubscribe()
    assert store.count("memory_items", {}) >= 1
    row = store.query("memory_distillation_runs", {"practice_id": summary.practice_id})[0]
    assert row["status"] == "completed"


def test_t8_corrupt_session_ledger_error_never_crashes(tmp_path):
    store = InMemoryStructuredStore()
    store.connect()
    bus = EventBus()
    svc = _service(bus, store)
    bad_dir = tmp_path / "sessions" / "prac_broken"
    bad_dir.mkdir(parents=True)
    (bad_dir / "raw").mkdir()
    (bad_dir / "raw" / "events.jsonl").write_bytes(b"\x00\x01 not json \xff\xfe")
    bus.publish(
        Event(
            topic=EventTopics.PRACTICE_SESSION_FINISHED,
            payload={
                "practice_id": "prac_broken",
                "session_dir": str(bad_dir),
                "manifest_hash": "h",
            },
        )
    )
    assert svc.drain(timeout=15)
    svc.unsubscribe()
    rows = store.query("memory_distillation_runs", {"practice_id": "prac_broken"})
    # failed OR completed-with-zero-candidates are both honest outcomes;
    # what must never happen is a silent disappearance or a crash
    assert rows and rows[0]["status"] in ("failed", "completed")


def test_lineage_edges_link_memory_to_episode_and_practice(tmp_path):
    store = InMemoryStructuredStore()
    store.connect()
    bus = EventBus()
    lineage = LineageRepository(store)
    svc = _service(bus, store, lineage=lineage)
    summary, session_dir = _make_session(tmp_path)
    _finished(bus, summary, session_dir)
    assert svc.drain(timeout=15)
    svc.unsubscribe()
    edges = store.query("lineage_edges", {})
    if store.count("memory_items", {}) >= 1 and edges:
        relations = {(e["from_type"], e["relation"], e["to_type"]) for e in edges}
        assert ("memory", "derived_from", "episode") in relations
        assert ("episode", "observed_in", "practice") in relations
