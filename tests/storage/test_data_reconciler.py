"""PR-DF-19 (phase-II §17-§26): DataReconciler — recovery after store outages.

Fault E2E coverage (§26):
* Case 1: retrieval down -> projection lag -> rebuild -> lag 0
* Case 2: structured down -> reconcile_required ledger -> catch-up, 0 dup
* Case 3: repeated reconcile runs -> counts unchanged (idempotency §24)
* Case 4: crash between practice close and distill -> discovered + finished
Plus the §21 catalog ledger columns and the §25 evolution spool replay.
"""

from __future__ import annotations

import json
import time

from rosclaw.memory.seekdb_client import InMemoryStructuredStore
from rosclaw.memory.v2.gate import MemoryWriteGate
from rosclaw.memory.v2.repository import MemoryRepository
from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.storage.reconciler import DataReconciler


def _store():
    s = InMemoryStructuredStore()
    s.connect()
    return s


def _make_session(data_root, task="slide puck"):
    cfg = PracticeConfig(
        robot_id="sim_bot",
        task_name=task,
        data_root=str(data_root),
        sources=SourceConfig(agent=True, runtime=True),
        mock=True,
        publish_to_event_bus=False,
    )
    coord = PracticeCoordinator(cfg)
    coord.initialize()
    coord.start()
    time.sleep(0.3)
    coord.stop()
    return coord.summary


def _reconciler(store, data_root, **kw):
    return DataReconciler(structured_store=store, data_root=data_root, **kw)


# -- §21 catalog ledger columns ----------------------------------------------


def test_catalog_has_reconcile_columns(tmp_path):
    summary = _make_session(tmp_path)
    rows = DataReconciler(
        structured_store=_store(), data_root=tmp_path
    )._catalog_rows(summary.practice_id)
    assert rows, "practice row missing"
    for col in (
        "fact_ingested",
        "memory_distilled",
        "last_fact_ingest_at",
        "last_memory_distill_at",
        "fact_ingest_error",
        "memory_distill_error",
        "reconcile_required",
    ):
        assert col in rows[0], f"column {col} missing"


# -- §22/§24 reconcile_practice + idempotency ---------------------------------


def test_reconcile_practice_end_to_end_and_idempotent(tmp_path):
    summary = _make_session(tmp_path)
    store = _store()
    rec = _reconciler(store, tmp_path)

    pending = rec.pending_practices()
    assert any(r["practice_id"] == summary.practice_id for r in pending)

    result = rec.reconcile_practice(summary.practice_id)
    assert result["processed"] is True
    assert result["steps"]["memory_distill"]["status"] == "completed"
    episodes_before = store.count("episodes", {})
    memories_before = store.count("memory_items", {})
    assert memories_before >= 1, "distillation must store at least the episode memory"

    row = rec._catalog_rows(summary.practice_id)[0]
    assert row["fact_ingested"] == 1
    assert row["memory_distilled"] == 1
    assert row["reconcile_required"] == 0

    # §24: reconcile again — 0 duplicate facts / memories
    again = rec.reconcile_practice(summary.practice_id)
    assert again["steps"]["fact_ingest"] == "already_done"
    assert again["steps"]["memory_distill"] == "already_done"
    assert store.count("episodes", {}) == episodes_before
    assert store.count("memory_items", {}) == memories_before


def test_reconcile_dry_run_writes_nothing(tmp_path):
    summary = _make_session(tmp_path)
    store = _store()
    rec = _reconciler(store, tmp_path)
    result = rec.reconcile_practice(summary.practice_id, dry_run=True)
    assert result["steps"]["fact_ingest"] == "would_run"
    assert result["steps"]["memory_distill"] == "would_run"
    assert store.count("memory_items", {}) == 0
    assert store.count("episodes", {}) == 0


# -- §26 Case 2: structured down -> ledger -> catch-up ------------------------


class _DeadStore:
    def __init__(self, inner):
        self._inner = inner
        self.alive = False

    def insert(self, *a, **k):
        if not self.alive:
            raise ConnectionError("structured store down")
        return self._inner.insert(*a, **k)

    def query(self, *a, **k):
        return self._inner.query(*a, **k)

    def count(self, *a, **k):
        return self._inner.count(*a, **k)


def test_case2_structured_down_marks_and_recovers(tmp_path):
    from rosclaw.memory.v2.distillation_service import MemoryDistillationService

    inner = _store()
    dead = _DeadStore(inner)
    summary = _make_session(tmp_path)

    # structured down: distillation must fail loudly into the ledger, never
    # fake a stored memory
    repo = MemoryRepository(dead)
    svc = MemoryDistillationService(None, repo, MemoryWriteGate(repo))
    svc._work(
        {
            "practice_id": summary.practice_id,
            "session_id": "sess_1",
            "episode_id": "ep_1",
            "session_dir": str(tmp_path / "sessions" / summary.practice_id),
            "manifest_hash": "",
        }
    )
    assert inner.count("memory_items", {}) == 0, "no fake memory stored while down"
    row = DataReconciler(structured_store=inner, data_root=tmp_path)._catalog_rows(
        summary.practice_id
    )[0]
    assert row["reconcile_required"] == 1
    assert row["memory_distilled"] == 0

    # store recovers: reconcile catches everything up
    dead.alive = True
    rec = _reconciler(inner, tmp_path)
    report = rec.reconcile_memory()
    assert report["processed"] == 1
    assert inner.count("memory_items", {}) >= 1
    row = rec._catalog_rows(summary.practice_id)[0]
    assert row["reconcile_required"] == 0

    # Case 3 (§26): repeated reconcile leaves counts unchanged
    memories = inner.count("memory_items", {})
    episodes = inner.count("episodes", {})
    rec.reconcile_memory()
    rec.reconcile_memory()
    assert inner.count("memory_items", {}) == memories
    assert inner.count("episodes", {}) == episodes


# -- §26 Case 4: crash between practice close and distill ---------------------


def test_case4_crash_before_distill_discovered_and_finished(tmp_path):
    summary = _make_session(tmp_path)  # practice closed, process "dies" here
    store = _store()
    rec = _reconciler(store, tmp_path)

    # restart: reconciler discovers the pending practice and finishes it
    pending = rec.pending_practices()
    assert [r["practice_id"] for r in pending] == [summary.practice_id]
    report = rec.reconcile_memory()
    assert report["processed"] == 1
    assert store.count("memory_items", {}) >= 1
    assert rec.pending_practices() == []


# -- §26 Case 1: retrieval down -> projection lag -> rebuild ------------------


class _FakeRetrievalStore:
    def __init__(self):
        self.rows = {}

    def connect(self):
        return None

    def insert(self, table, record):
        self.rows[record["id"]] = record
        return record["id"]

    def insert_many(self, table, records):
        for r in records:
            self.rows[r["id"]] = r
        return len(records)

    def refresh_index(self, table):
        return None

    def count(self, table):
        return len(self.rows)


def test_case1_projection_lag_rebuilds_to_zero(tmp_path):
    summary = _make_session(tmp_path)
    store = _store()
    # ingest + distill while retrieval is "down" (no projection writes)
    rec_no_retrieval = _reconciler(store, tmp_path)
    rec_no_retrieval.reconcile_practice(summary.practice_id)
    source = store.count("memory_items", {})
    assert source >= 1

    retrieval = _FakeRetrievalStore()  # empty: lag = source
    rec = _reconciler(store, tmp_path, retrieval_store=retrieval)
    status = rec.reconcile_projection(dry_run=True)
    assert status["lag_before"] == source
    assert status["action"] == "would_rebuild"

    done = rec.reconcile_projection()
    assert done["rebuilt"] == source
    assert done["lag_after"] == 0


def test_projection_skipped_without_retrieval_store(tmp_path):
    rec = _reconciler(_store(), tmp_path)
    assert rec.reconcile_projection()["skipped"] == "no retrieval store"


# -- §25 evolution spool replay ------------------------------------------------


def test_evolution_spool_reconcile_upserts_and_marks_synced(tmp_path):
    from rosclaw.evolution.orchestrator.storage.local_store import LocalStore

    store = _store()
    cache = LocalStore(str(tmp_path / "auto"))
    cache.save("proposals", "prop_1", {"id": "prop_1", "hypothesis": "h1"})
    cache.save("patches", "patch_1", {"id": "patch_1"})
    # one record already in the store (current — must not be re-upserted)
    store.insert(
        "evolution_records",
        {
            "id": "proposals:prop_0",
            "namespace": "proposals",
            "key": "prop_0",
            "data": json.dumps({"id": "prop_0"}),
            "updated_at": time.time() + 100,
        },
    )
    cache.save("proposals", "prop_0", {"id": "prop_0"})

    rec = _reconciler(store, tmp_path, evolution_cache=cache)
    dry = rec.reconcile_evolution_spool(dry_run=True)
    assert dry["upserted"] == 2
    assert store.count("evolution_records", {}) == 1, "dry-run must not write"

    done = rec.reconcile_evolution_spool()
    assert done["upserted"] == 2
    assert done["skipped_current"] == 1
    assert store.count("evolution_records", {}) == 3
    rows = store.query("evolution_records", {"id": "proposals:prop_1"})
    assert json.loads(rows[0]["data"])["hypothesis"] == "h1"
    # local copies stay (cache remains a cache), sync marks written
    assert cache.load("proposals", "prop_1") is not None
    assert cache.load("_reconcile", "proposals:prop_1") is not None
    # second run is a noop
    again = rec.reconcile_evolution_spool()
    assert again["upserted"] == 0


def test_evolution_spool_skipped_without_cache(tmp_path):
    rec = _reconciler(_store(), tmp_path)
    assert rec.reconcile_evolution_spool()["skipped"] == "no evolution cache"


# -- CLI ------------------------------------------------------------------------


def test_cmd_data_reconcile_json(tmp_path, capsys):
    summary = _make_session(tmp_path)
    store_path = tmp_path / "k.sqlite"
    from rosclaw.memory.seekdb_client import SQLiteStructuredStore

    store = SQLiteStructuredStore(str(store_path))
    store.connect()
    store.disconnect()

    import argparse

    from rosclaw.storage.cli import cmd_data_reconcile

    args = argparse.Namespace(
        practice=summary.practice_id,
        all=False,
        dry_run=False,
        json=True,
        data_root=str(tmp_path),
        spool_path=str(tmp_path / "auto"),
        backend="sqlite",
        url=None,
        path=str(store_path),
    )
    rc = cmd_data_reconcile(args)
    out = capsys.readouterr().out
    report = json.loads(out[out.index("{"):])
    assert rc == 0
    assert report["memory"]["processed"] is True
    assert report["memory"]["steps"]["memory_distill"]["status"] == "completed"

    store2 = SQLiteStructuredStore(str(store_path))
    store2.connect()
    assert store2.count("memory_items", {}) >= 1
    store2.disconnect()
