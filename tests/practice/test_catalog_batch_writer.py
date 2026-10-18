"""Tests for PracticeCatalog batched event/event-index writes."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from rosclaw.practice.storage.catalog import PracticeCatalog, _BatchWriter


def _event_record(i: int, practice_id: str = "p1") -> dict:
    return {
        "event_id": f"e{i:04d}",
        "practice_id": practice_id,
        "source": "runtime",
        "event_type": "test.tick",
        "timestamp_ns": i,
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "action_id": None,
        "task_id": None,
        "skill_id": None,
        "payload_ref": None,
        "tags": json.dumps(["test"]),
    }


def _index_record(i: int, session_id: str = "s1", episode_id: str = "ep1") -> dict:
    return {
        "event_id": f"e{i:04d}",
        "session_id": session_id,
        "episode_id": episode_id,
        "timestamp_ns": i,
        "event_type": "test.tick",
        "artifact_id": None,
        "byte_offset": i * 100,
        "summary_json": json.dumps({"seq": i}),
    }


@pytest.fixture
def catalog(tmp_path: Path) -> PracticeCatalog:
    catalog = PracticeCatalog(tmp_path / "catalog.sqlite")
    try:
        yield catalog
    finally:
        catalog.close()


def test_insert_event_synchronous_when_batch_size_one(tmp_path: Path) -> None:
    cat = PracticeCatalog(tmp_path / "catalog.sqlite", event_batch_size=1)
    cat.insert_event(_event_record(0))
    assert cat.count_events("p1") == 1
    cat.close()


def test_batch_flush_on_size(catalog: PracticeCatalog) -> None:
    for i in range(10):
        catalog.insert_event(_event_record(i))
    # With default batch_size=500, nothing should be flushed yet.
    assert catalog.count_events("p1") == 0
    catalog.flush()
    assert catalog.count_events("p1") == 10


def test_batch_flush_on_interval(tmp_path: Path) -> None:
    cat = PracticeCatalog(tmp_path / "catalog.sqlite", event_batch_size=500, event_flush_ms=50)
    cat.insert_event(_event_record(0))
    assert cat.count_events("p1") == 0
    time.sleep(0.15)
    assert cat.count_events("p1") == 1
    cat.close()


def test_close_flushes_pending_events(catalog: PracticeCatalog) -> None:
    for i in range(10):
        catalog.insert_event(_event_record(i))
    catalog.close()

    # Re-open to verify persistence.
    cat2 = PracticeCatalog(catalog._db_path, event_batch_size=1)
    assert cat2.count_events("p1") == 10
    cat2.close()


def test_close_flushes_pending_event_index(tmp_path: Path) -> None:
    cat = PracticeCatalog(tmp_path / "catalog.sqlite")
    for i in range(5):
        cat.insert_event_index(_index_record(i))
    cat.close()

    cat2 = PracticeCatalog(cat._db_path, event_batch_size=1)
    rows = cat2.list_event_index(session_id="s1")
    assert len(rows) == 5
    cat2.close()


def test_event_index_roundtrip(catalog: PracticeCatalog) -> None:
    catalog.insert_event_index(_index_record(7))
    catalog.flush()
    row = catalog.get_event_index("e0007")
    assert row is not None
    assert row["byte_offset"] == 700
    assert row["summary_json"]["seq"] == 7


def test_event_count_matches_inserted(catalog: PracticeCatalog) -> None:
    for i in range(100):
        catalog.insert_event(_event_record(i))
    catalog.flush()
    assert catalog.count_events("p1") == 100


def test_batch_writer_queue_full_falls_back_without_data_loss(tmp_path: Path) -> None:
    cat = PracticeCatalog(
        tmp_path / "catalog.sqlite",
        event_batch_size=500,
        event_flush_ms=300_000,  # Never flush by interval.
        event_max_queue=2,
    )
    # Saturating the queue must persist overflow records synchronously rather
    # than silently dropping physical-event evidence.
    for i in range(5):
        cat.insert_event(_event_record(i))
    cat.flush()
    assert cat.count_events("p1") == 5
    cat.close()


def test_batch_writer_close_retries_live_thread() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked_flush(_batch: list[dict]) -> None:
        entered.set()
        release.wait(timeout=2.0)

    writer = _BatchWriter(
        "blocked",
        blocked_flush,
        batch_size=2,
        flush_interval_ms=1.0,
    )
    writer.put({"_wm": 1})
    assert entered.wait(timeout=1.0)
    assert not writer.close(timeout=0.01)
    release.set()
    assert writer.close(timeout=1.0)


def test_catalog_suspended_queries_durability_and_resume(tmp_path):
    import sqlite3

    cat = PracticeCatalog(tmp_path / "owned.sqlite", event_flush_ms=300_000)
    original = (cat._event_writer._thread, cat._event_index_writer._thread)
    try:
        assert all(worker.is_alive() for worker in original)
        event = dict(_event_record(1), _wm=1)
        index = dict(_index_record(1), _wm=1)
        cat.insert_event(event)
        cat.insert_event_index(index)
        cat.suspend_writers()
        for worker in original:
            worker.join(timeout=1.0)
            assert not worker.is_alive()
        assert cat._event_writer is None
        assert cat._event_index_writer is None
        assert cat.count_events("p1") == 1
        assert cat.get_event_index("e0001")["summary_json"]["seq"] == 1
        assert cat.flush_until(1, timeout_sec=0.1)["ok"]
        cat.flush()
        cat.suspend_writers()
        assert cat._event_writer is None
        assert cat._event_index_writer is None
        with sqlite3.connect(str(cat._db_path)) as reader:
            assert reader.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
            assert reader.execute("SELECT COUNT(*) FROM practice_event_index").fetchone()[0] == 1
        cat.resume_writers()
        replacement = (cat._event_writer._thread, cat._event_index_writer._thread)
        assert all(worker.is_alive() for worker in replacement)
        assert not any(worker is old for worker in replacement for old in original)
        cat.insert_event(dict(_event_record(2), _wm=2))
        cat.insert_event_index(dict(_index_record(2), _wm=2))
        cat.close()
        cat.close()
        for worker in replacement:
            worker.join(timeout=1.0)
            assert not worker.is_alive()
        with sqlite3.connect(str(cat._db_path)) as reader:
            assert reader.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
            assert reader.execute("SELECT COUNT(*) FROM practice_event_index").fetchone()[0] == 2
        with pytest.raises(RuntimeError, match="closed"):
            cat.flush()
    finally:
        cat.close()


def test_catalog_second_worker_start_failure_unwinds_real_resources(tmp_path):
    import sqlite3
    from unittest.mock import patch

    import rosclaw.practice.storage.catalog as module

    connections = []
    started = []
    real_connect = sqlite3.connect
    real_start = threading.Thread.start

    def connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    def start(worker):
        if worker.name == "catalog-batch-event_index":
            assert len(started) == 1
            assert started[0].is_alive()
            raise OSError("second worker startup injected")
        result = real_start(worker)
        if worker.name == "catalog-batch-events":
            started.append(worker)
        return result

    with (
        patch.object(module.sqlite3, "connect", side_effect=connect),
        patch.object(threading.Thread, "start", new=start),
        pytest.raises(OSError, match="second worker startup injected"),
    ):
        PracticeCatalog(tmp_path / "partial.sqlite")
    assert len(started) == 1
    assert len(connections) == 1
    started[0].join(timeout=1.0)
    assert not started[0].is_alive()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
    # A new real connection can still read the initialized schema.
    with real_connect(str(tmp_path / "partial.sqlite")) as reader:
        assert reader.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
