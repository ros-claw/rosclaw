"""Tests for PracticeCatalog batched event/event-index writes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from rosclaw.practice.storage.catalog import PracticeCatalog


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
    return PracticeCatalog(tmp_path / "catalog.sqlite")


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
