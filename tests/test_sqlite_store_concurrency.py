"""SQLiteKnowledgeStore concurrency stress test.

Verifies that a single store instance shared across writer, reader, and
preloader threads does not raise ``ProgrammingError`` or lose updates.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path

import pytest

from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore


@pytest.fixture
def fresh_sqlite(tmp_path: Path) -> str:
    """Return a fresh SQLite path, cleaning up any prior file."""
    db_path = tmp_path / "concurrency.sqlite"
    # Clean up if a previous run left the file behind.
    for suffix in ("", "-wal", "-shm"):
        with contextlib.suppress(FileNotFoundError):
            os.remove(str(db_path) + suffix)
    return str(db_path)


def _writer(store: SQLiteKnowledgeStore, robot_id: str, count: int, errors: list) -> None:
    for i in range(count):
        try:
            store.insert(
                "experience_graph",
                {
                    "id": f"{robot_id}_{i}",
                    "event_type": "praxis",
                    "robot_id": robot_id,
                    "timestamp": time.time(),
                    "instruction": f"action {i}",
                    "outcome": "SUCCESS" if i % 2 == 0 else "FAILED",
                },
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(("writer", robot_id, i, exc))


def _reader(
    store: SQLiteKnowledgeStore, robot_ids: list[str], stop_event: threading.Event, errors: list
) -> None:
    while not stop_event.is_set():
        try:
            store.query(
                "experience_graph",
                filters={"robot_id": robot_ids[len(robot_ids) // 2]},
                order_by="-timestamp",
                limit=10,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(("reader", exc))
        time.sleep(0.001)


def _preloader(store: SQLiteKnowledgeStore, errors: list) -> None:
    """Simulate a background preloader that repeatedly connects and counts."""
    try:
        store.connect()
        store.count("experience_graph", {})
        store.query("experience_graph", limit=1)
    except Exception as exc:  # noqa: BLE001
        errors.append(("preloader", exc))


def test_sqlite_store_concurrent_writers_readers_preloader(fresh_sqlite: str) -> None:
    store = SQLiteKnowledgeStore(fresh_sqlite)
    store.connect()

    errors: list = []
    writer_count = 8
    records_per_writer = 250
    robot_ids = [f"robot_{i}" for i in range(writer_count)]

    stop_readers = threading.Event()
    reader_threads = [
        threading.Thread(target=_reader, args=(store, robot_ids, stop_readers, errors))
        for _ in range(8)
    ]
    for t in reader_threads:
        t.start()

    writer_threads = [
        threading.Thread(target=_writer, args=(store, robot_ids[i], records_per_writer, errors))
        for i in range(writer_count)
    ]
    for t in writer_threads:
        t.start()

    # Preloader thread that periodically reconnects/queries.
    preloader_threads = [
        threading.Thread(target=_preloader, args=(store, errors)) for _ in range(1)
    ]
    for t in preloader_threads:
        t.start()

    for t in writer_threads:
        t.join()

    stop_readers.set()
    for t in reader_threads:
        t.join()

    for t in preloader_threads:
        t.join()

    assert not errors, f"Concurrent operations raised exceptions: {errors[:10]}"

    total = store.count("experience_graph", {})
    assert total == writer_count * records_per_writer

    for i in range(writer_count):
        count = store.count("experience_graph", {"robot_id": robot_ids[i]})
        assert count == records_per_writer


def test_sqlite_store_upsert_no_lost_update(fresh_sqlite: str) -> None:
    """Concurrent inserts with the same id must not raise or duplicate rows."""
    store = SQLiteKnowledgeStore(fresh_sqlite)
    store.connect()

    errors: list = []
    threads: list[threading.Thread] = []

    def _upsert(i: int) -> None:
        try:
            store.insert(
                "experience_graph",
                {
                    "id": "shared",
                    "event_type": "praxis",
                    "robot_id": f"r{i}",
                    "timestamp": float(i),
                    "instruction": f"action {i}",
                    "outcome": "SUCCESS",
                },
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    for i in range(20):
        threads.append(threading.Thread(target=_upsert, args=(i,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Upserts raised exceptions: {errors[:10]}"
    rows = store.query("experience_graph", filters={"id": "shared"})
    assert len(rows) == 1
    assert rows[0]["robot_id"].startswith("r")
