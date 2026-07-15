"""Tests for rosclaw.storage.outbox."""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from unittest.mock import MagicMock
from urllib.request import Request, urlopen

import pytest

from rosclaw.storage.outbox import OutboxRecord, OutboxStore, OutboxWorker


@pytest.fixture
def outbox(tmp_path: Path) -> OutboxStore:
    store = OutboxStore(db_path=str(tmp_path / "outbox.sqlite"))
    store.connect()
    return store


def test_enqueue_and_pending(outbox: OutboxStore) -> None:
    rid = outbox.enqueue("seekdb_http", {"event": "test"})
    pending = outbox.pending(limit=10)
    assert len(pending) == 1
    assert pending[0].id == rid
    assert pending[0].payload == {"event": "test"}


def test_mark_delivered(outbox: OutboxStore) -> None:
    rid = outbox.enqueue("seekdb_http", {"event": "test"})
    outbox.mark_delivered(rid)
    assert outbox.pending(limit=10) == []
    assert outbox.stats()["total"] == 0


def test_mark_failed_backoff(outbox: OutboxStore) -> None:
    rid = outbox.enqueue("seekdb_http", {"event": "test"})
    outbox.mark_failed(rid, "timeout")
    # Immediately after failure, next_retry_at is in the future.
    assert outbox.pending(limit=10) == []
    rows = outbox.records()
    assert len(rows) == 1
    assert rows[0].retry_count == 1
    assert rows[0].next_retry_at is not None
    assert rows[0].next_retry_at > time.time()
    assert rows[0].error_log == "timeout"


def test_mark_failed_moves_to_dead_letters_when_exhausted(outbox: OutboxStore) -> None:
    rid = outbox.enqueue("seekdb_http", {"event": "test"})
    # Simulate max_retries exhausted.
    outbox.mark_failed(rid, "err1", max_retries=1)
    assert outbox.stats()["total"] == 0
    assert outbox.stats()["dead_letters"] == 1
    dead = outbox.dead_letters()
    assert len(dead) == 1
    assert dead[0].id == rid
    assert dead[0].retry_count == 1
    assert dead[0].error_log == "err1"


def test_capacity_drops_oldest(outbox: OutboxStore) -> None:
    outbox._max_records = 2
    first = outbox.enqueue("seekdb_http", {"seq": 1})
    time.sleep(0.01)
    second = outbox.enqueue("seekdb_http", {"seq": 2})
    time.sleep(0.01)
    third = outbox.enqueue("seekdb_http", {"seq": 3})
    stats = outbox.stats()
    assert stats["total"] == 2
    ids = {r.id for r in outbox.pending(limit=10)}
    assert first not in ids
    assert second in ids
    assert third in ids


def test_stats_oldest_pending_age(outbox: OutboxStore) -> None:
    outbox.enqueue("seekdb_http", {"event": "a"})
    time.sleep(0.05)
    stats = outbox.stats()
    assert stats["pending"] == 1
    assert stats["oldest_pending_sec"] is not None
    assert stats["oldest_pending_sec"] >= 0.04


def test_worker_drains_successfully(outbox: OutboxStore) -> None:
    committer = MagicMock()
    committer.save_to_seekdb_batch = None
    worker = OutboxWorker(outbox, committer, interval_sec=0.05, batch_size=10)
    outbox.enqueue("seekdb_http", {"event": "a"})
    outbox.enqueue("seekdb_http", {"event": "b"})
    worker.start()
    # Wait for worker to drain.
    for _ in range(100):
        if outbox.stats()["total"] == 0:
            break
        time.sleep(0.01)
    worker.stop()
    assert outbox.stats()["total"] == 0
    assert committer.save_to_seekdb.call_count == 2


def test_worker_retries_failed_commits(outbox: OutboxStore) -> None:
    committer = MagicMock()
    committer.save_to_seekdb_batch = None
    committer.save_to_seekdb.side_effect = RuntimeError("upstream down")
    worker = OutboxWorker(outbox, committer, interval_sec=0.05, batch_size=10, max_retries=2)
    outbox.enqueue("seekdb_http", {"event": "a"})
    worker.start()
    # Wait for the worker to exhaust retries.
    for _ in range(300):
        if committer.save_to_seekdb.call_count >= 2:
            break
        time.sleep(0.01)
    worker.stop()
    # Exhausted record should be in dead letters, not outbox.
    assert outbox.stats()["total"] == 0
    assert outbox.stats()["dead_letters"] == 1
    assert committer.save_to_seekdb.call_count >= 2
    dead = outbox.dead_letters()
    assert len(dead) == 1
    assert dead[0].retry_count >= 2
    assert "upstream down" in (dead[0].error_log or "")


def test_worker_batch_commit_uses_batch_method(outbox: OutboxStore) -> None:
    committer = MagicMock()
    committer.save_to_seekdb_batch = MagicMock()
    worker = OutboxWorker(outbox, committer, interval_sec=60.0, batch_size=10)
    outbox.enqueue("seekdb_http", {"event": "a"})
    outbox.enqueue("seekdb_http", {"event": "b"})
    drained = worker._drain_once()
    assert drained == 2
    assert outbox.stats()["total"] == 0
    committer.save_to_seekdb_batch.assert_called_once()
    payloads = committer.save_to_seekdb_batch.call_args[0][0]
    assert len(payloads) == 2
    committer.save_to_seekdb.assert_not_called()
    worker.stop()


def test_worker_batch_commit_falls_back_to_per_record(outbox: OutboxStore) -> None:
    committer = MagicMock()
    committer.save_to_seekdb_batch.side_effect = RuntimeError("batch rejected")
    worker = OutboxWorker(outbox, committer, interval_sec=60.0, batch_size=10)
    outbox.enqueue("seekdb_http", {"event": "a"})
    outbox.enqueue("seekdb_http", {"event": "b"})
    drained = worker._drain_once()
    assert drained == 2
    assert outbox.stats()["total"] == 0
    committer.save_to_seekdb_batch.assert_called_once()
    assert committer.save_to_seekdb.call_count == 2
    worker.stop()


def test_worker_flush_drains_ready_records(outbox: OutboxStore) -> None:
    committer = MagicMock()
    committer.save_to_seekdb_batch = None
    worker = OutboxWorker(outbox, committer, interval_sec=60.0, batch_size=10)
    outbox.enqueue("seekdb_http", {"event": "a"})
    # flush() can drain even before the worker thread is started.
    flushed = worker.flush()
    worker.stop()
    assert flushed == 1
    assert outbox.stats()["total"] == 0


def test_worker_stops_gracefully(outbox: OutboxStore) -> None:
    committer = MagicMock()
    worker = OutboxWorker(outbox, committer, interval_sec=60.0)
    worker.start()
    assert worker._thread is not None and worker._thread.is_alive()
    worker.stop(timeout=1.0)
    assert worker._thread is None or not worker._thread.is_alive()


def test_outbox_record_to_dict() -> None:
    record = OutboxRecord(
        id="r1",
        target="seekdb_http",
        payload={"x": 1},
        created_at=1.0,
        retry_count=0,
        next_retry_at=None,
        error_log=None,
    )
    d = record.to_dict()
    assert d["id"] == "r1"
    assert json.loads(d["payload_json"]) == {"x": 1}


class _RecordingHTTPHandler(BaseHTTPRequestHandler):
    """Simple handler that records JSON POST bodies and returns 200/500."""

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        state = self.server.state
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8"))
        state["received"].append(payload)
        if state["fail_next"]:
            state["fail_next"] = False
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()


@pytest.fixture
def http_committer(tmp_path: Path):
    """Start a local HTTP server and yield a committer that posts to it."""

    class _HTTPCommitter:
        def __init__(self, url: str) -> None:
            self.url = url

        def save_to_seekdb(self, payload: dict[str, Any]) -> None:
            req = Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=2.0) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"unexpected status {resp.status}")

        def save_to_seekdb_batch(self, payloads: list[dict[str, Any]]) -> None:
            req = Request(
                self.url,
                data=json.dumps(payloads).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=2.0) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"unexpected status {resp.status}")

    state: dict[str, Any] = {"received": [], "fail_next": False}
    server = HTTPServer(("127.0.0.1", 0), _RecordingHTTPHandler)
    server.state = state
    port = server.server_address[1]
    committer = _HTTPCommitter(f"http://127.0.0.1:{port}")
    committer.state = state
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield committer
    finally:
        server.shutdown()
        server.server_close()


def test_worker_drains_to_http_endpoint_per_record(outbox: OutboxStore, http_committer) -> None:
    committer = http_committer
    committer.save_to_seekdb_batch = None  # force per-record path
    worker = OutboxWorker(outbox, committer, interval_sec=60.0, batch_size=10)
    outbox.enqueue("seekdb_http", {"event": "a"})
    outbox.enqueue("seekdb_http", {"event": "b"})
    drained = worker.flush()
    worker.stop()
    assert drained == 2
    assert outbox.stats()["total"] == 0
    assert len(http_committer.state["received"]) == 2
    assert {"event": "a"} in http_committer.state["received"]
    assert {"event": "b"} in http_committer.state["received"]


def test_worker_drains_to_http_endpoint_in_batches(outbox: OutboxStore, http_committer) -> None:
    committer = http_committer
    worker = OutboxWorker(outbox, committer, interval_sec=60.0, batch_size=10)
    outbox.enqueue("seekdb_http", {"event": "a"})
    outbox.enqueue("seekdb_http", {"event": "b"})
    drained = worker._drain_once()
    worker.stop()
    assert drained == 2
    assert outbox.stats()["total"] == 0
    assert len(http_committer.state["received"]) == 1
    payloads = http_committer.state["received"][0]
    assert isinstance(payloads, list)
    assert {"event": "a"} in payloads
    assert {"event": "b"} in payloads


def test_worker_http_failure_retries_and_dead_letters(outbox: OutboxStore, http_committer) -> None:
    committer = http_committer
    http_committer.state["fail_next"] = True
    worker = OutboxWorker(outbox, committer, interval_sec=0.05, batch_size=10, max_retries=1)
    outbox.enqueue("seekdb_http", {"event": "fail_once"})
    worker.start()
    for _ in range(300):
        if outbox.stats()["dead_letters"] == 1:
            break
        time.sleep(0.01)
    worker.stop()
    assert outbox.stats()["total"] == 0
    assert outbox.stats()["dead_letters"] == 1
