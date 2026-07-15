"""Local outbox for asynchronous upstream persistence.

The outbox decouples event producers (e.g. :class:`SeekDBBridge`) from the
availability of the remote SeekDB HTTP adapter.  Events are written to a local
SQLite outbox synchronously; a background worker drains the outbox by calling
the upstream committer.  If the upstream is down, records accumulate with
exponential backoff until they succeed or exceed ``max_retries`` and are moved
to the dead-letter table.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.storage.outbox")


@dataclass
class OutboxRecord:
    """A single pending upstream commit record."""

    id: str
    target: str
    payload: dict[str, Any]
    created_at: float
    retry_count: int = 0
    next_retry_at: float | None = None
    error_log: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target": self.target,
            "payload_json": json.dumps(self.payload),
            "created_at": self.created_at,
            "retry_count": self.retry_count,
            "next_retry_at": self.next_retry_at,
            "error_log": self.error_log,
        }


class OutboxStore:
    """SQLite-backed durable outbox for upstream events."""

    def __init__(
        self,
        db_path: str = "~/.rosclaw/storage/outbox.sqlite",
        max_records: int = 100_000,
    ):
        self._db_path = db_path
        self._max_records = max_records
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def connect(self) -> None:
        with self._lock:
            if self._conn is not None:
                return
            path = Path(self._db_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._create_tables()

    @property
    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def _create_tables(self) -> None:
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS outbox (
                    id TEXT PRIMARY KEY,
                    target TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    next_retry_at REAL,
                    error_log TEXT
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_outbox_next_retry ON outbox(next_retry_at)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_outbox_created_at ON outbox(created_at)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS outbox_dead_letters (
                    id TEXT PRIMARY KEY,
                    target TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    retry_count INTEGER NOT NULL,
                    failed_at REAL NOT NULL,
                    error_log TEXT
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_dead_letters_created_at ON outbox_dead_letters(created_at)"
            )
            self._connection.commit()

    def enqueue(self, target: str, payload: dict[str, Any]) -> str:
        """Add *payload* to the outbox for *target*.  Drops oldest record if full."""
        record_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            # Enforce capacity by dropping the oldest record(s).
            count = self._connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
            if count >= self._max_records:
                oldest = self._connection.execute(
                    "SELECT id FROM outbox ORDER BY created_at ASC LIMIT 1"
                ).fetchone()
                if oldest:
                    self._connection.execute("DELETE FROM outbox WHERE id = ?", (oldest["id"],))
                    logger.warning(
                        "Outbox capacity exceeded; dropped oldest record %s", oldest["id"]
                    )
            self._connection.execute(
                """
                INSERT INTO outbox (id, target, payload_json, created_at, retry_count, next_retry_at)
                VALUES (?, ?, ?, ?, 0, ?)
                """,
                (record_id, target, json.dumps(payload), now, now),
            )
            self._connection.commit()
        return record_id

    def pending(self, limit: int = 100, max_retries: int = 10) -> list[OutboxRecord]:
        """Return records that are ready for retry."""
        now = time.time()
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, target, payload_json, created_at, retry_count, next_retry_at, error_log
                FROM outbox
                WHERE (next_retry_at IS NULL OR next_retry_at <= ?)
                  AND retry_count < ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (now, max_retries, limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def mark_delivered(self, record_id: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM outbox WHERE id = ?", (record_id,))
            self._connection.commit()

    def mark_failed(
        self, record_id: str, error: str | None = None, *, max_retries: int = 10
    ) -> None:
        now = time.time()
        with self._lock:
            row = self._connection.execute(
                "SELECT target, payload_json, created_at, retry_count FROM outbox WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                return
            retry_count = row["retry_count"] + 1
            if retry_count >= max_retries:
                # Exhausted: move to dead letters so the queue does not retry forever.
                self._connection.execute(
                    """
                    INSERT INTO outbox_dead_letters
                        (id, target, payload_json, created_at, retry_count, failed_at, error_log)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id,
                        row["target"],
                        row["payload_json"],
                        row["created_at"],
                        retry_count,
                        now,
                        error,
                    ),
                )
                self._connection.execute("DELETE FROM outbox WHERE id = ?", (record_id,))
            else:
                # Exponential backoff capped at 5 minutes.
                backoff = min(2**retry_count, 300)
                next_retry = now + backoff
                self._connection.execute(
                    """
                    UPDATE outbox
                    SET retry_count = ?, next_retry_at = ?, error_log = ?
                    WHERE id = ?
                    """,
                    (retry_count, next_retry, error, record_id),
                )
            self._connection.commit()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
            pending = self._connection.execute(
                "SELECT COUNT(*) FROM outbox WHERE next_retry_at IS NULL OR next_retry_at <= ?",
                (time.time(),),
            ).fetchone()[0]
            failed = self._connection.execute(
                "SELECT COUNT(*) FROM outbox WHERE retry_count > 0"
            ).fetchone()[0]
            dead_letters = self._connection.execute(
                "SELECT COUNT(*) FROM outbox_dead_letters"
            ).fetchone()[0]
            oldest_pending_row = self._connection.execute(
                """
                SELECT MIN(created_at) FROM outbox
                WHERE next_retry_at IS NULL OR next_retry_at <= ?
                """,
                (time.time(),),
            ).fetchone()
            oldest_created = oldest_pending_row[0] if oldest_pending_row else None
        oldest_pending_sec: float | None = None
        if oldest_created is not None:
            oldest_pending_sec = round(time.time() - oldest_created, 3)
        return {
            "total": total,
            "pending": pending,
            "failed": failed,
            "dead_letters": dead_letters,
            "oldest_pending_sec": oldest_pending_sec,
        }

    def records(self, limit: int | None = None) -> list[OutboxRecord]:
        """Return all records (ignoring backoff), newest first."""
        with self._lock:
            sql = "SELECT * FROM outbox ORDER BY created_at DESC"
            params: tuple = ()
            if limit is not None:
                sql += " LIMIT ?"
                params = (limit,)
            rows = self._connection.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def dead_letters(self, limit: int | None = None) -> list[OutboxRecord]:
        """Return exhausted records moved to the dead-letter table, newest first."""
        with self._lock:
            sql = "SELECT * FROM outbox_dead_letters ORDER BY failed_at DESC"
            params: tuple = ()
            if limit is not None:
                sql += " LIMIT ?"
                params = (limit,)
            rows = self._connection.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: sqlite3.Row) -> OutboxRecord:
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            payload = {}
        # Dead-letter rows do not have next_retry_at; normalize absent columns to None.
        keys = row.keys()
        return OutboxRecord(
            id=row["id"],
            target=row["target"],
            payload=payload,
            created_at=row["created_at"],
            retry_count=row["retry_count"],
            next_retry_at=row["next_retry_at"] if "next_retry_at" in keys else None,
            error_log=row["error_log"] if "error_log" in keys else None,
        )

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


class OutboxWorker:
    """Background worker that drains an :class:`OutboxStore` through a committer."""

    def __init__(
        self,
        outbox: OutboxStore,
        committer: Any,
        *,
        interval_sec: float = 5.0,
        batch_size: int = 100,
        max_retries: int = 10,
        name: str = "outbox-worker",
    ):
        self._outbox = outbox
        self._committer = committer
        self._interval_sec = interval_sec
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._name = name
        self._drain_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=self._name)
        self._thread.start()
        logger.info("Started %s", self._name)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("%s did not stop within %ss", self._name, timeout)
        self._thread = None

    def flush(self, timeout: float | None = None) -> int:
        """Drain all ready records immediately.

        Called before :meth:`stop` during graceful shutdown.  Returns the
        number of records delivered or moved to dead letters.
        """
        deadline = time.time() + timeout if timeout is not None else None
        total = 0
        while not self._stop.is_set():
            if deadline is not None and time.time() >= deadline:
                break
            try:
                with self._drain_lock:
                    drained = self._drain_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s flush cycle failed: %s", self._name, exc)
                drained = 0
            total += drained
            if drained == 0:
                break
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._drain_lock:
                    drained = self._drain_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s drain cycle failed: %s", self._name, exc)
                drained = 0
            # Sleep longer when idle, but still wake quickly for stop.
            sleep_for = 0.2 if self._stop.is_set() else self._interval_sec
            if drained == 0 and not self._stop.is_set():
                self._stop.wait(timeout=sleep_for)

    def _drain_once(self) -> int:
        records = self._outbox.pending(self._batch_size, self._max_retries)
        if not records:
            return 0

        batch_commit = getattr(self._committer, "save_to_seekdb_batch", None)
        if batch_commit is not None and len(records) > 1:
            try:
                batch_commit([record.payload for record in records])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Batch commit failed for %s record(s) to %s: %s; falling back to per-record",
                    len(records),
                    records[0].target,
                    exc,
                )
                batch_commit = None
            else:
                for record in records:
                    self._outbox.mark_delivered(record.id)
                return len(records)

        for record in records:
            try:
                self._committer.save_to_seekdb(record.payload)
                self._outbox.mark_delivered(record.id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to commit outbox record %s to %s: %s",
                    record.id,
                    record.target,
                    exc,
                )
                self._outbox.mark_failed(record.id, str(exc), max_retries=self._max_retries)
        return len(records)
