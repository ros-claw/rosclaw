"""Local outbox for asynchronous upstream persistence.

The outbox decouples event producers (e.g. :class:`SeekDBBridge`) from the
availability of the remote SeekDB HTTP adapter.  Events are written to a local
SQLite outbox synchronously; background workers drain the outbox by calling
the upstream committer.  If the upstream is down, records accumulate with
exponential backoff until they succeed or exceed ``max_retries`` and are moved
to the dead-letter table.

Delivery semantics (PR-DB-1):

* **At-least-once delivery** — records are claimed under a lease, only marked
  ``delivered`` after the remote commit succeeded, and reclaimed when a worker
  crashes mid-flight (lease expiry).
* **Idempotent enqueue** — every record carries an ``idempotency_key`` with a
  partial unique index; re-enqueueing the same logical entity returns the
  existing record instead of duplicating it.
* **Idempotent remote upsert** — the worker injects the key into the payload
  before handing it to the committer, so the remote side can upsert by key.
  At-least-once + idempotent upsert = effectively once; never advertised as
  strict exactly-once.

State machine::

    pending ──claim──▶ inflight ──commit ok──▶ delivered (retained, then purged)
    pending/inflight ──failure──▶ retry ──claim──▶ inflight
    retry over limit ──▶ outbox_dead_letters
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.storage.outbox")

_ACTIVE_STATUSES = ("pending", "retry", "inflight")
_ACTIVE_PLACEHOLDERS = ", ".join("?" for _ in _ACTIVE_STATUSES)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON encoding of *payload*."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    idempotency_key: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    payload_sha256: str | None = None
    status: str = "pending"
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    updated_at: float | None = None
    remote_revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target": self.target,
            "payload_json": json.dumps(self.payload),
            "created_at": self.created_at,
            "retry_count": self.retry_count,
            "next_retry_at": self.next_retry_at,
            "error_log": self.error_log,
            "idempotency_key": self.idempotency_key,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "payload_sha256": self.payload_sha256,
            "status": self.status,
            "lease_owner": self.lease_owner,
            "lease_expires_at": self.lease_expires_at,
            "updated_at": self.updated_at,
            "remote_revision": self.remote_revision,
        }


class OutboxStore:
    """SQLite-backed durable outbox for upstream events.

    Schema v2 adds idempotency keys, a pending/inflight/retry/delivered
    status machine, and lease-based multi-worker claiming.  Existing v1
    databases are migrated in place via ``ALTER TABLE ADD COLUMN``.
    """

    _NEW_OUTBOX_COLUMNS = {
        "idempotency_key": "TEXT",
        "entity_type": "TEXT",
        "entity_id": "TEXT",
        "payload_sha256": "TEXT",
        "status": "TEXT DEFAULT 'pending'",
        "lease_owner": "TEXT",
        "lease_expires_at": "REAL",
        "updated_at": "REAL",
        "remote_revision": "TEXT",
    }

    _NEW_DEAD_LETTER_COLUMNS = {
        "idempotency_key": "TEXT",
        "entity_type": "TEXT",
        "entity_id": "TEXT",
        "payload_sha256": "TEXT",
    }

    def __init__(
        self,
        db_path: str = "~/.rosclaw/storage/outbox.sqlite",
        max_records: int = 100_000,
        delivered_retention_sec: float = 3600.0,
    ):
        if max_records < 1:
            raise ValueError("max_records must be at least 1")
        self._db_path = db_path
        self._max_records = max_records
        self._delivered_retention_sec = delivered_retention_sec
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
            self._migrate_columns("outbox", self._NEW_OUTBOX_COLUMNS)
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_outbox_next_retry ON outbox(next_retry_at)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_outbox_created_at ON outbox(created_at)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_outbox_status_retry ON outbox(status, next_retry_at)"
            )
            # Partial unique index: NULL keys (legacy rows) are ignored, every
            # real idempotency key is claimable exactly once.
            self._connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_idempotency
                ON outbox(idempotency_key) WHERE idempotency_key IS NOT NULL
                """
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
            self._migrate_columns("outbox_dead_letters", self._NEW_DEAD_LETTER_COLUMNS)
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_dead_letters_created_at ON outbox_dead_letters(created_at)"
            )
            # Normalize legacy rows that predate the status column.
            self._connection.execute(
                "UPDATE outbox SET status = CASE WHEN retry_count > 0 THEN 'retry' ELSE 'pending' END "
                "WHERE status IS NULL"
            )
            self._connection.commit()

    def _migrate_columns(self, table: str, columns: dict[str, str]) -> None:
        existing = {row[1] for row in self._connection.execute(f"PRAGMA table_info({table})")}
        for name, ddl in columns.items():
            if name not in existing:
                logger.info("Migrating outbox table %s: adding column %s", table, name)
                self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(
        self,
        target: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> str:
        """Add *payload* to the outbox for *target*.

        Capacity exhaustion is reported to the producer instead of deleting an
        older, unacknowledged record. This preserves the durable-outbox
        guarantee and lets the caller apply backpressure or stop safely.

        Enqueue is idempotent on ``idempotency_key``: enqueueing the same
        logical entity twice returns the existing record id instead of
        creating a duplicate.  When no key is supplied one is derived as
        ``<entity_type|record>:<entity_id|uuid>:<payload-hash-prefix>``.
        """
        record_id = str(uuid.uuid4())
        now = time.time()
        payload_json = json.dumps(payload)
        payload_sha = _canonical_sha256(payload)
        if idempotency_key is None:
            idempotency_key = (
                f"{entity_type or 'record'}:{entity_id or record_id}:{payload_sha[:16]}"
            )
        with self._lock:
            count = self._connection.execute(
                f"SELECT COUNT(*) FROM outbox WHERE status IN ({_ACTIVE_PLACEHOLDERS})",
                _ACTIVE_STATUSES,
            ).fetchone()[0]
            if count >= self._max_records:
                raise OverflowError(
                    f"Outbox capacity exhausted ({self._max_records} records); "
                    "refusing to discard unacknowledged data"
                )
            try:
                self._connection.execute(
                    """
                    INSERT INTO outbox (
                        id, target, payload_json, created_at, retry_count, next_retry_at,
                        idempotency_key, entity_type, entity_id, payload_sha256,
                        status, updated_at
                    )
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        record_id,
                        target,
                        payload_json,
                        now,
                        now,
                        idempotency_key,
                        entity_type,
                        entity_id,
                        payload_sha,
                        now,
                    ),
                )
                self._connection.commit()
            except sqlite3.IntegrityError:
                self._connection.rollback()
                existing = self._connection.execute(
                    "SELECT id FROM outbox WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    logger.info(
                        "Outbox enqueue deduplicated by idempotency key %s (existing %s)",
                        idempotency_key,
                        existing["id"],
                    )
                    return str(existing["id"])
                raise
        return record_id

    # ------------------------------------------------------------------
    # Claim / delivery lifecycle
    # ------------------------------------------------------------------

    def claim(
        self,
        limit: int = 100,
        *,
        owner: str,
        lease_sec: float = 30.0,
        max_retries: int = 10,
    ) -> list[OutboxRecord]:
        """Atomically claim up to *limit* ready records under a lease.

        A record is claimable when it is ``pending``/``retry``, due for
        delivery, and under the retry cap — or when it is ``inflight`` with an
        expired lease (its previous worker crashed).  The UPDATE re-checks the
        predicate inside the same transaction so two workers can never claim
        the same record.  Claimed records whose stored payload no longer
        matches its SHA-256 are moved straight to dead letters (corrupt
        payloads must never be uploaded as ``{}``).
        """
        now = time.time()
        lease_expires = now + lease_sec
        with self._lock:
            candidates = self._connection.execute(
                """
                SELECT id FROM outbox
                WHERE (
                        (status IN ('pending', 'retry'))
                        AND (next_retry_at IS NULL OR next_retry_at <= ?)
                        AND retry_count < ?
                      )
                   OR (status = 'inflight' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?)
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (now, max_retries, now, limit),
            ).fetchall()
            candidate_ids = [row["id"] for row in candidates]
            if not candidate_ids:
                return []
            placeholders = ", ".join("?" for _ in candidate_ids)
            self._connection.execute(
                f"""
                UPDATE outbox
                SET status = 'inflight', lease_owner = ?, lease_expires_at = ?, updated_at = ?
                WHERE id IN ({placeholders})
                  AND (
                        ((status IN ('pending', 'retry'))
                         AND (next_retry_at IS NULL OR next_retry_at <= ?)
                         AND retry_count < ?)
                     OR (status = 'inflight' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?)
                      )
                """,
                (owner, lease_expires, now, *candidate_ids, now, max_retries, now),
            )
            self._connection.commit()
            rows = self._connection.execute(
                f"SELECT * FROM outbox WHERE lease_owner = ? AND status = 'inflight' "
                f"AND id IN ({placeholders}) ORDER BY created_at ASC",
                (owner, *candidate_ids),
            ).fetchall()

        records: list[OutboxRecord] = []
        for row in rows:
            record = self._row_to_record(row)
            stored_sha = (
                row["payload_sha256"] if "payload_sha256" in row.keys() else None  # noqa: SIM118
            )
            if stored_sha and not self._payload_hash_matches(row["payload_json"], stored_sha):
                logger.error(
                    "Outbox record %s payload checksum mismatch; moving to dead letters",
                    record.id,
                )
                self._move_to_dead_letters(
                    row, retry_count=row["retry_count"], error="payload checksum mismatch"
                )
                continue
            records.append(record)
        return records

    @staticmethod
    def _payload_hash_matches(payload_json: str, stored_sha: str) -> bool:
        """Compare the raw stored JSON against the canonical-form SHA.

        The enqueue hash is computed over the canonical (sorted-key) JSON
        while the column stores the original serialization, so verification
        re-parses and re-hashes the canonical form.  A payload that cannot be
        parsed or re-hashed is treated as corrupt.
        """
        try:
            payload = json.loads(payload_json)
        except (json.JSONDecodeError, TypeError):
            return False
        try:
            return _canonical_sha256(payload) == stored_sha
        except (TypeError, ValueError):
            return False

    def _move_to_dead_letters(
        self, row: sqlite3.Row, *, retry_count: int, error: str | None
    ) -> None:
        now = time.time()
        keys = row.keys()
        self._connection.execute(
            """
            INSERT OR REPLACE INTO outbox_dead_letters
                (id, target, payload_json, created_at, retry_count, failed_at, error_log,
                 idempotency_key, entity_type, entity_id, payload_sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["target"],
                row["payload_json"],
                row["created_at"],
                retry_count,
                now,
                error,
                row["idempotency_key"] if "idempotency_key" in keys else None,
                row["entity_type"] if "entity_type" in keys else None,
                row["entity_id"] if "entity_id" in keys else None,
                row["payload_sha256"] if "payload_sha256" in keys else None,
            ),
        )
        self._connection.execute("DELETE FROM outbox WHERE id = ?", (row["id"],))
        self._connection.commit()

    def pending(self, limit: int = 100, max_retries: int = 10) -> list[OutboxRecord]:
        """Return records that are ready for retry (read-only, no lease).

        Kept for inspection and backward compatibility; workers use
        :meth:`claim` so concurrent drainers cannot double-deliver.
        """
        now = time.time()
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM outbox
                WHERE status IN ('pending', 'retry')
                  AND (next_retry_at IS NULL OR next_retry_at <= ?)
                  AND retry_count < ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (now, max_retries, limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def mark_delivered(
        self,
        record_id: str,
        *,
        owner: str | None = None,
        remote_revision: str | None = None,
    ) -> None:
        """Mark a record delivered after the remote commit succeeded.

        Delivered rows are retained for ``delivered_retention_sec`` so the
        delivery is observable and reconcilable, then purged.  When *owner*
        is given, only the lease owner may mark delivery.
        """
        now = time.time()
        with self._lock:
            if owner is not None:
                cursor = self._connection.execute(
                    """
                    UPDATE outbox
                    SET status = 'delivered', lease_owner = NULL, lease_expires_at = NULL,
                        updated_at = ?, remote_revision = ?
                    WHERE id = ? AND lease_owner = ?
                    """,
                    (now, remote_revision, record_id, owner),
                )
                if cursor.rowcount == 0:
                    logger.warning(
                        "mark_delivered(%s) ignored: lease not owned by %s", record_id, owner
                    )
                    return
            else:
                self._connection.execute(
                    """
                    UPDATE outbox
                    SET status = 'delivered', lease_owner = NULL, lease_expires_at = NULL,
                        updated_at = ?, remote_revision = ?
                    WHERE id = ?
                    """,
                    (now, remote_revision, record_id),
                )
            self._connection.commit()

    def purge_delivered(self, retention_sec: float | None = None) -> int:
        """Delete delivered rows older than the retention window."""
        retention = self._delivered_retention_sec if retention_sec is None else retention_sec
        cutoff = time.time() - retention
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM outbox WHERE status = 'delivered' AND updated_at < ?",
                (cutoff,),
            )
            self._connection.commit()
        return cursor.rowcount

    def mark_failed(
        self,
        record_id: str,
        error: str | None = None,
        *,
        owner: str | None = None,
        max_retries: int = 10,
    ) -> None:
        now = time.time()
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM outbox WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                return
            if owner is not None and row["lease_owner"] not in (None, owner):
                logger.warning(
                    "mark_failed(%s) ignored: lease owned by %s, not %s",
                    record_id,
                    row["lease_owner"],
                    owner,
                )
                return
            retry_count = row["retry_count"] + 1
            if retry_count >= max_retries:
                # Exhausted: move to dead letters so the queue does not retry forever.
                self._move_to_dead_letters(row, retry_count=retry_count, error=error)
            else:
                # Exponential backoff capped at 5 minutes.
                backoff = min(2**retry_count, 300)
                next_retry = now + backoff
                self._connection.execute(
                    """
                    UPDATE outbox
                    SET retry_count = ?, next_retry_at = ?, error_log = ?,
                        status = 'retry', lease_owner = NULL, lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (retry_count, next_retry, error, now, record_id),
                )
                self._connection.commit()

    def requeue_dead_letter(self, record_id: str) -> bool:
        """Move a dead-letter record back to ``pending`` for another attempt."""
        now = time.time()
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM outbox_dead_letters WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                return False
            keys = row.keys()
            try:
                self._connection.execute(
                    """
                    INSERT INTO outbox (
                        id, target, payload_json, created_at, retry_count, next_retry_at,
                        idempotency_key, entity_type, entity_id, payload_sha256,
                        status, updated_at
                    )
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        row["id"],
                        row["target"],
                        row["payload_json"],
                        row["created_at"],
                        now,
                        row["idempotency_key"] if "idempotency_key" in keys else None,
                        row["entity_type"] if "entity_type" in keys else None,
                        row["entity_id"] if "entity_id" in keys else None,
                        row["payload_sha256"] if "payload_sha256" in keys else None,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                self._connection.rollback()
                logger.warning(
                    "requeue_dead_letter(%s) skipped: id already present in outbox", record_id
                )
                return False
            self._connection.execute(
                "DELETE FROM outbox_dead_letters WHERE id = ?",
                (record_id,),
            )
            self._connection.commit()
        return True

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            total = self._connection.execute(
                f"SELECT COUNT(*) FROM outbox WHERE status IN ({_ACTIVE_PLACEHOLDERS})",
                _ACTIVE_STATUSES,
            ).fetchone()[0]
            pending = self._connection.execute(
                """
                SELECT COUNT(*) FROM outbox
                WHERE status IN ('pending', 'retry')
                  AND (next_retry_at IS NULL OR next_retry_at <= ?)
                """,
                (now,),
            ).fetchone()[0]
            inflight = self._connection.execute(
                "SELECT COUNT(*) FROM outbox WHERE status = 'inflight'"
            ).fetchone()[0]
            retry = self._connection.execute(
                "SELECT COUNT(*) FROM outbox WHERE status = 'retry'"
            ).fetchone()[0]
            delivered = self._connection.execute(
                "SELECT COUNT(*) FROM outbox WHERE status = 'delivered'"
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
                WHERE status IN ('pending', 'retry')
                  AND (next_retry_at IS NULL OR next_retry_at <= ?)
                """,
                (now,),
            ).fetchone()
            oldest_created = oldest_pending_row[0] if oldest_pending_row else None
        oldest_pending_sec: float | None = None
        if oldest_created is not None:
            oldest_pending_sec = round(now - oldest_created, 3)
        return {
            "total": total,
            "pending": pending,
            "inflight": inflight,
            "retry": retry,
            "delivered": delivered,
            "failed": failed,
            "dead_letters": dead_letters,
            "oldest_pending_sec": oldest_pending_sec,
        }

    def records(self, limit: int | None = None) -> list[OutboxRecord]:
        """Return all records (ignoring backoff and status), newest first."""
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
        keys = row.keys()

        def col(name: str, default: Any = None) -> Any:
            return row[name] if name in keys else default

        return OutboxRecord(
            id=row["id"],
            target=row["target"],
            payload=payload,
            created_at=row["created_at"],
            retry_count=row["retry_count"],
            next_retry_at=col("next_retry_at"),
            error_log=col("error_log"),
            idempotency_key=col("idempotency_key"),
            entity_type=col("entity_type"),
            entity_id=col("entity_id"),
            payload_sha256=col("payload_sha256"),
            status=col("status", "pending") or "pending",
            lease_owner=col("lease_owner"),
            lease_expires_at=col("lease_expires_at"),
            updated_at=col("updated_at"),
            remote_revision=col("remote_revision"),
        )

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


class OutboxWorker:
    """Background worker that drains an :class:`OutboxStore` through a committer.

    The worker claims records under a lease identified by ``owner_id``; if it
    crashes, another worker (or its own restart) re-claims the records once
    the lease expires.  Before each commit the record's ``idempotency_key`` is
    injected into the payload so the remote side can upsert idempotently.
    """

    def __init__(
        self,
        outbox: OutboxStore,
        committer: Any,
        *,
        interval_sec: float = 5.0,
        batch_size: int = 100,
        max_retries: int = 10,
        name: str = "outbox-worker",
        owner_id: str | None = None,
        lease_sec: float = 30.0,
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
        self._owner_id = owner_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._lease_sec = lease_sec

    @property
    def owner_id(self) -> str:
        return self._owner_id

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=self._name)
        self._thread.start()
        logger.info("Started %s (owner=%s)", self._name, self._owner_id)

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
        records = self._outbox.claim(
            self._batch_size,
            owner=self._owner_id,
            lease_sec=self._lease_sec,
            max_retries=self._max_retries,
        )
        try:
            self._outbox.purge_delivered()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to purge delivered outbox rows: %s", exc)
        if not records:
            return 0

        batch_commit = getattr(self._committer, "save_to_seekdb_batch", None)
        if batch_commit is not None and len(records) > 1:
            try:
                batch_commit([self._payload_for_commit(record) for record in records])
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
                    self._outbox.mark_delivered(record.id, owner=self._owner_id)
                return len(records)

        for record in records:
            try:
                self._committer.save_to_seekdb(self._payload_for_commit(record))
                self._outbox.mark_delivered(record.id, owner=self._owner_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to commit outbox record %s to %s: %s",
                    record.id,
                    record.target,
                    exc,
                )
                self._outbox.mark_failed(
                    record.id, str(exc), owner=self._owner_id, max_retries=self._max_retries
                )
        return len(records)

    @staticmethod
    def _payload_for_commit(record: OutboxRecord) -> dict[str, Any]:
        """Return the payload with the idempotency key attached for upsert."""
        payload = dict(record.payload)
        if record.idempotency_key:
            payload.setdefault("idempotency_key", record.idempotency_key)
        return payload
