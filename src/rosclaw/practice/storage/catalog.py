"""Local SQLite catalog for practice sessions, events, and artifacts."""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.practice.storage.catalog")


class _BatchWriter:
    """Threaded batch inserter for catalog tables.

    Records are queued and flushed either when the batch reaches
    ``batch_size`` or ``flush_interval_ms`` elapse.  The worker is a daemon
    thread so a crashing process does not hang on a stuck writer; ``close()``
    drains the queue and joins the thread.

    Passing ``batch_size=1`` makes inserts effectively synchronous, which is
    useful in tests that require immediate read-after-write visibility.
    """

    _POISON = object()

    def __init__(
        self,
        name: str,
        flush_fn: Any,
        batch_size: int = 500,
        flush_interval_ms: float = 300.0,
        max_queue_size: int = 2000,
    ):
        self.name = name
        self._flush_fn = flush_fn
        self._batch_size = max(1, int(batch_size))
        self._flush_interval_s = max(0.001, float(flush_interval_ms)) / 1000.0
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(max_queue_size)))
        self._lock = threading.Lock()
        self._closed = False
        self._total_dropped = 0
        self._thread = threading.Thread(
            target=self._worker, name=f"catalog-batch-{name}", daemon=True
        )
        self._thread.start()

    def put(self, record: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError(f"Batch writer {self.name} is closed")
        # Synchronous path for unit tests that require read-after-write.
        if self._batch_size == 1:
            self._flush([record])
            return
        try:
            self._queue.put(record, block=False)
        except queue.Full:
            with self._lock:
                self._total_dropped += 1
            logger.error("Batch writer %s queue full; dropping record", self.name)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._queue.put(self._POISON, block=True, timeout=5.0)
        except queue.Full:
            logger.warning("Batch writer %s queue full on close; draining inline", self.name)
        self._thread.join(timeout=5.0)
        # Final drain in case the worker did not process everything.
        final: list[dict[str, Any]] = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item is not self._POISON:
                    final.append(item)
            except queue.Empty:
                break
        if final:
            self._flush_fn(final)
        if self._total_dropped:
            logger.warning(
                "Batch writer %s dropped %s records due to queue overflow",
                self.name,
                self._total_dropped,
            )

    def _worker(self) -> None:
        while True:
            batch: list[dict[str, Any]] = []
            deadline: float | None = None
            while len(batch) < self._batch_size:
                timeout: float | None = None
                if batch:
                    assert deadline is not None
                    timeout = max(0.0, deadline - time.monotonic())
                else:
                    timeout = None
                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is self._POISON:
                    self._flush(batch)
                    return
                batch.append(item)
                if deadline is None:
                    deadline = time.monotonic() + self._flush_interval_s
            if batch:
                self._flush(batch)

    def _flush(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        try:
            self._flush_fn(batch)
        except Exception:
            logger.exception("Batch writer %s flush failed", self.name)


class PracticeCatalog:
    """SQLite-backed index of practices, events, and artifacts.

    Version 2 adds ``practice_sessions``, ``practice_episodes``,
    ``practice_artifacts`` (with schema metadata), and ``practice_event_index``
    tables. Existing ``practices``, ``events``, ``failures``, and ``artifacts``
    tables are preserved for backward compatibility, and the new tables are
    created with ``IF NOT EXISTS`` so old catalogs migrate automatically.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS practices (
        practice_id TEXT PRIMARY KEY,
        session_id TEXT,
        episode_id TEXT,
        robot_id TEXT,
        robot_type TEXT,
        task_id TEXT,
        task_name TEXT,
        skill_id TEXT,
        start_time TEXT,
        end_time TEXT,
        duration_ms REAL,
        outcome TEXT,
        reward REAL,
        manifest_path TEXT,
        events_jsonl_path TEXT,
        replay_path TEXT,
        failure_report_path TEXT,
        seekdb_committed INTEGER
    );

    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        practice_id TEXT,
        source TEXT,
        event_type TEXT,
        timestamp_ns INTEGER,
        timestamp_utc TEXT,
        action_id TEXT,
        task_id TEXT,
        skill_id TEXT,
        payload_ref TEXT,
        tags TEXT
    );

    CREATE TABLE IF NOT EXISTS failures (
        failure_id TEXT PRIMARY KEY,
        practice_id TEXT,
        failure_type TEXT,
        severity TEXT,
        source TEXT,
        related_action_id TEXT,
        description TEXT,
        timestamp_ns INTEGER
    );

    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY,
        practice_id TEXT,
        artifact_type TEXT,
        path TEXT,
        checksum TEXT,
        size_bytes INTEGER,
        created_at TEXT
    );

    -- v2 tables ---------------------------------------------------------

    CREATE TABLE IF NOT EXISTS practice_sessions (
        session_id TEXT PRIMARY KEY,
        practice_id TEXT,
        body_id TEXT,
        task_name TEXT,
        started_at TEXT,
        ended_at TEXT,
        status TEXT,
        outcome TEXT,
        event_count INTEGER DEFAULT 0,
        artifact_count INTEGER DEFAULT 0,
        metadata_json TEXT
    );

    CREATE TABLE IF NOT EXISTS practice_episodes (
        episode_id TEXT PRIMARY KEY,
        session_id TEXT,
        body_id TEXT,
        skill_id TEXT,
        policy_id TEXT,
        started_at TEXT,
        ended_at TEXT,
        outcome TEXT,
        success INTEGER,
        failure_labels_json TEXT,
        metrics_json TEXT
    );

    CREATE TABLE IF NOT EXISTS practice_artifacts (
        artifact_id TEXT PRIMARY KEY,
        session_id TEXT,
        episode_id TEXT,
        artifact_type TEXT,
        path TEXT,
        sha256 TEXT,
        size_bytes INTEGER,
        schema_name TEXT,
        created_at TEXT,
        metadata_json TEXT
    );

    CREATE TABLE IF NOT EXISTS practice_event_index (
        event_id TEXT PRIMARY KEY,
        session_id TEXT,
        episode_id TEXT,
        timestamp_ns INTEGER,
        event_type TEXT,
        artifact_id TEXT,
        byte_offset INTEGER,
        summary_json TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_practices_robot ON practices(robot_id);
    CREATE INDEX IF NOT EXISTS idx_practices_task ON practices(task_id);
    CREATE INDEX IF NOT EXISTS idx_practices_outcome ON practices(outcome);
    CREATE INDEX IF NOT EXISTS idx_events_practice ON events(practice_id);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

    CREATE INDEX IF NOT EXISTS idx_sessions_body ON practice_sessions(body_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_status ON practice_sessions(status);
    CREATE INDEX IF NOT EXISTS idx_episodes_session ON practice_episodes(session_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_body ON practice_episodes(body_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_skill ON practice_episodes(skill_id);
    CREATE INDEX IF NOT EXISTS idx_artifacts_session ON practice_artifacts(session_id);
    CREATE INDEX IF NOT EXISTS idx_artifacts_episode ON practice_artifacts(episode_id);
    CREATE INDEX IF NOT EXISTS idx_artifacts_type ON practice_artifacts(artifact_type);
    CREATE INDEX IF NOT EXISTS idx_event_index_session ON practice_event_index(session_id);
    CREATE INDEX IF NOT EXISTS idx_event_index_episode ON practice_event_index(episode_id);
    CREATE INDEX IF NOT EXISTS idx_event_index_type ON practice_event_index(event_type);
    """

    _LEGACY_TABLES = {
        "practices": [
            "practice_id",
            "session_id",
            "episode_id",
            "robot_id",
            "robot_type",
            "task_id",
            "task_name",
            "skill_id",
            "start_time",
            "end_time",
            "duration_ms",
            "outcome",
            "reward",
            "manifest_path",
            "events_jsonl_path",
            "replay_path",
            "failure_report_path",
            "seekdb_committed",
        ],
        "events": [
            "event_id",
            "practice_id",
            "source",
            "event_type",
            "timestamp_ns",
            "timestamp_utc",
            "action_id",
            "task_id",
            "skill_id",
            "payload_ref",
            "tags",
        ],
        "failures": [
            "failure_id",
            "practice_id",
            "failure_type",
            "severity",
            "source",
            "related_action_id",
            "description",
            "timestamp_ns",
        ],
        "artifacts": [
            "artifact_id",
            "practice_id",
            "artifact_type",
            "path",
            "checksum",
            "size_bytes",
            "created_at",
        ],
    }

    def __init__(
        self,
        db_path: Path | str,
        *,
        event_batch_size: int = 500,
        event_flush_ms: float = 300.0,
        event_max_queue: int = 2000,
    ):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._event_batch_size = event_batch_size
        self._event_flush_ms = event_flush_ms
        self._event_max_queue = event_max_queue
        self._event_writer = _BatchWriter(
            "events",
            self._flush_events,
            batch_size=event_batch_size,
            flush_interval_ms=event_flush_ms,
            max_queue_size=event_max_queue,
        )
        self._event_index_writer = _BatchWriter(
            "event_index",
            self._flush_event_index,
            batch_size=event_batch_size,
            flush_interval_ms=event_flush_ms,
            max_queue_size=event_max_queue,
        )
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._migrate_legacy_tables()
            self._conn.executescript(self._SCHEMA)
            self._conn.commit()

    def _migrate_legacy_tables(self) -> None:
        """Add missing columns to legacy tables so old catalogs stay usable."""
        existing_tables = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table, expected_columns in self._LEGACY_TABLES.items():
            if table not in existing_tables:
                continue
            cursor = self._conn.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in cursor.fetchall()}
            for column in expected_columns:
                if column not in existing:
                    logger.info("Migrating table %s: adding column %s", table, column)
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column}")

    def close(self) -> None:
        # Close writers first (they need the lock to flush) before closing the
        # underlying connection.
        if self._event_writer is not None:
            self._event_writer.close()
            self._event_writer = None
        if self._event_index_writer is not None:
            self._event_index_writer.close()
            self._event_index_writer = None
        with self._lock:
            self._conn.close()

    def flush(self) -> None:
        """Flush any pending batched writes before reading."""
        if self._event_writer is not None:
            self._event_writer.close()
            self._event_writer = _BatchWriter(
                "events",
                self._flush_events,
                batch_size=self._event_batch_size,
                flush_interval_ms=self._event_flush_ms,
                max_queue_size=self._event_max_queue,
            )
        if self._event_index_writer is not None:
            self._event_index_writer.close()
            self._event_index_writer = _BatchWriter(
                "event_index",
                self._flush_event_index,
                batch_size=self._event_batch_size,
                flush_interval_ms=self._event_flush_ms,
                max_queue_size=self._event_max_queue,
            )

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    def insert_practice(self, record: dict[str, Any]) -> None:
        cols = ", ".join(record.keys())
        placeholders = ", ".join("?" for _ in record)
        values = list(record.values())
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO practices ({cols}) VALUES ({placeholders})",
                values,
            )
            self._conn.commit()

    def insert_event(self, record: dict[str, Any]) -> None:
        self._event_writer.put(record)

    def _flush_events(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        cols = ", ".join(batch[0].keys())
        placeholders = ", ".join("?" for _ in batch[0])
        values = [list(record.values()) for record in batch]
        with self._lock:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO events ({cols}) VALUES ({placeholders})",
                values,
            )
            self._conn.commit()

    def update_practice(
        self,
        practice_id: str,
        updates: dict[str, Any],
    ) -> bool:
        if not updates:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [practice_id]
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE practices SET {set_clause} WHERE practice_id = ?",
                values,
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def get_practice(self, practice_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM practices WHERE practice_id = ?",
                (practice_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_practices(
        self,
        robot_id: str | None = None,
        task_id: str | None = None,
        outcome: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if robot_id:
            where.append("robot_id = ?")
            params.append(robot_id)
        if task_id:
            where.append("task_id = ?")
            params.append(task_id)
        if outcome:
            where.append("outcome = ?")
            params.append(outcome)
        sql = "SELECT * FROM practices"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def count_events(self, practice_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE practice_id = ?",
                (practice_id,),
            ).fetchone()
        return row[0] if row else 0

    def count_source_events(self, practice_id: str) -> int:
        """Count practice events excluding runtime lifecycle markers."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE practice_id = ? "
                "AND event_type NOT IN ('runtime.start', 'runtime.stop')",
                (practice_id,),
            ).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # v2: sessions
    # ------------------------------------------------------------------

    def insert_session(self, record: dict[str, Any]) -> None:
        record = dict(record)
        if "metadata" in record and not isinstance(record["metadata"], str):
            record["metadata_json"] = json.dumps(record.pop("metadata"), ensure_ascii=False)
        elif "metadata_json" in record and not isinstance(record["metadata_json"], str):
            record["metadata_json"] = json.dumps(record["metadata_json"], ensure_ascii=False)
        cols = ", ".join(record.keys())
        placeholders = ", ".join("?" for _ in record)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO practice_sessions ({cols}) VALUES ({placeholders})",
                list(record.values()),
            )
            self._conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM practice_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return self._decode_json_field(dict(row), "metadata_json")

    def list_sessions(
        self,
        body_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if body_id:
            where.append("body_id = ?")
            params.append(body_id)
        if status:
            where.append("status = ?")
            params.append(status)
        sql = "SELECT * FROM practice_sessions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._decode_json_field(dict(row), "metadata_json") for row in rows]

    def update_session(self, session_id: str, updates: dict[str, Any]) -> bool:
        updates = dict(updates)
        if "metadata" in updates:
            updates["metadata_json"] = json.dumps(updates.pop("metadata"), ensure_ascii=False)
        if not updates:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [session_id]
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE practice_sessions SET {set_clause} WHERE session_id = ?",
                values,
            )
            self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # v2: episodes
    # ------------------------------------------------------------------

    def insert_episode(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record = self._encode_json_fields(record, ["failure_labels", "metrics"])
        if "success" in record:
            record["success"] = 1 if record["success"] else 0
        cols = ", ".join(record.keys())
        placeholders = ", ".join("?" for _ in record)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO practice_episodes ({cols}) VALUES ({placeholders})",
                list(record.values()),
            )
            self._conn.commit()

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM practice_episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data = self._decode_json_field(data, "failure_labels_json")
        data = self._decode_json_field(data, "metrics_json")
        if "success" in data and data["success"] is not None:
            data["success"] = bool(data["success"])
        return data

    def list_episodes(
        self,
        session_id: str | None = None,
        body_id: str | None = None,
        skill_id: str | None = None,
        outcome: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if body_id:
            where.append("body_id = ?")
            params.append(body_id)
        if skill_id:
            where.append("skill_id = ?")
            params.append(skill_id)
        if outcome:
            where.append("outcome = ?")
            params.append(outcome)
        sql = "SELECT * FROM practice_episodes"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._decode_episode_row(row) for row in rows]

    def update_episode(self, episode_id: str, updates: dict[str, Any]) -> bool:
        updates = dict(updates)
        updates = self._encode_json_fields(updates, ["failure_labels", "metrics"])
        if "success" in updates:
            updates["success"] = 1 if updates["success"] else 0
        if not updates:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [episode_id]
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE practice_episodes SET {set_clause} WHERE episode_id = ?",
                values,
            )
            self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # v2: artifacts
    # ------------------------------------------------------------------

    def insert_artifact_v2(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record = self._encode_json_fields(record, ["metadata"])
        cols = ", ".join(record.keys())
        placeholders = ", ".join("?" for _ in record)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO practice_artifacts ({cols}) VALUES ({placeholders})",
                list(record.values()),
            )
            self._conn.commit()

    def get_artifact_v2(self, artifact_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM practice_artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return self._decode_json_field(dict(row), "metadata_json")

    def list_artifacts_v2(
        self,
        session_id: str | None = None,
        episode_id: str | None = None,
        artifact_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if episode_id:
            where.append("episode_id = ?")
            params.append(episode_id)
        if artifact_type:
            where.append("artifact_type = ?")
            params.append(artifact_type)
        sql = "SELECT * FROM practice_artifacts"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._decode_json_field(dict(row), "metadata_json") for row in rows]

    # ------------------------------------------------------------------
    # v2: event index
    # ------------------------------------------------------------------

    def insert_event_index(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record = self._encode_json_fields(record, ["summary"])
        self._event_index_writer.put(record)

    def _flush_event_index(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        cols = ", ".join(batch[0].keys())
        placeholders = ", ".join("?" for _ in batch[0])
        values = [list(record.values()) for record in batch]
        with self._lock:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO practice_event_index ({cols}) VALUES ({placeholders})",
                values,
            )
            self._conn.commit()

    def get_event_index(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM practice_event_index WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        return self._decode_json_field(dict(row), "summary_json")

    def list_event_index(
        self,
        session_id: str | None = None,
        episode_id: str | None = None,
        event_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if episode_id:
            where.append("episode_id = ?")
            params.append(episode_id)
        if event_type:
            where.append("event_type = ?")
            params.append(event_type)
        sql = "SELECT * FROM practice_event_index"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY timestamp_ns LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._decode_json_field(dict(row), "summary_json") for row in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_json_fields(record: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        record = dict(record)
        for field_name in fields:
            json_field = f"{field_name}_json"
            if field_name in record and not isinstance(record[field_name], str):
                record[json_field] = json.dumps(record.pop(field_name), ensure_ascii=False)
            elif json_field in record and not isinstance(record[json_field], str):
                record[json_field] = json.dumps(record[json_field], ensure_ascii=False)
        return record

    @staticmethod
    def _decode_json_field(data: dict[str, Any], json_field: str) -> dict[str, Any]:
        data = dict(data)
        value = data.get(json_field)
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError):
                data[json_field] = json.loads(value)
        return data

    @classmethod
    def _decode_episode_row(cls, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data = cls._decode_json_field(data, "failure_labels_json")
        data = cls._decode_json_field(data, "metrics_json")
        if "success" in data and data["success"] is not None:
            data["success"] = bool(data["success"])
        return data

    def __enter__(self) -> PracticeCatalog:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
