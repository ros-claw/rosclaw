"""Local SQLite catalog for practice sessions, events, and artifacts."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.practice.storage.catalog")


class PracticeCatalog:
    """SQLite-backed index of practices, events, and artifacts."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS practices (
        practice_id TEXT PRIMARY KEY,
        session_id TEXT,
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

    CREATE INDEX IF NOT EXISTS idx_practices_robot ON practices(robot_id);
    CREATE INDEX IF NOT EXISTS idx_practices_task ON practices(task_id);
    CREATE INDEX IF NOT EXISTS idx_practices_outcome ON practices(outcome);
    CREATE INDEX IF NOT EXISTS idx_events_practice ON events(practice_id);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
    """

    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(self._SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

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
        cols = ", ".join(record.keys())
        placeholders = ", ".join("?" for _ in record)
        values = list(record.values())
        with self._lock:
            self._conn.execute(
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

    def __enter__(self) -> PracticeCatalog:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
