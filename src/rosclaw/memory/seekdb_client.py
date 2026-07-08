"""SeekDB Client - Knowledge Plane interface for ROSClaw.

Provides abstract SeekDBClient and concrete implementations:
- SeekDBMemoryClient: In-memory for testing
- SeekDBSQLiteClient: SQLite for single-machine deployment

Sprint 5 of DESIGN_SPRINT3_5.
"""

import contextlib
import json
import logging
import sqlite3
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SEEKDB_SCHEMAS: dict[str, Any] = {
    "experience_graph": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "event_type": "TEXT NOT NULL",
            "robot_id": "TEXT NOT NULL",
            "timestamp": "REAL NOT NULL",
            "instruction": "TEXT",
            "cot_trace": "TEXT",
            "trajectory": "TEXT",
            "outcome": "TEXT",
            "duration_sec": "REAL",
            "error_details": "TEXT",
            "tags": "TEXT",
            "metadata": "TEXT",
        },
        "indices": ["robot_id", "event_type", "outcome", "timestamp"],
    },
    "skill_metadata": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "skill_id": "TEXT NOT NULL",
            "name": "TEXT NOT NULL",
            "description": "TEXT",
            "category": "TEXT",
            "source": "TEXT",
            "success_count": "INTEGER DEFAULT 0",
            "failure_count": "INTEGER DEFAULT 0",
            "avg_duration_sec": "REAL",
            "last_used": "REAL",
            "prerequisites": "TEXT",
            "metadata": "TEXT",
        },
        "indices": ["name", "category", "source", "skill_id"],
    },
    "knowledge_graph": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "subject": "TEXT NOT NULL",
            "predicate": "TEXT NOT NULL",
            "object": "TEXT NOT NULL",
            "confidence": "REAL DEFAULT 1.0",
            "source": "TEXT",
            "timestamp": "REAL NOT NULL",
        },
        "indices": ["subject", "predicate", "object"],
    },
    "heuristic_rules": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "condition": "TEXT NOT NULL",
            "action": "TEXT NOT NULL",
            "priority": "INTEGER DEFAULT 0",
            "success_count": "INTEGER DEFAULT 0",
            "failure_count": "INTEGER DEFAULT 0",
            "last_triggered": "REAL",
        },
        "indices": ["priority"],
    },
    # Sprint 8 — Knowledge Plane tables
    "robots": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "name": "TEXT NOT NULL",
            "robot_type": "TEXT",
            "eurdf_profile": "TEXT",
            "status": "TEXT DEFAULT 'active'",
            "registered_at": "REAL",
            "metadata": "TEXT",
        },
        "indices": ["robot_type", "status"],
    },
    "providers": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "name": "TEXT NOT NULL",
            "capability_type": "TEXT NOT NULL",
            "endpoint": "TEXT",
            "status": "TEXT DEFAULT 'active'",
            "metadata": "TEXT",
        },
        "indices": ["capability_type", "status"],
    },
    "skills": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "name": "TEXT NOT NULL",
            "version": "TEXT DEFAULT '1.0'",
            "description": "TEXT",
            "parameters": "TEXT",
            "metadata": "TEXT",
        },
        "indices": ["name", "version"],
    },
    "tasks": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "name": "TEXT NOT NULL",
            "description": "TEXT",
            "robot_id": "TEXT",
            "status": "TEXT DEFAULT 'pending'",
            "created_at": "REAL",
            "metadata": "TEXT",
        },
        "indices": ["robot_id", "status", "created_at"],
    },
    "episodes": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "task_id": "TEXT",
            "robot_id": "TEXT NOT NULL",
            "started_at": "REAL",
            "ended_at": "REAL",
            "outcome": "TEXT",
            "artifact_uri": "TEXT",
            "metadata": "TEXT",
        },
        "indices": ["task_id", "robot_id", "outcome", "started_at"],
    },
    "praxis_events": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "episode_id": "TEXT",
            "robot_id": "TEXT NOT NULL",
            "event_type": "TEXT NOT NULL",
            "timestamp": "REAL NOT NULL",
            "payload": "TEXT",
            "metadata": "TEXT",
        },
        "indices": ["episode_id", "robot_id", "event_type", "timestamp"],
    },
    "failures": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "episode_id": "TEXT",
            "task_id": "TEXT",
            "robot_id": "TEXT NOT NULL",
            "body_id": "TEXT",
            "failure_type": "TEXT",
            "root_cause": "TEXT",
            "timestamp": "REAL",
            "recovery_hint": "TEXT",
            "metadata": "TEXT",
        },
        "indices": ["episode_id", "task_id", "robot_id", "body_id", "failure_type", "timestamp"],
    },
    "success_patterns": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "skill_id": "TEXT",
            "robot_id": "TEXT",
            "context_hash": "TEXT",
            "success_count": "INTEGER DEFAULT 0",
            "avg_duration_sec": "REAL",
            "metadata": "TEXT",
        },
        "indices": ["skill_id", "robot_id", "context_hash"],
    },
    "benchmarks": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "robot_id": "TEXT",
            "task_id": "TEXT",
            "score": "REAL",
            "metrics": "TEXT",
            "run_at": "REAL",
            "metadata": "TEXT",
        },
        "indices": ["robot_id", "task_id", "run_at"],
    },
    "artifacts": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "episode_id": "TEXT",
            "artifact_type": "TEXT NOT NULL",
            "uri": "TEXT NOT NULL",
            "size_bytes": "INTEGER",
            "created_at": "REAL",
            "metadata": "TEXT",
        },
        "indices": ["episode_id", "artifact_type", "created_at", "uri"],
    },
    # Sprint 8 — Practice distilled knowledge tables
    "how_interventions": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "failure_id": "TEXT",
            "episode_id": "TEXT",
            "robot_id": "TEXT NOT NULL",
            "task_id": "TEXT",
            "intervention_type": "TEXT",
            "description": "TEXT",
            "action_taken": "TEXT",
            "outcome": "TEXT",
            "timestamp": "REAL",
            "metadata": "TEXT",
        },
        "indices": ["failure_id", "episode_id", "robot_id", "outcome"],
    },
    "body_cognition": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "body_id": "TEXT NOT NULL",
            "robot_id": "TEXT NOT NULL",
            "episode_id": "TEXT",
            "session_id": "TEXT",
            "cognition_type": "TEXT",
            "data": "TEXT",
            "timestamp": "REAL",
            "metadata": "TEXT",
        },
        "indices": ["body_id", "robot_id", "episode_id", "cognition_type"],
    },
    "sim2real_deltas": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "body_id": "TEXT NOT NULL",
            "robot_id": "TEXT NOT NULL",
            "episode_id": "TEXT",
            "dofs": "TEXT",
            "sim_value": "TEXT",
            "real_value": "TEXT",
            "delta": "TEXT",
            "unit": "TEXT",
            "timestamp": "REAL",
            "metadata": "TEXT",
        },
        "indices": ["body_id", "robot_id", "episode_id"],
    },
    "skill_candidates": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "skill_id": "TEXT",
            "robot_id": "TEXT NOT NULL",
            "episode_id": "TEXT",
            "policy_id": "TEXT",
            "policy_type": "TEXT",
            "policy_params": "TEXT",
            "metrics": "TEXT",
            "status": "TEXT DEFAULT 'candidate'",
            "evidence_refs": "TEXT",
            "timestamp": "REAL",
            "metadata": "TEXT",
        },
        "indices": ["skill_id", "robot_id", "episode_id", "policy_id", "status"],
    },
    "promotion_results": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "candidate_id": "TEXT",
            "policy_id": "TEXT",
            "robot_id": "TEXT NOT NULL",
            "episode_id": "TEXT",
            "gate_name": "TEXT",
            "passed": "INTEGER",
            "metrics": "TEXT",
            "failures": "TEXT",
            "evidence_refs": "TEXT",
            "promoted_policy_ref": "TEXT",
            "timestamp": "REAL",
            "metadata": "TEXT",
        },
        "indices": ["candidate_id", "policy_id", "robot_id", "episode_id", "passed"],
    },
    "retries": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "failure_type": "TEXT",
            "rule_id": "TEXT",
            "parameter_patch": "TEXT",
            "max_retries": "INTEGER DEFAULT 3",
            "attempt_count": "INTEGER DEFAULT 0",
            "status": "TEXT DEFAULT 'pending'",
            "created_at": "REAL",
            "updated_at": "REAL",
            "original_outcome": "TEXT",
            "retry_outcome": "TEXT",
            "improvement": "TEXT",
        },
        "indices": ["status", "rule_id", "created_at"],
    },
    # Auto module tables
    "auto_proposals": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "task_id": "TEXT",
            "target_skill": "TEXT",
            "source": "TEXT",
            "hypothesis": "TEXT",
            "search_space": "TEXT",
            "status": "TEXT DEFAULT 'open'",
            "created_at": "REAL",
        },
        "indices": ["task_id", "status", "created_at"],
    },
    "auto_patches": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "proposal_id": "TEXT",
            "target_skill": "TEXT",
            "patch_type": "TEXT",
            "changes": "TEXT",
            "status": "TEXT DEFAULT 'created'",
            "created_at": "REAL",
        },
        "indices": ["proposal_id", "target_skill", "status"],
    },
    "auto_experiments": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "proposal_id": "TEXT",
            "patch_id": "TEXT",
            "task_id": "TEXT",
            "baseline_skill": "TEXT",
            "candidate_skill": "TEXT",
            "status": "TEXT",
            "metrics": "TEXT",
            "created_at": "REAL",
            "completed_at": "REAL",
        },
        "indices": ["task_id", "status", "created_at"],
    },
    "auto_results": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "experiment_id": "TEXT",
            "decision": "TEXT",
            "delta": "TEXT",
            "created_at": "REAL",
        },
        "indices": ["experiment_id", "decision"],
    },
    "champions": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "skill_id": "TEXT NOT NULL",
            "task_id": "TEXT",
            "level": "TEXT",
            "parent_skill": "TEXT",
            "metrics": "TEXT",
            "promoted_at": "REAL",
        },
        "indices": ["skill_id", "task_id", "level"],
    },
    "dead_ends": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "task_id": "TEXT",
            "direction": "TEXT",
            "rejection_reason": "TEXT",
            "registered_at": "REAL",
        },
        "indices": ["task_id", "registered_at"],
    },
}


class SeekDBClient(ABC):
    """Abstract interface to SeekDB Knowledge Plane."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def insert(self, table: str, record: dict) -> str: ...

    @abstractmethod
    def query(
        self,
        table: str,
        filters: dict | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> list[dict]: ...

    @abstractmethod
    def update(self, table: str, record_id: str, updates: dict) -> bool: ...

    @abstractmethod
    def count(self, table: str, filters: dict | None = None) -> int: ...

    @abstractmethod
    def delete(self, table: str, record_id: str) -> bool: ...

    @abstractmethod
    def delete_where(self, table: str, filters: dict) -> int: ...


class SeekDBMemoryClient(SeekDBClient):
    """In-memory SeekDB client for testing.

    Maintains inverted indexes on columns declared in SEEKDB_SCHEMAS
    to avoid full-table scans on the most common filter patterns.

    Thread-safe: all mutating operations are protected by a reentrant lock
    so that concurrent EventBus handlers and Runtime threads do not corrupt
    the inverted indexes.
    """

    MAX_SCAN_LIMIT = 10_000

    def __init__(self):
        self._tables: dict[str, dict[str, dict]] = {}
        # Inverted indexes: _indices[table][column][value] = set of record_ids
        self._indices: dict[str, dict[str, dict[Any, set[str]]]] = {}
        import threading

        self._lock = threading.RLock()

    def connect(self) -> None:
        for table_name, schema in SEEKDB_SCHEMAS.items():
            if table_name not in self._tables:
                self._tables[table_name] = {}
            if table_name not in self._indices:
                self._indices[table_name] = {}
            for col in schema.get("indices", []):
                if col not in self._indices[table_name]:
                    self._indices[table_name][col] = {}

    def is_connected(self) -> bool:
        return bool(self._tables)

    def disconnect(self) -> None:
        pass

    def insert(self, table: str, record: dict) -> str:
        with self._lock:
            if table not in self._tables:
                self._tables[table] = {}
                self._indices[table] = {}
            record_id = record.get("id") or str(uuid.uuid4())
            stored = dict(record)
            stored["id"] = record_id
            self._tables[table][record_id] = stored
            # Update inverted indexes
            if table in self._indices:
                for col, idx in self._indices[table].items():
                    val = stored.get(col)
                    if val is not None:
                        if val not in idx:
                            idx[val] = set()
                        idx[val].add(record_id)
            return record_id

    def query(
        self,
        table: str,
        filters: dict | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        with self._lock:
            if table not in self._tables:
                return []

            # Use inverted indexes when filters match indexed columns
            candidate_ids = None
            remaining_filters = {}
            if filters and table in self._indices:
                for k, v in filters.items():
                    if k in self._indices[table]:
                        idx = self._indices[table][k]
                        matched_ids = idx.get(v, set())
                        if candidate_ids is None:
                            candidate_ids = set(matched_ids)
                        else:
                            candidate_ids &= matched_ids
                    else:
                        remaining_filters[k] = v

            if candidate_ids is not None:
                records = [
                    dict(self._tables[table][rid])
                    for rid in candidate_ids
                    if rid in self._tables[table]
                ]
            else:
                records = [dict(r) for r in self._tables[table].values()]

            # Apply remaining non-indexed filters
            if remaining_filters:
                records = [
                    r for r in records if all(r.get(k) == v for k, v in remaining_filters.items())
                ]

            if order_by:
                reverse = order_by.startswith("-")
                key = order_by.lstrip("-")
                records.sort(key=lambda r: r.get(key, 0), reverse=reverse)
            return records[:limit]

    def update(self, table: str, record_id: str, updates: dict) -> bool:
        with self._lock:
            if table not in self._tables or record_id not in self._tables[table]:
                return False
            old_record = self._tables[table][record_id]
            # Remove old index entries
            if table in self._indices:
                for col, idx in self._indices[table].items():
                    old_val = old_record.get(col)
                    if old_val is not None and old_val in idx:
                        idx[old_val].discard(record_id)
            # Apply updates
            old_record.update(updates)
            # Add new index entries
            if table in self._indices:
                for col, idx in self._indices[table].items():
                    new_val = old_record.get(col)
                    if new_val is not None:
                        if new_val not in idx:
                            idx[new_val] = set()
                        idx[new_val].add(record_id)
            return True

    def count(self, table: str, filters: dict | None = None) -> int:
        with self._lock:
            # Use index for simple counts when possible
            if filters and table in self._indices:
                indexed_keys = [k for k in filters if k in self._indices[table]]
                if len(indexed_keys) == len(filters) and len(indexed_keys) > 0:
                    candidate_ids = None
                    for k in indexed_keys:
                        matched = self._indices[table][k].get(filters[k], set())
                        if candidate_ids is None:
                            candidate_ids = set(matched)
                        else:
                            candidate_ids &= matched
                    return len(candidate_ids) if candidate_ids else 0
            return len(self.query(table, filters, limit=self.MAX_SCAN_LIMIT))

    def delete(self, table: str, record_id: str) -> bool:
        with self._lock:
            if table not in self._tables or record_id not in self._tables[table]:
                return False
            record = self._tables[table].pop(record_id)
            # Remove from indexes
            if table in self._indices:
                for col, idx in self._indices[table].items():
                    val = record.get(col)
                    if val is not None and val in idx:
                        idx[val].discard(record_id)
            return True

    def delete_where(self, table: str, filters: dict) -> int:
        with self._lock:
            if table not in self._tables:
                return 0
            to_delete = []
            for rid, record in self._tables[table].items():
                if all(record.get(k) == v for k, v in filters.items()):
                    to_delete.append(rid)
            for rid in to_delete:
                self.delete(table, rid)
            return len(to_delete)


class SeekDBSQLiteClient(SeekDBClient):
    """SQLite-backed SeekDB client for single-machine deployment."""

    def __init__(self, db_path: str = "./seekdb.sqlite"):
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        if self._conn is not None:
            return
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def is_connected(self) -> bool:
        return self._conn is not None

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def _connection(self) -> Any:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    # Composite indexes for common query patterns
    _COMPOSITE_INDICES = [
        # (table, index_name, columns)
        ("experience_graph", "idx_exp_robot_ts", "robot_id, timestamp DESC"),
        ("experience_graph", "idx_exp_robot_outcome", "robot_id, outcome"),
        ("knowledge_graph", "idx_kg_subj_pred", "subject, predicate"),
        ("heuristic_rules", "idx_hr_priority_action", "priority DESC, action"),
        # Sprint 8 — Knowledge Plane composite indexes
        ("episodes", "idx_ep_robot_started", "robot_id, started_at DESC"),
        ("episodes", "idx_ep_task_outcome", "task_id, outcome"),
        ("praxis_events", "idx_pe_episode_ts", "episode_id, timestamp DESC"),
        ("failures", "idx_fail_robot_ts", "robot_id, timestamp DESC"),
        ("failures", "idx_fail_task_type", "task_id, failure_type"),
        ("artifacts", "idx_art_episode_type", "episode_id, artifact_type"),
        ("success_patterns", "idx_sp_skill_robot", "skill_id, robot_id"),
        ("benchmarks", "idx_bench_robot_run", "robot_id, run_at DESC"),
    ]

    def _create_tables(self) -> None:
        for table_name, schema in SEEKDB_SCHEMAS.items():
            cols = ", ".join(f"{k} {v}" for k, v in schema["columns"].items())
            self._connection.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols})")
        self._connection.commit()
        # Add any columns that were added to SEEKDB_SCHEMAS after the table
        # was first created before creating indexes that may reference them.
        self._migrate_missing_columns()
        # Create single-column and composite indexes
        for table_name, schema in SEEKDB_SCHEMAS.items():
            for idx_col in schema.get("indices", []):
                idx_name = f"idx_{table_name}_{idx_col}"
                self._connection.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({idx_col})"
                )
        for table_name, idx_name, columns in self._COMPOSITE_INDICES:
            self._connection.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({columns})"
            )
        self._connection.commit()

    def _migrate_missing_columns(self) -> None:
        """Add columns that were added to SEEKDB_SCHEMAS after table creation.

        SQLite does not allow adding a PRIMARY KEY column to an existing table,
        so the constraint is dropped for the migration; the column presence is
        enough for inserts/upserts to succeed.
        """
        for table_name, schema in SEEKDB_SCHEMAS.items():
            cursor = self._connection.execute(f"PRAGMA table_info({table_name})")
            existing = {row["name"] for row in cursor.fetchall()}
            if not existing:
                continue
            for col_name, col_type in schema["columns"].items():
                if col_name in existing:
                    continue
                # Strip PRIMARY KEY because ALTER TABLE ADD COLUMN rejects it.
                safe_type = col_type.replace(" PRIMARY KEY", "").strip()
                try:
                    self._connection.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN {col_name} {safe_type}"
                    )
                except Exception as exc:
                    logger.warning("Failed to add column %s to %s: %s", col_name, table_name, exc)
        self._connection.commit()

    def insert(self, table: str, record: dict) -> str:
        if table not in SEEKDB_SCHEMAS:
            raise ValueError(f"Unknown table: {table}")
        serialized = {}
        for k, v in record.items():
            serialized[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        # The id column is required (PRIMARY KEY). Assign a deterministic fallback
        # before building the INSERT so the column list always includes it.
        if "id" not in serialized:
            serialized["id"] = str(uuid.uuid4())
        cols = ", ".join(serialized.keys())
        placeholders = ", ".join("?" for _ in serialized)
        values = list(serialized.values())
        try:
            self._connection.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                values,
            )
        except Exception:
            # Conflict on id — update existing record (explicit upsert)
            self._connection.execute(
                f"REPLACE INTO {table} ({cols}) VALUES ({placeholders})",
                values,
            )
        self._connection.commit()
        return serialized["id"]

    def query(
        self,
        table: str,
        filters: dict | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        sql = f"SELECT * FROM {table}"
        params = []
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(f"{k} = ?")
                params.append(v)
            sql += " WHERE " + " AND ".join(conditions)
        if order_by:
            direction = "DESC" if order_by.startswith("-") else "ASC"
            key = order_by.lstrip("-")
            sql += f" ORDER BY {key} {direction}"
        sql += f" LIMIT {limit}"
        cursor = self._connection.execute(sql, params)
        rows = cursor.fetchall()
        results = []
        for row in rows:
            record = dict(row)
            for k, v in record.items():
                if isinstance(v, str) and (v.startswith("[") or v.startswith("{")):
                    with contextlib.suppress(json.JSONDecodeError):
                        record[k] = json.loads(v)
            results.append(record)
        return results

    def update(self, table: str, record_id: str, updates: dict) -> bool:
        if not updates:
            return False
        serialized = {}
        for k, v in updates.items():
            serialized[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        set_clause = ", ".join(f"{k} = ?" for k in serialized)
        values = list(serialized.values()) + [record_id]
        cursor = self._connection.execute(
            f"UPDATE {table} SET {set_clause} WHERE id = ?",
            values,
        )
        self._connection.commit()
        return cursor.rowcount > 0

    def count(self, table: str, filters: dict | None = None) -> int:
        sql = f"SELECT COUNT(*) FROM {table}"
        params = []
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(f"{k} = ?")
                params.append(v)
            sql += " WHERE " + " AND ".join(conditions)
        cursor = self._connection.execute(sql, params)
        return cursor.fetchone()[0]

    def delete(self, table: str, record_id: str) -> bool:
        cursor = self._connection.execute(
            f"DELETE FROM {table} WHERE id = ?",
            (record_id,),
        )
        self._connection.commit()
        return cursor.rowcount > 0

    def delete_where(self, table: str, filters: dict) -> int:
        if not filters:
            return 0
        conditions = []
        params = []
        for k, v in filters.items():
            conditions.append(f"{k} = ?")
            params.append(v)
        sql = f"DELETE FROM {table} WHERE " + " AND ".join(conditions)
        cursor = self._connection.execute(sql, params)
        self._connection.commit()
        return cursor.rowcount
