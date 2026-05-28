"""SeekDB Client - Knowledge Plane interface for ROSClaw.

Provides abstract SeekDBClient and concrete implementations:
- SeekDBMemoryClient: In-memory for testing
- SeekDBSQLiteClient: SQLite for single-machine deployment

Sprint 5 of DESIGN_SPRINT3_5.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import json
import time


SEEKDB_SCHEMAS = {
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
            "skill_id": "TEXT PRIMARY KEY",
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
        "indices": ["name", "category", "source"],
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
}


class SeekDBClient(ABC):
    """Abstract interface to SeekDB Knowledge Plane."""

    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def insert(self, table: str, record: dict) -> str:
        ...

    @abstractmethod
    def query(
        self,
        table: str,
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        ...

    @abstractmethod
    def update(self, table: str, record_id: str, updates: dict) -> bool:
        ...

    @abstractmethod
    def count(self, table: str, filters: Optional[dict] = None) -> int:
        ...


class SeekDBMemoryClient(SeekDBClient):
    """In-memory SeekDB client for testing.

    Maintains inverted indexes on columns declared in SEEKDB_SCHEMAS
    to avoid full-table scans on the most common filter patterns.
    """

    MAX_SCAN_LIMIT = 10_000

    def __init__(self):
        self._tables: dict[str, dict[str, dict]] = {}
        # Inverted indexes: _indices[table][column][value] = set of record_ids
        self._indices: dict[str, dict[str, dict[Any, set[str]]]] = {}

    def connect(self) -> None:
        for table_name, schema in SEEKDB_SCHEMAS.items():
            self._tables[table_name] = {}
            self._indices[table_name] = {}
            for col in schema.get("indices", []):
                self._indices[table_name][col] = {}

    def disconnect(self) -> None:
        pass

    def insert(self, table: str, record: dict) -> str:
        if table not in self._tables:
            self._tables[table] = {}
            self._indices[table] = {}
        record_id = record.get("id", str(len(self._tables[table])))
        self._tables[table][record_id] = dict(record)
        # Update inverted indexes
        if table in self._indices:
            for col, idx in self._indices[table].items():
                val = record.get(col)
                if val is not None:
                    if val not in idx:
                        idx[val] = set()
                    idx[val].add(record_id)
        return record_id

    def query(
        self,
        table: str,
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
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
                self._tables[table][rid]
                for rid in candidate_ids
                if rid in self._tables[table]
            ]
        else:
            records = list(self._tables[table].values())

        # Apply remaining non-indexed filters
        if remaining_filters:
            records = [
                r for r in records
                if all(r.get(k) == v for k, v in remaining_filters.items())
            ]

        if order_by:
            reverse = order_by.startswith("-")
            key = order_by.lstrip("-")
            records.sort(key=lambda r: r.get(key, 0), reverse=reverse)
        return records[:limit]

    def update(self, table: str, record_id: str, updates: dict) -> bool:
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

    def count(self, table: str, filters: Optional[dict] = None) -> int:
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


class SeekDBSQLiteClient(SeekDBClient):
    """SQLite-backed SeekDB client for single-machine deployment."""

    def __init__(self, db_path: str = "./seekdb.sqlite"):
        self._db_path = db_path
        self._conn = None

    def connect(self) -> None:
        import sqlite3
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    # Composite indexes for common query patterns
    _COMPOSITE_INDICES = [
        # (table, index_name, columns)
        ("experience_graph", "idx_exp_robot_ts", "robot_id, timestamp DESC"),
        ("experience_graph", "idx_exp_robot_outcome", "robot_id, outcome"),
        ("knowledge_graph", "idx_kg_subj_pred", "subject, predicate"),
        ("heuristic_rules", "idx_hr_priority_action", "priority DESC, action"),
    ]

    def _create_tables(self) -> None:
        for table_name, schema in SEEKDB_SCHEMAS.items():
            cols = ", ".join(f"{k} {v}" for k, v in schema["columns"].items())
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols})")
            for idx_col in schema.get("indices", []):
                idx_name = f"idx_{table_name}_{idx_col}"
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({idx_col})"
                )
        # Create composite indexes
        for table_name, idx_name, columns in self._COMPOSITE_INDICES:
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({columns})"
            )
        self._conn.commit()

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def insert(self, table: str, record: dict) -> str:
        if table not in SEEKDB_SCHEMAS:
            raise ValueError(f"Unknown table: {table}")
        serialized = {}
        for k, v in record.items():
            serialized[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        cols = ", ".join(serialized.keys())
        placeholders = ", ".join("?" for _ in serialized)
        values = list(serialized.values())
        record_id = serialized.get("id", str(int(time.time() * 1000)))
        if "id" not in serialized:
            serialized["id"] = record_id
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()
        return record_id

    def query(
        self,
        table: str,
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
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
        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()
        results = []
        for row in rows:
            record = dict(row)
            for k, v in record.items():
                if isinstance(v, str) and (v.startswith("[") or v.startswith("{")):
                    try:
                        record[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
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
        cursor = self._conn.execute(
            f"UPDATE {table} SET {set_clause} WHERE id = ?",
            values,
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self, table: str, filters: Optional[dict] = None) -> int:
        sql = f"SELECT COUNT(*) FROM {table}"
        params = []
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(f"{k} = ?")
                params.append(v)
            sql += " WHERE " + " AND ".join(conditions)
        cursor = self._conn.execute(sql, params)
        return cursor.fetchone()[0]
