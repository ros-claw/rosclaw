"""Knowledge Plane storage backends for ROSClaw.

Provides abstract SeekDBClient and concrete implementations:
- InMemoryKnowledgeStore: In-memory backend for testing
- SQLiteKnowledgeStore: SQLite backend for single-machine deployment
- SeekDBMySQLClient: MySQL-compatible SeekDB/OceanBase server deployment

Legacy aliases SeekDBMemoryClient and SeekDBSQLiteClient are kept for
backward compatibility but emit DeprecationWarning on instantiation.

Sprint 5 of DESIGN_SPRINT3_5.
"""

import contextlib
import json
import logging
import queue
import re
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

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
    # Memory 2.0 (PR-MEM-1) — typed embodied memory with evidence rows
    "memory_items": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "memory_type": "TEXT NOT NULL",
            "robot_id": "TEXT NOT NULL",
            "tenant_id": "TEXT",
            "project_id": "TEXT",
            "site_id": "TEXT",
            "body_id": "TEXT",
            "practice_id": "TEXT",
            "session_id": "TEXT",
            "episode_id": "TEXT",
            "task_id": "TEXT",
            "skill_id": "TEXT",
            "policy_id": "TEXT",
            "title": "TEXT",
            "document": "TEXT",
            "summary": "TEXT",
            "outcome": "TEXT",
            "reward": "REAL",
            "confidence": "REAL DEFAULT 1.0",
            "importance": "REAL DEFAULT 0.5",
            "novelty": "REAL DEFAULT 0.5",
            "quality_score": "REAL DEFAULT 0.5",
            "evidence_refs": "TEXT",
            "artifact_refs": "TEXT",
            "tags": "TEXT",
            "metadata": "TEXT",
            "embedding_model": "TEXT",
            "embedding_version": "TEXT",
            "content_hash": "TEXT",
            "event_time": "REAL",
            "created_at": "REAL",
            "updated_at": "REAL",
            "expires_at": "REAL",
            "schema_version": "TEXT",
            "status": "TEXT DEFAULT 'active'",
            "pinned": "INTEGER DEFAULT 0",
        },
        "indices": [
            "robot_id",
            "memory_type",
            "body_id",
            "task_id",
            "skill_id",
            "outcome",
            "event_time",
            "content_hash",
            "status",
            "practice_id",
        ],
    },
    "memory_evidence": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "memory_id": "TEXT NOT NULL",
            "evidence_type": "TEXT NOT NULL",
            "source_event_id": "TEXT",
            "artifact_uri": "TEXT",
            "byte_offset": "INTEGER",
            "sha256": "TEXT",
            "confidence": "REAL DEFAULT 1.0",
            "created_at": "REAL",
        },
        "indices": ["memory_id", "evidence_type", "source_event_id"],
    },
    # Memory 2.0 (PR-MEM-2) — embedding index lifecycle registry
    "memory_index_registry": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "table_or_collection": "TEXT NOT NULL",
            "embedder_type": "TEXT",
            "model_name": "TEXT",
            "model_revision": "TEXT",
            "dimension": "INTEGER",
            "distance_metric": "TEXT",
            "corpus_hash": "TEXT",
            "index_version": "INTEGER DEFAULT 1",
            "record_count": "INTEGER DEFAULT 0",
            "created_at": "REAL",
            "completed_at": "REAL",
            "status": "TEXT",
        },
        "indices": ["table_or_collection", "status", "model_name"],
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
            "body_condition_failure": "INTEGER",
            "body_sense_evidence": "TEXT",
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
            "body_condition_failure": "INTEGER",
            "body_sense_evidence": "TEXT",
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


class InMemoryKnowledgeStore(SeekDBClient):
    """In-memory knowledge store for testing.

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


class SQLiteKnowledgeStore(SeekDBClient):
    """SQLite-backed knowledge store for single-machine deployment.

    Thread-safe: the connection is opened with ``check_same_thread=False`` and
    all operations are protected by a recursive lock so that callers from
    background threads (preloader, event bus handlers, etc.) can safely share
    one store instance.
    """

    def __init__(
        self,
        db_path: str = "~/.rosclaw/memory/knowledge.sqlite",
        *,
        vector_enabled: bool = False,
        embedder: Any | None = None,
    ):
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._vector_enabled = vector_enabled
        self._embedder = embedder
        self._vector_store: Any | None = None
        self._embedder_warmed = False
        self._warmed_tables: set[str] = set()

    def _has_table(self, table: str) -> bool:
        """Return True if *table* exists in the SQLite database."""
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
        return row is not None

    _TABLE_TEXT_FIELDS: dict[str, list[str]] = {
        "knowledge_graph": ["subject", "predicate", "object"],
        "experience_graph": ["instruction", "tags", "error_details"],
        "heuristic_rules": ["condition", "action"],
        "failures": ["failure_type", "root_cause", "recovery_hint"],
        "praxis_events": ["event_type", "payload"],
        "episodes": ["task_id", "outcome"],
        "success_patterns": ["skill_id"],
        "artifacts": ["artifact_type", "uri", "metadata"],
    }

    # Columns that are persisted as JSON strings and should be re-hydrated on read.
    _JSON_COLUMN_NAMES: frozenset[str] = frozenset(
        {
            "body_sense_evidence",
            "changes",
            "cot_trace",
            "data",
            "delta",
            "dofs",
            "error_details",
            "evidence_refs",
            "failures",
            "improvement",
            "metrics",
            "metadata",
            "original_outcome",
            "parameter_patch",
            "payload",
            "policy_params",
            "prerequisites",
            "real_value",
            "retry_outcome",
            "search_space",
            "sim_value",
            "tags",
            "trajectory",
        }
    )

    def _json_columns(self, table: str) -> list[str]:
        """Return the columns of *table* that should be JSON-deserialized."""
        schema = SEEKDB_SCHEMAS.get(table, {})
        return [col for col in schema.get("columns", {}) if col in self._JSON_COLUMN_NAMES]

    def _extract_warmup_text(self, table: str, row: dict[str, Any]) -> str:
        """Build a single searchable text blob for *row* in *table*."""
        fields = self._TABLE_TEXT_FIELDS.get(table, [])
        if not fields:
            # Fallback: concatenate all string-ish values.
            fields = [k for k in row if k != "id"]
        parts: list[str] = []
        for field in fields:
            value = row.get(field)
            if value is None or value == "":
                continue
            if isinstance(value, (list, dict)):
                with contextlib.suppress(Exception):
                    parts.append(json.dumps(value, ensure_ascii=False))
            elif isinstance(value, str):
                parts.append(value)
        return " ".join(parts)

    def warmup_embedder(self, tables: list[str] | None = None) -> dict[str, int]:
        """Fit the embedder on existing rows and index them into the vector store.

        This eliminates cold-start latency for the first vector query and makes
        historical knowledge/experience/failure records searchable through the
        hybrid retrieval path.

        Returns a dict mapping table name to the number of rows indexed.
        """
        if not self._vector_enabled:
            return {}
        self._ensure_vector_store()
        if self._vector_store is None or self._embedder is None:
            self._embedder_warmed = True
            return {}

        tables = tables or list(self._TABLE_TEXT_FIELDS.keys())
        # Only warm tables that have not been warmed yet.
        tables = [t for t in tables if t not in self._warmed_tables]
        if not tables:
            self._embedder_warmed = True
            return {}

        counts: dict[str, int] = {}
        all_texts: list[str] = []
        rows_by_table: dict[str, list[tuple[str, str]]] = {}

        for table in tables:
            if not self._has_table(table):
                continue
            try:
                rows = self.query(table, limit=10_000)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Warmup query failed for %s: %s", table, exc)
                continue
            table_rows: list[tuple[str, str]] = []
            for row in rows:
                record_id = row.get("id")
                if not record_id:
                    continue
                text = self._extract_warmup_text(table, row)
                if not text:
                    continue
                all_texts.append(text)
                table_rows.append((str(record_id), text))
            rows_by_table[table] = table_rows

        # For the TF-IDF embedder, fit once on the entire corpus so that
        # subsequent encodings share a stable vocabulary.
        try:
            from rosclaw.storage.vector import TfidfEmbedder

            if (
                isinstance(self._embedder, TfidfEmbedder)
                and self._embedder._vocab is None
                and all_texts
            ):
                self._embedder.fit(all_texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedder warmup fit failed: %s", exc)

        for table, table_rows in rows_by_table.items():
            if not table_rows:
                self._warmed_tables.add(table)
                continue
            try:
                texts = [text for _, text in table_rows]
                if hasattr(self._embedder, "encode_batch"):
                    embeddings = self._embedder.encode_batch(texts)
                else:
                    embeddings = [self._embedder.encode(text) for text in texts]
                self._vector_store.upsert_many(
                    table,
                    [
                        (record_id, text, embedding)
                        for (record_id, text), embedding in zip(table_rows, embeddings, strict=True)
                    ],
                )
                counts[table] = len(table_rows)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Warmup embedding failed for %s: %s", table, exc)
            self._warmed_tables.add(table)

        self._embedder_warmed = True
        logger.info("Embedder warmup complete: %s", counts)
        return counts

    def _ensure_vector_store(self) -> None:
        """Create the embedder and vector store if vector support is enabled."""
        if not self._vector_enabled:
            return
        from rosclaw.storage.vector import SQLiteVectorStore, TfidfEmbedder

        if self._embedder is None:
            self._embedder = TfidfEmbedder()
        if self._vector_store is None:
            self._vector_store = SQLiteVectorStore(self)

    def similar(
        self,
        table: str,
        query_text: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Hybrid keyword + vector search on *table*.

        Requires ``vector_enabled=True`` and an embedder.  If vector support is
        not configured, falls back to a simple keyword search over text fields.
        """
        self._ensure_vector_store()
        if self._vector_store is not None and self._embedder is not None:
            if table not in self._warmed_tables:
                self.warmup_embedder([table])
            if table in self._warmed_tables:
                return self._vector_store.hybrid_search(
                    table, query_text, self._embedder, filters=filters, limit=limit
                )
        # Fallback: keyword search against string columns of the table.
        return self._keyword_fallback(table, query_text, filters, limit)

    def _index_vector(
        self,
        table: str,
        record_id: str,
        record: dict[str, Any],
    ) -> None:
        """Keep the vector index in sync with a freshly written record.

        Only indexes tables that have already been warmed, so inserts into
        unqueried tables do not pay the embedding cost.
        """
        if not self._vector_enabled:
            return
        if table not in self._warmed_tables:
            return
        self._ensure_vector_store()
        if self._vector_store is None or self._embedder is None:
            return
        text = self._extract_warmup_text(table, record)
        if not text:
            return
        try:
            embedding = self._embedder.encode(text)
            self._vector_store.upsert(table, record_id, text, embedding)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Vector index update failed for %s %s: %s", table, record_id, exc)

    def _keyword_fallback(
        self,
        table: str,
        query_text: str,
        filters: dict[str, Any] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        from rosclaw.storage.vector import _tokenize

        query_tokens = set(_tokenize(query_text))
        results = []
        rows = self.query(table, filters=filters, limit=limit * 10)
        for row in rows:
            row_text = self._extract_warmup_text(table, row)
            if not row_text:
                continue
            tokens = set(_tokenize(row_text))
            overlap = len(query_tokens & tokens)
            if overlap:
                results.append({**row, "score": overlap / max(len(query_tokens), 1)})
        results.sort(key=lambda r: -r["score"])
        return results[:limit]

    def connect(self) -> None:
        if self._conn is not None:
            return
        path = Path(self._db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                PRAGMA foreign_keys=ON;
                PRAGMA busy_timeout=5000;
                PRAGMA temp_store=MEMORY;
                """
            )
            self._create_tables()

    def is_connected(self) -> bool:
        return self._conn is not None

    def disconnect(self) -> None:
        with self._lock:
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
        with self._lock:
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
            from rosclaw.storage.migrations import MigrationRunner

            MigrationRunner().apply(self._connection, "sqlite")

    def _migrate_missing_columns(self) -> None:
        """Add columns that were added to SEEKDB_SCHEMAS after table creation.

        SQLite ``ALTER TABLE ADD COLUMN`` rejects adding a ``NOT NULL`` column
        without a default value. We therefore strip ``NOT NULL`` and add a safe
        default when migrating older databases.
        """
        with self._lock:
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
                    if "NOT NULL" in safe_type.upper() and "DEFAULT" not in safe_type.upper():
                        safe_type = safe_type.replace("NOT NULL", "").strip()
                        safe_type += " DEFAULT ''"
                    try:
                        self._connection.execute(
                            f"ALTER TABLE {table_name} ADD COLUMN {col_name} {safe_type}"
                        )
                    except sqlite3.Error as exc:
                        logger.warning(
                            "Failed to add column %s to %s: %s", col_name, table_name, exc
                        )
            self._connection.commit()

    @staticmethod
    def _validate_table_and_columns(
        table: str,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        if table not in SEEKDB_SCHEMAS:
            raise ValueError(f"Unknown table: {table}")
        schema = SEEKDB_SCHEMAS[table]
        if columns:
            unknown = set(columns) - set(schema["columns"])
            if unknown:
                raise ValueError(f"Unknown columns for {table}: {sorted(unknown)}")
        return schema

    def insert(self, table: str, record: dict) -> str:
        self._validate_table_and_columns(table, list(record))
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
        set_clause = ", ".join(f"{k}=excluded.{k}" for k in serialized)
        with self._lock:
            self._connection.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {set_clause}",
                values,
            )
            self._connection.commit()
        self._index_vector(table, serialized["id"], record)
        return serialized["id"]

    def query(
        self,
        table: str,
        filters: dict | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        self._validate_table_and_columns(table, list(filters or {}))
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
            self._validate_table_and_columns(table, [key])
            sql += f" ORDER BY {key} {direction}"
        sql += " LIMIT ?"
        params.append(max(0, int(limit)))
        with self._lock:
            cursor = self._connection.execute(sql, params)
            rows = cursor.fetchall()
        results = []
        json_columns = self._json_columns(table)
        for row in rows:
            record = dict(row)
            for k in json_columns:
                v = record.get(k)
                if isinstance(v, str) and (v.startswith("[") or v.startswith("{")):
                    with contextlib.suppress(json.JSONDecodeError):
                        record[k] = json.loads(v)
            results.append(record)
        return results

    def update(self, table: str, record_id: str, updates: dict) -> bool:
        if not updates:
            return False
        self._validate_table_and_columns(table, list(updates))
        serialized = {}
        for k, v in updates.items():
            serialized[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        set_clause = ", ".join(f"{k} = ?" for k in serialized)
        values = list(serialized.values()) + [record_id]
        with self._lock:
            cursor = self._connection.execute(
                f"UPDATE {table} SET {set_clause} WHERE id = ?",
                values,
            )
            self._connection.commit()
        if cursor.rowcount > 0:
            # Re-read the updated record to keep the vector index in sync.
            rows = self.query(table, filters={"id": record_id}, limit=1)
            if rows:
                self._index_vector(table, record_id, rows[0])
        return cursor.rowcount > 0

    def count(self, table: str, filters: dict | None = None) -> int:
        self._validate_table_and_columns(table, list(filters or {}))
        sql = f"SELECT COUNT(*) FROM {table}"
        params = []
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(f"{k} = ?")
                params.append(v)
            sql += " WHERE " + " AND ".join(conditions)
        with self._lock:
            cursor = self._connection.execute(sql, params)
            return cursor.fetchone()[0]

    def delete(self, table: str, record_id: str) -> bool:
        self._validate_table_and_columns(table)
        with self._lock:
            cursor = self._connection.execute(
                f"DELETE FROM {table} WHERE id = ?",
                (record_id,),
            )
            self._connection.commit()
        return cursor.rowcount > 0

    def delete_where(self, table: str, filters: dict) -> int:
        if not filters:
            return 0
        self._validate_table_and_columns(table, list(filters))
        conditions = []
        params = []
        for k, v in filters.items():
            conditions.append(f"{k} = ?")
            params.append(v)
        sql = f"DELETE FROM {table} WHERE " + " AND ".join(conditions)
        with self._lock:
            cursor = self._connection.execute(sql, params)
            self._connection.commit()
        return cursor.rowcount


class _PooledConnection:
    """Context manager that returns a connection to its pool on exit."""

    def __init__(self, pool: "_ConnectionPool", conn: Any) -> None:
        self._pool = pool
        self._conn = conn
        self._broken = False

    def __enter__(self) -> Any:
        return self._conn

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            self._broken = True
        self._pool._release(self._conn, self._broken)


class _ConnectionPool:
    """Simple blocking connection pool for PyMySQL connections."""

    def __init__(
        self,
        creator: Any,
        pool_size: int = 4,
        timeout: float = 10.0,
    ) -> None:
        self._creator = creator
        self._max_size = max(1, pool_size)
        self._timeout = timeout
        self._pool: queue.Queue[Any] = queue.Queue(maxsize=self._max_size)
        self._lock = threading.Lock()
        self._created = 0
        self._closed = False

    def acquire(self) -> _PooledConnection:
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        conn: Any | None = None
        while conn is None:
            try:
                candidate = self._pool.get_nowait()
            except queue.Empty:
                candidate = None
            if candidate is not None:
                try:
                    ping = getattr(candidate, "ping", None)
                    if ping is not None:
                        ping(reconnect=True)
                    conn = candidate
                except Exception:
                    self._close_conn(candidate)
                    with self._lock:
                        self._created = max(0, self._created - 1)
                    logger.warning("Discarded unhealthy MySQL connection from pool")
                    continue
                if conn is not None:
                    break

            # Decide whether we may create a new connection while holding the
            # lock, then create it outside the lock so slow I/O does not block
            # other threads. If creation fails, decrement _created so capacity
            # is not leaked.
            with self._lock:
                may_create = self._created < self._max_size
                if may_create:
                    self._created += 1
            if may_create:
                try:
                    conn = self._creator()
                except Exception:
                    with self._lock:
                        self._created = max(0, self._created - 1)
                    raise
            if conn is None:
                conn = self._pool.get(timeout=self._timeout)
                try:
                    ping = getattr(conn, "ping", None)
                    if ping is not None:
                        ping(reconnect=True)
                except Exception:
                    self._close_conn(conn)
                    with self._lock:
                        self._created = max(0, self._created - 1)
                    conn = None
        return _PooledConnection(self, conn)

    def _release(self, conn: Any, is_broken: bool) -> None:
        if self._closed or is_broken or conn is None:
            self._close_conn(conn)
            with self._lock:
                self._created = max(0, self._created - 1)
            return
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            self._close_conn(conn)
            with self._lock:
                self._created = max(0, self._created - 1)

    def _close_conn(self, conn: Any) -> None:
        if conn is None:
            return
        with contextlib.suppress(Exception):
            conn.close()

    def close(self) -> None:
        self._closed = True
        while True:
            try:
                conn = self._pool.get_nowait()
                self._close_conn(conn)
            except queue.Empty:
                break
        with self._lock:
            self._created = 0


class SeekDBMySQLClient(SeekDBClient):
    """Experimental MySQL-compatible backend (SeekDB/OceanBase SQL port).

    URLs use a database DSN, for example
    ``mysql://root@127.0.0.1:2881/rosclaw``. Port 2881 is the SeekDB SQL
    protocol, not an HTTP endpoint.
    """

    _SUPPORTED_SCHEMES = {"mysql", "mysql+pymysql", "seekdb"}

    def __init__(
        self,
        url: str = "mysql://root@127.0.0.1:2881/rosclaw",
        *,
        pool_size: int = 4,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        write_timeout: float = 10.0,
    ):
        parsed = urlparse(url)
        if parsed.scheme not in self._SUPPORTED_SCHEMES:
            raise ValueError(
                "SeekDB server URL must use mysql://, mysql+pymysql://, or seekdb://; "
                "port 2881 is not an HTTP API"
            )
        if not parsed.hostname:
            raise ValueError("SeekDB server URL must include a hostname")

        database = parsed.path.lstrip("/") or "rosclaw"
        self._validate_identifier(database)
        self._url = url
        self._host = parsed.hostname
        self._port = parsed.port or 2881
        self._user = unquote(parsed.username or "root")
        self._password = unquote(parsed.password or "")
        self._database = database
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._write_timeout = write_timeout
        self._initialized = False
        self._pool = _ConnectionPool(
            creator=self._create_raw_connection,
            pool_size=pool_size,
            timeout=max(connect_timeout, 10.0),
        )

    def _open_connection(self, database: str | None) -> Any:
        """Create a PyMySQL connection, optionally selecting *database*."""
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError(
                "PyMySQL is required for SeekDB server connections; install rosclaw again "
                "with current dependencies"
            ) from exc

        kwargs: dict[str, Any] = {
            "host": self._host,
            "port": self._port,
            "user": self._user,
            "password": self._password,
            "charset": "utf8mb4",
            "autocommit": False,
            "connect_timeout": int(self._connect_timeout),
            "read_timeout": int(self._read_timeout),
            "write_timeout": int(self._write_timeout),
            "cursorclass": pymysql.cursors.DictCursor,
        }
        if database is not None:
            kwargs["database"] = database
        return pymysql.connect(**kwargs)

    def _create_raw_connection(self) -> Any:
        """Create a pooled PyMySQL connection bound to the configured database."""
        return self._open_connection(self._database)

    @staticmethod
    def _validate_identifier(identifier: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier!r}")
        return identifier

    @classmethod
    def _quoted(cls, identifier: str) -> str:
        return f"`{cls._validate_identifier(identifier)}`"

    def connect(self) -> None:
        if self._initialized:
            return

        try:
            import pymysql  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "PyMySQL is required for SeekDB server connections; install rosclaw again "
                "with current dependencies"
            ) from exc

        bootstrap = self._open_connection(None)
        try:
            with bootstrap.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS {self._quoted(self._database)} "
                    "DEFAULT CHARACTER SET utf8mb4"
                )
            bootstrap.commit()
        finally:
            bootstrap.close()

        with self._pool.acquire() as connection:
            try:
                self._create_tables(connection)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        self._initialized = True

    def is_connected(self) -> bool:
        return self._initialized and not self._pool._closed

    def disconnect(self) -> None:
        self._pool.close()
        self._initialized = False

    @property
    def _connection(self) -> _PooledConnection:
        if not self._initialized:
            self.connect()
        return self._pool.acquire()

    @staticmethod
    def _mysql_column_type(
        column_name: str,
        declaration: str,
        *,
        indexed: bool,
    ) -> str:
        normalized = declaration.upper()
        not_null = "NOT NULL" in normalized
        primary_key = "PRIMARY KEY" in normalized
        default_match = re.search(r"\bDEFAULT\s+(.+)$", declaration, flags=re.IGNORECASE)

        if "INTEGER" in normalized:
            sql_type = "BIGINT"
        elif "REAL" in normalized:
            sql_type = "DOUBLE"
        elif primary_key or indexed:
            sql_type = "VARCHAR(255)"
        else:
            sql_type = "LONGTEXT"

        parts = [sql_type]
        if not_null:
            parts.append("NOT NULL")
        if primary_key:
            parts.append("PRIMARY KEY")
        if default_match and sql_type != "LONGTEXT":
            parts.append(f"DEFAULT {default_match.group(1)}")
        return " ".join(parts)

    @classmethod
    def _validate_table_and_columns(
        cls,
        table: str,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        if table not in SEEKDB_SCHEMAS:
            raise ValueError(f"Unknown table: {table}")
        schema = SEEKDB_SCHEMAS[table]
        if columns:
            unknown = set(columns) - set(schema["columns"])
            if unknown:
                raise ValueError(f"Unknown columns for {table}: {sorted(unknown)}")
        return schema

    def _create_tables(self, connection: Any) -> None:
        with connection.cursor() as cursor:
            for table_name, schema in SEEKDB_SCHEMAS.items():
                indexed = set(schema.get("indices", []))
                column_sql = ", ".join(
                    f"{self._quoted(column_name)} "
                    f"{self._mysql_column_type(column_name, declaration, indexed=column_name in indexed)}"
                    for column_name, declaration in schema["columns"].items()
                )
                cursor.execute(
                    f"CREATE TABLE IF NOT EXISTS {self._quoted(table_name)} ({column_sql}) "
                    "DEFAULT CHARACTER SET utf8mb4"
                )

                cursor.execute(f"SHOW COLUMNS FROM {self._quoted(table_name)}")
                existing_columns = {
                    row["Field"] if isinstance(row, dict) else row[0] for row in cursor.fetchall()
                }
                for column_name, declaration in schema["columns"].items():
                    if column_name in existing_columns:
                        continue
                    cursor.execute(
                        f"ALTER TABLE {self._quoted(table_name)} ADD COLUMN "
                        f"{self._quoted(column_name)} "
                        f"{self._mysql_column_type(column_name, declaration, indexed=column_name in indexed)}"
                    )

                cursor.execute(f"SHOW INDEX FROM {self._quoted(table_name)}")
                existing_indices = {
                    row["Key_name"] if isinstance(row, dict) else row[2]
                    for row in cursor.fetchall()
                }
                for column_name in indexed:
                    index_name = f"idx_{table_name}_{column_name}"
                    if index_name in existing_indices:
                        continue
                    cursor.execute(
                        f"CREATE INDEX {self._quoted(index_name)} "
                        f"ON {self._quoted(table_name)} ({self._quoted(column_name)})"
                    )
            from rosclaw.storage.migrations import MigrationRunner

            MigrationRunner().apply(connection, "mysql")

    @staticmethod
    def _serialize(value: Any) -> Any:
        return json.dumps(value) if isinstance(value, (list, dict)) else value

    @staticmethod
    def _deserialize_record(record: dict[str, Any]) -> dict[str, Any]:
        decoded = dict(record)
        for key, value in decoded.items():
            if isinstance(value, str) and (value.startswith("[") or value.startswith("{")):
                with contextlib.suppress(json.JSONDecodeError):
                    decoded[key] = json.loads(value)
        return decoded

    def insert(self, table: str, record: dict) -> str:
        schema = self._validate_table_and_columns(table, list(record))
        serialized = {key: self._serialize(value) for key, value in record.items()}
        serialized.setdefault("id", str(uuid.uuid4()))
        self._validate_table_and_columns(table, list(serialized))

        columns = list(serialized)
        placeholders = ", ".join("%s" for _ in columns)
        update_columns = [column for column in columns if column != "id"]
        update_clause = ", ".join(
            f"{self._quoted(column)} = VALUES({self._quoted(column)})" for column in update_columns
        )
        if not update_clause:
            update_clause = f"{self._quoted('id')} = {self._quoted('id')}"

        with self._connection as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"INSERT INTO {self._quoted(table)} "
                    f"({', '.join(self._quoted(column) for column in columns)}) "
                    f"VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_clause}",
                    list(serialized.values()),
                )
            conn.commit()
        assert "id" in schema["columns"]
        return str(serialized["id"])

    def query(
        self,
        table: str,
        filters: dict | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        self._validate_table_and_columns(table, list(filters or {}))
        sql = f"SELECT * FROM {self._quoted(table)}"
        params: list[Any] = []
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"{self._quoted(key)} = %s")
                params.append(value)
            sql += " WHERE " + " AND ".join(conditions)
        if order_by:
            descending = order_by.startswith("-")
            key = order_by.lstrip("-")
            self._validate_table_and_columns(table, [key])
            sql += f" ORDER BY {self._quoted(key)} {'DESC' if descending else 'ASC'}"
        sql += " LIMIT %s"
        params.append(max(0, int(limit)))

        with self._connection as conn, conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [self._deserialize_record(dict(row)) for row in rows]

    def update(self, table: str, record_id: str, updates: dict) -> bool:
        if not updates:
            return False
        self._validate_table_and_columns(table, list(updates))
        serialized = {key: self._serialize(value) for key, value in updates.items()}
        set_clause = ", ".join(f"{self._quoted(key)} = %s" for key in serialized)
        with self._connection as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {self._quoted(table)} SET {set_clause} WHERE {self._quoted('id')} = %s",
                    [*serialized.values(), record_id],
                )
                changed = cursor.rowcount > 0
            conn.commit()
        return changed

    def count(self, table: str, filters: dict | None = None) -> int:
        self._validate_table_and_columns(table, list(filters or {}))
        sql = f"SELECT COUNT(*) AS row_count FROM {self._quoted(table)}"
        params: list[Any] = []
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"{self._quoted(key)} = %s")
                params.append(value)
            sql += " WHERE " + " AND ".join(conditions)
        with self._connection as conn, conn.cursor() as cursor:
            cursor.execute(sql, params)
            row = cursor.fetchone()
        if isinstance(row, dict):
            return int(row["row_count"])
        return int(row[0])

    def delete(self, table: str, record_id: str) -> bool:
        self._validate_table_and_columns(table)
        with self._connection as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self._quoted(table)} WHERE {self._quoted('id')} = %s",
                    (record_id,),
                )
                deleted = cursor.rowcount > 0
            conn.commit()
        return deleted

    def delete_where(self, table: str, filters: dict) -> int:
        if not filters:
            return 0
        self._validate_table_and_columns(table, list(filters))
        conditions = []
        params = []
        for key, value in filters.items():
            conditions.append(f"{self._quoted(key)} = %s")
            params.append(value)
        with self._connection as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self._quoted(table)} WHERE " + " AND ".join(conditions),
                    params,
                )
                deleted = cursor.rowcount
            conn.commit()
        return int(deleted)


class SeekDBMemoryClient(InMemoryKnowledgeStore):
    """Deprecated alias for :class:`InMemoryKnowledgeStore`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import warnings

        warnings.warn(
            "SeekDBMemoryClient is deprecated; use InMemoryKnowledgeStore",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class SeekDBSQLiteClient(SQLiteKnowledgeStore):
    """Deprecated alias for :class:`SQLiteKnowledgeStore`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import warnings

        warnings.warn(
            "SeekDBSQLiteClient is deprecated; use SQLiteKnowledgeStore",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
