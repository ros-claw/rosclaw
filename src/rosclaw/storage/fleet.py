"""OceanBase Fleet Knowledge Plane (PR-OB-1, §8).

Center-side schema and sync for multi-robot fleets::

    robot local outbox → fleet ingest API (idempotent upsert) → sync watermark

Every shared table carries ``tenant_id / project_id / site_id / robot_id /
body_id`` so a fleet can isolate tenants, group by project/site, and trace
memories back to the exact robot body that produced them.  The center being
down never affects a robot's local task — sync is strictly outbox-driven.

Validated against the real container ``OceanBase seekdb-v1.3.0.0``
(quay.io/oceanbase/seekdb on :2881).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger("rosclaw.storage.fleet")

FLEET_SCHEMA_VERSION = "fleet.v1"

# §8.2 fleet tables — all tenant-scoped.
FLEET_DDL: dict[str, str] = {
    "fleet_robots": """
        CREATE TABLE IF NOT EXISTS fleet_robots (
            robot_id VARCHAR(128) NOT NULL,
            tenant_id VARCHAR(128) NOT NULL,
            project_id VARCHAR(128),
            site_id VARCHAR(128),
            robot_type VARCHAR(64),
            body_ids JSON,
            status VARCHAR(32) DEFAULT 'active',
            registered_at DOUBLE,
            metadata JSON,
            PRIMARY KEY (tenant_id, robot_id)
        )
    """,
    "fleet_episodes": """
        CREATE TABLE IF NOT EXISTS fleet_episodes (
            episode_id VARCHAR(128) NOT NULL,
            tenant_id VARCHAR(128) NOT NULL,
            project_id VARCHAR(128),
            site_id VARCHAR(128),
            robot_id VARCHAR(128) NOT NULL,
            body_id VARCHAR(128),
            practice_id VARCHAR(128),
            task_id VARCHAR(128),
            skill_id VARCHAR(128),
            outcome VARCHAR(32),
            event_count BIGINT,
            started_at DOUBLE,
            ended_at DOUBLE,
            metadata JSON,
            PRIMARY KEY (tenant_id, episode_id),
            KEY idx_fleet_episodes_robot (tenant_id, robot_id, started_at),
            KEY idx_fleet_episodes_task (tenant_id, task_id, outcome)
        )
    """,
    "fleet_memories": """
        CREATE TABLE IF NOT EXISTS fleet_memories (
            memory_id VARCHAR(128) NOT NULL,
            tenant_id VARCHAR(128) NOT NULL,
            project_id VARCHAR(128),
            site_id VARCHAR(128),
            robot_id VARCHAR(128) NOT NULL,
            body_id VARCHAR(128),
            memory_type VARCHAR(64) NOT NULL,
            title TEXT,
            document MEDIUMTEXT,
            outcome VARCHAR(32),
            confidence DOUBLE,
            importance DOUBLE,
            content_hash VARCHAR(128),
            event_time DOUBLE,
            tags JSON,
            metadata JSON,
            updated_at DOUBLE,
            PRIMARY KEY (tenant_id, memory_id),
            KEY idx_fleet_memories_robot (tenant_id, robot_id, memory_type),
            KEY idx_fleet_memories_body (tenant_id, body_id, memory_type),
            KEY idx_fleet_memories_hash (tenant_id, content_hash)
        )
    """,
    "fleet_failures": """
        CREATE TABLE IF NOT EXISTS fleet_failures (
            failure_id VARCHAR(128) NOT NULL,
            tenant_id VARCHAR(128) NOT NULL,
            robot_id VARCHAR(128) NOT NULL,
            body_id VARCHAR(128),
            memory_id VARCHAR(128),
            failure_type VARCHAR(64),
            root_cause TEXT,
            event_time DOUBLE,
            metadata JSON,
            PRIMARY KEY (tenant_id, failure_id),
            KEY idx_fleet_failures_robot (tenant_id, robot_id, event_time)
        )
    """,
    "fleet_interventions": """
        CREATE TABLE IF NOT EXISTS fleet_interventions (
            intervention_id VARCHAR(128) NOT NULL,
            tenant_id VARCHAR(128) NOT NULL,
            robot_id VARCHAR(128) NOT NULL,
            body_id VARCHAR(128),
            memory_id VARCHAR(128),
            action_template JSON,
            outcome VARCHAR(32),
            event_time DOUBLE,
            metadata JSON,
            PRIMARY KEY (tenant_id, intervention_id),
            KEY idx_fleet_interventions_body (tenant_id, body_id, outcome)
        )
    """,
    "fleet_skill_evidence": """
        CREATE TABLE IF NOT EXISTS fleet_skill_evidence (
            evidence_id VARCHAR(128) NOT NULL,
            tenant_id VARCHAR(128) NOT NULL,
            robot_id VARCHAR(128) NOT NULL,
            body_id VARCHAR(128),
            skill_id VARCHAR(128) NOT NULL,
            success_count BIGINT DEFAULT 0,
            failure_count BIGINT DEFAULT 0,
            sample_count BIGINT DEFAULT 0,
            event_time DOUBLE,
            metadata JSON,
            PRIMARY KEY (tenant_id, skill_id, evidence_id),
            KEY idx_fleet_skill_outcome (tenant_id, skill_id)
        )
    """,
    "fleet_sync_watermarks": """
        CREATE TABLE IF NOT EXISTS fleet_sync_watermarks (
            robot_id VARCHAR(128) NOT NULL,
            tenant_id VARCHAR(128) NOT NULL,
            entity_type VARCHAR(64) NOT NULL,
            last_entity_id VARCHAR(128),
            last_synced_at DOUBLE,
            records_synced BIGINT DEFAULT 0,
            PRIMARY KEY (tenant_id, robot_id, entity_type)
        )
    """,
}

# §8.3 retrieval projection — heap table with vector + fulltext for HYBRID_SEARCH.
# The ngram parser is required for CJK fulltext (OceanBase default parser
# tokenizes on whitespace and misses Chinese terms entirely).
FLEET_SEARCH_DDL = """
    CREATE TABLE IF NOT EXISTS fleet_memory_search (
        memory_id VARCHAR(128) NOT NULL,
        tenant_id VARCHAR(128) NOT NULL,
        robot_id VARCHAR(128) NOT NULL,
        body_id VARCHAR(128),
        memory_type VARCHAR(64),
        document MEDIUMTEXT,
        metadata JSON,
        embedding VECTOR(384),
        event_time DOUBLE,
        PRIMARY KEY (tenant_id, memory_id),
        FULLTEXT INDEX ft_doc (document) WITH PARSER ngram
    )
"""


def _now() -> float:
    return time.time()


def _content_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


class FleetPlane:
    """Center-side fleet plane over a MySQL-compatible OceanBase connection."""

    def __init__(self, connection: Any, *, tenant_id: str = "default"):
        self._conn = connection
        self._tenant = tenant_id

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def ensure_schema(self) -> list[str]:
        """Create all fleet tables (idempotent). Returns created/existing names."""
        created = []
        with self._conn.cursor() as cursor:
            for name, ddl in FLEET_DDL.items():
                cursor.execute(ddl)
                created.append(name)
            try:
                cursor.execute(FLEET_SEARCH_DDL)
                created.append("fleet_memory_search")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "fleet_memory_search DDL failed (vector/fulltext unsupported?): %s", exc
                )
        self._conn.commit()
        return created

    # ------------------------------------------------------------------
    # Ingest (idempotent upsert; outbox-driven)
    # ------------------------------------------------------------------

    def upsert_robot(self, robot: dict[str, Any]) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                REPLACE INTO fleet_robots
                    (robot_id, tenant_id, project_id, site_id, robot_type, body_ids, status, registered_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    robot["robot_id"],
                    robot.get("tenant_id", self._tenant),
                    robot.get("project_id"),
                    robot.get("site_id"),
                    robot.get("robot_type"),
                    json.dumps(robot.get("body_ids", [])),
                    robot.get("status", "active"),
                    robot.get("registered_at", _now()),
                    json.dumps(robot.get("metadata", {})),
                ),
            )
        self._conn.commit()

    def upsert_memory(self, memory: dict[str, Any]) -> None:
        """Idempotent fleet memory upsert keyed by (tenant_id, memory_id)."""
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                REPLACE INTO fleet_memories
                    (memory_id, tenant_id, project_id, site_id, robot_id, body_id,
                     memory_type, title, document, outcome, confidence, importance,
                     content_hash, event_time, tags, metadata, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    memory["memory_id"],
                    memory.get("tenant_id", self._tenant),
                    memory.get("project_id"),
                    memory.get("site_id"),
                    memory["robot_id"],
                    memory.get("body_id"),
                    memory["memory_type"],
                    memory.get("title"),
                    memory.get("document"),
                    memory.get("outcome"),
                    memory.get("confidence"),
                    memory.get("importance"),
                    memory.get("content_hash")
                    or _content_hash(memory.get("title", ""), memory.get("document", "")),
                    memory.get("event_time", _now()),
                    json.dumps(memory.get("tags", [])),
                    json.dumps(memory.get("metadata", {})),
                    _now(),
                ),
            )
        self._conn.commit()

    def upsert_episode(self, episode: dict[str, Any]) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                REPLACE INTO fleet_episodes
                    (episode_id, tenant_id, project_id, site_id, robot_id, body_id,
                     practice_id, task_id, skill_id, outcome, event_count, started_at, ended_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    episode["episode_id"],
                    episode.get("tenant_id", self._tenant),
                    episode.get("project_id"),
                    episode.get("site_id"),
                    episode["robot_id"],
                    episode.get("body_id"),
                    episode.get("practice_id"),
                    episode.get("task_id"),
                    episode.get("skill_id"),
                    episode.get("outcome"),
                    episode.get("event_count"),
                    episode.get("started_at"),
                    episode.get("ended_at"),
                    json.dumps(episode.get("metadata", {})),
                ),
            )
        self._conn.commit()

    def upsert_skill_evidence(self, evidence: dict[str, Any]) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                REPLACE INTO fleet_skill_evidence
                    (evidence_id, tenant_id, robot_id, body_id, skill_id,
                     success_count, failure_count, sample_count, event_time, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    evidence["evidence_id"],
                    evidence.get("tenant_id", self._tenant),
                    evidence["robot_id"],
                    evidence.get("body_id"),
                    evidence["skill_id"],
                    evidence.get("success_count", 0),
                    evidence.get("failure_count", 0),
                    evidence.get("sample_count", 0),
                    evidence.get("event_time", _now()),
                    json.dumps(evidence.get("metadata", {})),
                ),
            )
        self._conn.commit()

    def bump_watermark(
        self, robot_id: str, entity_type: str, last_entity_id: str, count: int
    ) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO fleet_sync_watermarks
                    (robot_id, tenant_id, entity_type, last_entity_id, last_synced_at, records_synced)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    last_entity_id = VALUES(last_entity_id),
                    last_synced_at = VALUES(last_synced_at),
                    records_synced = records_synced + VALUES(records_synced)
                """,
                (robot_id, self._tenant, entity_type, last_entity_id, _now(), count),
            )
        self._conn.commit()

    def watermark(self, robot_id: str, entity_type: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "SELECT last_entity_id, last_synced_at, records_synced FROM fleet_sync_watermarks "
                "WHERE tenant_id = %s AND robot_id = %s AND entity_type = %s",
                (self._tenant, robot_id, entity_type),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {"last_entity_id": row[0], "last_synced_at": row[1], "records_synced": row[2]}

    # ------------------------------------------------------------------
    # Tenant-scoped queries
    # ------------------------------------------------------------------

    def query_memories(
        self,
        *,
        robot_id: str | None = None,
        body_id: str | None = None,
        memory_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses = ["tenant_id = %s"]
        params: list[Any] = [self._tenant]
        if robot_id:
            clauses.append("robot_id = %s")
            params.append(robot_id)
        if body_id:
            clauses.append("body_id = %s")
            params.append(body_id)
        if memory_type:
            clauses.append("memory_type = %s")
            params.append(memory_type)
        params.append(limit)
        with self._conn.cursor() as cursor:
            cursor.execute(
                "SELECT memory_id, robot_id, body_id, memory_type, title, outcome, content_hash, event_time "
                f"FROM fleet_memories WHERE {' AND '.join(clauses)} "
                "ORDER BY event_time DESC LIMIT %s",
                params,
            )
            rows = cursor.fetchall()
        return [
            {
                "memory_id": row[0],
                "robot_id": row[1],
                "body_id": row[2],
                "memory_type": row[3],
                "title": row[4],
                "outcome": row[5],
                "content_hash": row[6],
                "event_time": row[7],
            }
            for row in rows
        ]

    def skill_success_pattern(self, skill_id: str) -> list[dict[str, Any]]:
        """Cross-robot success pattern for one skill within this tenant."""
        with self._conn.cursor() as cursor:
            cursor.execute(
                "SELECT robot_id, body_id, SUM(success_count), SUM(failure_count), SUM(sample_count) "
                "FROM fleet_skill_evidence WHERE tenant_id = %s AND skill_id = %s "
                "GROUP BY robot_id, body_id",
                (self._tenant, skill_id),
            )
            rows = cursor.fetchall()
        return [
            {
                "robot_id": row[0],
                "body_id": row[1],
                "success_count": int(row[2] or 0),
                "failure_count": int(row[3] or 0),
                "sample_count": int(row[4] or 0),
            }
            for row in rows
        ]


class FleetIngestCommitter:
    """Outbox committer projecting local memory records into the fleet plane."""

    def __init__(self, plane: FleetPlane, robot_id: str):
        self._plane = plane
        self._robot_id = robot_id

    def save_to_seekdb(self, payload: dict[str, Any]) -> None:
        memory = dict(payload)
        memory.pop("idempotency_key", None)
        memory.setdefault("robot_id", self._robot_id)
        memory.setdefault("memory_id", memory.get("id"))
        self._plane.upsert_memory(memory)
        self._plane.bump_watermark(self._robot_id, "memory", str(memory["memory_id"]), 1)

    def save_to_seekdb_batch(self, payloads: list[dict[str, Any]]) -> None:
        for payload in payloads:
            memory = dict(payload)
            memory.pop("idempotency_key", None)
            memory.setdefault("robot_id", self._robot_id)
            memory.setdefault("memory_id", memory.get("id"))
            self._plane.upsert_memory(memory)
        if payloads:
            last = payloads[-1]
            self._plane.bump_watermark(
                self._robot_id,
                "memory",
                str(last.get("memory_id") or last.get("id")),
                len(payloads),
            )
