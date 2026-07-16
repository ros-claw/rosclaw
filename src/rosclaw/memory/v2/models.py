"""ROSClaw Memory 2.0 — structured embodied memory data model (PR-MEM-1).

Replaces the generic ``experience_graph`` event storage with typed,
evidence-backed memory records.  Every memory item must be traceable to at
least one :class:`MemoryEvidence` row pointing at practice events, episode
summaries, MCAP, telemetry windows, human feedback, or critic results —
"禁止只存一段无来源文本".
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "memory.v2"


class MemoryType(StrEnum):
    """The ten canonical memory categories (数据库优化v2.md §5.2)."""

    EPISODIC = "episodic"
    FAILURE = "failure"
    INTERVENTION = "intervention"
    PROCEDURAL = "procedural"
    SEMANTIC = "semantic"
    BODY = "body"
    SPATIAL = "spatial"
    SKILL = "skill"
    HUMAN_FEEDBACK = "human_feedback"
    SIM2REAL = "sim2real"


class MemoryStatus(StrEnum):
    """Lifecycle status of a memory item."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"


class EvidenceType(StrEnum):
    """Where a piece of memory evidence originates."""

    PRACTICE_EVENT = "practice_event"
    EPISODE_SUMMARY = "episode_summary"
    MCAP = "mcap"
    FRAME = "frame"
    TELEMETRY_WINDOW = "telemetry_window"
    HUMAN_FEEDBACK = "human_feedback"
    CRITIC_RESULT = "critic_result"


class GateDecision(StrEnum):
    """MemoryWriteGate outcomes (数据库优化v2.md §5.5)."""

    STORE = "STORE"
    MERGE = "MERGE"
    UPDATE = "UPDATE"
    IGNORE = "IGNORE"
    QUARANTINE = "QUARANTINE"


def _content_hash(*parts: str) -> str:
    """Stable SHA-256 over the semantic identity of a memory."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


@dataclass
class MemoryItem:
    """Unified embodied memory record (数据库优化v2.md §5.3)."""

    memory_type: str
    robot_id: str
    title: str
    document: str

    memory_id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:16]}")
    summary: str = ""

    tenant_id: str | None = None
    project_id: str | None = None
    site_id: str | None = None

    body_id: str | None = None
    practice_id: str | None = None
    session_id: str | None = None
    episode_id: str | None = None
    task_id: str | None = None
    skill_id: str | None = None
    policy_id: str | None = None

    outcome: str | None = None
    reward: float | None = None
    confidence: float = 1.0
    importance: float = 0.5
    novelty: float = 0.5
    quality_score: float = 0.5

    evidence_refs: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    embedding_model: str | None = None
    embedding_version: str | None = None
    content_hash: str = ""

    event_time: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: float | None = None

    schema_version: str = SCHEMA_VERSION
    status: str = MemoryStatus.ACTIVE.value
    pinned: bool = False

    def compute_content_hash(self) -> str:
        """Semantic identity: type + robot/body + title + document."""
        return _content_hash(
            self.memory_type,
            self.robot_id or "",
            self.body_id or "",
            self.title.strip().lower(),
            self.document.strip(),
        )

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = self.compute_content_hash()

    def to_record(self) -> dict[str, Any]:
        """Serialize to a ``memory_items`` table row."""
        return {
            "id": self.memory_id,
            "memory_type": self.memory_type,
            "robot_id": self.robot_id,
            "tenant_id": self.tenant_id,
            "project_id": self.project_id,
            "site_id": self.site_id,
            "body_id": self.body_id,
            "practice_id": self.practice_id,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "skill_id": self.skill_id,
            "policy_id": self.policy_id,
            "title": self.title,
            "document": self.document,
            "summary": self.summary,
            "outcome": self.outcome,
            "reward": self.reward,
            "confidence": self.confidence,
            "importance": self.importance,
            "novelty": self.novelty,
            "quality_score": self.quality_score,
            "evidence_refs": json.dumps(self.evidence_refs, ensure_ascii=False),
            "artifact_refs": json.dumps(self.artifact_refs, ensure_ascii=False),
            "tags": json.dumps(self.tags, ensure_ascii=False),
            "metadata": json.dumps(self.metadata, ensure_ascii=False),
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "content_hash": self.content_hash,
            "event_time": self.event_time,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "schema_version": self.schema_version,
            "status": self.status,
            "pinned": 1 if self.pinned else 0,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> MemoryItem:
        """Deserialize from a ``memory_items`` table row."""

        def _json_list(value: Any) -> list:
            if isinstance(value, list):
                return value
            if isinstance(value, str) and value:
                try:
                    parsed = json.loads(value)
                    return parsed if isinstance(parsed, list) else []
                except json.JSONDecodeError:
                    return []
            return []

        def _json_dict(value: Any) -> dict:
            if isinstance(value, dict):
                return value
            if isinstance(value, str) and value:
                try:
                    parsed = json.loads(value)
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    return {}
            return {}

        return cls(
            memory_id=record.get("id", ""),
            memory_type=record.get("memory_type", "episodic"),
            robot_id=record.get("robot_id", ""),
            tenant_id=record.get("tenant_id"),
            project_id=record.get("project_id"),
            site_id=record.get("site_id"),
            body_id=record.get("body_id"),
            practice_id=record.get("practice_id"),
            session_id=record.get("session_id"),
            episode_id=record.get("episode_id"),
            task_id=record.get("task_id"),
            skill_id=record.get("skill_id"),
            policy_id=record.get("policy_id"),
            title=record.get("title", ""),
            document=record.get("document", ""),
            summary=record.get("summary", ""),
            outcome=record.get("outcome"),
            reward=record.get("reward"),
            confidence=float(record.get("confidence") or 1.0),
            importance=float(record.get("importance") or 0.5),
            novelty=float(record.get("novelty") or 0.5),
            quality_score=float(record.get("quality_score") or 0.5),
            evidence_refs=_json_list(record.get("evidence_refs")),
            artifact_refs=_json_list(record.get("artifact_refs")),
            tags=_json_list(record.get("tags")),
            metadata=_json_dict(record.get("metadata")),
            embedding_model=record.get("embedding_model"),
            embedding_version=record.get("embedding_version"),
            content_hash=record.get("content_hash", ""),
            event_time=float(record.get("event_time") or time.time()),
            created_at=float(record.get("created_at") or time.time()),
            updated_at=float(record.get("updated_at") or time.time()),
            expires_at=record.get("expires_at"),
            schema_version=record.get("schema_version", SCHEMA_VERSION),
            status=record.get("status", MemoryStatus.ACTIVE.value),
            pinned=bool(record.get("pinned")),
        )


@dataclass
class MemoryEvidence:
    """Independent evidence row linking a memory to its source (§5.4)."""

    memory_id: str
    evidence_type: str
    evidence_id: str = field(default_factory=lambda: f"evd_{uuid.uuid4().hex[:16]}")
    source_event_id: str | None = None
    artifact_uri: str | None = None
    byte_offset: int | None = None
    sha256: str | None = None
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.evidence_id,
            "memory_id": self.memory_id,
            "evidence_type": self.evidence_type,
            "source_event_id": self.source_event_id,
            "artifact_uri": self.artifact_uri,
            "byte_offset": self.byte_offset,
            "sha256": self.sha256,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> MemoryEvidence:
        return cls(
            evidence_id=record.get("id", ""),
            memory_id=record.get("memory_id", ""),
            evidence_type=record.get("evidence_type", "practice_event"),
            source_event_id=record.get("source_event_id"),
            artifact_uri=record.get("artifact_uri"),
            byte_offset=record.get("byte_offset"),
            sha256=record.get("sha256"),
            confidence=float(record.get("confidence") or 1.0),
            created_at=float(record.get("created_at") or time.time()),
        )
