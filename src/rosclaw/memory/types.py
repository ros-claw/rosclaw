"""Sprint 8 types for ROSClaw Knowledge Plane.

PraxisEvent  — unified spatiotemporal event for practice/sandbox/runtime.
FailureMemory — structured failure case with root cause and recovery hint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class PraxisEvent:
    """Unified practice event for the Knowledge Plane.

    Captures events from sandbox, provider, runtime, and practice modules
    in a single normalized format.
    """

    event_id: str
    robot_id: str
    event_type: str
    timestamp: float = field(default_factory=time.time)
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    payload: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_seekdb_record(self) -> dict:
        """Serialize to SeekDB praxis_events row."""
        return {
            "id": self.event_id,
            "episode_id": self.episode_id,
            "robot_id": self.robot_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_seekdb_record(cls, record: dict) -> PraxisEvent:
        """Deserialize from SeekDB praxis_events row."""
        return cls(
            event_id=record.get("id", ""),
            robot_id=record.get("robot_id", ""),
            event_type=record.get("event_type", ""),
            timestamp=record.get("timestamp", 0.0),
            episode_id=record.get("episode_id"),
            task_id=record.get("task_id"),
            payload=record.get("payload") or {},
            metadata=record.get("metadata") or {},
        )


@dataclass
class FailureMemory:
    """Structured failure case for root-cause analysis and recovery.

    Written when sandbox episodes fail, provider calls error out,
    or practice records a negative outcome.
    """

    failure_id: str
    robot_id: str
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    failure_type: str = "unknown"
    root_cause: str = ""
    timestamp: float = field(default_factory=time.time)
    recovery_hint: str = ""
    sandbox_intervened: bool = False
    category: str = ""
    metadata: dict = field(default_factory=dict)

    def to_seekdb_record(self) -> dict:
        """Serialize to SeekDB failures row."""
        return {
            "id": self.failure_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "robot_id": self.robot_id,
            "failure_type": self.failure_type,
            "root_cause": self.root_cause,
            "timestamp": self.timestamp,
            "recovery_hint": self.recovery_hint,
            "metadata": {
                **self.metadata,
                "sandbox_intervened": self.sandbox_intervened,
                "category": self.category,
            },
        }

    @classmethod
    def from_seekdb_record(cls, record: dict) -> FailureMemory:
        """Deserialize from SeekDB failures row."""
        meta = record.get("metadata") or {}
        return cls(
            failure_id=record.get("id", ""),
            robot_id=record.get("robot_id", ""),
            episode_id=record.get("episode_id"),
            task_id=record.get("task_id"),
            failure_type=record.get("failure_type", "unknown"),
            root_cause=record.get("root_cause", ""),
            timestamp=record.get("timestamp", 0.0),
            recovery_hint=record.get("recovery_hint", ""),
            sandbox_intervened=meta.get("sandbox_intervened", False),
            category=meta.get("category", ""),
            metadata=meta,
        )


@dataclass
class ArtifactRef:
    """Reference to a large artifact stored outside SeekDB.

    Artifacts (MCAP, video, replay) live in the local object store
    at ``./.rosclaw/artifacts/``.  SeekDB only keeps the URI and
    lightweight metadata.
    """

    artifact_id: str
    artifact_type: str
    uri: str
    episode_id: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def build_uri(
        artifact_type: str,
        date_str: str,
        episode_id: str,
        filename: str,
    ) -> str:
        """Build a canonical artifact URI.

        Example:
            artifact://episodes/2026-05-29/ep_0001/episode.mcap
        """
        return f"artifact://{artifact_type}/{date_str}/{episode_id}/{filename}"

    def to_seekdb_record(self) -> dict:
        return {
            "id": self.artifact_id,
            "episode_id": self.episode_id,
            "artifact_type": self.artifact_type,
            "uri": self.uri,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
