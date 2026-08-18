"""FailureCase — 失败案例定义."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass
class FailureCase:
    id: str
    praxis_event_id: str
    task_id: str
    skill_id: str
    phase: str = ""
    failure_mode: str = ""
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    evidence: dict = field(default_factory=dict)
    replay_uri: str | None = None
    mcap_uri: str | None = None
    diagnosis_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "praxis_event_id": self.praxis_event_id,
            "task_id": self.task_id,
            "skill_id": self.skill_id,
            "phase": self.phase,
            "failure_mode": self.failure_mode,
            "severity": self.severity,
            "evidence": self.evidence,
            "replay_uri": self.replay_uri,
            "mcap_uri": self.mcap_uri,
            "diagnosis_id": self.diagnosis_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FailureCase":
        return cls(
            id=d["id"],
            praxis_event_id=d["praxis_event_id"],
            task_id=d["task_id"],
            skill_id=d["skill_id"],
            phase=d.get("phase", ""),
            failure_mode=d.get("failure_mode", ""),
            severity=d.get("severity", "medium"),
            evidence=d.get("evidence", {}),
            replay_uri=d.get("replay_uri"),
            mcap_uri=d.get("mcap_uri"),
            diagnosis_id=d.get("diagnosis_id"),
            created_at=d.get("created_at", ""),
        )
