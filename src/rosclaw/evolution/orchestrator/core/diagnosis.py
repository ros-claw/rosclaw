"""Diagnosis — 失败诊断结果."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass
class Diagnosis:
    id: str
    failure_id: str
    task: str
    skill: str
    phase: str = ""
    failure_mode: str = ""
    root_cause_candidates: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    auto_repairable: bool = True
    confidence: float = 0.5
    risk_level: Literal["low", "medium", "high"] = "low"
    recommended_search_space: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "failure_id": self.failure_id,
            "task": self.task,
            "skill": self.skill,
            "phase": self.phase,
            "failure_mode": self.failure_mode,
            "root_cause_candidates": self.root_cause_candidates,
            "evidence": self.evidence,
            "severity": self.severity,
            "auto_repairable": self.auto_repairable,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "recommended_search_space": self.recommended_search_space,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Diagnosis":
        return cls(
            id=d["id"],
            failure_id=d["failure_id"],
            task=d["task"],
            skill=d["skill"],
            phase=d.get("phase", ""),
            failure_mode=d.get("failure_mode", ""),
            root_cause_candidates=d.get("root_cause_candidates", []),
            evidence=d.get("evidence", {}),
            severity=d.get("severity", "medium"),
            auto_repairable=d.get("auto_repairable", True),
            confidence=d.get("confidence", 0.5),
            risk_level=d.get("risk_level", "low"),
            recommended_search_space=d.get("recommended_search_space", {}),
            created_at=d.get("created_at", ""),
        )
