"""Hypothesis — 改进假设."""

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class Hypothesis:
    id: str
    failure_case_id: str
    statement: str = ""
    mechanism: str = ""
    confidence: float = 0.0
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    testable: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "failure_case_id": self.failure_case_id,
            "statement": self.statement,
            "mechanism": self.mechanism,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "testable": self.testable,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        return cls(
            id=d["id"],
            failure_case_id=d["failure_case_id"],
            statement=d.get("statement", ""),
            mechanism=d.get("mechanism", ""),
            confidence=d.get("confidence", 0.0),
            supporting_evidence=d.get("supporting_evidence", []),
            contradicting_evidence=d.get("contradicting_evidence", []),
            testable=d.get("testable", True),
            created_at=d.get("created_at", ""),
        )
