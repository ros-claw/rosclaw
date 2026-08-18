"""EvaluationResult — 实验评估结果."""

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class EvaluationResult:
    id: str
    experiment_id: str
    baseline_metrics: dict = field(default_factory=dict)
    candidate_metrics: dict = field(default_factory=dict)
    delta: dict = field(default_factory=dict)
    safety_result: dict = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    decision: str = ""
    diagnosis: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "baseline_metrics": self.baseline_metrics,
            "candidate_metrics": self.candidate_metrics,
            "delta": self.delta,
            "safety_result": self.safety_result,
            "failure_modes": self.failure_modes,
            "decision": self.decision,
            "diagnosis": self.diagnosis,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluationResult":
        return cls(
            id=d["id"],
            experiment_id=d["experiment_id"],
            baseline_metrics=d.get("baseline_metrics", {}),
            candidate_metrics=d.get("candidate_metrics", {}),
            delta=d.get("delta", {}),
            safety_result=d.get("safety_result", {}),
            failure_modes=d.get("failure_modes", []),
            decision=d.get("decision", ""),
            diagnosis=d.get("diagnosis", ""),
            created_at=d.get("created_at", ""),
        )
