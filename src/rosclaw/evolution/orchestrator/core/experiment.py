"""ExperimentSpec — 实验规格定义."""

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class ExperimentSpec:
    id: str
    proposal_id: str
    patch_id: str
    task: str = ""
    robot: str = ""
    environment: dict = field(default_factory=dict)
    baseline_skill_id: str = ""
    candidate_skill_id: str = ""
    evaluation: dict = field(
        default_factory=lambda: {
            "episodes": 50,
            "seeds": [0, 1, 2],
            "metrics": ["success_rate", "collision_rate", "completion_time"],
        }
    )
    safety: dict = field(default_factory=dict)
    promotion: dict = field(default_factory=dict)
    patch_context: dict = field(default_factory=dict)
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "proposal_id": self.proposal_id,
            "patch_id": self.patch_id,
            "task": self.task,
            "robot": self.robot,
            "environment": self.environment,
            "baseline_skill_id": self.baseline_skill_id,
            "candidate_skill_id": self.candidate_skill_id,
            "evaluation": self.evaluation,
            "safety": self.safety,
            "promotion": self.promotion,
            "patch_context": self.patch_context,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentSpec":
        return cls(
            id=d["id"],
            proposal_id=d["proposal_id"],
            patch_id=d["patch_id"],
            task=d.get("task", ""),
            robot=d.get("robot", ""),
            environment=d.get("environment", {}),
            baseline_skill_id=d.get("baseline_skill_id", ""),
            candidate_skill_id=d.get("candidate_skill_id", ""),
            evaluation=d.get(
                "evaluation",
                {
                    "episodes": 50,
                    "seeds": [0, 1, 2],
                    "metrics": ["success_rate", "collision_rate", "completion_time"],
                },
            ),
            safety=d.get("safety", {}),
            promotion=d.get("promotion", {}),
            patch_context=d.get("patch_context", {}),
            status=d.get("status", "pending"),
            created_at=d.get("created_at", ""),
        )
