"""Champion — 晋升技能冠军."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class Champion:
    id: str
    skill_id: str
    task_id: str
    level: Literal["baseline", "sim", "sandbox", "real_candidate", "real"] = "baseline"
    parent_skill_id: str = ""
    patch_id: str = ""
    metrics: dict = field(default_factory=dict)
    validation_summary: dict = field(default_factory=dict)
    known_limits: list[str] = field(default_factory=list)
    rollback_to: str | None = None
    experiment_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id, "skill_id": self.skill_id, "task_id": self.task_id,
            "level": self.level, "parent_skill_id": self.parent_skill_id,
            "patch_id": self.patch_id, "metrics": self.metrics,
            "validation_summary": self.validation_summary, "known_limits": self.known_limits,
            "rollback_to": self.rollback_to, "experiment_id": self.experiment_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Champion":
        return cls(
            id=d["id"], skill_id=d["skill_id"], task_id=d["task_id"],
            level=d.get("level", "baseline"), parent_skill_id=d.get("parent_skill_id", ""),
            patch_id=d.get("patch_id", ""), metrics=d.get("metrics", {}),
            validation_summary=d.get("validation_summary", {}),
            known_limits=d.get("known_limits", []), rollback_to=d.get("rollback_to"),
            experiment_id=d.get("experiment_id", ""), created_at=d.get("created_at", ""),
        )
