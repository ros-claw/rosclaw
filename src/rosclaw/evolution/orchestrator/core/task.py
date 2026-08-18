"""AutoTask — 自进化任务定义."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass
class AutoTask:
    id: str
    name: str
    task_type: Literal[
        "skill_tuning", "failure_repair", "benchmark_optimization", "sim2real_validation"
    ]
    robot_id: str
    environment_id: str
    target_skill_id: str
    objective: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type,
            "robot_id": self.robot_id,
            "environment_id": self.environment_id,
            "target_skill_id": self.target_skill_id,
            "objective": self.objective,
            "constraints": self.constraints,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AutoTask":
        return cls(
            id=d["id"],
            name=d["name"],
            task_type=d["task_type"],
            robot_id=d["robot_id"],
            environment_id=d["environment_id"],
            target_skill_id=d["target_skill_id"],
            objective=d.get("objective", {}),
            constraints=d.get("constraints", {}),
            status=d.get("status", "pending"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )
