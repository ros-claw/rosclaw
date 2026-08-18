"""DeadEnd — 死胡同/无效方向记录."""

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class DeadEnd:
    id: str
    task_id: str
    direction: str = ""
    tested_patches: list[str] = field(default_factory=list)
    observed_effect: dict = field(default_factory=dict)
    rejection_reason: str = ""
    avoid_conditions: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "direction": self.direction,
            "tested_patches": self.tested_patches,
            "observed_effect": self.observed_effect,
            "rejection_reason": self.rejection_reason,
            "avoid_conditions": self.avoid_conditions,
            "evidence": self.evidence,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeadEnd":
        return cls(
            id=d["id"],
            task_id=d["task_id"],
            direction=d.get("direction", ""),
            tested_patches=d.get("tested_patches", []),
            observed_effect=d.get("observed_effect", {}),
            rejection_reason=d.get("rejection_reason", ""),
            avoid_conditions=d.get("avoid_conditions", {}),
            evidence=d.get("evidence", []),
            created_at=d.get("created_at", ""),
        )
