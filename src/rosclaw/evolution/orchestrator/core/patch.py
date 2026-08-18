"""Patch — 改进补丁."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass
class Patch:
    id: str
    proposal_id: str
    patch_level: int = 2
    patch_type: Literal[
        "config_patch",
        "skill_parameter_patch",
        "skill_graph_patch",
        "policy_checkpoint_patch",
        "code_patch",
    ] = "skill_parameter_patch"
    target_skill: str = ""
    changes: list[dict] = field(default_factory=list)
    rollback_plan: dict = field(default_factory=dict)
    human_approval_required: bool = False
    status: Literal["draft", "approved", "applied", "rejected", "rolled_back"] = "draft"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "proposal_id": self.proposal_id,
            "patch_level": self.patch_level,
            "patch_type": self.patch_type,
            "target_skill": self.target_skill,
            "changes": self.changes,
            "rollback_plan": self.rollback_plan,
            "human_approval_required": self.human_approval_required,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Patch":
        return cls(
            id=d["id"],
            proposal_id=d["proposal_id"],
            patch_level=d.get("patch_level", 2),
            patch_type=d.get("patch_type", "skill_parameter_patch"),
            target_skill=d.get("target_skill", ""),
            changes=d.get("changes", []),
            rollback_plan=d.get("rollback_plan", {}),
            human_approval_required=d.get("human_approval_required", False),
            status=d.get("status", "draft"),
            created_at=d.get("created_at", ""),
        )
