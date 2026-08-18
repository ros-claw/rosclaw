"""Proposal — 改进提案."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass
class Proposal:
    id: str
    source: Literal[
        "failure_guided",
        "benchmark_guided",
        "memory_guided",
        "how_guided",
        "know_guided",
        "darwin_guided",
    ] = "failure_guided"
    event_id: str = ""
    task: str = ""
    target_skill_id: str = ""
    hypothesis_id: str = ""
    hypothesis_statement: str = ""
    patch_type: Literal[
        "config_patch",
        "skill_parameter_patch",
        "skill_graph_patch",
        "policy_checkpoint_patch",
        "code_patch",
    ] = "skill_parameter_patch"
    search_space: dict = field(default_factory=dict)
    expected_effect: dict = field(default_factory=dict)
    risk_level: Literal["low", "medium", "high"] = "low"
    required_gates: list[str] = field(default_factory=list)
    status: Literal["draft", "approved", "rejected", "pending"] = "draft"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "event_id": self.event_id,
            "task": self.task,
            "target_skill_id": self.target_skill_id,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_statement": self.hypothesis_statement,
            "patch_type": self.patch_type,
            "search_space": self.search_space,
            "expected_effect": self.expected_effect,
            "risk_level": self.risk_level,
            "required_gates": self.required_gates,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Proposal":
        return cls(
            id=d["id"],
            source=d.get("source", "failure_guided"),
            event_id=d.get("event_id", ""),
            task=d.get("task", ""),
            target_skill_id=d.get("target_skill_id", ""),
            hypothesis_id=d.get("hypothesis_id", ""),
            hypothesis_statement=d.get("hypothesis_statement", ""),
            patch_type=d.get("patch_type", "skill_parameter_patch"),
            search_space=d.get("search_space", {}),
            expected_effect=d.get("expected_effect", {}),
            risk_level=d.get("risk_level", "low"),
            required_gates=d.get("required_gates", []),
            status=d.get("status", "draft"),
            created_at=d.get("created_at", ""),
        )
