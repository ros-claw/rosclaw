"""Sandbox decision and session schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SandboxAction = Literal["ALLOW", "BLOCK", "MODIFY", "REQUIRE_CONFIRMATION"]


@dataclass
class SandboxDecision:
    """Firewall decision for a proposed action."""

    decision_id: str = ""
    action: SandboxAction = "ALLOW"
    skill_id: str = ""
    task_id: str = ""
    robot_id: str = ""
    original_action: dict[str, Any] = field(default_factory=dict)
    modified_action: dict[str, Any] | None = None
    rejection_reason: str = ""
    safety_checks: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    human_approval_required: bool = False
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "action": self.action,
            "skill_id": self.skill_id,
            "task_id": self.task_id,
            "robot_id": self.robot_id,
            "original_action": self.original_action,
            "modified_action": self.modified_action,
            "rejection_reason": self.rejection_reason,
            "safety_checks": self.safety_checks,
            "risk_score": self.risk_score,
            "human_approval_required": self.human_approval_required,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SandboxDecision:
        return cls(
            decision_id=d.get("decision_id", ""),
            action=d.get("action", "ALLOW"),
            skill_id=d.get("skill_id", ""),
            task_id=d.get("task_id", ""),
            robot_id=d.get("robot_id", ""),
            original_action=dict(d.get("original_action", {})),
            modified_action=d.get("modified_action"),
            rejection_reason=d.get("rejection_reason", ""),
            safety_checks=list(d.get("safety_checks", [])),
            risk_score=d.get("risk_score", 0.0),
            human_approval_required=d.get("human_approval_required", False),
            created_at=d.get("created_at", ""),
        )


@dataclass
class SandboxSession:
    """Sandbox simulation session record."""

    session_id: str = ""
    task_id: str = ""
    skill_id: str = ""
    robot_id: str = ""
    world_file: str = ""
    episodes: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending | running | completed | failed
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "skill_id": self.skill_id,
            "robot_id": self.robot_id,
            "world_file": self.world_file,
            "episodes": self.episodes,
            "results": self.results,
            "aggregate_metrics": self.aggregate_metrics,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SandboxSession:
        return cls(
            session_id=d.get("session_id", ""),
            task_id=d.get("task_id", ""),
            skill_id=d.get("skill_id", ""),
            robot_id=d.get("robot_id", ""),
            world_file=d.get("world_file", ""),
            episodes=d.get("episodes", 0),
            results=list(d.get("results", [])),
            aggregate_metrics=dict(d.get("aggregate_metrics", {})),
            status=d.get("status", "pending"),
            created_at=d.get("created_at", ""),
        )
