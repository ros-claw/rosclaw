"""Auto module schemas — canonical shapes for Proposal, Patch, Experiment, etc."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PatchType = Literal[
    "config_patch",
    "skill_parameter_patch",
    "skill_graph_patch",
    "policy_checkpoint_patch",
    "code_patch",
]

ProposalSource = Literal[
    "failure_guided",
    "benchmark_guided",
    "memory_guided",
    "how_guided",
    "know_guided",
    "darwin_guided",
]

ChampionLevel = Literal[
    "baseline_champion",
    "sim_champion",
    "sandbox_champion",
    "real_candidate",
    "real_champion",
    "deprecated",
]


@dataclass
class AutoProposal:
    """Canonical AutoProposal schema (Section 10.5)."""

    proposal_id: str = ""
    source: ProposalSource = "failure_guided"
    event_id: str = ""
    task_id: str = ""
    target_skill_id: str = ""
    hypothesis_id: str = ""
    hypothesis_statement: str = ""
    patch_type: PatchType = "skill_parameter_patch"
    search_space: dict[str, Any] = field(default_factory=dict)
    expected_effect: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    required_gates: list[str] = field(default_factory=list)
    status: str = "draft"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source": self.source,
            "event_id": self.event_id,
            "task_id": self.task_id,
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
    def from_dict(cls, d: dict[str, Any]) -> AutoProposal:
        return cls(
            proposal_id=d.get("proposal_id", ""),
            source=d.get("source", "failure_guided"),
            event_id=d.get("event_id", ""),
            task_id=d.get("task_id", ""),
            target_skill_id=d.get("target_skill_id", ""),
            hypothesis_id=d.get("hypothesis_id", ""),
            hypothesis_statement=d.get("hypothesis_statement", ""),
            patch_type=d.get("patch_type", "skill_parameter_patch"),
            search_space=dict(d.get("search_space", {})),
            expected_effect=dict(d.get("expected_effect", {})),
            risk_level=d.get("risk_level", "low"),
            required_gates=list(d.get("required_gates", [])),
            status=d.get("status", "draft"),
            created_at=d.get("created_at", ""),
        )


@dataclass
class AutoPatch:
    """Canonical AutoPatch schema."""

    patch_id: str = ""
    proposal_id: str = ""
    patch_level: int = 0
    patch_type: PatchType = "skill_parameter_patch"
    target_skill_id: str = ""
    changes: list[dict[str, Any]] = field(default_factory=list)
    rollback_plan: dict[str, Any] = field(default_factory=dict)
    human_approval_required: bool = False
    status: str = "draft"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "proposal_id": self.proposal_id,
            "patch_level": self.patch_level,
            "patch_type": self.patch_type,
            "target_skill_id": self.target_skill_id,
            "changes": self.changes,
            "rollback_plan": self.rollback_plan,
            "human_approval_required": self.human_approval_required,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AutoPatch:
        return cls(
            patch_id=d.get("patch_id", ""),
            proposal_id=d.get("proposal_id", ""),
            patch_level=d.get("patch_level", 0),
            patch_type=d.get("patch_type", "skill_parameter_patch"),
            target_skill_id=d.get("target_skill_id", ""),
            changes=list(d.get("changes", [])),
            rollback_plan=dict(d.get("rollback_plan", {})),
            human_approval_required=d.get("human_approval_required", False),
            status=d.get("status", "draft"),
            created_at=d.get("created_at", ""),
        )


@dataclass
class ExperimentSpec:
    """Canonical ExperimentSpec schema."""

    experiment_id: str = ""
    proposal_id: str = ""
    patch_id: str = ""
    task_id: str = ""
    robot_id: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    baseline_skill_id: str = ""
    candidate_skill_id: str = ""
    evaluation: dict[str, Any] = field(
        default_factory=lambda: {
            "episodes": 50,
            "seeds": [0, 1, 2],
            "metrics": ["success_rate", "collision_rate", "completion_time"],
        }
    )
    safety: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "proposal_id": self.proposal_id,
            "patch_id": self.patch_id,
            "task_id": self.task_id,
            "robot_id": self.robot_id,
            "environment": self.environment,
            "baseline_skill_id": self.baseline_skill_id,
            "candidate_skill_id": self.candidate_skill_id,
            "evaluation": self.evaluation,
            "safety": self.safety,
            "promotion": self.promotion,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentSpec:
        return cls(
            experiment_id=d.get("experiment_id", ""),
            proposal_id=d.get("proposal_id", ""),
            patch_id=d.get("patch_id", ""),
            task_id=d.get("task_id", ""),
            robot_id=d.get("robot_id", ""),
            environment=dict(d.get("environment", {})),
            baseline_skill_id=d.get("baseline_skill_id", ""),
            candidate_skill_id=d.get("candidate_skill_id", ""),
            evaluation=dict(d.get("evaluation", {"episodes": 50, "seeds": [0, 1, 2], "metrics": ["success_rate", "collision_rate", "completion_time"]})),
            safety=dict(d.get("safety", {})),
            promotion=dict(d.get("promotion", {})),
            status=d.get("status", "pending"),
            created_at=d.get("created_at", ""),
        )


@dataclass
class EvaluationResult:
    """Canonical EvaluationResult schema."""

    result_id: str = ""
    experiment_id: str = ""
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    candidate_metrics: dict[str, float] = field(default_factory=dict)
    delta: dict[str, float] = field(default_factory=dict)
    safety_result: dict[str, Any] = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    decision: str = ""
    diagnosis: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
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
    def from_dict(cls, d: dict[str, Any]) -> EvaluationResult:
        return cls(
            result_id=d.get("result_id", ""),
            experiment_id=d.get("experiment_id", ""),
            baseline_metrics=dict(d.get("baseline_metrics", {})),
            candidate_metrics=dict(d.get("candidate_metrics", {})),
            delta=dict(d.get("delta", {})),
            safety_result=dict(d.get("safety_result", {})),
            failure_modes=list(d.get("failure_modes", [])),
            decision=d.get("decision", ""),
            diagnosis=d.get("diagnosis", ""),
            created_at=d.get("created_at", ""),
        )


@dataclass
class Champion:
    """Canonical Champion schema."""

    champion_id: str = ""
    skill_id: str = ""
    task_id: str = ""
    level: ChampionLevel = "baseline_champion"
    parent_skill_id: str = ""
    patch_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    known_limits: list[str] = field(default_factory=list)
    rollback_to: str = ""
    experiment_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion_id": self.champion_id,
            "skill_id": self.skill_id,
            "task_id": self.task_id,
            "level": self.level,
            "parent_skill_id": self.parent_skill_id,
            "patch_id": self.patch_id,
            "metrics": self.metrics,
            "validation_summary": self.validation_summary,
            "known_limits": self.known_limits,
            "rollback_to": self.rollback_to,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Champion:
        return cls(
            champion_id=d.get("champion_id", ""),
            skill_id=d.get("skill_id", ""),
            task_id=d.get("task_id", ""),
            level=d.get("level", "baseline_champion"),
            parent_skill_id=d.get("parent_skill_id", ""),
            patch_id=d.get("patch_id", ""),
            metrics=dict(d.get("metrics", {})),
            validation_summary=dict(d.get("validation_summary", {})),
            known_limits=list(d.get("known_limits", [])),
            rollback_to=d.get("rollback_to", ""),
            experiment_id=d.get("experiment_id", ""),
            created_at=d.get("created_at", ""),
        )


@dataclass
class DeadEnd:
    """Canonical DeadEnd schema."""

    deadend_id: str = ""
    task_id: str = ""
    direction: str = ""
    tested_patches: list[str] = field(default_factory=list)
    observed_effect: dict[str, Any] = field(default_factory=dict)
    rejection_reason: str = ""
    avoid_conditions: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "deadend_id": self.deadend_id,
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
    def from_dict(cls, d: dict[str, Any]) -> DeadEnd:
        return cls(
            deadend_id=d.get("deadend_id", ""),
            task_id=d.get("task_id", ""),
            direction=d.get("direction", ""),
            tested_patches=list(d.get("tested_patches", [])),
            observed_effect=dict(d.get("observed_effect", {})),
            rejection_reason=d.get("rejection_reason", ""),
            avoid_conditions=dict(d.get("avoid_conditions", {})),
            evidence=list(d.get("evidence", [])),
            created_at=d.get("created_at", ""),
        )
