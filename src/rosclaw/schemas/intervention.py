"""Intervention and Evidence schemas for How / Know / Auto integration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ObjectiveDirection = Literal["maximize", "minimize"]


@dataclass
class InterventionRequest:
    """Runtime intervention request shape (v1.5 compatible)."""

    run_id: str = ""
    task_name: str = ""
    task_family: str = ""
    domain: str = ""
    objective_direction: ObjectiveDirection = "maximize"
    metric_name: str = ""
    current_iteration: int = 0
    budget_iterations: int = 0
    previous_scores: list[float] = field(default_factory=list)
    previous_valid: list[bool] = field(default_factory=list)
    best_score: float | None = None
    last_score: float | None = None
    error_log: str = ""
    safety_events: list[str] = field(default_factory=list)
    constraint_violations: list[str] = field(default_factory=list)
    code_excerpt: str = ""
    last_diff: str = ""
    changed_files: list[str] = field(default_factory=list)
    agent_id: str = ""
    recent_pattern_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "domain": self.domain,
            "objective_direction": self.objective_direction,
            "metric_name": self.metric_name,
            "current_iteration": self.current_iteration,
            "budget_iterations": self.budget_iterations,
            "previous_scores": self.previous_scores,
            "previous_valid": self.previous_valid,
            "best_score": self.best_score,
            "last_score": self.last_score,
            "error_log": self.error_log,
            "safety_events": self.safety_events,
            "constraint_violations": self.constraint_violations,
            "code_excerpt": self.code_excerpt,
            "last_diff": self.last_diff,
            "changed_files": self.changed_files,
            "agent_id": self.agent_id,
            "recent_pattern_ids": self.recent_pattern_ids,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InterventionRequest:
        return cls(
            run_id=d.get("run_id", ""),
            task_name=d.get("task_name", ""),
            task_family=d.get("task_family", ""),
            domain=d.get("domain", ""),
            objective_direction=d.get("objective_direction", "maximize"),
            metric_name=d.get("metric_name", ""),
            current_iteration=d.get("current_iteration", 0),
            budget_iterations=d.get("budget_iterations", 0),
            previous_scores=list(d.get("previous_scores", [])),
            previous_valid=list(d.get("previous_valid", [])),
            best_score=d.get("best_score"),
            last_score=d.get("last_score"),
            error_log=d.get("error_log", ""),
            safety_events=list(d.get("safety_events", [])),
            constraint_violations=list(d.get("constraint_violations", [])),
            code_excerpt=d.get("code_excerpt", ""),
            last_diff=d.get("last_diff", ""),
            changed_files=list(d.get("changed_files", [])),
            agent_id=d.get("agent_id", ""),
            recent_pattern_ids=list(d.get("recent_pattern_ids", [])),
        )


@dataclass
class InterventionDecision:
    """Decision returned from the intervention pipeline."""

    strategy: str = ""  # SAFETY | FREE_EXPLORATION | CATALYST | NOOP | ...
    snippet: str = ""
    injected: bool = False
    diagnosis: str = ""
    next_experiment: str = ""
    code_target: str = ""
    expected_signal: str = ""
    stop_condition: str = ""
    symptom: str = ""
    matched_symptom: str = ""
    similarity: float = 0.0
    injection_id: str = ""
    pattern_id: str = ""
    cluster_id: str = ""
    is_staging: bool = False
    requires_sandbox_validation: bool = False
    sandbox_checks: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "snippet": self.snippet,
            "injected": self.injected,
            "diagnosis": self.diagnosis,
            "next_experiment": self.next_experiment,
            "code_target": self.code_target,
            "expected_signal": self.expected_signal,
            "stop_condition": self.stop_condition,
            "symptom": self.symptom,
            "matched_symptom": self.matched_symptom,
            "similarity": self.similarity,
            "injection_id": self.injection_id,
            "pattern_id": self.pattern_id,
            "cluster_id": self.cluster_id,
            "is_staging": self.is_staging,
            "requires_sandbox_validation": self.requires_sandbox_validation,
            "sandbox_checks": self.sandbox_checks,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InterventionDecision:
        return cls(
            strategy=d.get("strategy", ""),
            snippet=d.get("snippet", ""),
            injected=d.get("injected", False),
            diagnosis=d.get("diagnosis", ""),
            next_experiment=d.get("next_experiment", ""),
            code_target=d.get("code_target", ""),
            expected_signal=d.get("expected_signal", ""),
            stop_condition=d.get("stop_condition", ""),
            symptom=d.get("symptom", ""),
            matched_symptom=d.get("matched_symptom", ""),
            similarity=d.get("similarity", 0.0),
            injection_id=d.get("injection_id", ""),
            pattern_id=d.get("pattern_id", ""),
            cluster_id=d.get("cluster_id", ""),
            is_staging=d.get("is_staging", False),
            requires_sandbox_validation=d.get("requires_sandbox_validation", False),
            sandbox_checks=list(d.get("sandbox_checks", [])),
            reason=d.get("reason", ""),
        )


@dataclass
class InterventionTrace:
    """Record of a single intervention issued by How."""

    injection_id: str = ""
    run_id: str = ""
    task_id: str = ""
    strategy: str = ""
    pattern_id: str = ""
    snippet_hash: str = ""
    timestamp: str = ""
    source: str = "rosclaw-how"

    def to_dict(self) -> dict[str, Any]:
        return {
            "injection_id": self.injection_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "strategy": self.strategy,
            "pattern_id": self.pattern_id,
            "snippet_hash": self.snippet_hash,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InterventionTrace:
        return cls(
            injection_id=d.get("injection_id", ""),
            run_id=d.get("run_id", ""),
            task_id=d.get("task_id", ""),
            strategy=d.get("strategy", ""),
            pattern_id=d.get("pattern_id", ""),
            snippet_hash=d.get("snippet_hash", ""),
            timestamp=d.get("timestamp", ""),
            source=d.get("source", "rosclaw-how"),
        )


@dataclass
class EvidenceTrace:
    """Full evidence chain: pattern → injection → diff → verifier → outcome.

    Section 10.4 of the optimization doc.
    """

    injection_id: str = ""
    pattern_id: str = ""
    run_id: str = ""
    task_name: str = ""
    pre_score: float = 0.0
    post_score_1: float = 0.0
    post_score_3: float = 0.0
    code_diff_summary: list[str] = field(default_factory=list)
    used_hint: bool = False
    verifier_status: str = ""
    objective_direction: ObjectiveDirection = "maximize"
    timestamp: str = ""
    source: str = "rosclaw-how"

    def to_dict(self) -> dict[str, Any]:
        return {
            "injection_id": self.injection_id,
            "pattern_id": self.pattern_id,
            "run_id": self.run_id,
            "task_name": self.task_name,
            "pre_score": self.pre_score,
            "post_score_1": self.post_score_1,
            "post_score_3": self.post_score_3,
            "code_diff_summary": self.code_diff_summary,
            "used_hint": self.used_hint,
            "verifier_status": self.verifier_status,
            "objective_direction": self.objective_direction,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvidenceTrace:
        return cls(
            injection_id=d.get("injection_id", ""),
            pattern_id=d.get("pattern_id", ""),
            run_id=d.get("run_id", ""),
            task_name=d.get("task_name", ""),
            pre_score=d.get("pre_score", 0.0),
            post_score_1=d.get("post_score_1", 0.0),
            post_score_3=d.get("post_score_3", 0.0),
            code_diff_summary=list(d.get("code_diff_summary", [])),
            used_hint=d.get("used_hint", False),
            verifier_status=d.get("verifier_status", ""),
            objective_direction=d.get("objective_direction", "maximize"),
            timestamp=d.get("timestamp", ""),
            source=d.get("source", "rosclaw-how"),
        )

    @property
    def score_delta(self) -> float:
        """Net score improvement (respects objective direction)."""
        delta = self.post_score_3 - self.pre_score
        if self.objective_direction == "minimize":
            delta = -delta
        return delta
