"""rosclaw_how.schemas — v1.5 unified request/decision/state schemas.

Introduced by the v1.5 optimization plan to lift the API beyond the v1
"error_log + scores + iteration" triplet into a richer runtime context
(task / optimization / safety / artifact). All v1 endpoints continue to
work through ``from_v1_prompt_build()`` which adapts the old shape into
the new one without breaking any caller.

Design rules:

* Every field has a default so legacy v1 callers (which only fill
  ``error_log`` / ``previous_scores`` / ``current_iteration``) keep
  working through the adapter — no validation errors on partial bodies.
* No I/O, no side effects — pure Pydantic.
* Frozen-style helpers: never mutate; always return new objects.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ObjectiveDirection = Literal["maximize", "minimize"]

OptimizationState = Literal[
    "early",
    "improving",
    "plateau",
    "regressing",
    "oscillating",
    "invalid_heavy",
    "late_budget",
    "unknown",
]

FeasibilityState = Literal[
    "mostly_valid",
    "invalid_heavy",
    "all_invalid",
    "unknown",
]

SafetyState = Literal[
    "safe",
    "warning",
    "constraint_violation",
    "hazard",
    "emergency",
]

SafetySeverity = Literal["S0", "S1", "S2", "S3", "S4"]

# v1 + v1.5 union. v1 callers only ever see the first three.
StrategyV15 = Literal[
    # v1 strategies (preserved for backward compat)
    "SAFETY",
    "FREE_EXPLORATION",
    "CATALYST",
    # v1.5 strategies
    "NOOP",
    "DIAGNOSE",
    "FEASIBILITY_REPAIR",
    "STABILIZE",
    "DIVERSIFY",
    "EXPLOIT_BEST",
    "ESCALATE_TO_KNOW",
    "STOP_UNSAFE",
    "RESOURCE_REPAIR",
]


class TaskContext(BaseModel):
    """What the agent is trying to do."""

    task_name: str | None = None
    benchmark: str | None = None
    task_family: str | None = None
    domain: str | None = None
    embodiment_type: str | None = None
    artifact_language: str | None = None
    objective_direction: ObjectiveDirection = "maximize"
    metric_name: str | None = None
    hard_constraints: list[str] = Field(default_factory=list)


class OptimizationContext(BaseModel):
    """How the agent is progressing on the task."""

    current_iteration: int = Field(0, ge=0)
    budget_iterations: int | None = Field(None, ge=0)
    previous_scores: list[float] = Field(default_factory=list)
    previous_valid: list[bool] = Field(default_factory=list)
    best_score: float | None = None
    last_score: float | None = None
    invalid_count: int = Field(0, ge=0)
    timeout_count: int = Field(0, ge=0)


class SafetyContext(BaseModel):
    """Free-form safety signal from the verifier / simulator / robot."""

    error_log: str = ""
    safety_events: list[str] = Field(default_factory=list)
    constraint_violations: list[str] = Field(default_factory=list)
    severity_hint: SafetySeverity | None = None


class ArtifactContext(BaseModel):
    """What the agent's candidate / code looks like right now."""

    code_excerpt: str | None = None
    last_diff: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    candidate_returncode: int | None = None


class AgentContext(BaseModel):
    """How the calling agent identifies itself + which patterns it has
    seen recently this run (used for cooldown / anti-repeat)."""

    run_id: str | None = None
    agent_id: str | None = None
    recent_pattern_ids: list[str] = Field(default_factory=list)


class InterventionRequest(BaseModel):
    """Full v1.5 request shape for ``POST /runtime/v1/intervene``.

    Every section is optional except ``task_context`` (we accept it
    empty for v1 callers via the adapter)."""

    run_id: str | None = None
    task_context: TaskContext = Field(default_factory=TaskContext)
    optimization_context: OptimizationContext = Field(default_factory=OptimizationContext)
    safety_context: SafetyContext = Field(default_factory=SafetyContext)
    artifact_context: ArtifactContext = Field(default_factory=ArtifactContext)
    agent_context: AgentContext = Field(default_factory=AgentContext)


class RuntimeState(BaseModel):
    """Diagnoser output: what's actually happening in the agent loop."""

    optimization_state: OptimizationState = "unknown"
    feasibility_state: FeasibilityState = "unknown"
    safety_state: SafetyState = "safe"
    safety_severity: SafetySeverity = "S0"
    safety_symptom: str | None = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class InterventionDecision(BaseModel):
    """Decision returned from the v1.5 intervention pipeline.

    Carries enough metadata for the agent to act on the snippet *and*
    for the operator dashboard / outcomes layer to attribute the
    intervention back to a strategy and (when applicable) a pattern.
    """

    strategy: StrategyV15
    runtime_state: RuntimeState
    snippet: str = ""
    injected: bool = False
    diagnosis: str | None = None
    next_experiment: str | None = None
    code_target: str | None = None
    expected_signal: str | None = None
    stop_condition: str | None = None
    symptom: str | None = None
    matched_symptom: str | None = None
    similarity: float | None = None
    injection_id: str | None = None
    pattern_id: str | None = None
    cluster_id: str | None = None
    is_staging: bool = False
    requires_sandbox_validation: bool = False
    sandbox_checks: list[str] = Field(default_factory=list)
    reason: str | None = None


# ── v1 → v1.5 adapter ────────────────────────────────────────────────────


def from_v1_prompt_build(
    error_log: str,
    previous_scores: list[float],
    current_iteration: int,
    *,
    objective_direction: ObjectiveDirection = "maximize",
    run_id: str | None = None,
    recent_pattern_ids: list[str] | None = None,
) -> InterventionRequest:
    """Turn a v1 ``/wiki/v1/prompt/build`` payload into a v1.5 request."""
    return InterventionRequest(
        run_id=run_id,
        task_context=TaskContext(objective_direction=objective_direction),
        optimization_context=OptimizationContext(
            current_iteration=int(current_iteration),
            previous_scores=list(previous_scores),
            last_score=float(previous_scores[-1]) if previous_scores else None,
        ),
        safety_context=SafetyContext(error_log=str(error_log or "")),
        agent_context=AgentContext(
            run_id=run_id,
            recent_pattern_ids=list(recent_pattern_ids or []),
        ),
    )


def decision_as_v1_response(
    decision: InterventionDecision,
    *,
    start_latency_ms: int,
) -> dict[str, Any]:
    """Map a v1.5 ``InterventionDecision`` into the v1 PromptBuildResponse
    shape so the existing ``/wiki/v1/prompt/build`` clients stay green.

    Only v1 strategies are returned (the new states collapse to the
    legacy ones the v1 schema allows)."""
    strategy = decision.strategy
    v1_strategy: str = strategy
    # Collapse new strategies that the v1 enum doesn't know about into
    # the closest legacy value so JSONResponse validation never fails.
    if strategy in ("STOP_UNSAFE", "RESOURCE_REPAIR"):
        v1_strategy = "SAFETY"
    elif strategy in (
        "FEASIBILITY_REPAIR",
        "STABILIZE",
        "DIVERSIFY",
        "EXPLOIT_BEST",
        "ESCALATE_TO_KNOW",
        "DIAGNOSE",
    ):
        v1_strategy = "CATALYST"
    elif strategy == "NOOP":
        v1_strategy = "FREE_EXPLORATION"

    payload: dict[str, Any] = {
        "prompt_snippet": decision.snippet,
        "injected": decision.injected,
        "strategy": v1_strategy,
        "latency_ms": start_latency_ms,
    }
    if decision.symptom is not None:
        payload["symptom"] = decision.symptom
    if decision.matched_symptom is not None:
        payload["matched_symptom"] = decision.matched_symptom
    if decision.similarity is not None:
        payload["similarity"] = decision.similarity
    if decision.reason is not None:
        payload["reason"] = decision.reason
    if decision.injection_id is not None:
        payload["injection_id"] = decision.injection_id
    if decision.pattern_id is not None:
        payload["pattern_id"] = decision.pattern_id
    if decision.is_staging:
        payload["is_staging"] = True
    return payload


__all__ = [
    "AgentContext",
    "ArtifactContext",
    "FeasibilityState",
    "InterventionDecision",
    "InterventionRequest",
    "ObjectiveDirection",
    "OptimizationContext",
    "OptimizationState",
    "RuntimeState",
    "SafetyContext",
    "SafetySeverity",
    "SafetyState",
    "StrategyV15",
    "TaskContext",
    "decision_as_v1_response",
    "from_v1_prompt_build",
]
