"""rosclaw.how.intervention — Runtime Intervention Controller.

This subpackage adds the multi-dimension diagnose → policy → composer
pipeline on top of the reactive :class:`HeuristicEngine`. It is intentionally
pure: no FastAPI, no SeekDB, no embedding model. The hot path is rules
+ deterministic decisions, suitable for inline use from the runtime's
recovery callbacks.

Public surface (re-exported from ``rosclaw.how``):

* ``InterventionRequest``       — Pydantic input shape.
* ``InterventionDecision``      — strategy + snippet + sandbox checks.
* ``RuntimeState``              — diagnoser output.
* ``diagnose``                  — request → state.
* ``decide_strategy``           — state → ``StrategyV15``.
* ``compose``                   — strategy + state → decision.
* ``diagnose_safety``           — extended safety taxonomy (S0-S4).
* ``SAFETY_TAXONOMY``           — canonical 15-symptom dict.

The reactive HeuristicEngine continues to own ``record_outcome`` and the
EventBus subscription. The intervention layer adds a richer decision; the
engine wraps it with outcome tracking via ``HeuristicEngine.decide_recovery``.
"""
from __future__ import annotations

from .intervention_policy import COOLDOWN_WINDOW, decide_strategy, is_blocking
from .runtime_diagnoser import diagnose
from .safety_router import (
    SAFETY_TAXONOMY,
    diagnose_safety,
    safety_state_from_severity,
    v1_safety_keywords,
)
from .safety_router import (
    is_blocking as safety_is_blocking,
)
from .schemas import (
    AgentContext,
    ArtifactContext,
    FeasibilityState,
    InterventionDecision,
    InterventionRequest,
    ObjectiveDirection,
    OptimizationContext,
    OptimizationState,
    RuntimeState,
    SafetyContext,
    SafetySeverity,
    SafetyState,
    StrategyV15,
    TaskContext,
    decision_as_v1_response,
    from_v1_prompt_build,
)
from .score_normalizer import (
    delta_normalized,
    is_improving_normalized,
    normalize_score,
    normalize_scores,
)
from .snippet_composer import compose

__all__ = [
    "AgentContext",
    "ArtifactContext",
    "COOLDOWN_WINDOW",
    "FeasibilityState",
    "InterventionDecision",
    "InterventionRequest",
    "ObjectiveDirection",
    "OptimizationContext",
    "OptimizationState",
    "RuntimeState",
    "SAFETY_TAXONOMY",
    "SafetyContext",
    "SafetySeverity",
    "SafetyState",
    "StrategyV15",
    "TaskContext",
    "compose",
    "decide_strategy",
    "decision_as_v1_response",
    "delta_normalized",
    "diagnose",
    "diagnose_safety",
    "from_v1_prompt_build",
    "is_blocking",
    "is_improving_normalized",
    "normalize_score",
    "normalize_scores",
    "safety_is_blocking",
    "safety_state_from_severity",
    "v1_safety_keywords",
]
