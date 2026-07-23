"""rosclaw.how — Heuristic Rules & Recovery Strategies.

Two coexisting layers:

  * **Reactive (event-driven)** — :class:`HeuristicEngine`, :class:`RecoveryEngine`,
    :class:`RecoveryLoop`. Subscribes to ``praxis.failed`` /
    ``firewall.action_blocked`` / ``safety.violation`` on the EventBus,
    looks up a rule, emits a recovery hint, manages retry bookkeeping.
    Backed by SeekDB; outcome counters update on every success/failure.

  * **Proactive (intervention)** — :mod:`rosclaw.how.intervention`. Pure-rules
    pipeline: ``InterventionRequest`` → :func:`intervention.diagnose`
    (RuntimeState across optimization × feasibility × safety axes) →
    :func:`intervention.decide_strategy` (one of 11 strategies with cooldown
    awareness) → :func:`intervention.compose` (strategy-specific markdown
    snippet). Standalone — no FastAPI, no embedding model — and integrated
    into :meth:`HeuristicEngine.decide_recovery` so proactive decisions still
    feed outcome tracking via ``record_outcome``.

The two layers share the SAFETY_TAXONOMY (S0-S4 vocabulary). The engine
consults the taxonomy before its substring matcher, then falls back to
reactive rules. Existing reactive callers / tests are unchanged.

Integration:
  Runtime    -> _how = HeuristicEngine(seekdb_client)
  Firewall   -> on block: query _how.suggest_recovery() -> EventBus
  Agent      -> analyze_failure(): try _how first, fall back to LLM
  Practice   -> on praxis.failed: record outcome -> update rule stats
  RuntimeLoop-> proactive: _how.decide_recovery(InterventionRequest) ->
                full decision + rule_id (when safety-attributable).
"""

from .client import HowClient
from .engine import HeuristicEngine
from .intervention import (
    SAFETY_TAXONOMY,
    AgentContext,
    ArtifactContext,
    FeasibilityState,
    InterventionDecision,
    InterventionRequest,
    InterventionStrategy,
    ObjectiveDirection,
    OptimizationContext,
    OptimizationState,
    RuntimeState,
    SafetyContext,
    SafetySeverity,
    SafetyState,
    TaskContext,
    compose,
    decide_strategy,
    decision_as_v1_response,
    diagnose,
    diagnose_safety,
    from_v1_prompt_build,
    is_blocking,
)
from .recovery import RecoveryEngine, RecoveryFormatter, format_recovery_suggestion
from .recovery_loop import RecoveryLoop
from .retry_orchestrator import CandidatePatch, RetryExecutionResult, RetryOrchestrator

__all__ = [
    # service client
    "HowClient",
    # reactive
    "HeuristicEngine",
    "RecoveryEngine",
    "RecoveryFormatter",
    "format_recovery_suggestion",
    "RecoveryLoop",
    "CandidatePatch",
    "RetryExecutionResult",
    "RetryOrchestrator",
    # intervention schemas
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
    "InterventionStrategy",
    "TaskContext",
    # intervention ops
    "SAFETY_TAXONOMY",
    "compose",
    "decide_strategy",
    "decision_as_v1_response",
    "diagnose",
    "diagnose_safety",
    "from_v1_prompt_build",
    "is_blocking",
]
