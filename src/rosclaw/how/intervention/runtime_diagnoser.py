"""rosclaw_how.runtime_diagnoser — v1.5 multi-dimension state diagnosis.

Replaces the v1 ``state_router`` heuristic ("first 3 iters or recent[-1]
> recent[0]") with a structured ``RuntimeState`` covering three
orthogonal axes:

* ``optimization_state`` — improving / plateau / regressing /
  oscillating / late_budget / early / invalid_heavy.
* ``feasibility_state``  — mostly_valid / invalid_heavy / all_invalid.
* ``safety_state``       — safe / warning / constraint_violation /
  hazard / emergency (delegated to ``safety_router``).

The diagnoser is the single place where direction-folded score
arithmetic lives — every downstream strategy decision uses normalized
deltas, so minimize-tasks behave correctly without each strategy
needing to know about direction.

Pure rules, no I/O. Hot-path safe.
"""
from __future__ import annotations

import statistics
from typing import Final

from .safety_router import diagnose_safety, safety_state_from_severity
from .schemas import (
    FeasibilityState,
    InterventionRequest,
    OptimizationState,
    RuntimeState,
)
from .score_normalizer import normalize_scores

# Window over which we evaluate progress / regression.
DEFAULT_WINDOW: Final[int] = 3

# Minimum iterations before we leave the "early" state.
EARLY_ITERATIONS: Final[int] = 3

# Fraction of total budget that counts as "late" (90%).
LATE_BUDGET_FRAC: Final[float] = 0.9

# Variation-of-mean ratio above which a window counts as "oscillating".
# i.e. stdev / max(abs(mean), epsilon) > this → bouncing.
OSCILLATION_RATIO: Final[float] = 0.25

# Invalid-rate thresholds for feasibility classification.
INVALID_HEAVY_RATE: Final[float] = 0.4   # ≥40% invalid
ALL_INVALID_RATE: Final[float] = 0.95    # ≥95% invalid

# Plateau detection: relative |delta| smaller than this is "no change".
PLATEAU_EPSILON: Final[float] = 1e-6


def _feasibility(req: InterventionRequest) -> FeasibilityState:
    """Classify the candidate-validity rate."""
    opt = req.optimization_context
    flags = list(opt.previous_valid)
    if not flags:
        # Fall back to invalid_count when valid flags weren't provided.
        if opt.invalid_count > 0 and opt.current_iteration > 0:
            rate = opt.invalid_count / max(1, opt.current_iteration)
            if rate >= ALL_INVALID_RATE:
                return "all_invalid"
            if rate >= INVALID_HEAVY_RATE:
                return "invalid_heavy"
            return "mostly_valid"
        return "unknown"
    invalid = sum(1 for v in flags if not v)
    rate = invalid / max(1, len(flags))
    if rate >= ALL_INVALID_RATE:
        return "all_invalid"
    if rate >= INVALID_HEAVY_RATE:
        return "invalid_heavy"
    return "mostly_valid"


def _optimization(req: InterventionRequest) -> tuple[OptimizationState, list[str]]:
    """Classify the score trajectory; returns (state, human-readable reasons)."""
    opt = req.optimization_context
    direction = req.task_context.objective_direction
    iteration = opt.current_iteration
    budget = opt.budget_iterations
    reasons: list[str] = []

    if budget and iteration >= int(budget * LATE_BUDGET_FRAC):
        reasons.append(f"iteration {iteration}/{budget} past late-budget threshold")
        return "late_budget", reasons

    if iteration < EARLY_ITERATIONS or len(opt.previous_scores) < DEFAULT_WINDOW:
        reasons.append(f"only {iteration} iterations / {len(opt.previous_scores)} scores")
        return "early", reasons

    window = opt.previous_scores[-(DEFAULT_WINDOW + 1) :]
    norm = normalize_scores(window, direction)

    # Oscillation test: high spread relative to mean magnitude.
    if len(norm) >= 3:
        spread = max(norm) - min(norm)
        try:
            stdev = statistics.pstdev(norm)
        except statistics.StatisticsError:
            stdev = 0.0
        mean_abs = max(abs(statistics.mean(norm)), 1e-6)
        if spread > 0 and stdev / mean_abs > OSCILLATION_RATIO:
            # Distinguish "noisy improvement" (overall trending up) from
            # genuine oscillation around a fixed mean. We require BOTH
            # (a) net change small relative to spread, AND (b) the
            # sign changes at least twice over the window.
            net = norm[-1] - norm[0]
            sign_changes = 0
            for i in range(2, len(norm)):
                d1 = norm[i - 1] - norm[i - 2]
                d2 = norm[i] - norm[i - 1]
                if d1 * d2 < 0:
                    sign_changes += 1
            if abs(net) < spread * 0.5 or sign_changes >= 2:
                reasons.append(
                    f"window stdev/mean={stdev / mean_abs:.2f} > "
                    f"{OSCILLATION_RATIO} with {sign_changes} sign changes"
                )
                return "oscillating", reasons

    first, last = norm[0], norm[-1]
    delta = last - first
    rel = abs(delta) / max(abs(first), 1e-6)

    if rel < PLATEAU_EPSILON:
        reasons.append(f"|delta|={abs(delta):.3g} below plateau epsilon")
        return "plateau", reasons
    if delta > 0:
        reasons.append(f"normalized score improved by {delta:.3g} over window")
        return "improving", reasons
    if delta < 0:
        reasons.append(f"normalized score regressed by {delta:.3g} over window")
        return "regressing", reasons

    reasons.append("no signal in score window")
    return "plateau", reasons


def diagnose(req: InterventionRequest) -> RuntimeState:
    """Compute the v1.5 ``RuntimeState`` for an ``InterventionRequest``."""
    safety_ctx = req.safety_context
    sym, severity, _strategy = diagnose_safety(
        safety_ctx.error_log,
        safety_ctx.safety_events,
        safety_ctx.constraint_violations,
        safety_ctx.severity_hint,
    )
    safety_state = safety_state_from_severity(severity)

    feasibility = _feasibility(req)
    optimization, opt_reasons = _optimization(req)

    # Promote optimization_state to "invalid_heavy" when feasibility
    # says so — even if scores look stable. The agent isn't really
    # plateaued: it's failing to produce valid candidates.
    if feasibility in ("invalid_heavy", "all_invalid"):
        optimization = "invalid_heavy"
        opt_reasons = [f"feasibility={feasibility} dominates score trajectory"]

    confidence = 0.85 if safety_state != "safe" else 0.75
    if optimization == "early":
        confidence = 0.55

    return RuntimeState(
        optimization_state=optimization,
        feasibility_state=feasibility,
        safety_state=safety_state,
        safety_severity=severity,
        safety_symptom=sym,
        confidence=confidence,
        reasons=opt_reasons,
    )


__all__ = [
    "ALL_INVALID_RATE",
    "DEFAULT_WINDOW",
    "EARLY_ITERATIONS",
    "INVALID_HEAVY_RATE",
    "LATE_BUDGET_FRAC",
    "OSCILLATION_RATIO",
    "PLATEAU_EPSILON",
    "diagnose",
]
