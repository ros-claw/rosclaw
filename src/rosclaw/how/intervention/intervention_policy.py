"""rosclaw_how.intervention_policy — v1.5 state → strategy decision.

Inputs (pure data):

* ``RuntimeState`` from ``runtime_diagnoser.diagnose``.
* ``AgentContext.recent_pattern_ids`` from the request (cooldown signal).

Output: a v1.5 ``StrategyV15`` plus a list of human-readable reasons
the operator can read off the dashboard.

The decision tree is **deterministic** and **pure**: same input →
same output. No I/O, no randomness. This makes the policy
reproducible / cacheable / easy to test.

Cooldown rule: when the diagnoser says CATALYST but the agent has
already received an injection in the last N iterations, swap to
DIVERSIFY so the same pattern doesn't snowball.
"""
from __future__ import annotations

from typing import Final

from .safety_router import is_blocking
from .schemas import (
    AgentContext,
    InterventionRequest,
    RuntimeState,
    StrategyV15,
)

# How many recent injections to consider for the cooldown swap.
COOLDOWN_WINDOW: Final[int] = 3


def _cooldown_violated(agent: AgentContext, pattern_id: str | None) -> bool:
    if not pattern_id:
        return False
    recent = list(agent.recent_pattern_ids[-COOLDOWN_WINDOW:])
    return pattern_id in recent


def decide_strategy(
    req: InterventionRequest,
    state: RuntimeState,
    *,
    recent_pattern_id: str | None = None,
) -> tuple[StrategyV15, list[str]]:
    """Map ``RuntimeState`` → ``StrategyV15`` with cooldown awareness.

    ``recent_pattern_id`` is the pattern the router is *about to*
    inject; if the same pattern is in the agent's ``recent_pattern_ids``
    window the decision flips to DIVERSIFY (the snippet composer will
    return an "explore a different direction" template instead).
    """
    reasons: list[str] = []

    # ── 1. Safety override ───────────────────────────────────────────
    if state.safety_state == "emergency":
        reasons.append("safety severity S4 — emergency stop")
        return "STOP_UNSAFE", reasons
    if state.safety_state == "hazard":
        # Hazard symptoms either need to halt (collision/fall) or
        # inject a hard constraint (torque/velocity). Both are SAFETY
        # at the v1 level, so we route the more conservative ones to
        # STOP_UNSAFE and the rest stay as SAFETY.
        if state.safety_symptom in (
            "Collision_Risk",
            "Self_Collision",
            "Fall_Risk",
            "Emergency_Stop",
            "Human_Proximity",
        ):
            reasons.append(f"safety hazard {state.safety_symptom} — stop unsafe")
            return "STOP_UNSAFE", reasons
        reasons.append(f"safety hazard {state.safety_symptom} — inject hard constraint")
        return "SAFETY", reasons
    if state.safety_state == "constraint_violation":
        # The v1.5 safety taxonomy may want resource repair vs stabilize
        # at S2 — but the legacy SAFETY enum is still the right top-level.
        if state.safety_symptom == "Memory_Exhaustion":
            reasons.append("S2 memory exhaustion — resource repair")
            return "RESOURCE_REPAIR", reasons
        if state.safety_symptom == "Numerical_Instability":
            reasons.append("S2 numerical instability — stabilize")
            return "STABILIZE", reasons
        reasons.append(f"S2 constraint violation {state.safety_symptom}")
        return "SAFETY", reasons
    if state.safety_state == "warning":
        # S1 warnings: compile errors must take the FEASIBILITY repair
        # path BEFORE the optimization state machine swallows them
        # under plateau-CATALYST. Other S1 signals (battery_low /
        # gripper_overload) fall through to the optimization path; if
        # the safety_router exposed a strategy, honor it.
        if state.safety_symptom == "Compile_Error":
            reasons.append("S1 compile error — feasibility repair")
            return "FEASIBILITY_REPAIR", reasons
        if state.safety_symptom == "Gripper_Overload":
            reasons.append("S1 gripper overload — safety")
            return "SAFETY", reasons
        # battery_low / unknown S1 → fall through to the optimization path.

    # ── 2. Feasibility ───────────────────────────────────────────────
    if state.feasibility_state in ("all_invalid", "invalid_heavy"):
        reasons.append(f"feasibility={state.feasibility_state}")
        return "FEASIBILITY_REPAIR", reasons

    # ── 3. Optimization state machine ────────────────────────────────
    opt = state.optimization_state
    if opt == "early":
        reasons.append("early iterations — let the agent explore")
        return "FREE_EXPLORATION", reasons
    if opt == "improving":
        reasons.append("score improving — no injection")
        return "NOOP", reasons
    if opt == "late_budget":
        reasons.append("late in iteration budget — exploit best candidate")
        return "EXPLOIT_BEST", reasons
    if opt == "oscillating":
        reasons.append("score oscillating — stabilize before next mutation")
        return "STABILIZE", reasons
    if opt == "regressing":
        reasons.append("score regressing — diagnose root cause")
        return "DIAGNOSE", reasons
    if opt == "invalid_heavy":
        reasons.append("invalid-heavy run — feasibility repair")
        return "FEASIBILITY_REPAIR", reasons
    if opt == "plateau":
        if _cooldown_violated(req.agent_context, recent_pattern_id):
            reasons.append(
                f"plateau but pattern '{recent_pattern_id}' fired in last "
                f"{COOLDOWN_WINDOW} injections — diversify instead"
            )
            return "DIVERSIFY", reasons
        reasons.append("plateau — fire catalyst")
        return "CATALYST", reasons

    # Unknown / no signal — be conservative.
    reasons.append("no actionable signal; deferring to free exploration")
    return "FREE_EXPLORATION", reasons


__all__ = ["COOLDOWN_WINDOW", "decide_strategy", "is_blocking"]
