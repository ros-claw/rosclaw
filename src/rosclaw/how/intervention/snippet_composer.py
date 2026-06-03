"""rosclaw_how.snippet_composer — v1.5 strategy-aware snippet templates.

Replaces the v1 ``assemble_inspiration`` "Cross-Domain + Suggested action
+ Diff" trio with a strategy-specific composition:

* CATALYST            → Diagnosis / Likely Cause / Next Experiment /
                        Code Target / Patch Sketch / Expected Signal /
                        Stop Condition.
* SAFETY              → Hard Constraint / Stop Condition.
* STOP_UNSAFE         → Halt + sandbox-validation directive.
* FEASIBILITY_REPAIR  → Invalid Cause / Minimal Repair / Verifier Check.
* STABILIZE           → Symptom / Reduce / Clamp / Stop Condition.
* DIVERSIFY           → Cooldown notice / Orthogonal direction.
* EXPLOIT_BEST        → Preserve best / Local mutation only.
* DIAGNOSE            → What changed / What to log / Probe.
* ESCALATE_TO_KNOW    → Blind-spot report + research request.

The v1 ``assemble_inspiration`` is kept as a fallback for the legacy
CATALYST path so existing tests don't change shape; the new templates
plug into ``/runtime/v1/intervene`` and the v1 endpoint when the new
strategy enum is engaged.

No I/O. Returns plain markdown.
"""
from __future__ import annotations

import json
from typing import Any, Final

from .schemas import InterventionDecision, RuntimeState, StrategyV15

# In rosclaw-how the cap came from settings.max_injection_tokens (default 400).
# In the monorepo there is no FastAPI settings layer to wire to, so we inline
# the same default. Multiply by 4 to get a char budget (matches the original
# ``max_chars = max_injection_tokens * 4`` calculation).
_DEFAULT_MAX_INJECTION_TOKENS: Final[int] = 400


# SAFETY snippets used by the SAFETY / RESOURCE_REPAIR composers. Ported
# verbatim from ``rosclaw_how.semantic_router.SAFETY_RULES`` — we copy the
# constant here so the v1.5 layer stays standalone (no vector router /
# embedding model dependency).
_SAFETY_RULES: Final[dict[str, str]] = {
    "Torque_Overflow": (
        "## ⚠️ SAFETY: Torque Overflow\n"
        "Hardware limit on Unitree-G1 = 237 N·m per joint. Check IMMEDIATELY:\n"
        "  1) PID Kp gain — likely too large.\n"
        "  2) Missing output saturation / `torch.clamp`.\n"
        "  3) Integral term wind-up (apply anti-windup)."
    ),
    "Velocity_Divergence": (
        "## ⚠️ SAFETY: Velocity Divergence\n"
        "Linear/angular velocity reached unbounded magnitude. Check:\n"
        "  1) Integral term accumulates without leak — add anti-windup.\n"
        "  2) Integration step `dt` too small.\n"
        "  3) Apply `torch.clamp(v, -v_max, v_max)`."
    ),
    "Memory_Exhaustion": (
        "## ⚠️ SAFETY: Memory Exhaustion\n"
        "Process exceeded available memory. Check:\n"
        "  1) Sequence / batch size too large.\n"
        "  2) KV-cache never released — apply sliding window.\n"
        "  3) Gradient accumulation without `.detach()`."
    ),
    "Numerical_Instability": (
        "## ⚠️ SAFETY: Numerical Instability\n"
        "NaN/Inf produced during training or rollout. Check:\n"
        "  1) Learning rate too aggressive — try /3.\n"
        "  2) Apply gradient clipping (`clip_grad_norm_=1.0`).\n"
        "  3) Loss function divides by zero or feeds a log of <=0."
    ),
    "Compile_Error": (
        "## ⚠️ SAFETY: Compile / Import Error\n"
        "The code fails to load. Fix the syntax / import path BEFORE\n"
        "calling the Wiki — this is not a deadlock, it's a typo."
    ),
}

# Hard-coded fallback snippets for STOP_UNSAFE / SAFETY when the
# semantic router didn't produce a match. Each is < 400 tokens.
STOP_UNSAFE_TEMPLATES: Final[dict[str, str]] = {
    "Collision_Risk": (
        "## 🛑 STOP — Collision Risk\n"
        "The simulator / sensor reported a collision. **Halt the current "
        "action** and:\n"
        "  1) Roll back to the last known-safe pose.\n"
        "  2) Re-plan with collision checking in the constraint set.\n"
        "  3) Do NOT retry the same trajectory."
    ),
    "Self_Collision": (
        "## 🛑 STOP — Self-Collision\n"
        "Detected link-on-link contact. Halt:\n"
        "  1) Reduce joint velocity to zero.\n"
        "  2) Re-plan with self-collision checks enabled.\n"
        "  3) Tighten joint-range limits before retry."
    ),
    "Human_Proximity": (
        "## 🛑 STOP — Human Proximity\n"
        "A human is inside the robot's workspace. Halt all motion until "
        "the operator clears the area. Do NOT proceed with optimization."
    ),
    "Fall_Risk": (
        "## 🛑 STOP — Fall Risk\n"
        "IMU / tilt sensor indicates the platform is about to fall. Halt:\n"
        "  1) Lock joints in current configuration.\n"
        "  2) Issue zero velocity to all actuators.\n"
        "  3) Investigate root cause before retry."
    ),
    "Emergency_Stop": (
        "## 🛑 EMERGENCY STOP\n"
        "An emergency-stop signal was raised. Halt all activity, log the "
        "trigger, and require explicit operator clearance before resume."
    ),
}


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def _first_analogy(matched: dict[str, Any] | None) -> dict[str, Any] | None:
    if not matched:
        return None
    analogies = matched.get("analogies") or []
    if not analogies:
        return None
    first = analogies[0]
    if isinstance(first, str):
        try:
            first = json.loads(first)
        except (ValueError, json.JSONDecodeError):
            return None
    return first if isinstance(first, dict) else None


def _first_pattern(matched: dict[str, Any] | None) -> str | None:
    if not matched:
        return None
    patterns = matched.get("patterns") or []
    for p in patterns:
        if isinstance(p, str) and p:
            return p
    return None


def _extract_diff(pattern_md: str, max_lines: int = 12) -> str:
    if not pattern_md:
        return ""
    lines = [ln for ln in pattern_md.splitlines() if ln.startswith(("+", "-", "@@"))]
    return "\n".join(lines[:max_lines])


# ── per-strategy templates ───────────────────────────────────────────────


def _compose_catalyst(matched: dict[str, Any] | None, state: RuntimeState) -> dict[str, str]:
    """Build the structured fields for a CATALYST intervention."""
    analogy = _first_analogy(matched) or {}
    pattern_md = _first_pattern(matched) or ""
    diff = _extract_diff(pattern_md)

    diagnosis = (
        f"Optimization state = {state.optimization_state}; the agent has "
        "stopped making progress on the current objective."
    )
    likely_cause = str(analogy.get("insight") or "").strip()
    next_experiment = str(analogy.get("action_suggestion") or "").strip()
    if not next_experiment:
        next_experiment = (
            "Change one component of the candidate (e.g. tighten one gain "
            "or add one clamp). Do not retune everything at once."
        )

    code_target = ""
    symptom = str((matched or {}).get("symptom") or "")
    if symptom:
        code_target = f"Look for symbols related to: {symptom}."

    parts = ["## 🔧 ROSClaw-How Intervention — CATALYST"]
    parts.append(f"**Diagnosis:** {diagnosis}")
    if likely_cause:
        parts.append(f"**Likely Cause:** {_truncate(likely_cause, 300)}")
    parts.append(f"**Next Experiment:** {_truncate(next_experiment, 300)}")
    if code_target:
        parts.append(f"**Code Target:** {code_target}")
    if diff:
        parts.append("**Patch Sketch:**\n```diff\n" + diff + "\n```")
    parts.append(
        "**Expected Verifier Signal:** validity preserved; score moves in "
        "the agreed direction within 3-5 iterations."
    )
    parts.append(
        "**Stop Condition:** if validity drops or oscillation increases, "
        "revert this change."
    )
    return {
        "snippet": "\n\n".join(parts),
        "diagnosis": diagnosis,
        "next_experiment": next_experiment,
        "code_target": code_target,
        "expected_signal": (
            "validity preserved; normalized score improves within 3-5 iterations"
        ),
        "stop_condition": "validity drops or oscillation increases",
    }


def _compose_safety(state: RuntimeState) -> dict[str, str]:
    sym = state.safety_symptom or "Numerical_Instability"
    snippet = _SAFETY_RULES.get(sym) or (
        f"## ⚠️ SAFETY: {sym}\nThe verifier flagged a safety constraint. "
        "Halt the unsafe action and add a guard before retrying."
    )
    return {
        "snippet": snippet,
        "diagnosis": f"safety constraint violated: {sym}",
        "stop_condition": "rerun only after the guard is in place",
        "expected_signal": "safety event count drops to zero",
    }


def _compose_stop_unsafe(state: RuntimeState) -> dict[str, str]:
    sym = state.safety_symptom or "Emergency_Stop"
    snippet = STOP_UNSAFE_TEMPLATES.get(sym) or (
        "## 🛑 STOP — Unsafe Condition\n"
        f"The verifier reported a hazard ({sym}). Halt and require "
        "operator clearance / sandbox validation before resume."
    )
    return {
        "snippet": snippet,
        "diagnosis": f"hazard: {sym}",
        "stop_condition": "halt all motion until operator clearance",
    }


def _compose_feasibility(state: RuntimeState) -> dict[str, str]:
    snippet = (
        "## 🛠 ROSClaw-How — FEASIBILITY REPAIR\n"
        "**Diagnosis:** the agent is producing candidates that fail "
        f"the verifier's hard constraints ({state.feasibility_state}).\n\n"
        "**Minimal Repair:**\n"
        "  1) Inspect the most recent invalid candidate's return code / "
        "stderr — fix the boundary it violates.\n"
        "  2) Add an explicit guard for that constraint before retrying.\n"
        "  3) Do NOT chase score until validity ≥ 80%.\n\n"
        "**Verifier Check:** rerun the validator on the repaired candidate "
        "BEFORE the next optimization step."
    )
    return {
        "snippet": snippet,
        "diagnosis": f"feasibility = {state.feasibility_state}",
        "next_experiment": "repair the hard-constraint violation, then re-verify",
        "expected_signal": "validity rate climbs above 80%",
        "stop_condition": "all candidates still invalid after the repair",
    }


def _compose_stabilize() -> dict[str, str]:
    snippet = (
        "## 〰️ ROSClaw-How — STABILIZE\n"
        "**Diagnosis:** score is oscillating; the candidate is unstable.\n\n"
        "**Reduce:** halve the most aggressive gain / step size.\n"
        "**Clamp:** add explicit output saturation around the noisy variable.\n"
        "**Smooth:** apply a low-pass filter on the action signal.\n\n"
        "**Stop Condition:** if oscillation persists after the clamp, revert "
        "and inspect the integrator / observer."
    )
    return {
        "snippet": snippet,
        "diagnosis": "oscillation in the score window",
        "expected_signal": "score variance shrinks within 2-3 iterations",
        "stop_condition": "oscillation persists after the clamp",
    }


def _compose_diversify(recent_pattern_id: str | None) -> dict[str, str]:
    snippet = (
        "## 🎲 ROSClaw-How — DIVERSIFY\n"
        "**Diagnosis:** plateau detected, but the same pattern fired "
        f"recently ({recent_pattern_id}). Repeating it would not help.\n\n"
        "**Next Experiment:** pick an *orthogonal* mutation — change a "
        "different module of the candidate, or alter the optimization "
        "objective surface (e.g. add a regularizer instead of tuning gains).\n\n"
        "**Stop Condition:** if the orthogonal direction also plateaus, "
        "escalate (request a fresh hypothesis from rosclaw-know)."
    )
    return {
        "snippet": snippet,
        "diagnosis": "plateau with cooldown — diversify",
        "next_experiment": "mutate an orthogonal component of the candidate",
        "stop_condition": "orthogonal direction plateaus too",
    }


def _compose_exploit_best() -> dict[str, str]:
    snippet = (
        "## 🎯 ROSClaw-How — EXPLOIT BEST\n"
        "**Diagnosis:** iteration budget is almost exhausted.\n\n"
        "**Action:** lock in the best-known candidate and make only local "
        "(small-magnitude) mutations to refine it. Do NOT explore new "
        "branches.\n\n"
        "**Stop Condition:** every remaining trial must preserve validity "
        "and beat the running best."
    )
    return {
        "snippet": snippet,
        "diagnosis": "late-budget — exploit",
        "expected_signal": "best score holds or improves",
        "stop_condition": "trial regresses below best",
    }


def _compose_diagnose() -> dict[str, str]:
    snippet = (
        "## 🔎 ROSClaw-How — DIAGNOSE\n"
        "**Observation:** score is regressing. Before mutating further, log:\n"
        "  - the diff between the last improving and last regressing candidate;\n"
        "  - the verifier output for both, side-by-side;\n"
        "  - any new validity / safety events.\n\n"
        "**Probe:** revert to the previous candidate and re-run once to "
        "rule out verifier nondeterminism.\n\n"
        "**Stop Condition:** until the cause of regression is identified, "
        "do not commit further changes."
    )
    return {
        "snippet": snippet,
        "diagnosis": "regression — need root cause",
        "stop_condition": "no further changes until cause identified",
    }


def _compose_escalate() -> dict[str, str]:
    snippet = (
        "## 📡 ROSClaw-How — ESCALATE\n"
        "**Observation:** repeated unknown errors / no matching pattern.\n\n"
        "This run is being reported to rosclaw-know as a blind spot. The "
        "agent should pause this optimization branch and ask for a fresh "
        "hypothesis (e.g. via ``/wiki/v1/prompt/init`` with a more specific "
        "task summary)."
    )
    return {
        "snippet": snippet,
        "diagnosis": "blind spot — escalate to know",
        "stop_condition": "wait for know-side research",
    }


# ── public composer ──────────────────────────────────────────────────────


def compose(
    strategy: StrategyV15,
    state: RuntimeState,
    *,
    matched: dict[str, Any] | None = None,
    recent_pattern_id: str | None = None,
) -> InterventionDecision:
    """Build an ``InterventionDecision`` for the chosen strategy.

    ``matched`` is the optional ``SemanticRouter.find_nearest`` output;
    it's only used by CATALYST. The composer never queries the router
    itself — keeping I/O at the api boundary."""
    max_chars = _DEFAULT_MAX_INJECTION_TOKENS * 4

    if strategy in ("NOOP", "FREE_EXPLORATION"):
        return InterventionDecision(
            strategy=strategy,
            runtime_state=state,
            snippet="",
            injected=False,
            reason="; ".join(state.reasons) if state.reasons else None,
        )

    pieces: dict[str, str]
    if strategy == "CATALYST":
        pieces = _compose_catalyst(matched, state)
    elif strategy == "SAFETY":
        pieces = _compose_safety(state)
    elif strategy == "STOP_UNSAFE":
        pieces = _compose_stop_unsafe(state)
    elif strategy == "FEASIBILITY_REPAIR":
        pieces = _compose_feasibility(state)
    elif strategy == "STABILIZE":
        pieces = _compose_stabilize()
    elif strategy == "DIVERSIFY":
        pieces = _compose_diversify(recent_pattern_id)
    elif strategy == "EXPLOIT_BEST":
        pieces = _compose_exploit_best()
    elif strategy == "DIAGNOSE":
        pieces = _compose_diagnose()
    elif strategy == "ESCALATE_TO_KNOW":
        pieces = _compose_escalate()
    elif strategy == "RESOURCE_REPAIR":
        # Treat resource exhaustion as a safety + repair compound.
        pieces = _compose_safety(state)
    else:  # pragma: no cover — exhaustive enum, but safe default
        pieces = {"snippet": "", "diagnosis": None}

    snippet = _truncate(pieces.get("snippet", ""), max_chars)
    requires_sandbox = strategy == "STOP_UNSAFE"
    sandbox_checks: list[str] = []
    if requires_sandbox and state.safety_symptom:
        sandbox_checks = _sandbox_checks_for(state.safety_symptom)

    return InterventionDecision(
        strategy=strategy,
        runtime_state=state,
        snippet=snippet,
        injected=bool(snippet),
        diagnosis=pieces.get("diagnosis"),
        next_experiment=pieces.get("next_experiment"),
        code_target=pieces.get("code_target"),
        expected_signal=pieces.get("expected_signal"),
        stop_condition=pieces.get("stop_condition"),
        symptom=state.safety_symptom,
        requires_sandbox_validation=requires_sandbox,
        sandbox_checks=sandbox_checks,
    )


def _sandbox_checks_for(symptom: str) -> list[str]:
    """Suggest sandbox validation steps for a hazard."""
    mapping = {
        "Collision_Risk": ["collision_check", "trajectory_replan"],
        "Self_Collision": ["self_collision_check", "joint_range_tighten"],
        "Fall_Risk": ["imu_stability_check", "joint_lock"],
        "Human_Proximity": ["workspace_clear_check", "operator_acknowledge"],
        "Emergency_Stop": ["operator_acknowledge"],
        "Torque_Overflow": ["torque_limit_check"],
        "Velocity_Divergence": ["velocity_limit_check"],
    }
    return mapping.get(symptom, [])


__all__ = ["compose"]
