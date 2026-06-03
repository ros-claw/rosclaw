"""rosclaw_how.safety_router — v1.5 multi-level safety taxonomy.

The v1 ``state_router._matches_safety`` covers five engineering-grade
safety symptoms (torque / velocity / OOM / NaN / compile error). For
physical AI / embodied agents that vocabulary is too narrow — real
robots fail in ways that v1 lumps into ``Unknown_Error`` and the
CATALYST path then surfaces unsafe optimization hints.

This module classifies error logs + structured safety events into:

* a **symptom label** (extended vocabulary)
* a **severity** (S0 info → S4 emergency)
* a **strategy mapping** the api / runtime decider can dispatch on

The v1 SAFETY_KEYWORDS dict is preserved as a subset so the legacy
``state_router`` keeps matching the original five symptoms.

Pure rules. No I/O. Hot-path safe.
"""
from __future__ import annotations

from typing import Final, Literal

from .schemas import SafetySeverity, SafetyState

SafetyStrategy = Literal[
    "NOOP",
    "SAFETY",
    "STABILIZE",
    "FEASIBILITY_REPAIR",
    "RESOURCE_REPAIR",
    "STOP_UNSAFE",
]


# Each entry: symptom_label → (severity, strategy, keyword tuples).
# Ordered most-severe-first so substring scans surface the worst signal.
SAFETY_TAXONOMY: Final[dict[str, dict[str, object]]] = {
    # ── S4 emergency: stop now ─────────────────────────────────────────
    "Human_Proximity": {
        "severity": "S4",
        "strategy": "STOP_UNSAFE",
        "keywords": ("human proximity", "person detected", "operator nearby"),
    },
    "Emergency_Stop": {
        "severity": "S4",
        "strategy": "STOP_UNSAFE",
        "keywords": ("emergency stop", "e-stop", "estop triggered"),
    },
    # ── S3 hazard: physical or safety hazard ──────────────────────────
    "Collision_Risk": {
        "severity": "S3",
        "strategy": "STOP_UNSAFE",
        "keywords": ("collision detected", "collision risk", "contact unexpected"),
    },
    "Self_Collision": {
        "severity": "S3",
        "strategy": "STOP_UNSAFE",
        "keywords": ("self-collision", "self collision", "link collision"),
    },
    "Torque_Overflow": {
        "severity": "S3",
        "strategy": "SAFETY",
        "keywords": ("torque overflow", "exceeded limit", "torque saturation"),
    },
    "Velocity_Divergence": {
        "severity": "S3",
        "strategy": "SAFETY",
        "keywords": ("velocity diverg", "velocity explod", "velocity inf"),
    },
    "Fall_Risk": {
        "severity": "S3",
        "strategy": "STOP_UNSAFE",
        "keywords": ("imu tilt", "tip-over", "fall detected", "robot fall"),
    },
    # ── S2 constraint violation: safety constraint exceeded ───────────
    "Joint_Limit_Violation": {
        "severity": "S2",
        "strategy": "SAFETY",
        "keywords": ("joint limit", "joint range exceeded", "joint position limit"),
    },
    "Workspace_Boundary": {
        "severity": "S2",
        "strategy": "SAFETY",
        "keywords": ("workspace bound", "out of workspace", "reach limit"),
    },
    "Numerical_Instability": {
        "severity": "S2",
        "strategy": "STABILIZE",
        "keywords": ("nan", "inf detected", "numerical instabil", "numerical error"),
    },
    "Memory_Exhaustion": {
        "severity": "S2",
        "strategy": "RESOURCE_REPAIR",
        "keywords": ("oom", "out of memory", "cuda out of", "cudamalloc"),
    },
    "Thermal_Limit": {
        "severity": "S2",
        "strategy": "SAFETY",
        "keywords": ("thermal limit", "overheat", "temperature high"),
    },
    # ── S1 warning: soft / recoverable issue ──────────────────────────
    "Compile_Error": {
        "severity": "S1",
        "strategy": "FEASIBILITY_REPAIR",
        "keywords": ("syntaxerror", "indentationerror", "nameerror", "typeerror", "importerror"),
    },
    "Battery_Low": {
        "severity": "S1",
        "strategy": "NOOP",
        "keywords": ("battery low", "low voltage", "battery critical"),
    },
    "Gripper_Overload": {
        "severity": "S1",
        "strategy": "SAFETY",
        "keywords": ("gripper overload", "grip force exceed"),
    },
}


# Severity → high-level SafetyState ("safe"/"warning"/...)
SEVERITY_TO_STATE: Final[dict[SafetySeverity, SafetyState]] = {
    "S0": "safe",
    "S1": "warning",
    "S2": "constraint_violation",
    "S3": "hazard",
    "S4": "emergency",
}


def diagnose_safety(
    error_log: str,
    safety_events: list[str] | None = None,
    constraint_violations: list[str] | None = None,
    severity_hint: SafetySeverity | None = None,
) -> tuple[str | None, SafetySeverity, SafetyStrategy]:
    """Classify the safety signal in a request.

    Returns ``(symptom, severity, strategy)``:
      * ``symptom`` — label from ``SAFETY_TAXONOMY`` or None.
      * ``severity`` — S0 (safe) … S4 (emergency); defaults to S0 when
        nothing matched.
      * ``strategy`` — what the runtime should dispatch.

    Match priority (highest first):
      1. ``severity_hint`` — caller-asserted severity wins when present.
      2. Structured ``safety_events`` / ``constraint_violations`` —
         each entry is normalized and matched against the taxonomy.
      3. Free-form ``error_log`` keyword scan.

    A label is returned for any non-S0 match; S0 returns ``(None, "S0",
    "NOOP")`` so callers can skip the safety branch entirely.
    """
    haystack_parts: list[str] = []
    if error_log:
        haystack_parts.append(error_log)
    for ev in safety_events or []:
        if ev:
            haystack_parts.append(str(ev))
    for cv in constraint_violations or []:
        if cv:
            haystack_parts.append(str(cv))
    haystack = "\n".join(haystack_parts).lower()

    # Walk the taxonomy in declared order (S4 → S1) so the most severe
    # match wins when several keyword groups overlap.
    matched: tuple[str, SafetySeverity, SafetyStrategy] | None = None
    for symptom, entry in SAFETY_TAXONOMY.items():
        keywords: tuple[str, ...] = entry["keywords"]  # type: ignore[assignment]
        if any(kw in haystack for kw in keywords):
            matched = (
                symptom,
                entry["severity"],  # type: ignore[arg-type]
                entry["strategy"],  # type: ignore[arg-type]
            )
            break

    if severity_hint is not None:
        sym = matched[0] if matched else None
        if matched:
            return matched[0], severity_hint, matched[2]
        # No keyword match but caller asserted severity — promote to a
        # generic safety strategy at the asserted level.
        strategy: SafetyStrategy = "SAFETY" if severity_hint in ("S2", "S3") else (
            "STOP_UNSAFE" if severity_hint == "S4" else "NOOP"
        )
        return sym, severity_hint, strategy

    if matched is None:
        return None, "S0", "NOOP"
    return matched


def safety_state_from_severity(severity: SafetySeverity) -> SafetyState:
    """Map a severity to its high-level ``SafetyState``."""
    return SEVERITY_TO_STATE.get(severity, "safe")


def is_blocking(strategy: SafetyStrategy) -> bool:
    """``True`` when the strategy should preempt optimization entirely
    (i.e. SAFETY / STOP_UNSAFE / RESOURCE_REPAIR — the hot path must
    not fall through to CATALYST)."""
    return strategy in ("SAFETY", "STOP_UNSAFE", "RESOURCE_REPAIR")


# ── v1 backward compat ───────────────────────────────────────────────────


def v1_safety_keywords() -> dict[str, tuple[str, ...]]:
    """Return the v1 ``SAFETY_KEYWORDS`` subset (used by ``state_router``).

    Built dynamically from the taxonomy so the v1 keyword list never
    drifts out of sync with the v1.5 source of truth."""
    out: dict[str, tuple[str, ...]] = {}
    for symptom in (
        "Torque_Overflow",
        "Velocity_Divergence",
        "Memory_Exhaustion",
        "Numerical_Instability",
        "Compile_Error",
    ):
        entry = SAFETY_TAXONOMY.get(symptom)
        if entry is None:
            continue
        out[symptom] = entry["keywords"]  # type: ignore[assignment]
    return out


__all__ = [
    "SAFETY_TAXONOMY",
    "SEVERITY_TO_STATE",
    "SafetyStrategy",
    "diagnose_safety",
    "is_blocking",
    "safety_state_from_severity",
    "v1_safety_keywords",
]
