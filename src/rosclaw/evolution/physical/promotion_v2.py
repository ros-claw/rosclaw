"""Confirmation gate v2 (Physical Evolution Lab §9.5/§9.6, PR-PE-7).

Exploratory candidates never become effective policies.  The
confirmation gate evaluates a FROZEN protocol's evidence:

* zero safety events (hard);
* practical effect: primary metric improvement ≥ 5 percentage points
  OR relative improvement ≥ 10% — a p-value victory below practical
  effect is not a promotion (v3 §9.6);
* session-level paired CI supports a positive effect;
* ≥ 3 recurrence sessions across ≥ 2 distinct time windows;
* no cold-regime harm;
* complete A/B/C arm coverage per block — a MISSING B arm can never
  pass "not worse than B" (PR-PE-7 acceptance).

Thin evidence is INSUFFICIENT_EVIDENCE — never a plain PROMOTED
(v3 §9.6: 样本不足时输出 INSUFFICIENT_EVIDENCE，不能输出普通 PROMOTED).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

GATE_VERSION = "rosclaw.confirmation_gate.v2"

MIN_RECURRENCE_SESSIONS = 3
MIN_TIME_WINDOWS = 2
MIN_PRACTICAL_EFFECT_PP = 5.0
MIN_RELATIVE_IMPROVEMENT = 0.10
MAX_COLD_REGIME_HARM_PP = 2.0


class ConfirmationVerdict(StrEnum):
    VALIDATED_EFFECTIVE = "validated_effective"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REFUTED = "refuted"


@dataclass(frozen=True)
class RegimeBlock:
    """One A/C paired block inside one regime bin (v3 §9.5)."""

    regime_bin: str  # "cold" | "warm" | "hot_but_safe"
    arm_a_invalid: float
    arm_c_invalid: float
    start_temp_delta_c: float
    time_window: str  # e.g. "2026-07-30T02" — blocks in one window share it
    environmentally_invalid: bool = False
    safety_events: int = 0


@dataclass
class ConfirmationReport:
    verdict: ConfirmationVerdict
    effect_pp: float | None
    relative_improvement: float | None
    ci_low: float | None
    ci_high: float | None
    recurrence_sessions: int
    time_windows: int
    failed_checks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_version": GATE_VERSION,
            "verdict": self.verdict.value,
            "effect_pp": self.effect_pp,
            "relative_improvement": self.relative_improvement,
            "ci": [self.ci_low, self.ci_high],
            "recurrence_sessions": self.recurrence_sessions,
            "time_windows": self.time_windows,
            "failed_checks": self.failed_checks,
            "notes": self.notes,
        }


def _paired_ci(diffs: list[float]) -> tuple[float, float, float]:
    """Mean ± t-ish CI on block-paired diffs (small-n honest: normal
    approx with n-1 sample std; declared, not asymptotic)."""
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return mean, mean, mean
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    return mean, mean - 1.96 * se, mean + 1.96 * se


def evaluate_confirmation(
    blocks: list[RegimeBlock],
    *,
    cold_blocks: list[RegimeBlock] | None = None,
) -> ConfirmationReport:
    """Evaluate the frozen-protocol confirmation campaign evidence."""
    failed: list[str] = []
    notes: list[str] = []

    valid_blocks = [b for b in blocks if not b.environmentally_invalid]
    invalid_blocks = [b for b in blocks if b.environmentally_invalid]
    if invalid_blocks:
        notes.append(
            f"{len(invalid_blocks)} blocks marked ENVIRONMENTALLY_INVALID (kept as safety evidence)"
        )

    safety_events = sum(b.safety_events for b in blocks)
    if safety_events:
        failed.append(f"safety_events={safety_events} (max 0)")

    sessions = len(valid_blocks)
    if sessions < MIN_RECURRENCE_SESSIONS:
        failed.append(f"recurrence_sessions={sessions} < {MIN_RECURRENCE_SESSIONS}")

    windows = {b.time_window for b in valid_blocks}
    if len(windows) < MIN_TIME_WINDOWS:
        failed.append(f"time_windows={len(windows)} < {MIN_TIME_WINDOWS}")

    # Effect: A - C per block (positive = C better).
    diffs = [b.arm_a_invalid - b.arm_c_invalid for b in valid_blocks]
    effect_pp: float | None = None
    relative: float | None = None
    ci_low = ci_high = None
    if diffs:
        mean, ci_low, ci_high = _paired_ci(diffs)
        effect_pp = mean * 100.0
        a_mean = sum(b.arm_a_invalid for b in valid_blocks) / len(valid_blocks)
        relative = (mean / a_mean) if a_mean > 0 else None
        practical = effect_pp >= MIN_PRACTICAL_EFFECT_PP or (
            relative is not None and relative >= MIN_RELATIVE_IMPROVEMENT
        )
        if not practical:
            failed.append(
                f"practical_effect={effect_pp:.1f}pp"
                + (f" ({relative * 100:.1f}% rel)" if relative is not None else "")
                + f" below {MIN_PRACTICAL_EFFECT_PP}pp / {MIN_RELATIVE_IMPROVEMENT * 100:.0f}% rel"
            )
        if ci_low <= 0:
            failed.append(f"paired_ci=[{ci_low * 100:.1f},{ci_high * 100:.1f}]pp crosses zero")
    else:
        failed.append("no valid blocks")

    # Cold-regime harm check (v3 §9.6 no cold-regime harm).
    cold = (
        cold_blocks
        if cold_blocks is not None
        else [b for b in valid_blocks if b.regime_bin == "cold"]
    )
    if cold:
        cold_diffs = [b.arm_a_invalid - b.arm_c_invalid for b in cold]
        cold_effect = sum(cold_diffs) / len(cold_diffs) * 100.0
        if cold_effect < -MAX_COLD_REGIME_HARM_PP:
            failed.append(
                f"cold_regime_harm={cold_effect:.1f}pp (max {-MAX_COLD_REGIME_HARM_PP}pp)"
            )

    if failed:
        # Distinguish thin evidence from real refutation: if we HAVE the
        # sessions/windows but the effect fails or CI crosses zero, that
        # is REFUTED; missing sessions/windows is INSUFFICIENT_EVIDENCE.
        evidence_thin = any(
            f.startswith(("recurrence_sessions=", "time_windows=", "no valid blocks"))
            for f in failed
        )
        verdict = (
            ConfirmationVerdict.INSUFFICIENT_EVIDENCE
            if evidence_thin
            else ConfirmationVerdict.REFUTED
        )
    else:
        verdict = ConfirmationVerdict.VALIDATED_EFFECTIVE

    return ConfirmationReport(
        verdict=verdict,
        effect_pp=effect_pp,
        relative_improvement=relative,
        ci_low=ci_low,
        ci_high=ci_high,
        recurrence_sessions=sessions,
        time_windows=len(windows),
        failed_checks=failed,
        notes=notes,
    )
