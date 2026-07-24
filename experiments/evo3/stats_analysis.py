"""EVO-3 statistical analysis (数据库优化v4 §12).

Rules:

* Never pool raw round counts as if independent (§12.1): report
  Round/Session/Seed/Day/Hand/Gesture levels; the session is the unit of
  inference.
* Promotion conclusions carry effect size + 95% CI + p-value + session
  distribution (§12.5) — never a bare pooled rate (§17.10).

Implemented without heavyweight deps (numpy only); statsmodels is used
for the mixed-effects model when available (§12.2), with an honest
fallback report when it is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SessionRecord:
    """One session's aggregate outcome (the inference unit, v4 §12.1)."""

    session_id: str
    arm: str
    rounds: int
    invalid_count: int
    failure_count: int
    first_failure_round: int | None
    verified_count: int
    peak_temperature_c: float | None = None
    temperature_slope: float | None = None
    position_error_p95: float | None = None
    day: str | None = None
    hand: str | None = None
    seed: int | None = None
    memory_hurt_events: int = 0
    unsafe_actions: int = 0

    @property
    def invalid_rate(self) -> float:
        return self.invalid_count / self.rounds if self.rounds else 0.0

    @property
    def verified_rate(self) -> float:
        return self.verified_count / self.rounds if self.rounds else 0.0


# ---------------------------------------------------------------------------
# Session-level aggregation (§12.1)
# ---------------------------------------------------------------------------


def aggregate_by_arm(records: list[SessionRecord]) -> dict[str, Any]:
    arms: dict[str, list[SessionRecord]] = {}
    for record in records:
        arms.setdefault(record.arm, []).append(record)
    return {
        arm: {
            "sessions": len(rows),
            "rounds": sum(r.rounds for r in rows),
            "invalid_rate_mean": float(np.mean([r.invalid_rate for r in rows])),
            "invalid_rate_sd": float(np.std([r.invalid_rate for r in rows], ddof=1))
            if len(rows) > 1
            else 0.0,
            "invalid_rate_per_session": [round(r.invalid_rate, 4) for r in rows],
            "verified_rate_mean": float(np.mean([r.verified_rate for r in rows])),
            "memory_hurt_events": sum(r.memory_hurt_events for r in rows),
            "unsafe_actions": sum(r.unsafe_actions for r in rows),
        }
        for arm, rows in arms.items()
    }


# ---------------------------------------------------------------------------
# Paired bootstrap (§12.4)
# ---------------------------------------------------------------------------


def paired_bootstrap(
    before: list[float],
    after: list[float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap CI for the mean paired difference (after − before)."""
    if len(before) != len(after) or not before:
        raise ValueError("paired bootstrap requires equal non-empty pairs")
    diffs = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot[i] = rng.choice(diffs, size=n, replace=True).mean()
    alpha = (1.0 - ci) / 2.0
    return {
        "pairs": n,
        "mean_diff": float(diffs.mean()),
        "ci": [float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))],
        "p_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
    }


# ---------------------------------------------------------------------------
# McNemar (§12.4)
# ---------------------------------------------------------------------------


def mcnemar(b: int, c: int) -> dict[str, Any]:
    """Exact McNemar test on discordant pairs.

    b = pairs where arm1 failed and arm2 did not;
    c = pairs where arm2 failed and arm1 did not.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_exact": None, "note": "no discordant pairs"}
    k = min(b, c)
    # Two-sided exact binomial test with p=0.5.
    from math import comb

    p = 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return {"b": b, "c": c, "discordant": n, "p_exact": min(1.0, p)}


# ---------------------------------------------------------------------------
# Survival analysis (§12.3): first invalid round as time-to-event
# ---------------------------------------------------------------------------


def kaplan_meier(times: list[float], events: list[bool]) -> list[dict[str, float]]:
    """Kaplan-Meier survival curve S(t) = P(first failure after t)."""
    order = sorted(zip(times, events, strict=True), key=lambda pair: pair[0])
    curve: list[dict[str, float]] = []
    survival = 1.0
    at_risk = len(order)
    for time, event in order:
        if event:
            survival *= (at_risk - 1) / at_risk if at_risk > 0 else 1.0
        curve.append({"time": float(time), "survival": survival, "at_risk": at_risk})
        at_risk -= 1
    return curve


def median_first_failure(times: list[float], events: list[bool]) -> float | None:
    """Median time to first invalid (None if never reached)."""
    curve = kaplan_meier(times, events)
    for point in curve:
        if point["survival"] <= 0.5:
            return point["time"]
    return None


def restricted_mean_survival(times: list[float], events: list[bool], horizon: float) -> float:
    """RMST: expected failure-free rounds within [0, horizon]."""
    curve = kaplan_meier(times, events)
    area = 0.0
    last_t = 0.0
    last_s = 1.0
    for point in curve:
        if point["time"] > horizon:
            break
        area += (point["time"] - last_t) * last_s
        last_t = point["time"]
        last_s = point["survival"]
    area += (horizon - last_t) * last_s
    return area


# ---------------------------------------------------------------------------
# Effect size + promotion report (§12.5)
# ---------------------------------------------------------------------------


def promotion_report(
    records: list[SessionRecord],
    *,
    arm_a: str,
    arm_b: str,
    horizon_rounds: int = 100,
    n_bootstrap: int = 10_000,
) -> dict[str, Any]:
    """Session-level comparison of two arms (§12.5).

    Reports effect size (Cohen's d on session invalid rates), a paired
    bootstrap on sessions matched by order index (same protocol slot),
    McNemar on session-level failure presence, RMST survival gain, and
    the session distribution — never a pooled round count.
    """
    a = sorted((r for r in records if r.arm == arm_a), key=lambda r: r.session_id)
    b = sorted((r for r in records if r.arm == arm_b), key=lambda r: r.session_id)
    if not a or not b:
        raise ValueError("both arms need at least one session")

    rate_a = np.asarray([r.invalid_rate for r in a])
    rate_b = np.asarray([r.invalid_rate for r in b])
    pooled_sd = math.sqrt(
        ((len(a) - 1) * rate_a.var(ddof=1) + (len(b) - 1) * rate_b.var(ddof=1))
        / max(len(a) + len(b) - 2, 1)
    )
    # 0/0 effect size is undefined, not zero — report it honestly.
    cohens_d = float((rate_b.mean() - rate_a.mean()) / pooled_sd) if pooled_sd > 0 else None

    pairs = min(len(a), len(b))
    boot = paired_bootstrap(
        [r.invalid_rate for r in a[:pairs]],
        [r.invalid_rate for r in b[:pairs]],
        n_bootstrap=n_bootstrap,
    )
    discord_b = sum(
        1
        for x, y in zip(a[:pairs], b[:pairs], strict=True)
        if x.invalid_count and not y.invalid_count
    )
    discord_c = sum(
        1
        for x, y in zip(a[:pairs], b[:pairs], strict=True)
        if y.invalid_count and not x.invalid_count
    )
    rmst_a = restricted_mean_survival(
        [r.first_failure_round if r.first_failure_round is not None else horizon_rounds for r in a],
        [r.first_failure_round is not None for r in a],
        horizon_rounds,
    )
    rmst_b = restricted_mean_survival(
        [r.first_failure_round if r.first_failure_round is not None else horizon_rounds for r in b],
        [r.first_failure_round is not None for r in b],
        horizon_rounds,
    )
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "sessions_a": len(a),
        "sessions_b": len(b),
        "invalid_rate_mean_a": float(rate_a.mean()),
        "invalid_rate_mean_b": float(rate_b.mean()),
        "cohens_d": cohens_d,
        "paired_bootstrap": boot,
        "mcnemar": mcnemar(discord_b, discord_c),
        "rmst_a": rmst_a,
        "rmst_b": rmst_b,
        "rmst_gain": rmst_b - rmst_a,
        "session_distribution_a": [round(r.invalid_rate, 4) for r in a],
        "session_distribution_b": [round(r.invalid_rate, 4) for r in b],
        "unsafe_actions_a": sum(r.unsafe_actions for r in a),
        "unsafe_actions_b": sum(r.unsafe_actions for r in b),
        "note": "session-level inference only; pooled round counts are never evidence",
    }


def mixed_effects_invalid_rate(
    records: list[SessionRecord],
) -> dict[str, Any]:
    """v4 §12.2 mixed-effects model when statsmodels is available.

        invalid ~ arm + temperature + temperature_slope + (1 | day)

    Returns an honest unavailable-report otherwise (never a silent OLS
    dressed up as a mixed model).
    """
    try:
        import pandas as pd  # type: ignore
        import statsmodels.formula.api as smf  # type: ignore
    except ImportError:
        return {
            "available": False,
            "reason": "statsmodels/pandas not installed on this host",
            "fallback": "use promotion_report (session-level bootstrap + McNemar)",
        }
    frame = pd.DataFrame(
        {
            "invalid": [r.invalid_rate for r in records],
            "arm": [r.arm for r in records],
            "temperature": [r.peak_temperature_c or 0.0 for r in records],
            "temperature_slope": [r.temperature_slope or 0.0 for r in records],
            "day": [r.day or "d0" for r in records],
        }
    )
    model = smf.mixedlm(
        "invalid ~ arm + temperature + temperature_slope", frame, groups=frame["day"]
    )
    fit = model.fit(reml=True)
    return {"available": True, "summary": str(fit.summary())}
