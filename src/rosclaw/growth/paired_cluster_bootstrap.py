"""Offline paired binary-effect estimates; never execution or promotion authority.

The operator supplies a preregistered balanced cluster shape. Every parent and
candidate outcome must be present, paired in that same order, and an ordinary
bool. Resampling whole clusters retains within-cluster dependence. Independent
clusters, untouched holdouts, ordering, evidence integrity and safety retention
are external requirements, not facts this numeric calculator can establish.

The percentile interval uses NumPy's explicit linear quantile convention and a
local PCG64 generator. Family-wise alpha is Bonferroni divided across the declared
comparisons, not chosen from whichever candidate looks best. A positive interval
and minimum effect are only a numerical condition, never a promotion decision.
Percentile bootstrap coverage is not guaranteed for a small or homogeneous
cluster population; reported intervals are not exact binomial confidence bounds.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PairedClusterPlan:
    """Explicit bounded design; configure before observing exam outcomes."""

    clusters: int
    members_per_cluster: int
    resamples: int
    seed: int
    comparisons: int
    familywise_alpha: float
    minimum_absolute_gain: float

    def __post_init__(self) -> None:
        for value, low, high in (
            (self.clusters, 2, 256),
            (self.members_per_cluster, 1, 4096),
            (self.resamples, 1000, 1_000_000),
            (self.seed, 0, 2**64 - 1),
            (self.comparisons, 1, 64),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("ordinary integer within paired bootstrap design bounds required")
        if self.clusters * self.resamples > 20_000_000:
            raise ValueError("paired bootstrap allocation budget exceeded")
        for probability in (self.familywise_alpha, self.minimum_absolute_gain):
            if type(probability) not in (int, float) or not math.isfinite(probability):
                raise ValueError("finite ordinary probability or effect required")
        if not 0 < self.familywise_alpha < 0.5:
            raise ValueError("family-wise alpha must be strictly between zero and one half")
        if 1 - self.familywise_alpha / self.comparisons / 2 == 1:
            raise ValueError("corrected interval tail must be representable")
        if not 0 <= self.minimum_absolute_gain <= 1:
            raise ValueError("absolute gain must be between zero and one")


@dataclass(frozen=True)
class PairedClusterEstimate:
    """Numeric evidence only; no freshness, safety, or authority attestation."""

    input_sha256: str
    clusters: int
    paired_cases: int
    parent_successes: int
    candidate_successes: int
    paired_wins: int
    paired_losses: int
    gain: float
    lower: float
    upper: float
    per_comparison_confidence: float
    numerical_gain_condition_met: bool
    cluster_net_wins: tuple[int, ...]
    promotion_authorized: bool = False
    hardware_authorized: bool = False


def _own_outcomes(
    outcomes: tuple[tuple[bool, ...], ...], plan: PairedClusterPlan
) -> tuple[tuple[bool, ...], ...]:
    if type(outcomes) is not tuple or len(outcomes) != plan.clusters:
        raise ValueError("complete preregistered cluster tuple required")
    for row in outcomes:
        if (
            type(row) is not tuple
            or len(row) != plan.members_per_cluster
            or any(type(value) is not bool for value in row)
        ):
            raise ValueError("complete balanced ordinary-bool outcome tuples required")
    return tuple(tuple(value for value in row) for row in outcomes)


def estimate_paired_cluster_gain(
    parent: tuple[tuple[bool, ...], ...],
    candidate: tuple[tuple[bool, ...], ...],
    *,
    plan: PairedClusterPlan,
) -> PairedClusterEstimate:
    """Paired cluster percentile bootstrap; no global RNG or file mutations.

    Reject partial or unbalanced data rather than imputing missing failures or
    dropping rows. This rejects a missing cluster only relative to the supplied
    plan; callers must independently bind that plan to the original protocol.
    """
    if type(plan) is not PairedClusterPlan:
        raise ValueError("exact preregistered paired cluster plan required")
    # Own and revalidate even a frozen instance: object.__setattr__ can bypass it.
    owned = PairedClusterPlan(**asdict(plan))
    old = _own_outcomes(parent, owned)
    new = _own_outcomes(candidate, owned)
    digest = hashlib.sha256(
        json.dumps(
            {"plan": asdict(owned), "parent": old, "candidate": new},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    net = tuple(
        sum(int(b) - int(a) for a, b in zip(old_row, new_row, strict=True))
        for old_row, new_row in zip(old, new, strict=True)
    )
    count = owned.clusters * owned.members_per_cluster
    gain = sum(net) / count
    rng = np.random.Generator(np.random.PCG64(owned.seed))
    draws = rng.integers(0, owned.clusters, size=(owned.resamples, owned.clusters))
    distribution = np.asarray(net, dtype=np.int64)[draws].sum(axis=1) / count
    comparison_alpha = owned.familywise_alpha / owned.comparisons
    tails = comparison_alpha / 2
    lower, upper = np.quantile(distribution, (tails, 1 - tails), method="linear")
    pairs = tuple(
        (a, b)
        for old_row, new_row in zip(old, new, strict=True)
        for a, b in zip(old_row, new_row, strict=True)
    )
    return PairedClusterEstimate(
        input_sha256=digest,
        clusters=owned.clusters,
        paired_cases=count,
        parent_successes=sum(sum(row) for row in old),
        candidate_successes=sum(sum(row) for row in new),
        paired_wins=sum(not a and b for a, b in pairs),
        paired_losses=sum(a and not b for a, b in pairs),
        gain=gain,
        lower=float(lower),
        upper=float(upper),
        per_comparison_confidence=1 - comparison_alpha,
        numerical_gain_condition_met=bool(gain >= owned.minimum_absolute_gain and lower > 0),
        cluster_net_wins=net,
    )
