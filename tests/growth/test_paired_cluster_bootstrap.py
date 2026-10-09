"""Synthetic calculator checks, NOT physical exam evidence."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from rosclaw.growth.paired_cluster_bootstrap import (
    PairedClusterPlan,
    estimate_paired_cluster_gain,
)


def design(**changes):
    return replace(PairedClusterPlan(10, 4, 20_000, 20261001310, 2, 0.05, 0.15), **changes)


def rows(value):
    return ((value,) * 4,) * 10


def estimate(old, new, **changes):
    return estimate_paired_cluster_gain(old, new, plan=design(**changes))


def test_uniform_improvement_and_no_authority():
    result = estimate(rows(False), rows(True))
    assert result.gain == result.lower == result.upper == 1
    assert result.paired_cases == result.paired_wins == 40
    assert result.paired_losses == result.parent_successes == 0
    assert result.candidate_successes == 40
    assert result.per_comparison_confidence == 0.975
    assert result.numerical_gain_condition_met
    assert type(result.numerical_gain_condition_met) is bool
    assert json.loads(json.dumps(asdict(result)))["numerical_gain_condition_met"] is True
    assert not result.promotion_authorized
    assert not result.hardware_authorized


@pytest.mark.parametrize("value", [False, True])
def test_no_change_fails_strict_positive_bound(value):
    result = estimate(rows(value), rows(value), minimum_absolute_gain=0)
    assert result.gain == result.lower == result.upper == 0
    assert not result.numerical_gain_condition_met


def test_regression_is_negative():
    result = estimate(rows(True), rows(False))
    assert result.gain == result.lower == result.upper == -1
    assert result.paired_losses == 40
    assert not result.numerical_gain_condition_met


def test_one_exceptional_cluster_not_forty_independent_trials():
    new = ((True,) * 4,) + ((False,) * 4,) * 9
    result = estimate(rows(False), new, minimum_absolute_gain=0.05)
    assert result.gain == 0.1
    assert result.lower == 0
    assert not result.numerical_gain_condition_met


def test_effect_threshold_inclusive():
    new = ((True, False, False, False),) * 10
    assert estimate(rows(False), new, minimum_absolute_gain=0.25).numerical_gain_condition_met
    assert not estimate(rows(False), new, minimum_absolute_gain=0.251).numerical_gain_condition_met


def test_reference_implementation_and_paired_losses():
    old = tuple((i % 3 == 0, True, False, i % 2 == 0) for i in range(10))
    new = tuple((True, i % 3 == 0, i % 2 == 0, True) for i in range(10))
    result = estimate(old, new)
    net = [sum(b) - sum(a) for a, b in zip(old, new, strict=True)]
    rng = np.random.Generator(np.random.PCG64(20261001310))
    # Independent scalar implementation resamples complete four-case groups.
    samples = [sum(net[int(i)] for i in rng.integers(0, 10, 10)) / 40 for _ in range(20_000)]
    bounds = np.quantile(samples, [0.0125, 0.9875], method="linear")
    assert (result.lower, result.upper) == tuple(bounds)
    assert result.cluster_net_wins == tuple(net)
    assert result.paired_losses > 0
    assert result.gain == (result.paired_wins - result.paired_losses) / 40
    assert estimate(old, new) == result


def test_does_not_change_global_random_state():
    before = np.random.get_state()
    estimate(rows(False), rows(True))
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_input_hash_binds_pair_order_and_design():
    new = ((True,) * 4,) + ((False,) * 4,) * 9
    base = estimate(rows(False), new)
    reordered = estimate(rows(False), tuple(reversed(new)))
    assert base.input_sha256 != reordered.input_sha256
    assert base.input_sha256 != estimate(rows(False), new, seed=1).input_sha256
    assert base.input_sha256 != estimate(rows(False), new, comparisons=1).input_sha256
    assert base.input_sha256 != estimate(new, rows(False)).input_sha256


def test_more_comparisons_widen_interval():
    new = tuple((i < 8, i < 6, i < 4, False) for i in range(10))
    one = estimate(rows(False), new, comparisons=1)
    many = estimate(rows(False), new, comparisons=4)
    assert many.lower <= one.lower <= one.upper <= many.upper


@pytest.mark.parametrize(
    "change",
    [
        {"clusters": True},
        {"clusters": 1},
        {"clusters": 257},
        {"members_per_cluster": 0},
        {"members_per_cluster": 4097},
        {"resamples": 999},
        {"resamples": 1_000_001},
        {"seed": -1},
        {"seed": 2**64},
        {"seed": 1.0},
        {"comparisons": 0},
        {"comparisons": 65},
        {"comparisons": False},
        {"familywise_alpha": 0},
        {"familywise_alpha": 0.5},
        {"familywise_alpha": float("nan")},
        {"familywise_alpha": float("inf")},
        {"familywise_alpha": True},
        {"familywise_alpha": 5e-324},
        {"familywise_alpha": 1e-20},
        {"minimum_absolute_gain": -0.1},
        {"minimum_absolute_gain": 1.1},
        {"minimum_absolute_gain": float("nan")},
        {"clusters": 256, "resamples": 1_000_000},
    ],
)
def test_invalid_plan(change):
    with pytest.raises(ValueError):
        design(**change)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        rows(False)[:-1],
        rows(False) + ((False,) * 4,),
        ((False,) * 3,) * 10,
        ((False,) * 5,) * 10,
        ((0,) * 4,) * 10,
        ((np.bool_(True),) * 4,) * 10,
        ([False] * 4,) * 10,
        ((None,) * 4,) * 10,
    ],
)
@pytest.mark.parametrize("side", ["parent", "candidate"])
def test_incomplete_unbalanced_non_bool_outcomes_rejected(bad, side):
    values = {"parent": rows(False), "candidate": rows(True), side: bad}
    with pytest.raises(ValueError):
        estimate_paired_cluster_gain(**values, plan=design())


def test_frozen_plan_is_revalidated():
    plan = design()
    object.__setattr__(plan, "resamples", 2**60)
    with pytest.raises(ValueError):
        estimate_paired_cluster_gain(rows(False), rows(True), plan=plan)


def test_wrong_plan_type():
    with pytest.raises(ValueError):
        estimate_paired_cluster_gain(rows(False), rows(True), plan=None)
