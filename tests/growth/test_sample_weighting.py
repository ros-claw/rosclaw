import copy

import numpy as np
import pytest

from rosclaw.growth.proposal_advantage_regression import fit_proposal_advantage_residual
from rosclaw.growth.sample_weighting import (
    balanced_partition_weights,
    sample_weight_receipt,
    validate_sample_weights,
)
from tests.growth.test_proposal_advantage_regression import inputs


def test_partition_balance_retains_every_row_and_owns_readonly_result():
    labels = np.array([2] * 8 + [4] * 2 + [9] * 2)
    original = labels.copy()
    weights = balanced_partition_weights(labels)
    assert len(weights) == len(labels)
    for label in np.unique(labels):
        assert weights[labels == label].sum() == pytest.approx(4)
    assert weights.mean() == 1
    assert not weights.flags.writeable
    np.testing.assert_array_equal(labels, original)
    receipt = sample_weight_receipt(weights)
    assert receipt["all_numeric_rows_retained"] is True
    assert receipt["row_count"] == 12
    assert receipt["physical_batch_verified"] is False
    assert receipt["runtime_execution_authorized"] is False


@pytest.mark.parametrize(
    "value",
    [
        [0, 0, 0, 1],
        [1, 1, 1, 1.1],
        [True] * 4,
        [float("nan")] * 4,
        [float("inf")] * 4,
        [1, 1, 1],
        [[1] * 4],
        [1, 1, 1, -1],
    ],
)
def test_invalid_weight_vectors_fail_closed(value):
    with pytest.raises(ValueError):
        validate_sample_weights(value, 4)


@pytest.mark.parametrize(
    "labels",
    [
        [0, 0, 0, 16],
        [0, 0, 0, -1],
        [0.0, 0.0, 1.0, 1.0],
        [True] * 4,
        [0] * 100 + [1],
        [[0, 1], [0, 1]],
    ],
)
def test_invalid_or_excessively_imbalanced_partitions_fail_closed(labels):
    with pytest.raises(ValueError):
        balanced_partition_weights(labels)


def test_explicit_unit_weights_preserve_default_numerics_exactly():
    pytest.importorskip("torch")
    data = inputs()
    old = fit_proposal_advantage_residual(**data)
    new = fit_proposal_advantage_residual(**data, sample_weights=np.ones(len(data["context"])))
    assert {k: v for k, v in new.items() if k != "sample_weighting"} == old
    assert new["sample_weighting"]["row_count"] == len(data["context"])


def test_nonuniform_weights_change_learning_but_not_kl_or_authority_bounds():
    pytest.importorskip("torch")
    data = inputs(budget=0.05, large_actions=True)
    before = copy.deepcopy(data)
    labels = np.zeros(len(data["context"]), dtype=int)
    labels[: len(labels) // 4] = 1
    weighted = fit_proposal_advantage_residual(
        **data, sample_weights=balanced_partition_weights(labels)
    )
    uniform = fit_proposal_advantage_residual(**data)
    assert weighted["layers"] != uniform["layers"]
    assert max(weighted["exact_mean_conditional_kl"], weighted["exact_mean_marginal_kl"]) <= 0.05
    for key in (
        "hardware_authorized",
        "promotion_authorized",
        "physical_batch_verified",
        "runtime_execution_authorized",
        "distributional_retention_guaranteed",
    ):
        assert weighted[key] is False
    np.testing.assert_array_equal(data["context"], before["context"])
    np.testing.assert_array_equal(data["gates"], before["gates"])


def test_weighting_never_bypasses_conditional_behavior_density():
    data = inputs()
    data["old_log_probability"][0] += 0.1
    with pytest.raises(ValueError, match="conditional behavior"):
        fit_proposal_advantage_residual(**data, sample_weights=np.ones(len(data["context"])))


def test_scalar_weight_receipt_fails_closed():
    with pytest.raises(ValueError, match="receipt"):
        sample_weight_receipt(1)
