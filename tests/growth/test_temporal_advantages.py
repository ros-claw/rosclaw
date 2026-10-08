"""Numeric temporal credit boundaries, not simulator capability evidence."""

import numpy as np
import pytest

from rosclaw.growth.temporal_advantages import generalized_advantage_targets


def estimate(rewards, values, next_values, ends, terminal, **kwargs):
    return generalized_advantage_targets(
        np.array(rewards),
        np.array(values),
        np.array(next_values),
        np.array(ends, dtype=bool),
        np.array(terminal, dtype=bool),
        discount=kwargs.get("discount", 1.0),
        trace_decay=kwargs.get("trace_decay", 1.0),
    )


def test_terminal_has_no_bootstrap_but_truncation_uses_actual_final_observation():
    result = estimate(
        [[2.0], [2.0]],
        [[0.5], [0.5]],
        [[99.0], [3.0]],
        [[True], [True]],
        [[True], [False]],
        discount=0.9,
    )
    np.testing.assert_allclose(result.advantages[:, 0], [1.5, 4.2])
    np.testing.assert_allclose(result.value_targets[:, 0], [2.0, 4.7])


def test_episode_boundary_prevents_next_episode_reward_leak():
    result = estimate(
        [[1.0, 2.0, 100.0]],
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [[False, True, True]],
        [[False, True, True]],
    )
    np.testing.assert_array_equal(result.advantages, [[3.0, 2.0, 100.0]])


def test_internal_truncation_bootstraps_without_future_advantage_leak():
    result = estimate(
        [[1.0, 2.0, 100.0]],
        [[0.0, 0.0, 0.0]],
        [[0.0, 3.0, 0.0]],
        [[False, True, True]],
        [[False, False, True]],
        discount=0.9,
    )
    np.testing.assert_allclose(result.advantages, [[5.23, 4.7, 100.0]])


def test_lambda_one_recovers_discounted_mc_minus_frozen_value():
    result = estimate(
        [[1.0, 2.0, 3.0]],
        [[0.1, 0.2, 0.3]],
        [[0.2, 0.3, 99.0]],
        [[False, False, True]],
        [[False, False, True]],
        discount=0.5,
    )
    np.testing.assert_allclose(result.value_targets, [[2.75, 3.5, 3.0]])
    np.testing.assert_allclose(result.advantages, [[2.65, 3.3, 2.7]])


def test_lambda_zero_recovers_td_without_later_reward_credit():
    result = estimate(
        [[1.0, 100.0]],
        [[0.2, 0.3]],
        [[0.3, 0.0]],
        [[False, True]],
        [[False, True]],
        discount=0.5,
        trace_decay=0.0,
    )
    np.testing.assert_allclose(result.advantages, [[0.95, 99.7]])
    np.testing.assert_array_equal(result.advantages, result.td_residuals)


def test_collection_cutoff_bootstraps_without_invented_unrecorded_rewards():
    result = estimate([[2.0]], [[0.5]], [[3.0]], [[False]], [[False]], discount=0.9)
    np.testing.assert_allclose(result.value_targets, [[4.7]])


def test_zero_discount_removes_all_future_credit():
    result = estimate(
        [[2.0, 100.0]], [[0.5, 0.0]], [[30.0, 0.0]], [[False, True]], [[False, True]], discount=0.0
    )
    np.testing.assert_array_equal(result.value_targets, [[2.0, 100.0]])


def test_owned_outputs_are_readonly_and_inputs_unchanged():
    reward = np.array([[1.0, 2.0]])
    value = np.zeros_like(reward)
    ends = np.array([[False, True]])
    result = generalized_advantage_targets(
        reward, value, value, ends, ends, discount=1.0, trace_decay=1.0
    )
    np.testing.assert_array_equal(reward, [[1.0, 2.0]])
    np.testing.assert_array_equal(value, [[0.0, 0.0]])
    reward[:] = 99.0
    for array in (result.advantages, result.value_targets, result.td_residuals):
        assert array.dtype == np.float64 and not array.flags.writeable
        assert not np.shares_memory(array, reward)
    np.testing.assert_array_equal(result.value_targets, [[3.0, 2.0]])


@pytest.mark.parametrize("key", ["discount", "trace_decay"])
@pytest.mark.parametrize("invalid", [True, -0.1, 1.1, np.nan, np.inf, "0.9"])
def test_invalid_hyperparameters_rejected(key, invalid):
    with pytest.raises(ValueError):
        estimate([[1.0]], [[0.0]], [[0.0]], [[True]], [[True]], **{key: invalid})


@pytest.mark.parametrize("field", range(3))
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_numeric_rows_rejected_even_unused_terminal_bootstrap(field, invalid):
    args = [np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))]
    args[field][0, 0] = invalid
    with pytest.raises(ValueError):
        generalized_advantage_targets(
            *args,
            np.ones((1, 1), dtype=bool),
            np.ones((1, 1), dtype=bool),
            discount=1.0,
            trace_decay=1.0,
        )


def test_termination_requires_an_episode_boundary():
    with pytest.raises(ValueError):
        estimate([[1.0]], [[0.0]], [[0.0]], [[False]], [[True]])


def test_integer_boundary_masks_are_not_silently_converted():
    with pytest.raises(ValueError):
        generalized_advantage_targets(
            [[1.0]], [[0.0]], [[0.0]], [[1]], [[1]], discount=1.0, trace_decay=1.0
        )


@pytest.mark.parametrize("shape", [(0, 1), (1, 0), (1,), (1, 1, 1), (1, 4097)])
def test_bad_or_unbounded_shapes_rejected(shape):
    x = np.zeros(shape)
    with pytest.raises(ValueError):
        generalized_advantage_targets(
            x,
            x,
            x,
            np.zeros(shape, dtype=bool),
            np.zeros(shape, dtype=bool),
            discount=1.0,
            trace_decay=1.0,
        )


def test_mismatched_value_shapes_rejected():
    with pytest.raises(ValueError):
        estimate([[1.0, 2.0]], [[0.0]], [[0.0, 0.0]], [[False, True]], [[False, True]])


def test_overflow_rejected_instead_of_returning_invalid_training_labels():
    with pytest.raises(ValueError):
        estimate([[1e308, 1e308]], [[0.0, 0.0]], [[0.0, 0.0]], [[False, True]], [[False, True]])
