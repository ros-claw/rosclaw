import numpy as np
import pytest

from rosclaw.growth.grouped_terminal_credit import grouped_terminal_credit


def batch():
    returns = np.repeat(np.array([1.0, 2.0, 4.0, 8.0, -10.0, -5.0, -2.0, 0.0]), 3)
    groups = np.repeat(np.arange(8), 3)
    contexts = np.repeat([11, 22], 4)
    return returns, groups, contexts


def test_other_episode_baseline_and_whole_episode_rows():
    reward, groups, contexts = batch()
    result = grouped_terminal_credit(reward, groups, trajectory_context_ids=contexts)
    expected = np.array([14 / 3, 13 / 3, 11 / 3, 7 / 3, -7 / 3, -4, -5, -17 / 3])
    assert np.array_equal(result["episode_baselines"], expected)
    assert result["trials_per_context"] == {11: 4, 22: 4}
    assert result["declared_episode_count"] == 8
    assert result["declared_context_count"] == 2
    assert np.array_equal(result["advantages"], np.repeat(result["episode_advantages"], 3))
    assert result["baseline_contains_target_episode"] is False
    assert result["all_declared_episodes_retained"] is True
    assert result["physical_batch_verified"] is False
    assert result["actor_updated"] is False
    assert result["promotion_authorized"] is False
    assert result["hardware_authorized"] is False


def test_target_episode_does_not_influence_its_own_baseline():
    reward, groups, contexts = batch()
    first = grouped_terminal_credit(reward, groups, trajectory_context_ids=contexts)
    reward[groups == 0] = 1e6
    second = grouped_terminal_credit(reward, groups, trajectory_context_ids=contexts)
    assert second["episode_baselines"][0] == first["episode_baselines"][0]
    assert second["episode_baselines"][1] != first["episode_baselines"][1]


def test_context_offsets_cancel_and_all_failures_are_retained():
    reward, groups, contexts = batch()
    before = grouped_terminal_credit(reward, groups, trajectory_context_ids=contexts)
    shifted = reward + np.where(contexts[groups] == 11, 100.0, -200.0)
    after = grouped_terminal_credit(shifted, groups, trajectory_context_ids=contexts)
    np.testing.assert_allclose(after["advantages"], before["advantages"], atol=1e-14, rtol=0)
    # Positive relative credit within an all-negative context is not success.
    assert np.all(after["episode_returns"][4:] < 0)
    assert after["episode_advantages"][-1] > 0
    assert after["all_declared_episodes_retained"] is True


def test_constant_returns_have_zero_finite_credit_without_learning_claim():
    reward, groups, contexts = batch()
    result = grouped_terminal_credit(np.ones_like(reward), groups, trajectory_context_ids=contexts)
    assert np.array_equal(result["advantages"], np.zeros_like(reward))
    assert result["actual_critic_fits"] == 0
    assert result["online_ppo_claimed"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "nan",
        "large",
        "different_row_return",
        "float_groups",
        "bool_contexts",
        "missing_group",
        "singleton",
        "shape",
    ],
)
def test_incomplete_or_malformed_terminal_batch_rejected(fault):
    reward, groups, contexts = batch()
    if fault == "nan":
        reward[0] = np.nan
    elif fault == "large":
        reward[:] = 1e7
    elif fault == "different_row_return":
        reward[0] += 0.1
    elif fault == "float_groups":
        groups = groups.astype(float)
    elif fault == "bool_contexts":
        contexts = contexts.astype(bool)
    elif fault == "missing_group":
        groups[groups == 0] = 1
    elif fault == "singleton":
        contexts[0] = 33
    elif fault == "shape":
        reward = reward[:, None]
    with pytest.raises(ValueError):
        grouped_terminal_credit(reward, groups, trajectory_context_ids=contexts)
