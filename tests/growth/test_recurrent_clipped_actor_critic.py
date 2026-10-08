"""Numeric PPO/MC proofs, not rollout provenance or robot authorization."""

import copy

import numpy as np
import pytest

from rosclaw.growth.causal_residual_memory import CausalResidualMemory, initial_parameters
from rosclaw.growth.correlated_exploration import stationary_noise
from rosclaw.growth.recurrent_clipped_actor_critic import (
    RecurrentClippedActorCriticConfig,
    fit_recurrent_clipped_actor_critic,
    initial_value_parameters,
)


def dataset(*, std=0.1, rho=0.9):
    rng = np.random.default_rng(23)
    n, t, d, k = 8, 6, 5, 2
    action = np.stack(
        [std * stationary_noise(seed=41 + i, rho=rho, count=t, dimension=k) for i in range(n)]
    )
    conditional = np.zeros_like(action)
    conditional[:, 1:] = rho * action[:, :-1]
    scales = np.full((n, t), std * np.sqrt(1 - rho * rho))
    scales[:, 0] = std
    logp = np.sum(
        -0.5 * ((action - conditional) / scales[:, :, None]) ** 2
        - np.log(scales[:, :, None])
        - 0.5 * np.log(2 * np.pi),
        axis=2,
    )
    advantages = np.sign(action[:, :, 0])
    return {
        "actor_parameters": initial_parameters(d, k, hidden_dimension=8, seed=17),
        "critic_parameters": initial_value_parameters(d, seed=19),
        "context": rng.normal(size=(n, t, d)),
        "baseline": np.zeros_like(action),
        "gates": np.ones((n, t)),
        "latent_actions": action,
        "behavior_log_probabilities": logp,
        "advantages": advantages,
        "returns": 3 + 0.4 * advantages,
    }


def config(**options):
    return RecurrentClippedActorCriticConfig(**dict(steps=4, seed=29, **options))


def fit(data, **options):
    pytest.importorskip("torch")
    return fit_recurrent_clipped_actor_critic(**data, config=config(**options))


def replay_means(parameters, data, cap):
    result = np.empty_like(data["baseline"])
    for episode in range(len(result)):
        memory = CausalResidualMemory(parameters)
        for frame in range(result.shape[1]):
            result[episode, frame] = data["baseline"][episode, frame] + (
                cap
                * data["gates"][episode, frame]
                * memory.step(data["context"][episode, frame], index=frame)
            )
    return result


def test_actual_actor_and_mc_critic_updates_with_exact_full_sequence_kl():
    data = dataset()
    original = copy.deepcopy(data)
    result = fit(data)
    assert result["accepted_joint_optimizer_steps"] == 4
    assert result["rejected_and_rolled_back_updates"] == 0
    assert result["actor_parameters_changed"]
    assert result["original_critic_parameters_hash"] != result["fitted_critic_parameters_hash"]
    assert result["final_mc_value_mse"] < np.mean(data["returns"] ** 2)
    assert result["positive_advantage_rows"] > 0 and result["negative_advantage_rows"] > 0
    assert result["all_rows_behavior_density_validated"] and result["all_rows_in_kl_checks"]
    for key in (
        "promotion_authorized",
        "runtime_execution_authorized",
        "hardware_authorized",
        "physical_batch_verified",
        "on_policy_collection_provenance_verified",
        "td_bootstrapping",
        "frozen_advantages_are_actor_inputs",
        "returns_are_actor_inputs",
    ):
        assert result[key] is False
    old = replay_means(original["actor_parameters"], data, 0.2)
    new = replay_means(result["actor_parameters"], data, 0.2)
    assert result["full_batch_marginal_mean_kl"] == pytest.approx(
        np.mean(np.sum((new - old) ** 2 / (2 * 0.1**2), axis=2)), abs=1e-12
    )
    old_conditional, new_conditional = old.copy(), new.copy()
    old_conditional[:, 1:] += 0.9 * (data["latent_actions"][:, :-1] - old[:, :-1])
    new_conditional[:, 1:] += 0.9 * (data["latent_actions"][:, :-1] - new[:, :-1])
    scales = np.full(data["gates"].shape, 0.1 * np.sqrt(1 - 0.9**2))
    scales[:, 0] = 0.1
    assert result["full_batch_conditional_mean_kl"] == pytest.approx(
        np.mean(
            np.sum((new_conditional - old_conditional) ** 2 / (2 * scales[:, :, None] ** 2), axis=2)
        ),
        abs=1e-12,
    )
    assert result["full_batch_marginal_mean_kl"] <= 0.02
    assert result["full_batch_conditional_mean_kl"] <= 0.02
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(data[key], value)
        else:
            assert data[key] == value
    result["actor_parameters"]["head_bias"][0] = 99
    assert data["actor_parameters"]["head_bias"][0] == 0


def test_failure_advantages_change_gradient_and_protected_zero_gates_do_not():
    data = dataset()
    positive = fit(data)
    negative = copy.deepcopy(data)
    negative["advantages"] *= -1
    opposite = fit(negative)
    assert positive["fitted_actor_parameters_hash"] != opposite["fitted_actor_parameters_hash"]
    protected = copy.deepcopy(data)
    protected["gates"][:] = 0
    result = fit(protected)
    assert not result["actor_parameters_changed"]
    assert result["original_actor_parameters_hash"] == result["fitted_actor_parameters_hash"]
    assert result["original_critic_parameters_hash"] != result["fitted_critic_parameters_hash"]


def test_joint_kl_rejection_rolls_back_actor_and_critic():
    data = dataset(std=0.001)
    result = fit(data, std=0.001, learning_rate=0.01, residual_cap=1.0, maximum_mean_kl=0.0001)
    assert result["accepted_joint_optimizer_steps"] == 0
    assert result["rejected_and_rolled_back_updates"] == 1
    assert result["original_actor_parameters_hash"] == result["fitted_actor_parameters_hash"]
    assert result["original_critic_parameters_hash"] == result["fitted_critic_parameters_hash"]
    assert result["full_batch_conditional_mean_kl"] == pytest.approx(0, abs=1e-15)
    assert result["accepted_update_history"] == []


@pytest.mark.parametrize("field", ["context", "advantages", "critic_parameters"])
def test_signed_integer_minimum_cannot_bypass_numeric_magnitude_bound(field):
    data = dataset()
    if field == "critic_parameters":
        data[field]["bias_1"] = [np.iinfo(np.int64).min]
    else:
        data[field] = np.full(data[field].shape, np.iinfo(np.int64).min, dtype=np.int64)
    with pytest.raises(ValueError, match="finite.*bounded|finite aligned"):
        fit_recurrent_clipped_actor_critic(**data, config=config())


def test_rng_threads_determinism_and_repeatability_restored():
    torch = pytest.importorskip("torch")
    torch_rng = torch.random.get_rng_state().clone()
    np_rng = np.random.get_state()
    threads = torch.get_num_threads()
    deterministic, warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    first, second = fit(dataset()), fit(dataset())
    assert first == second
    assert torch.equal(torch_rng, torch.random.get_rng_state())
    after = np.random.get_state()
    assert np_rng[0] == after[0] and np_rng[2:] == after[2:]
    np.testing.assert_array_equal(np_rng[1], after[1])
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn


@pytest.mark.parametrize(
    "field",
    [
        "context",
        "baseline",
        "gates",
        "latent_actions",
        "behavior_log_probabilities",
        "advantages",
        "returns",
    ],
)
def test_nonfinite_rows_rejected_including_zero_advantage_rows(field):
    data = dataset()
    data["advantages"][0, 0] = 0
    data[field].flat[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_recurrent_clipped_actor_critic(**data, config=config())


def test_incorrect_conditional_log_probability_never_fits():
    data = dataset()
    data["advantages"][0, 1] = 0
    data["behavior_log_probabilities"][0, 1] += 0.01
    with pytest.raises(ValueError, match="density"):
        fit_recurrent_clipped_actor_critic(**data, config=config())
    # Using marginal noise for every AR frame is a different behavior law.
    data = dataset()
    data["behavior_log_probabilities"] = np.sum(
        -0.5 * (data["latent_actions"] / 0.1) ** 2 - np.log(0.1) - 0.5 * np.log(2 * np.pi), axis=2
    )
    with pytest.raises(ValueError, match="density"):
        fit_recurrent_clipped_actor_critic(**data, config=config())


@pytest.mark.parametrize(
    "change",
    [
        {"steps": True},
        {"batch_episodes": 0},
        {"seed": -1},
        {"std": 0.0},
        {"rho": 1.0},
        {"clip_epsilon": True},
        {"maximum_mean_kl": 0.0},
        {"compute_device": "cuda"},
        {"compute_device": "REAL"},
        {"value_loss_weight": float("inf")},
    ],
)
def test_bad_configuration_rejected_before_optional_torch(change):
    with pytest.raises(ValueError, match="configuration"):
        fit_recurrent_clipped_actor_critic(
            **dataset(), config=RecurrentClippedActorCriticConfig(**change)
        )


def test_value_parameters_and_aligned_shapes_fail_closed():
    data = dataset()
    data["critic_parameters"]["weight_0"][0] = [1]
    with pytest.raises(ValueError):
        fit_recurrent_clipped_actor_critic(**data, config=config())
    data = dataset()
    data["latent_actions"] = data["latent_actions"][:, :-1]
    with pytest.raises(ValueError, match="aligned"):
        fit_recurrent_clipped_actor_critic(**data, config=config())
