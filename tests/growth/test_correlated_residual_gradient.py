import copy

import numpy as np
import pytest

from rosclaw.growth.correlated_residual_gradient import (
    ResidualGradientConfig,
    conditional_means,
    fit_correlated_residual,
    terminal_crossfit_advantages,
)


def batch():
    rng = np.random.default_rng(385)
    n, d, m = 48, 5, 3
    x = rng.normal(size=(n, d))
    first = np.arange(n) % 12 == 0
    base = rng.normal(scale=0.03, size=(n, m))
    std = np.full(n, 0.1)
    noise = rng.normal(size=(n, m))
    for i in range(1, n):
        if not first[i]:
            noise[i] = 0.9 * noise[i - 1] + np.sqrt(1 - 0.9**2) * noise[i]
    action = base + std[:, None] * noise
    conditional = conditional_means(base, action, first, 0.9)
    scale = std * np.where(first, 1, np.sqrt(1 - 0.9**2))
    logp = np.sum(
        -0.5 * ((action - conditional) / scale[:, None]) ** 2
        - np.log(scale[:, None])
        - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    gates = np.ones(n)
    gates[:2] = 0
    return {
        "layers": [
            (rng.normal(scale=0.1, size=(7, d)), np.zeros(7)),
            (np.zeros((m, 7)), np.zeros(m)),
        ],
        "context": x,
        "baseline": base,
        "gates": gates,
        "actions": action,
        "marginal_std": std,
        "first": first,
        "advantages": noise[:, 0],
        "old_log_probability": logp,
    }


def test_reset_uses_current_mean_not_previous_episode():
    mu = np.array([[1.0], [2.0], [3.0], [4.0]])
    action = np.array([[10.0], [20.0], [30.0], [40.0]])
    assert np.array_equal(
        conditional_means(mu, action, np.array([True, False, True, False]), 0.9),
        np.array([[1.0], [10.1], [3.0], [28.3]]),
    )


def test_learning_is_deterministic_bounded_and_does_not_mutate_host_or_batch():
    torch = pytest.importorskip("torch")
    data = batch()
    original = copy.deepcopy(data)
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    result = fit_correlated_residual(**data)
    again = fit_correlated_residual(**data)
    assert result == again
    assert torch.equal(rng, torch.get_rng_state())
    assert threads == torch.get_num_threads()
    assert deterministic == torch.are_deterministic_algorithms_enabled()
    for k in data:
        if k == "layers":
            for a, b in zip(data[k], original[k], strict=True):
                assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
        else:
            assert np.array_equal(data[k], original[k])
    assert 1 <= result["completed_optimizer_steps"] <= 160
    assert all(
        a > b
        for a, b in zip(
            result["full_batch_loss_history"],
            result["full_batch_loss_history"][1:],
            strict=False,
        )
    )
    assert 0 < result["exact_mean_conditional_kl"] <= 0.005
    assert 0 < result["exact_mean_marginal_kl"] <= 0.005
    hidden = data["context"]
    for layer in result["layers"]:
        hidden = np.tanh(hidden @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"]))
    final = data["baseline"] + 0.05 * data["gates"][:, None] * hidden
    assert np.array_equal(final[:2], data["baseline"][:2])
    assert np.max(np.abs(final - data["baseline"])) <= 0.05
    assert result["physical_batch_verified"] is False
    assert result["distributional_retention_guaranteed"] is False
    assert result["promotion_authorized"] is False
    assert result["hardware_authorized"] is False


def test_wrong_iid_density_rejected_before_training():
    data = batch()
    with pytest.raises(ValueError, match="history-conditioned"):
        fit_correlated_residual(**data, config=ResidualGradientConfig(rho=0))


@pytest.mark.parametrize(
    "field,value",
    [
        ("residual_cap", 0.21),
        ("learning_rate", 0.001),
        ("rho", 0.96),
        ("steps", 161),
        ("steps", True),
        ("seed", -1),
        ("rho", float("nan")),
    ],
)
def test_no_unbounded_optimizer_configuration(field, value):
    with pytest.raises(ValueError, match="bounded"):
        fit_correlated_residual(**batch(), config=ResidualGradientConfig(**{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("first", np.zeros(48, dtype=bool)),
        ("first", np.ones(48)),
        ("gates", np.full(48, 1.1)),
        ("actions", np.full((48, 3), np.nan)),
        ("marginal_std", np.full(48, 0.01)),
        ("context", np.array(1)),
    ],
)
def test_invalid_batch_fails_closed(field, value):
    data = batch()
    data[field] = value
    with pytest.raises(ValueError):
        fit_correlated_residual(**data)


def test_expanded_cap_is_explicit_not_silent_default_drift():
    pytest.importorskip("torch")
    data = batch()
    result = fit_correlated_residual(
        **data,
        config=ResidualGradientConfig(residual_cap=0.2, learning_rate=4e-4),
    )
    assert result["residual_cap"] == 0.2
    assert result["learning_rate"] == 4e-4
    assert result["exact_mean_conditional_kl"] <= 0.005
    assert result["exact_mean_marginal_kl"] <= 0.005


def test_terminal_critic_crossfit_uses_whole_trajectories():
    rng = np.random.default_rng(385)
    phi = rng.normal(size=(8 * 90, 9))
    group = np.repeat(np.arange(8), 90)
    phase = np.tile(np.repeat(np.arange(3), 30), 8)
    reward = np.repeat(np.arange(8.0), 90)
    result = terminal_crossfit_advantages(phi, phase, group, reward)
    changed = reward.copy()
    changed[group == 0] += 10
    other = terminal_crossfit_advantages(phi, phase, group, changed)
    assert result["crossfit_unit"] == "whole_rollout"
    assert result["critic_readout"].shape == (3, 9)
    assert np.isfinite(other["advantages"]).all()
    broken = reward.copy()
    broken[1] += 1
    with pytest.raises(ValueError, match="one actual terminal return"):
        terminal_crossfit_advantages(phi, phase, group, broken)
    with pytest.raises(ValueError, match="whole-trajectory"):
        terminal_crossfit_advantages(phi, phase, group[::-1], reward)


def test_exception_restores_torch_state():
    torch = pytest.importorskip("torch")
    data = batch()
    data["advantages"][:] = 0
    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    with pytest.raises(ValueError, match="no actual"):
        fit_correlated_residual(**data)
    assert torch.equal(state, torch.get_rng_state())
    assert threads == torch.get_num_threads()
