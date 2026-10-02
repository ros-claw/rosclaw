import copy
from dataclasses import replace

import numpy as np
import pytest

from rosclaw.growth import bounded_advantage_regression as regression
from rosclaw.growth.correlated_residual_gradient import conditional_means


def batch():
    n = 64
    action = np.repeat(np.tile([0.012, -0.012], 4), 8)[:, None]
    first = np.arange(n) % 8 == 0
    marginal = np.full(n, 0.1)
    base = np.zeros((n, 1))
    std = marginal * np.where(first, 1, np.sqrt(1 - 0.9**2))
    mu = conditional_means(base, action, first, 0.9)
    logp = (
        -0.5 * ((action - mu) / std[:, None]) ** 2 - np.log(std[:, None]) - 0.5 * np.log(2 * np.pi)
    ).sum(1)
    gate = np.ones(n)
    gate[0] = 0
    return {
        "layers": [(np.full((4, 3), 0.1), np.zeros(4)), (np.zeros((1, 4)), np.zeros(1))],
        "context": np.ones((n, 3)),
        "baseline": base,
        "gates": gate,
        "actions": action,
        "marginal_std": marginal,
        "first": first,
        "advantages": np.sign(action[:, 0]),
        "old_log_probability": logp,
        "config": regression.AdvantageRegressionConfig(steps=40),
    }


def test_real_numeric_regression_prefers_advantage_actions_with_guard_and_kl():
    pytest.importorskip("torch")
    inputs = batch()
    original = copy.deepcopy(inputs)
    learned = regression.fit_advantage_residual(**inputs)
    hidden = inputs["context"]
    for layer in learned["layers"]:
        hidden = np.tanh(hidden @ np.array(layer["weight"]).T + np.array(layer["bias"]))
    mean = inputs["baseline"] + 0.2 * inputs["gates"][:, None] * hidden
    assert mean[0, 0] == 0 and mean[1, 0] > 0
    assert learned["algorithm"] == "BOUNDED_ADVANTAGE_WEIGHTED_RESIDUAL_REGRESSION_V1"
    assert 0 < learned["completed_optimizer_steps"] <= 40
    history = learned["full_batch_loss_history"]
    assert all(a > b for a, b in zip(history, history[1:], strict=False))
    assert 0 <= learned["exact_mean_conditional_kl"] <= 0.005
    assert 0 <= learned["exact_mean_marginal_kl"] <= 0.005
    for key in (
        "physical_batch_verified",
        "distributional_retention_guaranteed",
        "promotion_authorized",
        "hardware_authorized",
    ):
        assert learned[key] is False
    for key in ("context", "baseline", "gates", "actions", "old_log_probability"):
        np.testing.assert_array_equal(inputs[key], original[key])
    for (a, b), (old_a, old_b) in zip(inputs["layers"], original["layers"], strict=True):
        np.testing.assert_array_equal(a, old_a)
        np.testing.assert_array_equal(b, old_b)


def test_advantage_weights_preserve_order_and_equal_advantages():
    values = regression.advantage_weights(
        [-100, 0, 1, 1, 100], regression.AdvantageRegressionConfig()
    )
    assert np.isfinite(values).all() and np.all(values > 0)
    assert values[0] < values[1] < values[2] == values[3] < values[4]
    assert values.mean() == pytest.approx(1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", float("nan")),
        ("temperature", 0),
        ("maximum_weight", True),
        ("maximum_weight", 21),
        ("steps", 161),
        ("residual_cap", 0.3),
    ],
)
def test_unsafe_unbounded_configs_rejected(field, value):
    with pytest.raises(ValueError):
        replace(regression.AdvantageRegressionConfig(), **{field: value}).validate()


@pytest.mark.parametrize("fault", ["density", "nan", "gate", "reset"])
def test_invalid_behavior_or_numeric_batch_rejected(fault):
    inputs = batch()
    if fault == "density":
        inputs["old_log_probability"][0] += 0.01
    elif fault == "nan":
        inputs["context"][0, 0] = float("nan")
    elif fault == "gate":
        inputs["gates"][0] = -1
    else:
        inputs["first"][0] = False
    with pytest.raises(ValueError):
        regression.fit_advantage_residual(**inputs)


def test_torch_host_settings_and_rng_restored_even_on_failure(monkeypatch):
    torch = pytest.importorskip("torch")
    state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    def fail(*args, **kwargs):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(regression, "_fit", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        regression.fit_advantage_residual(**batch())
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
