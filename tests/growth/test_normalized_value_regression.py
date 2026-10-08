"""Offline value calibration is neither actor learning nor physical success."""

import copy

import numpy as np
import pytest

from rosclaw.growth.normalized_value_regression import (
    NormalizedValueRegressionConfig,
    fit_normalized_value_regression,
)
from rosclaw.growth.recurrent_clipped_actor_critic import initial_value_parameters


def data():
    context = np.random.default_rng(13).normal(size=(8, 12, 3))
    parameters = initial_value_parameters(3, seed=17)
    hidden = np.tanh(context @ np.asarray(parameters["weight_0"]).T)
    return {
        "context": context,
        "value_targets": 2 + 3 * hidden[:, :, 0],
        "critic_parameters": parameters,
    }


def fit(values, **kwargs):
    pytest.importorskip("torch")
    return fit_normalized_value_regression(
        **values, config=NormalizedValueRegressionConfig(**dict(steps=8, **kwargs))
    )


def test_actual_critic_learning_preserves_units_without_actor_or_authority():
    pytest.importorskip("torch")
    values = data()
    old = copy.deepcopy(values)
    result = fit_normalized_value_regression(
        **values, config=NormalizedValueRegressionConfig(steps=256, learning_rate=0.01, seed=29)
    )
    p = result["critic_parameters"]
    hidden = np.tanh(values["context"] @ np.asarray(p["weight_0"]).T + np.asarray(p["bias_0"]))
    prediction = (hidden @ np.asarray(p["weight_1"]).T + np.asarray(p["bias_1"])).squeeze(-1)
    assert result["final_value_mse"] == pytest.approx(
        np.mean((prediction - values["value_targets"]) ** 2)
    )
    assert result["final_value_mse"] < result["training_mean_baseline_mse"] * 0.1
    assert result["original_critic_parameters_hash"] != result["fitted_critic_parameters_hash"]
    assert result["accepted_value_optimizer_steps"] == 256
    assert len(result["normalized_training_loss_history"]) == 256
    assert result["maximum_initial_output_preservation_error"] < 1e-12
    assert result["target_mean"] == pytest.approx(values["value_targets"].mean())
    assert result["target_scale"] == pytest.approx(values["value_targets"].std())
    assert result["frame_rows"] == 96
    assert result["actor_optimizer_updates"] == 0
    assert "actor_parameters" not in result
    for key in (
        "adaptive_online_popart",
        "physical_batch_verified",
        "held_out_calibration_verified",
        "physical_gain_verified",
        "promotion_authorized",
        "runtime_execution_authorized",
        "hardware_authorized",
    ):
        assert result[key] is False
    for key in ("context", "value_targets"):
        np.testing.assert_array_equal(values[key], old[key])
    assert values["critic_parameters"] == old["critic_parameters"]


def test_initial_nonzero_predictions_preserved_and_fits_repeat_exactly():
    values = data()
    values["critic_parameters"]["weight_1"] = [[0.1] * 64]
    values["critic_parameters"]["bias_1"] = [7.0]
    a, b = fit(values, seed=11), fit(values, seed=11)
    assert a == b
    assert a["maximum_initial_output_preservation_error"] < 1e-12


@pytest.mark.parametrize("constant", [0.0, 7.0, -100.0])
def test_constant_targets_use_finite_floor_without_clipping(constant):
    values = data()
    values["value_targets"][:] = constant
    result = fit(values)
    assert result["target_mean"] == constant
    assert result["target_scale"] == 0.0001
    assert result["training_mean_baseline_mse"] == 0
    assert np.isfinite(result["final_value_mse"])
    assert result["initial_value_mse"] == constant**2


@pytest.mark.parametrize("field", ["context", "value_targets"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 1_000_001.0])
def test_reject_invalid_numeric_rows_before_torch(field, invalid):
    values = data()
    values[field].flat[0] = invalid
    with pytest.raises(ValueError, match="complete bounded finite"):
        fit_normalized_value_regression(**values, config=NormalizedValueRegressionConfig())


@pytest.mark.parametrize(
    "field,value",
    [
        ("context", np.zeros((2, 3))),
        ("context", np.zeros((0, 12, 3))),
        ("context", np.zeros((8, 0, 3))),
        ("context", np.zeros((8, 12, 0))),
        ("context", np.ones((8, 12, 3), dtype=bool)),
        ("context", np.zeros((8, 12, 3), dtype=complex)),
        ("value_targets", np.zeros((8, 11))),
        ("value_targets", np.ones((8, 12), dtype=bool)),
        ("value_targets", np.full((8, 12), "1", dtype=object)),
    ],
)
def test_reject_wrong_shapes_and_non_numeric_types(field, value):
    values = data()
    values[field] = value
    with pytest.raises(ValueError, match="complete bounded finite"):
        fit_normalized_value_regression(**values, config=NormalizedValueRegressionConfig())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": True},
        {"steps": 0},
        {"steps": 8193},
        {"batch_episodes": 65},
        {"seed": -1},
        {"seed": 2**32},
        {"learning_rate": 1},
        {"learning_rate": np.nan},
        {"minimum_target_scale": 0.0},
        {"maximum_gradient_norm": np.inf},
        {"compute_device": "cuda"},
        {"compute_device": "cuda:-1"},
        {"compute_device": None},
    ],
)
def test_config_rejects_invalid_or_ambiguous_budget(kwargs):
    with pytest.raises(ValueError, match="configuration"):
        NormalizedValueRegressionConfig(**kwargs).validate()


def test_explicit_config_and_complete_critic_required():
    values = data()
    with pytest.raises(ValueError, match="configuration"):
        fit_normalized_value_regression(**values, config=None)
    values["critic_parameters"].pop("bias_1")
    with pytest.raises(ValueError, match="complete owned"):
        fit_normalized_value_regression(**values, config=NormalizedValueRegressionConfig())


def test_seed_and_numeric_hashes_bind_actual_input():
    values = data()
    a = fit(values)
    values["value_targets"][0, 0] += 1
    b = fit(values)
    assert a["input_numeric_hash"] != b["input_numeric_hash"]
    assert a["fitted_critic_parameters_hash"] != b["fitted_critic_parameters_hash"]


def test_torch_process_settings_and_rng_restored_on_success_and_failure(monkeypatch):
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng = torch.get_rng_state().clone()
    fit(data())
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    assert torch.equal(rng, torch.get_rng_state())

    def poisoned_step(optimizer):
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("nan"))

    monkeypatch.setattr(torch.optim.Adam, "step", poisoned_step)
    with pytest.raises(ValueError, match="finite fitted"):
        fit(data())
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    assert torch.equal(rng, torch.get_rng_state())


def test_unavailable_cuda_is_not_silently_fitted_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="explicit available"):
        fit(data(), compute_device="cuda:0")
