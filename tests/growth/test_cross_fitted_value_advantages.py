"""Numeric exclusions/credit do not certify optimizer execution or physics."""

import copy

import numpy as np
import pytest

from rosclaw.growth.cross_fitted_value_advantages import cross_fitted_value_advantages
from rosclaw.growth.normalized_value_regression import (
    NormalizedValueRegressionConfig,
    fit_normalized_value_regression,
)
from rosclaw.growth.recurrent_clipped_actor_critic import initial_value_parameters


@pytest.fixture(scope="module")
def fitted_data():
    pytest.importorskip("torch")
    x = np.random.default_rng(21).normal(size=(8, 6, 2))
    y = np.arange(8)[:, None] * 3.0 + np.arange(6)[None, :] * 0.5
    y[0] -= 100  # A bad complete episode must not disappear from credit.
    ids = np.repeat(np.arange(4), 2)
    parameters = initial_value_parameters(2, seed=19)
    fits = []
    for fold in range(2):
        train = ids % 2 != fold
        receipt = fit_normalized_value_regression(
            context=x[train],
            value_targets=y[train],
            critic_parameters=parameters,
            config=NormalizedValueRegressionConfig(steps=4, seed=fold),
        )
        fits.append(
            {
                "fold_index": fold,
                "train_context_ids": np.unique(ids[train]).tolist(),
                "held_out_context_ids": np.unique(ids[~train]).tolist(),
                "value_fit": receipt,
            }
        )
    return {
        "context": x,
        "value_targets": y,
        "episode_context_ids": ids,
        "initial_critic_parameters": parameters,
        "fold_fits": fits,
    }


def test_all_temporal_rows_repeated_contexts_and_original_targets_preserved(fitted_data):
    data = copy.deepcopy(fitted_data)
    result = cross_fitted_value_advantages(**data)
    predictions = np.empty_like(data["value_targets"])
    for fold, fit in enumerate(data["fold_fits"]):
        p = {k: np.asarray(v) for k, v in fit["value_fit"]["critic_parameters"].items()}
        mask = data["episode_context_ids"] % 2 == fold
        hidden = np.tanh(data["context"][mask] @ p["weight_0"].T + p["bias_0"])
        predictions[mask] = (hidden @ p["weight_1"].T + p["bias_1"]).squeeze(-1)
        binding = result["fold_bindings"][fold]
        assert set(binding["train_context_ids"]).isdisjoint(binding["held_out_context_ids"])
        assert binding["predicted_episode_indices"] == np.flatnonzero(mask).tolist()
    raw = data["value_targets"] - predictions
    np.testing.assert_array_equal(result["predictions"], predictions)
    np.testing.assert_array_equal(result["raw_advantages"], raw)
    np.testing.assert_array_equal(result["advantages"], (raw - raw.mean()) / raw.std())
    assert result["held_out_mse"] == pytest.approx(np.mean(raw**2))
    assert result["advantages"].shape == (8, 6)
    assert result["raw_advantages"][0].mean() < -90
    for key in ("context", "value_targets", "episode_context_ids"):
        np.testing.assert_array_equal(data[key], fitted_data[key])
    assert data["fold_fits"] == fitted_data["fold_fits"]
    assert data["initial_critic_parameters"] == fitted_data["initial_critic_parameters"]
    for key in ("predictions", "raw_advantages", "advantages", "episode_fold_ids"):
        assert not result[key].flags.writeable
    assert result["critic_fits_performed_here"] == result["actor_optimizer_updates"] == 0
    for key in (
        "context_ids_are_actor_observations",
        "optimizer_execution_independently_verified",
        "physical_batch_verified",
        "private_fresh_verified",
        "physical_gain_verified",
        "runtime_execution_authorized",
        "promotion_authorized",
        "hardware_authorized",
    ):
        assert result[key] is False


@pytest.mark.parametrize("key", ["train_context_ids", "held_out_context_ids"])
def test_leaked_or_reordered_context_declarations_rejected(fitted_data, key):
    data = copy.deepcopy(fitted_data)
    data["fold_fits"][0][key] = list(reversed(data["fold_fits"][0][key]))
    with pytest.raises(ValueError, match="excluded fit declarations"):
        cross_fitted_value_advantages(**data)


def test_shuffled_fit_order_is_not_implicitly_repaired(fitted_data):
    data = copy.deepcopy(fitted_data)
    data["fold_fits"].reverse()
    with pytest.raises(ValueError, match="excluded fit declarations"):
        cross_fitted_value_advantages(**data)


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_hash", "changed"),
        ("parameter_contract_source_hash", "changed"),
        ("input_numeric_hash", "changed"),
        ("original_critic_parameters_hash", "changed"),
        ("episode_count", True),
        ("episode_horizon", 5),
        ("frame_rows", 23),
        ("accepted_value_optimizer_steps", 3),
        ("actor_optimizer_updates", False),
        ("target_mean", 0.0),
        ("target_scale", 1.0),
        ("adaptive_online_popart", True),
        ("physical_batch_verified", True),
        ("held_out_calibration_verified", True),
        ("physical_gain_verified", True),
        ("promotion_authorized", True),
        ("runtime_execution_authorized", True),
        ("hardware_authorized", True),
    ],
)
def test_false_numeric_source_statistics_counts_or_authority_rejected(fitted_data, key, value):
    data = copy.deepcopy(fitted_data)
    data["fold_fits"][0]["value_fit"][key] = value
    with pytest.raises(ValueError, match="excluded-context training subset"):
        cross_fitted_value_advantages(**data)


@pytest.mark.parametrize(
    "key", ["initial_value_mse", "final_value_mse", "training_mean_baseline_mse"]
)
def test_false_training_metrics_rejected(fitted_data, key):
    data = copy.deepcopy(fitted_data)
    data["fold_fits"][0]["value_fit"][key] += 1
    with pytest.raises(ValueError, match="training measurements"):
        cross_fitted_value_advantages(**data)


@pytest.mark.parametrize("value", [[], [0.0] * 3, [0.0, 0.0, 0.0, np.nan], [False] * 4])
def test_complete_finite_loss_history_required(fitted_data, value):
    data = copy.deepcopy(fitted_data)
    data["fold_fits"][0]["value_fit"]["normalized_training_loss_history"] = value
    with pytest.raises(ValueError, match="optimizer loss receipt"):
        cross_fitted_value_advantages(**data)


def test_parameter_mutation_does_not_silently_supply_another_baseline(fitted_data):
    data = copy.deepcopy(fitted_data)
    data["fold_fits"][0]["value_fit"]["critic_parameters"]["bias_1"][0] += 1
    with pytest.raises(ValueError, match="fitted value parameters"):
        cross_fitted_value_advantages(**data)


def test_full_data_fit_cannot_bind_a_fold_by_changing_context_claims(fitted_data):
    data = copy.deepcopy(fitted_data)
    data["fold_fits"][0]["value_fit"] = fit_normalized_value_regression(
        context=data["context"],
        value_targets=data["value_targets"],
        critic_parameters=data["initial_critic_parameters"],
        config=NormalizedValueRegressionConfig(steps=4),
    )
    with pytest.raises(ValueError, match="excluded-context training subset"):
        cross_fitted_value_advantages(**data)


@pytest.mark.parametrize("field", ["context", "value_targets"])
@pytest.mark.parametrize("value", [np.nan, np.inf, 1_000_001.0])
def test_invalid_numeric_rows_rejected_before_predicting(fitted_data, field, value):
    data = copy.deepcopy(fitted_data)
    data[field].flat[0] = value
    with pytest.raises(ValueError, match="complete bounded finite"):
        cross_fitted_value_advantages(**data)


@pytest.mark.parametrize(
    "ids",
    [
        np.zeros(8, dtype=bool),
        np.zeros(8),
        np.full(8, -1),
        np.full(8, 2**32, dtype=np.uint64),
        np.arange(7),
    ],
)
def test_invalid_context_split_labels_rejected(fitted_data, ids):
    data = copy.deepcopy(fitted_data)
    data["episode_context_ids"] = ids
    with pytest.raises(ValueError, match="complete bounded finite"):
        cross_fitted_value_advantages(**data)


def test_empty_fold_not_repaired_by_reassigning_contexts(fitted_data):
    data = copy.deepcopy(fitted_data)
    data["episode_context_ids"] = np.repeat(np.arange(4) * 2, 2)
    with pytest.raises(ValueError, match="nonempty held-out"):
        cross_fitted_value_advantages(**data)


def test_numeric_adapter_does_not_import_torch_or_fit_anything(fitted_data, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("numeric credit must not import the optional trainer")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = cross_fitted_value_advantages(**fitted_data)
    assert result["critic_fits_performed_here"] == 0


@pytest.mark.parametrize("config", [None, {}, {"unknown": 1}, {"steps": True}])
def test_incomplete_or_invalid_configuration_rejected(fitted_data, config):
    data = copy.deepcopy(fitted_data)
    data["fold_fits"][0]["value_fit"]["config"] = config
    with pytest.raises(ValueError):
        cross_fitted_value_advantages(**data)


def test_zero_targets_and_zero_predictions_use_finite_advantage_floor(fitted_data):
    data = copy.deepcopy(fitted_data)
    data["value_targets"][:] = 0
    for fold, item in enumerate(data["fold_fits"]):
        train = data["episode_context_ids"] % 2 != fold
        item["value_fit"] = fit_normalized_value_regression(
            context=data["context"][train],
            value_targets=data["value_targets"][train],
            critic_parameters=data["initial_critic_parameters"],
            config=NormalizedValueRegressionConfig(steps=4),
        )
    result = cross_fitted_value_advantages(**data)
    np.testing.assert_array_equal(result["predictions"], np.zeros((8, 6)))
    np.testing.assert_array_equal(result["advantages"], np.zeros((8, 6)))
    assert result["raw_advantage_std"] == result["held_out_mse"] == 0


@pytest.mark.parametrize("frame_budget", [1, 6, 13, 65536])
def test_episode_chunking_keeps_every_credit_and_receipt_bit_exact(
    fitted_data, monkeypatch, frame_budget
):
    import rosclaw.growth.cross_fitted_value_advantages as module

    expected = cross_fitted_value_advantages(**fitted_data)
    monkeypatch.setattr(module, "_VALUE_PREDICTION_FRAME_BATCH", frame_budget)
    actual = cross_fitted_value_advantages(**fitted_data)
    for key in ("predictions", "raw_advantages", "advantages", "episode_fold_ids"):
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in set(actual) - {"predictions", "raw_advantages", "advantages", "episode_fold_ids"}:
        assert actual[key] == expected[key]


def test_large_low_dimension_prediction_bounds_hidden_allocation(monkeypatch):
    import rosclaw.growth.cross_fitted_value_advantages as module

    x = np.random.default_rng(91).normal(size=(17, 4096, 2))
    parameters = {k: np.asarray(v) for k, v in initial_value_parameters(2, seed=71).items()}
    parameters["weight_1"][:] = np.random.default_rng(52).normal(size=(1, 64))
    hidden = np.tanh(x @ parameters["weight_0"].T + parameters["bias_0"])
    expected = (hidden @ parameters["weight_1"].T + parameters["bias_1"]).squeeze(-1)
    del hidden
    original_tanh = np.tanh
    observed_frames = []

    def measured_tanh(value):
        observed_frames.append(value.shape[0] * value.shape[1])
        assert value.shape[2] == 64
        assert observed_frames[-1] <= module._VALUE_PREDICTION_FRAME_BATCH
        return original_tanh(value)

    monkeypatch.setattr(module.np, "tanh", measured_tanh)
    actual = module._predict(x, parameters)
    np.testing.assert_array_equal(actual, expected)
    assert observed_frames == [65536, 4096]
