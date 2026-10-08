import copy

import numpy as np
import pytest

from rosclaw.growth.context_prediction_mlp import _hash, fit_context_predictor, predict


def data():
    rng = np.random.default_rng(469)
    x = rng.normal(size=(256, 4))
    y = np.stack((0.4 * x[:, 0] - 0.2 * x[:, 1], 0.3 * x[:, 2]), axis=1)
    return x, y, np.repeat(np.arange(4), 64)


@pytest.fixture(scope="module")
def result():
    pytest.importorskip("torch")
    x, y, groups = data()
    return fit_context_predictor(x, y, groups, held_out_contexts=(3,), seed=469, epochs=100)


def test_learns_disjoint_prediction_without_motor_updates_and_preserves_inputs(result):
    x, y, groups = data()
    assert result["train_contexts"] == [0, 1, 2]
    assert result["held_out_contexts"] == [3]
    assert result["train_rows"] == 192 and result["held_out_rows"] == 64
    assert result["optimizer_updates"] == 200
    assert (
        result["held_out_standardized_mse"] < 0.1 * result["train_mean_baseline_standardized_mse"]
    )
    assert result["motor_policy_updates"] == 0
    assert not result["model"]["motor_policy"]
    assert not result["hardware_authorized"]
    np.testing.assert_array_equal(result["model"]["input_mean"], x[groups != 3].mean(axis=0))
    before = x.copy()
    output = predict(result["model"], x)
    assert output.shape == y.shape and np.isfinite(output).all()
    np.testing.assert_array_equal(x, before)


def test_rng_and_threads_restored():
    torch = pytest.importorskip("torch")
    x, y, groups = data()
    before, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    fit_context_predictor(x, y, groups, held_out_contexts=(3,), seed=469, epochs=1)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.get_num_threads() == threads


@pytest.mark.parametrize(
    "fault", ["nan", "boolean", "missing_context", "only_holdout", "seed_bool", "epochs_zero"]
)
def test_invalid_data_and_config_fail_before_training(fault):
    x, y, groups = data()
    kwargs = {"held_out_contexts": (3,), "epochs": 1}
    if fault == "nan":
        x[0, 0] = np.nan
    elif fault == "boolean":
        x = x.astype(bool)
    elif fault == "missing_context":
        kwargs["held_out_contexts"] = (9,)
    elif fault == "only_holdout":
        kwargs["held_out_contexts"] = (0, 1, 2, 3)
    elif fault == "seed_bool":
        kwargs["seed"] = True
    else:
        kwargs["epochs"] = 0
    with pytest.raises(ValueError):
        fit_context_predictor(x, y, groups, **kwargs)


@pytest.mark.parametrize("fault", ["authority", "ceiling", "missing_normalization", "bad_layer"])
def test_resealed_non_prediction_model_rejected(result, fault):
    model = copy.deepcopy(result["model"])
    if fault == "authority":
        model["hardware_authorized"] = True
    elif fault == "ceiling":
        model["activation_ceiling"] = "REAL"
    elif fault == "missing_normalization":
        del model["input_scale"]
    else:
        model["layers"][0] = {}
    model["model_hash"] = _hash({k: v for k, v in model.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        predict(model, data()[0])
