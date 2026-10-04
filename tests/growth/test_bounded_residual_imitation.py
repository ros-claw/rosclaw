import copy

import numpy as np
import pytest

from rosclaw.growth.bounded_residual_imitation import (
    BoundedResidualImitationConfig,
    fit_bounded_residual_imitation,
)


def batch():
    x = np.linspace(-1, 1, 128)[:, None]
    return {
        "layers": [(np.zeros((1, 1)), np.zeros(1))],
        "context": x,
        "baseline": np.zeros((128, 1)),
        "gates": np.ones(128),
        "targets": 0.12 * np.tanh(2 * x),
        "training_weights": np.ones(128),
    }


def test_fit_improves_preserves_inputs_and_restores_torch_state():
    torch = pytest.importorskip("torch")
    data = batch()
    before = copy.deepcopy(data)
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    result = fit_bounded_residual_imitation(
        **data,
        config=BoundedResidualImitationConfig(steps=200, batch_size=128, learning_rate=0.001),
    )
    assert result["training_loss_improved"]
    assert result["final_full_positive_weight_loss"] < result["initial_full_positive_weight_loss"]
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    for key in data:
        if key != "layers":
            np.testing.assert_array_equal(data[key], before[key])
    np.testing.assert_array_equal(data["layers"][0][0], before["layers"][0][0])
    assert result["physical_batch_verified"] is False
    assert result["online_rl_claimed"] is False
    assert result["hardware_authorized"] is False


def test_zero_weights_do_not_train_failures_or_claim_new_episodes():
    pytest.importorskip("torch")
    data = batch()
    data["training_weights"][64:] = 0
    fitted = fit_bounded_residual_imitation(**data, config=BoundedResidualImitationConfig(steps=2))
    changed = batch()
    changed["training_weights"][64:] = 0
    changed["targets"][64:] = 1000
    replay = fit_bounded_residual_imitation(
        **changed, config=BoundedResidualImitationConfig(steps=2)
    )
    assert fitted["layers"] == replay["layers"]
    assert fitted["original_input_rows"] == 128
    assert fitted["zero_weight_rows"] == fitted["positive_weight_rows"] == 64
    assert fitted["zero_weight_rows_are_not_negative_gradient_examples"] is True


def test_zero_gate_output_cannot_change():
    pytest.importorskip("torch")
    data = batch()
    data["gates"][:] = 0
    value = fit_bounded_residual_imitation(**data, config=BoundedResidualImitationConfig(steps=2))
    assert value["layers"] == [{"weight": [[0.0]], "bias": [0.0]}]
    assert value["training_loss_improved"] is False


@pytest.mark.parametrize(
    "fault", ["nan", "negative", "empty", "gate", "boolean", "shape", "integer_min", "layer"]
)
def test_malformed_data_rejected_before_training(fault):
    data = batch()
    if fault == "nan":
        data["targets"][0] = np.nan
    elif fault == "negative":
        data["training_weights"][0] = -1
    elif fault == "empty":
        data["training_weights"][:] = 0
    elif fault == "gate":
        data["gates"][0] = 1.01
    elif fault == "boolean":
        data["context"] = data["context"].astype(bool)
    elif fault == "shape":
        data["targets"] = np.zeros((128, 2))
    elif fault == "integer_min":
        data["targets"] = np.full((128, 1), np.iinfo(np.int64).min, dtype=np.int64)
    else:
        data["layers"] = [(np.zeros((2, 1)), np.zeros(1))]
    with pytest.raises(ValueError):
        fit_bounded_residual_imitation(**data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": True},
        {"steps": 4097},
        {"batch_size": 0},
        {"learning_rate": float("nan")},
        {"residual_cap": 0.201},
        {"compute_device": "cuda"},
        {"seed": -1},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        BoundedResidualImitationConfig(**kwargs).validate()
