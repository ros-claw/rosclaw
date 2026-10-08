import copy

import numpy as np
import pytest

from rosclaw.growth.causal_residual_memory import CausalResidualMemory, initial_parameters
from rosclaw.growth.recurrent_residual_imitation import (
    RecurrentResidualImitationConfig,
    fit_recurrent_residual_imitation,
)


def dataset():
    context = np.zeros((4, 12, 2))
    context[:, 0, 0] = [-1, 1, -1, 1]
    return {
        "parameters": initial_parameters(2, 1, hidden_dimension=4, seed=3),
        "context": context,
        "baseline": np.zeros((4, 12, 1)),
        "gates": np.ones((4, 12)),
        "targets": np.repeat(context[:, :1, :1] * 0.05, 12, axis=1),
        "training_weights": np.ones((4, 12)),
    }


def test_actual_recurrent_fit_reduces_loss_and_restores_rng_and_threads():
    torch = pytest.importorskip("torch")
    data = dataset()
    before = copy.deepcopy(data)
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    result = fit_recurrent_residual_imitation(
        **data, config=RecurrentResidualImitationConfig(steps=64)
    )
    assert result["completed_optimizer_steps"] == len(result["minibatch_loss_history"]) == 64
    assert (
        result["training_loss_improved"]
        and result["final_full_positive_weight_loss"] < result["initial_full_positive_weight_loss"]
    )
    assert torch.equal(rng, torch.get_rng_state()) and torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert result["teacher_targets_are_recurrent_inputs"] is False
    assert result["physical_batch_verified"] is result["promotion_authorized"] is False
    for key in ("context", "baseline", "gates", "targets", "training_weights"):
        np.testing.assert_array_equal(data[key], before[key])
    assert data["parameters"] == before["parameters"]
    memory = CausalResidualMemory(result["parameters"])
    for tick in range(12):
        assert np.max(np.abs(memory.step(data["context"][0, tick], index=tick))) <= 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", True),
        ("steps", 2049),
        ("batch_episodes", 0),
        ("learning_rate", float("nan")),
        ("residual_cap", 0.21),
        ("compute_device", "cuda"),
    ],
)
def test_invalid_config_rejected(field, value):
    config = RecurrentResidualImitationConfig(**{field: value})
    with pytest.raises(ValueError):
        fit_recurrent_residual_imitation(**dataset(), config=config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("context", np.zeros((4, 12, 3))),
        ("targets", np.full((4, 12, 1), np.nan)),
        ("gates", np.full((4, 12), 1.1)),
        ("training_weights", np.zeros((4, 12))),
        ("training_weights", np.full((4, 12), -1)),
    ],
)
def test_all_rows_checked_before_fitting(field, value):
    data = dataset()
    data[field] = value
    with pytest.raises(ValueError):
        fit_recurrent_residual_imitation(**data, config=RecurrentResidualImitationConfig(steps=1))


def test_positive_weight_underflow_rejected_not_silently_dropped():
    data = dataset()
    data["training_weights"][:] = np.nextafter(0.0, 1.0)
    data["training_weights"][0, 0] = 1e6
    with pytest.raises(ValueError, match="representation"):
        fit_recurrent_residual_imitation(**data, config=RecurrentResidualImitationConfig(steps=1))
