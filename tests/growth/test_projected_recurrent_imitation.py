import copy

import numpy as np
import pytest

from rosclaw.growth.recurrent_residual_imitation import (
    RecurrentResidualImitationConfig,
    fit_recurrent_residual_imitation,
)
from rosclaw.growth.staged_action_projection import staged_action_projection
from tests.growth.test_recurrent_residual_imitation import dataset


def projected_dataset():
    data = dataset()
    shape = data["targets"].shape
    prior, low, high = np.zeros(shape), np.full(shape, -0.16), np.full(shape, 0.16)
    projection = {
        "previous": prior,
        "lower": low,
        "upper": high,
        "executed_targets": staged_action_projection(
            data["targets"], prior, low, high, cap=0.16, slew=0.012
        ),
        "cap": 0.16,
        "slew": 0.012,
        "raw_loss_weight": 0.05,
    }
    return data, projection


def test_actual_projected_fit_improves_objective_owned_data_and_rng():
    torch = pytest.importorskip("torch")
    data, projection = projected_dataset()
    before = copy.deepcopy(projection)
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    result = fit_recurrent_residual_imitation(
        **data, action_projection=projection, config=RecurrentResidualImitationConfig(steps=64)
    )
    assert result["training_loss_improved"]
    assert result["completed_optimizer_steps"] == 64
    assert torch.equal(rng, torch.get_rng_state()) and torch.get_num_threads() == threads
    objective = result["executed_action_objective"]
    assert objective["all_original_rows_validated"] == 48
    assert objective["teacher_forced_previous_actions_not_closed_loop_rollout"] is True
    assert objective["projection_labels_are_recurrent_inputs"] is False
    assert objective["physical_batch_verified"] is objective["hardware_authorized"] is False
    for key in ("previous", "lower", "upper", "executed_targets"):
        np.testing.assert_array_equal(projection[key], before[key])
    expected = np.mean((projection["executed_targets"] / 0.16) ** 2) + 0.05 * np.mean(
        data["targets"] ** 2
    )
    assert result["initial_full_positive_weight_loss"] == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cap", True),
        ("slew", 0.2),
        ("raw_loss_weight", 0.0),
        ("raw_loss_weight", True),
        ("previous", np.full((4, 12, 1), np.nan)),
        ("lower", np.full((4, 12, 1), 0.01)),
        ("executed_targets", np.ones((4, 12, 1))),
    ],
)
def test_bad_projection_rejected_before_fit(field, value):
    data, projection = projected_dataset()
    projection[field] = value
    with pytest.raises(ValueError):
        fit_recurrent_residual_imitation(
            **data, action_projection=projection, config=RecurrentResidualImitationConfig(steps=1)
        )


def test_zero_weight_rows_still_validate_recorded_projection():
    data, projection = projected_dataset()
    data["training_weights"][0] = 0
    projection["executed_targets"][0, 0, 0] += 0.001
    with pytest.raises(ValueError, match="every recorded"):
        fit_recurrent_residual_imitation(
            **data, action_projection=projection, config=RecurrentResidualImitationConfig(steps=1)
        )


def test_default_retains_latent_only_receipt():
    pytest.importorskip("torch")
    result = fit_recurrent_residual_imitation(
        **dataset(), config=RecurrentResidualImitationConfig(steps=1)
    )
    assert "executed_action_objective" not in result
