import copy

import numpy as np
import pytest

from rosclaw.growth.context_prediction_mlp import fit_context_predictor
from rosclaw.growth.neural_context_advantages import neural_context_advantages


@pytest.fixture(scope="module")
def fitted():
    pytest.importorskip("torch")
    groups = np.repeat(np.arange(8), 40)
    contexts = np.repeat(np.arange(4), 2)
    features = np.column_stack((groups * 0.1, np.tile(np.linspace(0, 1, 40), 8)))
    phases = np.zeros(320, dtype=np.int64)
    returns = groups.astype(float) - 3.0
    x = np.column_stack((features, phases))
    results = [
        fit_context_predictor(
            x,
            returns[:, None],
            contexts[groups],
            held_out_contexts=(i,),
            epochs=20,
            batch_size=64,
            seed=10 + i,
        )
        for i in range(4)
    ]
    return features, phases, groups, returns, contexts, results


def evaluate(fitted, results=None):
    x, p, g, y, c, original = fitted
    return neural_context_advantages(
        x,
        p,
        g,
        y,
        trajectory_context_ids=c,
        fold_fit_results=original if results is None else results,
    )


def test_neural_mc_keeps_complete_disjoint_predictions_and_no_authority(fitted):
    result = evaluate(fitted)
    assert result["crossfit_predictions"].shape == (320,)
    assert result["linear_diagnostic_critic_readout"].shape == (1, 2)
    assert result["raw_return_prediction_mse"] >= 0
    assert len(result["critic_model_hashes"]) == 4
    assert result["overlapping_context_count"] == 0
    assert abs(result["advantages"].mean()) < 1e-12
    assert result["critic_kind"] == "WHOLE_CONTEXT_NEURAL_CROSSFIT_MC_NOT_TD_OR_GAE"
    assert result["actor_updated"] is False
    assert result["physical_batch_verified"] is False
    assert result["hardware_authorized"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "partial",
        "swapped",
        "train_count",
        "normalization",
        "authority",
        "chosen_on_holdout",
        "bool_context",
        "no_update",
    ],
)
def test_reject_incomplete_leaked_or_authorizing_fit(fitted, fault):
    results = copy.deepcopy(fitted[-1])
    if fault == "partial":
        results.pop()
    elif fault == "swapped":
        results[0], results[1] = results[1], results[0]
    elif fault == "train_count":
        results[0]["train_rows"] += 1
    elif fault == "normalization":
        results[0]["model"]["input_mean"][0] += 0.1
    elif fault == "authority":
        results[0]["hardware_authorized"] = True
    elif fault == "bool_context":
        results[0]["held_out_contexts"] = [False]
    elif fault == "no_update":
        results[0]["optimizer_updates"] = 0
    else:
        results[0]["hyperparameters_chosen_on_holdout"] = True
    with pytest.raises(ValueError):
        evaluate(fitted, results)
