import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

import rosclaw.growth.context_prediction_mlp as prediction_module
from rosclaw.growth.context_prediction_mlp import _hash, predict
from rosclaw.growth.zero_anchored_effect_prediction import predict_zero_anchored_effect


def model():
    weights = [np.zeros((128, 3)), np.zeros((64, 128)), np.zeros((1, 64))]
    for w in weights:
        w[0, 0] = 1
    weights[0][0, 2] = 2
    result = {
        "schema": "rosclaw.growth.context_prediction_mlp.v1",
        "source_hash": "sha256:"
        + hashlib.sha256(Path(prediction_module.__file__).read_bytes()).hexdigest(),
        "input_mean": [1.0, -2.0, 0.4],
        "input_scale": [0.5, 1.0, 2.0],
        "target_mean": [8.0],
        "target_scale": [3.0],
        "layers": [{"weight": w.tolist(), "bias": np.zeros(len(w)).tolist()} for w in weights],
        "prediction_only": True,
        "motor_policy": False,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
    result["model_hash"] = _hash(result)
    return result


def test_exact_zero_reference_in_original_units_and_owned_inputs():
    x = np.asarray([[1.0, 2.0, 0.0], [2.0, 3.0, 0.2]])
    before = x.copy()
    network = model()
    model_before = copy.deepcopy(network)
    result = predict_zero_anchored_effect(network, x, intervention_columns=(2,))
    reference = x.copy()
    reference[:, 2] = 0
    np.testing.assert_array_equal(result, predict(network, x) - predict(network, reference))
    assert result[0, 0] == 0
    assert result[1, 0] != 0
    result[:] = 99
    np.testing.assert_array_equal(x, before)
    assert network == model_before


@pytest.mark.parametrize("columns", [(), [], (True,), (-1,), (3,), (2, 2)])
def test_invalid_columns(columns):
    with pytest.raises(ValueError):
        predict_zero_anchored_effect(model(), np.zeros((2, 3)), intervention_columns=columns)


def test_model_authority_and_nonfinite_features_remain_rejected():
    network = model()
    network["hardware_authorized"] = True
    network["model_hash"] = _hash({k: v for k, v in network.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        predict_zero_anchored_effect(network, np.zeros((2, 3)), intervention_columns=(2,))
    with pytest.raises(ValueError):
        predict_zero_anchored_effect(model(), np.full((2, 3), np.nan), intervention_columns=(2,))
