import copy

import numpy as np
import pytest

from rosclaw.growth.conditional_response_jacobian import fit_conditional_response_jacobian, predict
from rosclaw.growth.context_prediction_mlp import _hash


def data():
    rng = np.random.default_rng(729)
    x = rng.normal(size=(256, 3))
    a = rng.normal(size=(256, 2)) * 0.003
    y = np.column_stack(
        ((0.4 + 0.3 * np.tanh(x[:, 0])) * a[:, 0], (-0.2 + 0.4 * np.tanh(x[:, 1])) * a[:, 1])
    )
    return x, a, y, np.repeat(np.arange(4), 64)


@pytest.fixture(scope="module")
def result():
    pytest.importorskip("torch")
    return fit_conditional_response_jacobian(
        *data(), held_out_contexts=(3,), seed=729, epochs=150, batch_size=64, learning_rate=0.003
    )


def test_learns_conditional_effect_better_than_global_jacobian(result):
    assert result["held_out_standardized_mse"] < 0.2 * result["global_jacobian_standardized_mse"]
    assert result["optimizer_updates"] == 450
    assert result["train_contexts"] == [0, 1, 2]
    assert result["held_out_contexts"] == [3]
    assert result["motor_policy_updates"] == 0


def test_exact_zero_response_and_inputs_owned(result):
    x, a, _, _ = data()
    before_x, before_a, before_model = x.copy(), a.copy(), copy.deepcopy(result["model"])
    np.testing.assert_array_equal(predict(result["model"], x, np.zeros_like(a)), np.zeros((256, 2)))
    output = predict(result["model"], x, a)
    output[:] = 99
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(a, before_a)
    assert result["model"] == before_model


def test_rng_and_threads_restored():
    torch = pytest.importorskip("torch")
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    fit_conditional_response_jacobian(*data(), held_out_contexts=(3,), epochs=1)
    assert torch.equal(rng, torch.get_rng_state())
    assert threads == torch.get_num_threads()


@pytest.mark.parametrize(
    "fault", ["nan", "boolean", "bad_context", "bad_rows", "bad_epochs", "seed_bool"]
)
def test_invalid_training_rejected(fault):
    x, a, y, groups = data()
    kw = {"held_out_contexts": (3,), "epochs": 1}
    if fault == "nan":
        a[0, 0] = np.nan
    elif fault == "boolean":
        a = a.astype(bool)
    elif fault == "bad_context":
        kw["held_out_contexts"] = (5,)
    elif fault == "bad_rows":
        y = y[:-1]
    elif fault == "bad_epochs":
        kw["epochs"] = 0
    else:
        kw["seed"] = True
    with pytest.raises(ValueError):
        fit_conditional_response_jacobian(x, a, y, groups, **kw)


@pytest.mark.parametrize("fault", ["authority", "source", "normalization", "layer"])
def test_resealed_invalid_models_rejected(result, fault):
    m = copy.deepcopy(result["model"])
    if fault == "authority":
        m["hardware_authorized"] = True
    elif fault == "source":
        m["source_hash"] = "sha256:" + "0" * 64
    elif fault == "normalization":
        m["intervention_scale"][0] = 0
    else:
        m["layers"][0] = {}
    m["model_hash"] = _hash({k: v for k, v in m.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        predict(m, data()[0], data()[1])
