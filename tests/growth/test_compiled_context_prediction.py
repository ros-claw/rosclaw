import copy
import hashlib

import numpy as np
import pytest

from rosclaw.growth.compiled_context_prediction import CompiledContextPrediction
from rosclaw.growth.context_prediction_mlp import fit_context_predictor, predict


def fitted():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(96, 4))
    y = np.column_stack((x[:, 0] * 0.5, x[:, 1] * 0.2))
    groups = np.repeat(np.arange(3), 32)
    return fit_context_predictor(x, y, groups, held_out_contexts=(2,), epochs=2, batch_size=32)[
        "model"
    ]


def test_reference_exact_caller_ownership_and_readonly_parameters(tmp_path):
    model = fitted()
    pristine = copy.deepcopy(model)
    compiled = CompiledContextPrediction(model)
    for count in (1, 17, 96):
        x = np.random.default_rng(count).normal(size=(count, 4))
        assert np.array_equal(compiled.predict(x), predict(pristine, x))
    model["layers"][0]["weight"][0][0] += 1
    assert np.array_equal(compiled.predict(np.zeros((1, 4))), predict(pristine, np.zeros((1, 4))))
    assert all(not a.flags.writeable for layer in compiled._layers for a in layer)
    assert compiled.contract()["motor_policy"] is False
    for x in (np.zeros((1, 3)), np.full((1, 4), np.nan), np.zeros((1, 4), dtype=bool)):
        with pytest.raises(ValueError):
            compiled.predict(x)
    path = tmp_path / "dependency.py"
    path.write_text("original")
    compiled._pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        compiled.predict(np.zeros((1, 4)))


@pytest.mark.parametrize("key,value", [("hardware_authorized", 0), ("motor_policy", True)])
def test_reference_rejects_type_forgery_and_authority(key, value):
    model = fitted()
    model[key] = value
    with pytest.raises(ValueError):
        CompiledContextPrediction(model)
