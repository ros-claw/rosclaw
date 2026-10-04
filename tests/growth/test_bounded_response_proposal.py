import numpy as np
import pytest

from rosclaw.growth.bounded_response_proposal import bounded_response_proposal


def test_improvement_zero_and_ownership_never_execution():
    matrix = np.eye(3)
    error = np.ones(3) * 0.01
    weights = np.ones(3)
    result = bounded_response_proposal(
        matrix, error, weights, maximum_increment=0.02, regularization=0.1
    )
    assert result["predicted_candidate_cost"] < result["predicted_baseline_cost"]
    assert np.max(np.abs(result["target_increment"])) <= 0.02
    assert np.array_equal(matrix, np.eye(3))
    assert result["runtime_execution_authorized"] is False
    zero = bounded_response_proposal(
        matrix, np.zeros(3), weights, maximum_increment=0.02, regularization=0.1
    )
    assert np.array_equal(zero["target_increment"], np.zeros(3))


@pytest.mark.parametrize("fault", ["nan", "bool", "weights", "shape", "bound", "regularization"])
def test_nonfinite_types_shapes_and_parameters_rejected(fault):
    matrix = np.eye(3)
    error = np.ones(3)
    weights = np.ones(3)
    bound = 0.02
    regularization = 0.1
    if fault == "nan":
        matrix[0, 0] = np.nan
    elif fault == "bool":
        matrix = matrix.astype(bool)
    elif fault == "weights":
        weights[:] = 0
    elif fault == "shape":
        error = np.ones(2)
    elif fault == "bound":
        bound = True
    else:
        regularization = 0.0
    with pytest.raises(ValueError):
        bounded_response_proposal(
            matrix, error, weights, maximum_increment=bound, regularization=regularization
        )
