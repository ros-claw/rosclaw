import numpy as np
import pytest

from rosclaw.growth.context_balanced_rehearsal import context_balanced_rehearsal_weights


def test_mass_context_balance_and_ownership():
    primary = np.zeros((6, 4))
    primary[0] = [1, 2, 3, 4]
    selected = np.array([False, True, True, True, False, False])
    contexts = np.array([0, 1, 1, 2, 2, 3])
    result = context_balanced_rehearsal_weights(
        primary, selected, contexts, rehearsal_mass_ratio=0.3
    )
    np.testing.assert_array_equal(result[~selected], primary[~selected])
    assert result[selected].sum() == pytest.approx(3)
    assert result[1:3].sum() == pytest.approx(result[3].sum())
    result[0, 0] = 9
    assert primary[0, 0] == 1


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1, float("nan"), True, 1])
def test_invalid_ratio(ratio):
    with pytest.raises(ValueError):
        context_balanced_rehearsal_weights(
            [[1.0, 1.0], [0.0, 0.0]], [False, True], [0, 1], rehearsal_mass_ratio=ratio
        )


def test_overlap_nonfinite_and_missing_rehearsal_rejected():
    for weights, selected, contexts in (
        ([[1, 1], [1, 1]], [False, True], [0, 1]),
        ([[1, float("inf")], [0, 0]], [False, True], [0, 1]),
        ([[1, 1], [0, 0]], [False, False], [0, 1]),
        ([[1, 1], [0, 0]], [False, True], [0.0, 1.0]),
        ([[1, 1], [0, 0]], [False, True], [0, -1]),
    ):
        with pytest.raises(ValueError):
            context_balanced_rehearsal_weights(
                weights, selected, contexts, rehearsal_mass_ratio=0.1
            )
