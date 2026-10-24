import numpy as np
import pytest

from rosclaw.control.vector_envelope import intersect_vector_bounds


def test_intersection_is_simultaneous_owned_and_readonly() -> None:
    low, high = np.array([-2.0, 0.0]), np.array([2.0, 4.0])
    result = intersect_vector_bounds(((low, high), (np.array([1.0, -1.0]), np.array([3.0, 0.0]))))
    assert result is not None
    assert np.array_equal(result[0], [1.0, 0.0])
    assert np.array_equal(result[1], [2.0, 0.0])
    low[:] = -100
    assert result[0][0] == 1.0
    assert not result[0].flags.writeable
    assert not result[1].flags.writeable


def test_disjoint_envelopes_are_explicitly_infeasible() -> None:
    assert (
        intersect_vector_bounds(
            ((np.array([0.0]), np.array([1.0])), (np.array([2.0]), np.array([3.0])))
        )
        is None
    )


@pytest.mark.parametrize(
    "low,high",
    [
        ([], []),
        ([np.nan], [1.0]),
        ([0.0], [np.inf]),
        ([2.0], [1.0]),
        ([[0.0]], [[1.0]]),
        ([0.0], [1.0, 2.0]),
        (["0"], ["1"]),
    ],
)
def test_invalid_bounds_rejected(low: object, high: object) -> None:
    with pytest.raises(ValueError):
        intersect_vector_bounds(((np.asarray(low), np.asarray(high)),))


def test_empty_collection_rejected() -> None:
    with pytest.raises(ValueError):
        intersect_vector_bounds(())
