import numpy as np
import pytest

from rosclaw.growth.staged_action_projection import staged_action_projection


def test_original_order_randomized_and_ownership():
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(200, 12))
    previous = rng.uniform(-0.16, 0.16, raw.shape)
    lower = rng.uniform(-0.16, 0, raw.shape)
    upper = rng.uniform(0, 0.16, raw.shape)
    result = staged_action_projection(raw, previous, lower, upper, cap=0.16, slew=0.012)
    expected = np.clip(
        previous + np.clip(0.16 * np.tanh(raw) - previous, -0.012, 0.012), lower, upper
    )
    np.testing.assert_array_equal(result, expected)
    assert not np.shares_memory(result, raw)
    assert not np.shares_memory(result, previous)


def test_final_moving_box_can_override_slew_and_preserves_zero():
    result = staged_action_projection([1.0], [0.1], [-0.01], [0.01], cap=0.16, slew=0.012)
    np.testing.assert_array_equal(result, [0.01])
    assert abs(result[0] - 0.1) > 0.012
    result = staged_action_projection([1.0], [0.0], [0.0], [0.0], cap=0.16, slew=0.012)
    np.testing.assert_array_equal(result, [0.0])


@pytest.mark.parametrize("operand", range(4))
def test_nonfinite_rejected(operand):
    values = [[0.1], [0.0], [-0.1], [0.1]]
    values[operand] = [float("nan")]
    with pytest.raises(ValueError):
        staged_action_projection(*values, cap=0.16, slew=0.012)


@pytest.mark.parametrize("cap,slew", [(True, 0.012), (0.16, 0.0), (0.16, 0.2), (2.0, 0.01)])
def test_bad_limits(cap, slew):
    with pytest.raises(ValueError):
        staged_action_projection([0.0], [0.0], [-0.1], [0.1], cap=cap, slew=slew)
