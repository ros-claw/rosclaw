import numpy as np
import pytest

from rosclaw.growth.advantage_weights import statewise_advantage_weights


def test_positive_cap_does_not_erase_terminal_quality_bonus():
    weights = statewise_advantage_weights([[1], [1], [1]], [-10, 3, 13], [-9] * 3)
    assert weights[2] == 1
    assert weights[1] == pytest.approx(np.exp(-20))
    assert 0 < weights[0] < weights[1]


def test_baseline_translation_and_batch_order_do_not_change_weights():
    states, returns = [[0], [1], [0], [1]], [3, 9, 13, 10]
    first = statewise_advantage_weights(states, returns, [0] * 4)
    assert first == statewise_advantage_weights(states, returns, [1e100] * 4)
    order = [3, 2, 1, 0]
    permuted = statewise_advantage_weights(
        [states[i] for i in order], [returns[i] for i in order], [-4] * 4
    )
    assert permuted == [first[i] for i in order]


def test_different_values_for_identical_state_rejected():
    with pytest.raises(ValueError, match="state value"):
        statewise_advantage_weights([[0], [0]], [3, 13], [0, 1])


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError):
        statewise_advantage_weights([[0]], [1], [0], temperature=temperature)


def test_nonfinite_or_overflow_cannot_produce_training_weights():
    for rewards in ([float("nan"), 1], [-1e308, 1e308]):
        with pytest.raises(ValueError):
            statewise_advantage_weights([[0], [0]], rewards, [0, 0])
