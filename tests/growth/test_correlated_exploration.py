import numpy as np
import pytest

from rosclaw.growth.correlated_exploration import (
    conditional_mean,
    conditional_scale,
    stationary_noise,
)


def test_white_limit_is_exactly_original_per_frame_sampling():
    noise = stationary_noise(seed=123, rho=0, count=270, dimension=12, first_frame=30)
    expected = np.stack(
        [
            np.random.default_rng(np.random.SeedSequence(123, spawn_key=(frame,))).normal(size=12)
            for frame in range(30, 300)
        ]
    )
    assert np.array_equal(noise, expected)
    assert not noise.flags.writeable


def test_ar_recursion_is_stationary_and_smoother_without_reducing_variance():
    noise = stationary_noise(seed=17, rho=0.9, count=3000, dimension=64)
    white = stationary_noise(seed=17, rho=0, count=3000, dimension=64)
    assert abs(float(noise.mean())) < 0.04
    assert abs(float(noise.var()) - 1) < 0.05
    correlation = np.corrcoef(noise[:-1].ravel(), noise[1:].ravel())[0, 1]
    assert abs(correlation - 0.9) < 0.01
    assert np.mean(np.diff(noise, axis=0) ** 2) < 0.15 * np.mean(np.diff(white, axis=0) ** 2)


def test_candidate_previous_mean_is_part_of_the_likelihood():
    current = np.array([0.2, 0.3])
    previous = np.array([0.1, 0.2])
    action = np.array([0.5, 0.4])
    old = conditional_mean(current, previous, action, 0.9)
    candidate = conditional_mean(current + 0.05, previous + 0.05, action, 0.9)
    assert np.allclose(candidate - old, 0.005, atol=1e-15, rtol=0)
    assert not np.allclose(candidate, old + 0.05)
    assert conditional_scale(0.1, 0.9, first=True) == 0.1
    assert conditional_scale(0.1, 0.9, first=False) == pytest.approx(0.1 * np.sqrt(0.19))


@pytest.mark.parametrize("rho", [-1, 1, np.nan, np.inf, True])
def test_invalid_correlations_fail_closed(rho):
    with pytest.raises(ValueError):
        stationary_noise(seed=1, rho=rho, count=3, dimension=2)


def test_invalid_shapes_and_nonfinite_context_are_rejected():
    with pytest.raises(ValueError):
        conditional_mean([1, 2], [0], [1, 2], 0.9)
    with pytest.raises(ValueError):
        conditional_mean([np.nan], [0], [0], 0.9)
    with pytest.raises(ValueError):
        conditional_scale(0, 0.9, first=False)
    with pytest.raises(ValueError):
        stationary_noise(seed=-1, rho=0.9, count=3, dimension=2)
    with pytest.raises(ValueError, match="arithmetic"):
        conditional_mean([1e308], [-1e308], [1e308], 0.9)
