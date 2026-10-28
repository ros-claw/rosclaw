"""Independent closed-rectangle distance checks for the full swept capsule."""

import itertools

import numpy as np
import pytest

from rosclaw.connectors.ros.public_coverage.geometry import CircleSpace


def rectangle_distance_squared(start, finish, low, high):
    """Minimize piecewise-quadratic point-to-box distance along t in [0, 1].

    This visits all parameter breakpoints, rather than using the production
    endpoint/corner-to-segment calculation or its obstacle-window selection.
    """
    delta = finish - start
    parameters = [0.0, 1.0]
    for axis in range(2):
        if delta[axis] != 0:
            parameters += [
                float((edge - start[axis]) / delta[axis])
                for edge in [low[axis], high[axis]]
                if 0 < (edge - start[axis]) / delta[axis] < 1
            ]
    breaks = sorted(set(parameters))
    for left, right in zip(breaks, breaks[1:], strict=False):
        middle = start + (left + right) / 2 * delta
        active = (middle < low) | (middle > high)
        slope = delta[active]
        if np.dot(slope, slope) > 0:
            box = np.clip(middle, low, high)
            intercept = (start - box)[active]
            stationary = -float(np.dot(intercept, slope) / np.dot(slope, slope))
            parameters.append(float(np.clip(stationary, left, right)))
    return min(
        float(np.sum((point - np.clip(point, low, high)) ** 2))
        for point in (start + t * delta for t in parameters)
    )


def oracle(space, start, finish):
    points = np.asarray([start, finish]) - space.origin
    extent = np.asarray(space.free.shape[::-1]) * space.resolution
    if not ((points > space.radius) & (points < extent - space.radius)).all():
        return False
    # Exhaust all blocked cells, independent of production's bounding window.
    for cell in np.argwhere(~space.free):
        low = space.origin + cell[::-1] * space.resolution
        high = low + space.resolution
        if rectangle_distance_squared(start, finish, low, high) <= space.radius**2 + 1e-14:
            return False
    return True


@pytest.mark.parametrize("radius", [0.0, 0.20, 0.23, 0.25])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("gap", [-1e-6, 0.0, 1e-6])
def test_capsule_middle_closed_cell_tangency(radius, transpose, side, gap):
    free = np.ones((64, 64), dtype=bool)
    free[30, 30] = False
    edge = 1.55 if side > 0 else 1.50
    height = edge + side * (radius + gap)
    start, finish = np.array([0.50, height]), np.array([2.50, height])
    if transpose:
        free, start, finish = free.T, start[::-1], finish[::-1]
    space = CircleSpace(free, 0.05, np.zeros(2), radius)
    expected = oracle(space, start, finish)
    if gap == 0:
        assert expected is False
        assert space.safe_point(start) and space.safe_point(finish)
    assert space.safe_segment(start, finish) == expected
    assert space.safe_segment(finish, start) == expected


def test_existing_distance_tolerance_is_included_in_the_candidate_window():
    free = np.ones((64, 64), dtype=bool)
    free[30, 30] = False
    space = CircleSpace(free, 0.05, np.zeros(2), 0.0)
    start, finish = np.array([0.5, 1.55 + 5e-8]), np.array([2.5, 1.55 + 5e-8])
    assert oracle(space, start, finish) is False
    assert space.safe_segment(start, finish) is False


def test_full_blocked_cell_oracle_across_scales_translations_and_directions():
    rng = np.random.default_rng(20261027)
    saw_safe = saw_blocked = False
    for resolution, radius in itertools.product([0.025, 0.05, 0.10], [0.0, 0.20, 0.23, 0.25]):
        free = np.ones((48, 48), dtype=bool)
        blocked = rng.integers(6, 42, size=(8, 2))
        free[tuple(blocked.T)] = False
        origin = np.array([-2.3, 1.2])
        space = CircleSpace(free, resolution, origin, radius)
        for _ in range(24):
            start, finish = origin + rng.uniform(
                radius + 0.05, 48 * resolution - radius - 0.05, (2, 2)
            )
            expected = oracle(space, start, finish)
            saw_safe |= expected
            saw_blocked |= not expected
            assert space.safe_segment(start, finish) == expected
            assert space.safe_segment(finish, start) == expected
    assert saw_safe and saw_blocked
