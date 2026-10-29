"""Dense projection must preserve exact closed-cell point semantics."""

import numpy as np
import pytest

from rosclaw.connectors.ros.public_coverage.geometry import CircleSpace


def partitioned(space, points):
    # The small-batch public contract also covers live per-point execution.
    # Dense projection must agree regardless of how callers partition inputs.
    return np.concatenate([space.safe_points(chunk) for chunk in np.array_split(points, 7)])


@pytest.mark.parametrize(
    "radius,resolution", [(0.0, 0.05), (0.01, 0.01), (0.20, 0.025), (0.23, 0.05)]
)
def test_dense_projection_agrees_with_small_batches_at_exact_contacts(radius, resolution):
    free = np.ones((130, 140), dtype=bool)
    free[30:60, 50:52] = False
    free[80, 75] = False
    origin = np.array([-1.3, -0.7])
    rng = np.random.default_rng(321)
    points = origin + rng.uniform(-1, 141, (8192, 2)) * resolution
    center = origin + (np.array([75, 80]) + 0.5) * resolution
    offsets = np.linspace(-0.000001, 0.000001, 2048)
    points[:2048] = center + np.column_stack([radius + resolution / 2 + offsets, np.zeros(2048)])
    points[2048:2051] = [[np.nan, 0], [np.inf, 1], [-np.inf, 0]]
    space = CircleSpace(free, resolution, origin, radius)
    assert np.array_equal(space.safe_points(points), partitioned(space, points))


@pytest.mark.parametrize("origin", [(0.0, 0.0), (1000000.0, -1000000.0)])
def test_dense_batches_preserve_mutation_and_large_origin_fallback(origin):
    rng = np.random.default_rng(125)
    free = rng.random((100, 110)) > 0.04
    points = np.array(origin) + rng.uniform(0, 110, (6000, 2)) * 0.05
    space = CircleSpace(free, 0.05, origin, 0.23)
    assert np.array_equal(space.safe_points(points), partitioned(space, points))
    # No old EDT/certificate may survive a mutable map change.
    free[10:90, 10:90] = False
    assert np.array_equal(space.safe_points(points), partitioned(space, points))


def test_dense_all_free_and_all_blocked_maps_preserve_borders():
    points = np.random.default_rng(12).uniform(-0.1, 5.1, (5000, 2))
    for value in [False, True]:
        space = CircleSpace(np.full((100, 100), value, bool), 0.05, (0.0, 0.0), 0.2)
        assert np.array_equal(space.safe_points(points), partitioned(space, points))


def test_dense_certificates_handle_subcell_radius_and_narrow_corridor():
    free = np.zeros((100, 100), bool)
    free[:, 30:32] = True
    points = np.random.default_rng(333).uniform(0.0, 5.0, (5000, 2))
    points[:2048, 0] = np.linspace(1.49, 1.61, 2048)
    for radius in [0.000001, 0.025, 0.050000001]:
        space = CircleSpace(free, 0.05, (0.0, 0.0), radius)
        assert np.array_equal(space.safe_points(points), partitioned(space, points))
