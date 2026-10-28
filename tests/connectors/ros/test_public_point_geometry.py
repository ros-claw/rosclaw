"""Closed-cell footprint oracle, including tangent and regional boundaries."""

import math

import numpy as np
import pytest

from rosclaw.connectors.ros.public_coverage.geometry import CircleSpace, RegionalCircleConstraint


@pytest.mark.parametrize("radius,resolution", [(0.25, 0.125), (0.20, 0.025), (0.23, 0.01)])
@pytest.mark.parametrize("axis", [0, 1])
def test_closed_cell_tangency_is_blocked_in_scalar_and_batch(radius, resolution, axis):
    free = np.ones((128, 128), bool)
    free[64, 64] = False
    point = np.array([64.5 * resolution, 64.5 * resolution])
    point[axis] = 65 * resolution + radius
    space = CircleSpace(free, resolution, [0.0, 0.0], radius)
    assert not space.safe_points([point])[0]
    assert not space.safe_point(point)


def oracle(space, point):
    point = np.asarray(point, float)
    relative = point - space.origin
    height, width = space.free.shape
    if (
        not np.isfinite(point).all()
        or (relative <= space.radius).any()
        or (
            relative
            >= [width * space.resolution - space.radius, height * space.resolution - space.radius]
        ).any()
    ):
        return False
    extent = np.array([width, height]) * space.resolution
    if (relative**2 <= space.radius**2 + 1e-14).any() or (
        (extent - relative) ** 2 <= space.radius**2 + 1e-14
    ).any():
        return False
    cells = np.argwhere(~space.free)
    centers = space.origin + (cells[:, ::-1] + 0.5) * space.resolution
    delta = np.maximum(np.abs(centers - point) - space.resolution / 2, 0.0)
    return bool((np.sum(delta * delta, axis=1) > space.radius**2 + 1e-14).all())


@pytest.mark.parametrize("radius", [0.0, 0.20, 0.23])
def test_scalar_singleton_and_batch_match_all_blocked_cell_oracle(radius):
    rng = np.random.default_rng(1026)
    free = np.ones((60, 64), dtype=bool)
    free[30, 32] = free[2, 2] = free[15, 45] = False
    space = CircleSpace(free, 0.025, [-0.7, 0.2], radius)
    points = rng.uniform([-0.8, 0.1], [1.0, 1.8], (150, 2))
    points = np.vstack(
        [points, [np.nan, 1.0], [np.inf, 1.0], [-0.7 + radius, 0.7], [-0.7 + radius + 1e-15, 0.8]]
    )
    expected = np.array([oracle(space, point) for point in points])
    assert expected.any() and not expected.all()
    assert np.array_equal(space.safe_points(points), expected)
    assert np.array_equal([space.safe_point(point) for point in points], expected)
    assert np.array_equal([space.safe_points([point])[0] for point in points], expected)


@pytest.mark.parametrize("yaw", [0.0, 0.7, -math.pi / 2])
def test_regional_single_point_matches_batch_inside_outside_and_at_window_edges(yaw):
    free = np.ones((20, 24), bool)
    free[8, 9] = False
    region = RegionalCircleConstraint(free, 0.025, [-0.3, -0.2], 0.20, [1.2, -0.7, yaw])
    rng = np.random.default_rng(1027)
    local = np.vstack(
        [rng.uniform([-0.8, -0.7], [0.8, 0.7], (120, 2)), region.low, region.high, [np.nan, 0.0]]
    )
    points = local @ region.rotation.T + region.translation
    expected = region.safe_points(points)
    assert np.array_equal([region.safe_point(point) for point in points], expected)


def test_single_point_offset_cache_cannot_hide_geometry_or_obstacle_changes():
    free = np.ones((40, 40), dtype=bool)
    space = CircleSpace(free, 0.025, [0.0, 0.0], 0.20)
    point = [0.5, 0.5]
    assert space.safe_points([point])[0]
    space.free[20, 20] = False
    assert not space.safe_points([point])[0]
    space.radius = 0.13
    space.resolution = 0.05
    points = np.array([[0.8, 0.8], [0.5, 0.5]])
    assert np.array_equal(
        [space.safe_points([point])[0] for point in points],
        space.safe_points(points),
    )
