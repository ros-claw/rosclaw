"""Public-map return routes cannot cut an obstacle corner or cross unknown."""

from types import SimpleNamespace as Namespace

import numpy as np

from rosclaw.connectors.ros.public_coverage.grid import Grid
from rosclaw.connectors.ros.public_coverage.route_costs import RouteCosts


def geometry(data, robot):
    h, w = data.shape
    message = Namespace(
        data=data.ravel(),
        info=Namespace(
            width=w,
            height=h,
            resolution=0.05,
            origin=Namespace(
                position=Namespace(x=0.0, y=0.0), orientation=Namespace(x=0.0, y=0.0, z=0.0)
            ),
        ),
    )
    return Grid(message, robot, clearance=0.20, tracking_margin=0.03)


def test_exact_route_goes_around_observed_obstacle_instead_of_cutting_corner():
    data = np.zeros((40, 40), dtype=int)
    data[19:22, 19:22] = 100
    robot = np.array([0.5, 1.025])
    target = np.array([1.5, 1.025])
    grid = geometry(data, robot)
    assert not grid.safe_segment(robot, target)
    route = RouteCosts(grid, robot).path(target)
    assert route is not None and grid.motion_path_safe(route)
    assert np.linalg.norm(np.diff(route, axis=0), axis=1).sum() > np.linalg.norm(target - robot)
    assert np.allclose(route[0], robot) and np.allclose(route[-1], target)


def test_unknown_wall_is_not_crossed_to_reach_a_disconnected_goal():
    data = np.zeros((40, 40), dtype=int)
    data[:, 19:22] = -1
    robot = np.array([0.5, 1.0])
    grid = geometry(data, robot)
    assert RouteCosts(grid, robot).path([1.5, 1.0]) is None


def test_existing_short_physical_safe_reentry_prefix_is_preserved():
    data = np.zeros((40, 40), dtype=int)
    robot = np.array([0.205, 1.0])
    grid = geometry(data, robot)
    assert grid.physical_pose_safe(robot) and not grid.safe_pose(robot)
    route = RouteCosts(grid, robot).path([1.0, 1.0])
    assert route is not None and grid.motion_path_safe(route)
    assert np.allclose(route[0], robot)
