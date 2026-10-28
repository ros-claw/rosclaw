"""Local observed hazards supplement, never replace, global path geometry."""

from types import SimpleNamespace as Namespace

import numpy as np
import pytest

from rosclaw.connectors.ros.public_coverage.geometry import RegionalCircleConstraint
from rosclaw.connectors.ros.public_coverage.grid import Grid, UnsafeRobotPose


def test_registered_local_hazard_blocks_an_otherwise_clear_global_path():
    free = np.ones((20, 20), bool)
    free[10, 10] = False
    constraint = RegionalCircleConstraint(free, 0.1, [-1.0, -1.0], 0.2, [2.0, 3.0, np.pi / 2])
    assert not constraint.safe_point([1.95, 3.05])
    assert not constraint.safe_segment([1.95, 2.0], [1.95, 4.0])
    assert constraint.safe_segment([3.0, 2.0], [3.0, 4.0])


def test_region_boundary_cannot_hide_a_touched_obstacle_cell():
    free = np.ones((10, 10), bool)
    free[5, 0] = False
    constraint = RegionalCircleConstraint(free, 0.1, [0.0, 0.0], 0.2, [0.0, 0.0, 0.0])
    assert not constraint.safe_point([-0.1, 0.55])
    assert not constraint.safe_segment([-0.5, 0.55], [1.5, 0.55])
    assert constraint.safe_segment([-0.5, -0.5], [1.5, -0.5])
    assert constraint.safe_point([3.0, 3.0])  # Global space remains mandatory there.


def test_unknown_inside_the_window_is_never_cleared():
    free = np.ones((10, 10), bool)
    free[5, 5] = False
    constraint = RegionalCircleConstraint(free, 0.1, [0.0, 0.0], 0.0, [0.0, 0.0, 0.0])
    assert not constraint.safe_point([0.55, 0.55])
    assert not constraint.safe_segment([-0.5, 0.55], [1.5, 0.55])


@pytest.mark.parametrize("registration", [None, [0, 0], [0, 0, float("nan")]])
def test_absent_or_invalid_registration_is_not_identity(registration):
    with pytest.raises(ValueError):
        RegionalCircleConstraint(np.ones((10, 10), bool), 0.1, [0, 0], 0.2, registration)


def message():
    origin = Namespace(
        position=Namespace(x=-2.0, y=-2.0), orientation=Namespace(x=0.0, y=0.0, z=0.0)
    )
    return Namespace(
        data=np.zeros(1600, int), info=Namespace(width=40, height=40, resolution=0.1, origin=origin)
    )


def local_costmap(value):
    origin = Namespace(
        position=Namespace(x=-1.0, y=-1.0), orientation=Namespace(x=0.0, y=0.0, z=0.0)
    )
    costs = np.zeros((20, 20), np.uint8)
    costs[10, 10] = value
    return Namespace(
        data=costs.ravel(), metadata=Namespace(size_x=20, size_y=20, resolution=0.1, origin=origin)
    )


@pytest.mark.parametrize("value", [254, 255])
def test_current_local_footprint_is_checked_even_if_global_grid_is_free(value):
    assert Grid(message(), [0.05, 0.05], tracking_margin=0.03).physical_pose_safe([0.05, 0.05])
    with pytest.raises(UnsafeRobotPose):
        Grid(
            message(),
            [0.05, 0.05],
            tracking_margin=0.03,
            local_navigation=local_costmap(value),
            map_from_local=[0.0, 0.0, 0.0],
        )


def test_global_obstacles_are_not_removed_by_a_local_free_window():
    occupancy = message()
    occupancy.data[20 * 40 + 30] = 100
    empty = local_costmap(0)
    grid = Grid(
        occupancy,
        [0.0, -0.5],
        tracking_margin=0.03,
        local_navigation=empty,
        map_from_local=[0, 0, 0],
    )
    assert not grid.safe_pose([1.05, 0.05])
    assert not grid.safe_segment([0.0, 0.05], [1.5, 0.05])


def test_local_inscribed_center_cost_is_not_treated_as_lethal_footprint_geometry():
    grid = Grid(
        message(),
        [-0.5, -0.5],
        tracking_margin=0.03,
        local_navigation=local_costmap(253),
        map_from_local=[0, 0, 0],
    )
    assert grid.physical_pose_safe([0.05, 0.05])
    assert not grid.safe_pose([0.05, 0.05])
    assert grid.safe_pose([0.25, 0.05])
