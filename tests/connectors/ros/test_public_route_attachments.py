"""A rejected global tangent is not proof that no current local detour exists."""

from types import SimpleNamespace as Namespace

import numpy as np
import pytest

from rosclaw.connectors.ros.public_coverage.grid import Grid
from rosclaw.connectors.ros.public_coverage.route_costs import RouteCosts


def scene(*, barrier=False, unknown=False):
    occupancy = np.zeros((80, 100), int)
    local = occupancy.copy()
    local[:, 45] = 255 if unknown else 254
    if not barrier:
        local[:40, 45] = 0
        local[41:, 45] = 0
        local[:, 60:63] = 255
    origin = Namespace(
        position=Namespace(x=-2.0, y=-2.0), orientation=Namespace(x=0.0, y=0.0, z=0.0)
    )
    message = Namespace(
        data=occupancy.ravel(), info=Namespace(width=100, height=80, resolution=0.05, origin=origin)
    )
    costmap = Namespace(
        data=local.ravel(),
        metadata=Namespace(size_x=100, size_y=80, resolution=0.05, origin=origin),
    )
    robot = np.array([0.019999, 0.025])
    goal = np.array([1.5, 0.025])
    grid = Grid(
        message,
        robot,
        clearance=0.20,
        tracking_margin=0.03,
        local_navigation=costmap,
        map_from_local=[0.0, 0.0, 0.0],
    )
    return grid, robot, goal


def test_live_safe_attachment_exists_when_global_tangent_and_full_local_goal_fail():
    grid, robot, goal = scene()
    costs = RouteCosts(grid, robot)
    route = np.array([robot, goal])
    original = route.copy()
    assert grid.safe_pose(robot) and grid.safe_pose(goal)
    assert not grid.motion_path_safe(route) and costs.path(goal) is None
    proposals = list(costs.attachment_paths(route))
    assert proposals and len(proposals) <= 48
    for path, proof in proposals:
        assert grid.motion_path_safe(path)
        assert np.array_equal(path[0], robot)
        assert proof["mission_target_changed"] is False
        assert 1 <= proof["sampled_attachments"] <= proof["maximum_attachments"] == 48
        assert np.allclose(path[-1], proof["attachment_xy"])
    assert proposals[0][0][1, 0] < robot[0]
    assert np.array_equal(route, original)


@pytest.mark.parametrize("unknown", [False, True])
def test_unknown_or_occupied_separation_never_yields_a_crossing(unknown):
    grid, robot, goal = scene(barrier=True, unknown=unknown)
    assert not list(RouteCosts(grid, robot).attachment_paths([robot, goal]))


@pytest.mark.parametrize("bound", [0, 49, True, 1.5])
def test_search_budget_cannot_be_disabled_or_widened(bound):
    grid, robot, goal = scene()
    with pytest.raises(ValueError, match="1..48"):
        list(RouteCosts(grid, robot).attachment_paths([robot, goal], maximum_candidates=bound))


@pytest.mark.parametrize("route", [[[float("nan"), 0], [1, 0]], [[0, 0]], [[2, 2], [3, 3]]])
def test_nonfinite_degenerate_or_old_start_route_is_refused(route):
    grid, robot, _ = scene()
    with pytest.raises(ValueError):
        list(RouteCosts(grid, robot).attachment_paths(route))


def test_remote_out_of_bounds_intent_cannot_allocate_an_unbounded_dense_route():
    grid, robot, _ = scene()
    assert not list(RouteCosts(grid, robot).attachment_paths([robot, [1e12, 0.025]]))


def test_existing_planning_service_can_interrupt_attachment_search():
    grid, robot, goal = scene()

    class InterruptedError(RuntimeError):
        pass

    calls = []

    def service():
        calls.append(True)
        raise InterruptedError("owned task canceled")

    with pytest.raises(InterruptedError, match="owned task canceled"):
        list(RouteCosts(grid, robot).attachment_paths([robot, goal], service=service))
    assert len(calls) == 1
