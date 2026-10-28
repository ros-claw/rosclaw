"""Public transfer proposals cannot discard local obstacles or configured body geometry."""

from types import SimpleNamespace as N

import numpy as np
import pytest

from rosclaw.connectors.ros.public_coverage.grid import Grid
from rosclaw.connectors.ros.public_coverage.online import TransferUnavailableError, propose_transfer


def origin(x=0.0, y=0.0):
    return N(position=N(x=x, y=y), orientation=N(x=0.0, y=0.0, z=0.0))


def occupancy(data):
    h, w = data.shape
    return N(data=data.ravel(), info=N(width=w, height=h, resolution=0.05, origin=origin()))


def costmap(data, x=0.0, y=0.0):
    h, w = data.shape
    return N(
        data=data.ravel(), metadata=N(size_x=w, size_y=h, resolution=0.05, origin=origin(x, y))
    )


def configured_grid(message, robot, **kwargs):
    kwargs.setdefault("tracking_margin", 0.03)
    return Grid(message, robot, **kwargs)


def test_local_unknown_ring_allows_only_a_short_live_safe_prefix_of_global_intent():
    data = np.zeros((120, 120), int)
    local = np.zeros((60, 60), np.uint8)
    yy, xx = np.mgrid[:60, :60]
    radius = np.hypot((xx + 0.5) * 0.05 - 1.5, (yy + 0.5) * 0.05 - 1.5)
    local[(radius >= 0.7) & (radius <= 0.9)] = 255
    before = local.copy()
    robot, target = [2.5, 2.5], [4.8, 2.5]
    grid = Grid(
        occupancy(data), robot, local_navigation=costmap(local, 1, 1), map_from_local=[0, 0, 0]
    )
    assert grid.refresh_target(target, max_shift=0.1) is None
    assert not grid.motion_path_safe([robot, target])
    proposal = propose_transfer(grid, robot, target)
    path = np.asarray(proposal.prefix_xy)
    assert proposal.intent_source == "global_proposal_live_prefix_required"
    assert proposal.intent_length_m > 2
    assert 0.005 < np.linalg.norm(np.diff(path, axis=0), axis=1).sum() <= 0.60 + 1e-12
    assert np.allclose(path[0], robot)
    assert grid.motion_path_safe(path)
    assert np.array_equal(local, before)
    assert np.all(data == 0)


def test_available_local_detour_is_kept_when_global_map_looks_straight():
    local = np.zeros((120, 120), np.uint8)
    local[40:61, 58:63] = 254
    robot, target = [2.0, 2.5], [4.0, 2.5]
    grid = Grid(
        occupancy(np.zeros((120, 120), int)),
        robot,
        local_navigation=costmap(local),
        map_from_local=[0, 0, 0],
    )
    assert not grid.motion_path_safe([robot, target])
    proposal = propose_transfer(grid, robot, target)
    assert proposal.intent_source == "live_global_local_route"
    assert proposal.intent_length_m > 2.0
    assert grid.motion_path_safe(proposal.prefix_xy)
    assert np.linalg.norm(np.asarray(proposal.prefix_xy)[-1] - robot) <= 0.60 + 1e-12


def test_global_intent_keeps_the_configured_larger_body_and_margin():
    data = np.zeros((100, 120), int)
    data[:, 60] = 100
    data[44:56, 60] = 0  # 0.60m gap cannot admit a 0.86m planning disk.
    robot, target = [1.5, 2.5], [4.5, 2.5]
    assert (
        configured_grid(
            occupancy(data), robot, clearance=0.20, tracking_margin=0.03
        ).refresh_target(target)
        is not None
    )
    grid = configured_grid(occupancy(data), robot, clearance=0.40, tracking_margin=0.03)
    with pytest.raises(TransferUnavailableError):
        propose_transfer(grid, robot, target)


@pytest.mark.parametrize("cost", [254, 255])
def test_global_navigation_hazard_is_not_dropped_during_fallback(cost):
    navigation = np.zeros((100, 120), np.uint8)
    navigation[:, 60] = cost
    robot = [1.5, 2.5]
    grid = configured_grid(
        occupancy(np.zeros_like(navigation)), robot, navigation=costmap(navigation)
    )
    with pytest.raises(TransferUnavailableError):
        propose_transfer(grid, robot, [4.5, 2.5])
    assert np.all(navigation[:, 60] == cost)


def test_work_snap_and_explicit_original_return_snap_are_separate_contracts():
    robot, target = [1.0, 2.0], [0.205, 2.0]
    grid = configured_grid(occupancy(np.zeros((100, 100), int)), robot)
    with pytest.raises(TransferUnavailableError):
        propose_transfer(grid, robot, target)
    proposal = propose_transfer(grid, robot, target, max_target_shift=0.10)
    assert 0.05 < proposal.target_shift_m <= 0.10
    assert grid.motion_path_safe(proposal.prefix_xy)
    assert proposal.requested_target_xy == tuple(target)
    assert proposal.selected_target_xy != tuple(target)


@pytest.mark.parametrize("limit", [-0.01, 0.100001, float("nan"), float("inf")])
def test_invalid_or_expanded_target_shift_is_rejected(limit):
    robot = [1.0, 2.0]
    grid = configured_grid(occupancy(np.zeros((100, 100), int)), robot)
    with pytest.raises(TransferUnavailableError):
        propose_transfer(grid, robot, [2, 2], max_target_shift=limit)


@pytest.mark.parametrize("target", [[float("nan"), 2], [2], [2, 2, 2]])
def test_missing_finite_target_cannot_become_a_default_goal(target):
    robot = [1.0, 2.0]
    grid = configured_grid(occupancy(np.zeros((100, 100), int)), robot)
    with pytest.raises(TransferUnavailableError):
        propose_transfer(grid, robot, target)


def test_no_translation_does_not_claim_arrival_or_task_completion():
    robot = [1.0, 2.0]
    grid = configured_grid(occupancy(np.zeros((100, 100), int)), robot)
    with pytest.raises(TransferUnavailableError, match="No transfer translation"):
        propose_transfer(grid, robot, robot)


def test_global_proposal_uses_original_snapshot_even_if_callers_mutate_their_message():
    data = np.zeros((100, 120), int)
    data[:, 60] = 100
    original = occupancy(data)
    robot = [1.5, 2.5]
    grid = configured_grid(original, robot)
    original.data[:] = 0
    with pytest.raises(TransferUnavailableError):
        propose_transfer(grid, robot, [4.5, 2.5])


def test_a_changed_actual_pose_cannot_borrow_the_old_safe_start():
    data = np.zeros((100, 100), int)
    data[40:45, 40:45] = 100
    grid = configured_grid(occupancy(data), [1.0, 1.0])
    with pytest.raises(TransferUnavailableError, match="Actual public full footprint"):
        propose_transfer(grid, [2.1, 2.1], [3.5, 3.5])


def test_original_short_physical_safe_reentry_still_uses_live_constraints():
    actual = [0.205, 2.0]
    grid = configured_grid(occupancy(np.zeros((100, 100), int)), actual)
    assert grid.physical_pose_safe(actual) and not grid.safe_pose(actual)
    proposal = propose_transfer(grid, actual, [1.0, 2.0])
    assert np.allclose(proposal.prefix_xy[0], actual)
    assert grid.motion_path_safe(proposal.prefix_xy)
