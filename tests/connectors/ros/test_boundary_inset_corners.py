"""Inset corner candidates cannot expand legal centers or weaken action gates."""

import copy
import hashlib
import math
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.mission.boundary_pass import (
    inset_corner_boundary_targets,
    rectangular_boundary_targets,
)
from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor


def centers():
    return [(round(x / 10, 1), round(y / 10, 1)) for x in range(-3, 4) for y in range(-3, 4)]


def test_corner_inset_keeps_edge_midpoints_legal_domain_closure_and_inputs():
    legal = centers()
    current = {"x": 0.29, "y": 0.29}
    before = copy.deepcopy((legal, current))
    old = rectangular_boundary_targets(legal, current, edge_midpoints=True)
    new = inset_corner_boundary_targets(legal, current, resolution=0.1)
    assert len(new) == 9 and new[0] == new[-1]
    assert (new[0]["x"], new[0]["y"]) == (0.2, 0.2)
    assert all((p["x"], p["y"]) in legal and math.isfinite(p["yaw"]) for p in new)
    assert [(p["x"], p["y"]) for p in new[1:-1:2]] == [(p["x"], p["y"]) for p in old[1:-1:2]]
    assert all(abs(p["x"]) == abs(p["y"]) == 0.2 for p in new[:-1:2])
    assert (legal, current) == before
    assert rectangular_boundary_targets(legal, current, edge_midpoints=True) == old


@pytest.mark.parametrize("bad", [0, -0.1, True, float("nan"), float("inf"), "0.1"])
def test_invalid_corner_grid_resolution_is_refused(bad):
    with pytest.raises(ValueError, match="resolution"):
        inset_corner_boundary_targets(centers(), {"x": 0, "y": 0}, resolution=bad)


@pytest.mark.parametrize("case", ["missing_corner", "missing_column", "too_small", "empty"])
def test_missing_or_irregular_legal_domain_produces_no_corner_candidate(case):
    legal = centers()
    if case == "missing_corner":
        legal.remove((0.2, 0.2))
    elif case == "missing_column":
        legal = [p for p in legal if p[0] != 0]
    elif case == "too_small":
        legal = [(x / 10, y / 10) for x in range(-1, 2) for y in range(-1, 2)]
    else:
        legal = []
    assert inset_corner_boundary_targets(legal, {"x": 0, "y": 0}, resolution=0.1) == ()


def test_corner_candidate_dispatch_retains_daemon_action_deadline_and_timeout(tmp_path):
    (tmp_path / "boundary-through-poses.xml").write_bytes(b"owned fixture BT bytes")
    executor = RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=SimpleNamespace(fresh=lambda: {"x": 0.29, "y": 0.29}),
        output=tmp_path / "actions",
        body_id="fixture",
        body_snapshot_hash="body",
        grid={"frame_id": "map", "resolution": 0.1},
        recovery_centers=centers(),
        boundary_pass=True,
        boundary_strategy="through_poses_tracking_inset_corners",
        boundary_tracking_bt_sha256=hashlib.sha256(b"owned fixture BT bytes").hexdigest(),
    )
    calls = []
    terminal = {"status": 5, "timed_out": True, "result": {"error_code": 0}}
    executor._run_goal = lambda *args, **kwargs: calls.append((args, kwargs)) or terminal
    result = executor._boundary({"status": 4}, "root", 123)
    assert result["status"] == "FAILED" and result["nav_goal_result"] is terminal
    args, kwargs = calls[0]
    assert args[:2] == ("/navigate_through_poses", "nav2_msgs/action/NavigateThroughPoses")
    assert args[3:] == ("root:boundary", 123)
    assert kwargs == {"goal_timeout_sec": 180, "stage": "BOUNDARY_PASS"}
    assert args[2]["behavior_tree"] == "/evidence/boundary-through-poses.xml"
    assert len(args[2]["poses"]) == result["waypoint_count"] == 9
    assert all(
        (p["pose"]["position"]["x"], p["pose"]["position"]["y"]) in centers()
        for p in args[2]["poses"]
    )
    assert "coverage_ratio" not in result
    assert executor._boundary({"status": 6}, "root", 123)["status"] == "SKIPPED"
    executor.boundary_centers = tuple(p for p in centers() if p != (0.2, 0.2))
    assert executor._boundary({"status": 4}, "root", 123)["status"] == "SKIPPED"
    executor.boundary_pass = False
    assert executor._boundary({"status": 4}, "root", 123)["status"] == "DISABLED"
    assert len(calls) == 1


@pytest.mark.parametrize("bad", [[(True, 0)], [(float("nan"), 0)], [("0", 0)], [(0,)]])
def test_invalid_legal_corner_coordinates_are_refused(bad):
    with pytest.raises(ValueError, match="finite original"):
        inset_corner_boundary_targets(bad, {"x": 0, "y": 0}, resolution=0.1)
