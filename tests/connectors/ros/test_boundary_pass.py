"""Boundary coordination cannot invent legal targets or grant predicted credit."""

import copy

import pytest

from rosclaw.connectors.ros.mission.boundary_pass import rectangular_boundary_targets


def test_ring_uses_existing_corners_closes_and_starts_near_observed_pose():
    centers = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0), (0.0, 0.0)]
    current = {"x": 0.8, "y": 0.9}
    before = copy.deepcopy((centers, current))
    targets = rectangular_boundary_targets(centers, current)
    assert len(targets) == 5
    assert (targets[0]["x"], targets[0]["y"]) == (1.0, 1.0)
    assert targets[0] == targets[-1]
    assert all((p["x"], p["y"]) in centers for p in targets)
    assert (centers, current) == before


def test_missing_corner_or_degenerate_map_produces_no_boundary_action():
    assert rectangular_boundary_targets([(-1, -1), (-1, 1), (1, -1)], {"x": 0, "y": 0}) == ()
    assert rectangular_boundary_targets([(0, 0), (0, 1)], {"x": 0, "y": 0}) == ()
    assert rectangular_boundary_targets([], {"x": 0, "y": 0}) == ()
    with pytest.raises(ValueError, match="finite"):
        rectangular_boundary_targets([(float("nan"), 0), (1, 1)], {"x": 0, "y": 0})


@pytest.mark.parametrize("nav_status, expected", [(4, "SUCCEEDED"), (6, "FAILED")])
def test_boundary_dispatch_is_daemon_owned_bounded_and_never_grants_credit(
    tmp_path, nav_status, expected
):
    from types import SimpleNamespace

    from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor

    witness = SimpleNamespace(fresh=lambda: {"x": 0.9, "y": 0.9})
    executor = RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=witness,
        output=tmp_path,
        body_id="fixture",
        body_snapshot_hash="body",
        grid={"frame_id": "map"},
        recovery_centers=[(-1, -1), (-1, 1), (1, -1), (1, 1)],
        boundary_pass=True,
    )
    calls = []
    executor._run_goal = lambda *args, **kwargs: (
        calls.append((args, kwargs)) or {"status": nav_status}
    )
    result = executor._boundary({"status": 4}, "root", 123)
    assert result["status"] == expected
    args, kwargs = calls[0]
    assert args[:2] == ("/navigate_through_poses", "nav2_msgs/action/NavigateThroughPoses")
    assert args[3:] == ("root:boundary", 123)
    assert kwargs == {"goal_timeout_sec": 180, "stage": "BOUNDARY_PASS"}
    assert len(args[2]["poses"]) == result["waypoint_count"] == 5
    assert "coverage_ratio" not in result
    assert executor._boundary({"status": 6}, "root", 123)["status"] == "SKIPPED"
    assert len(calls) == 1
    executor.boundary_pass = False
    assert executor._boundary({"status": 4}, "root", 123)["status"] == "DISABLED"
    assert len(calls) == 1


def test_midpoints_remain_existing_legal_centers_and_require_every_edge():
    centers = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]
    targets = rectangular_boundary_targets(centers, {"x": 1, "y": 1}, edge_midpoints=True)
    assert len(targets) == 9
    assert targets[0] == targets[-1]
    assert all((p["x"], p["y"]) in centers for p in targets)
    assert rectangular_boundary_targets(centers[:-1], {"x": 0, "y": 0}, edge_midpoints=True) == ()


@pytest.mark.parametrize("failure", [False, True, "budget"])
def test_sequential_boundary_audits_each_goal_and_stops_on_failure_or_budget(
    tmp_path, monkeypatch, failure
):
    from types import SimpleNamespace

    from rosclaw.connectors.ros.mission import executor as module

    executor = module.RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=SimpleNamespace(fresh=lambda: {"x": 1, "y": 1}),
        output=tmp_path,
        body_id="fixture",
        body_snapshot_hash="body",
        grid={"frame_id": "map"},
        recovery_centers=[(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)],
        boundary_pass=True,
        boundary_strategy="sequential",
    )
    calls = []
    executor._run_goal = lambda *args, **kwargs: (
        calls.append((args, kwargs)) or {"status": 6 if failure is True else 4}
    )
    if failure == "budget":
        stamps = iter([0, 0, 181])
        monkeypatch.setattr(module.time, "monotonic", lambda: next(stamps))
    result = executor._boundary({"status": 4}, "root", 1000)
    assert result["status"] == ("FAILED" if failure else "SUCCEEDED")
    assert result["waypoint_count"] == 9
    assert len(calls) == (1 if failure else 9)
    for index, (args, kwargs) in enumerate(calls):
        assert args[:2] == ("/navigate_to_pose", "nav2_msgs/action/NavigateToPose")
        assert args[3:] == (f"root:boundary:{index}", 1000)
        assert kwargs["stage"] == "BOUNDARY_PASS"
        assert 0 < kwargs["goal_timeout_sec"] <= 45
    assert "coverage_ratio" not in result


def test_inset_ring_uses_complete_existing_inner_edges_without_mutating_mask():
    from rosclaw.connectors.ros.mission.boundary_pass import inset_rectangular_boundary_targets

    centers = [(float(x), float(y)) for x in range(7) for y in range(7)]
    before = copy.deepcopy(centers)
    targets = inset_rectangular_boundary_targets(centers, {"x": 6, "y": 6}, resolution=1)
    assert len(targets) == 9 and targets[0] == targets[-1]
    assert all(
        (p["x"], p["y"]) in centers and 1 <= p["x"] <= 5 and 1 <= p["y"] <= 5 for p in targets
    )
    assert centers == before
    missing = [p for p in centers if p != (3, 1)]
    assert inset_rectangular_boundary_targets(missing, {"x": 6, "y": 6}, resolution=1) == ()
    assert inset_rectangular_boundary_targets(centers, {"x": 6, "y": 6}, resolution=0.5) == ()


@pytest.mark.parametrize(
    "inset,resolution", [(True, 1), (0, 1), (5, 1), (1.5, 1), (1, True), (1, 0), (1, float("nan"))]
)
def test_inset_ring_rejects_unregistered_or_nonfinite_grid_parameters(inset, resolution):
    from rosclaw.connectors.ros.mission.boundary_pass import inset_rectangular_boundary_targets

    with pytest.raises(ValueError):
        inset_rectangular_boundary_targets(
            [(0, 0)], {"x": 0, "y": 0}, resolution=resolution, inset_cells=inset
        )


@pytest.mark.parametrize("failure", [False, True, "budget", "thin"])
def test_extra_ring_remains_bounded_daemon_goals_and_never_credits_cells(
    tmp_path, monkeypatch, failure
):
    from types import SimpleNamespace

    from rosclaw.connectors.ros.mission import executor as module

    extent = 3 if failure == "thin" else 7
    centers = [(float(x), float(y)) for x in range(extent) for y in range(extent)]
    executor = module.RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=SimpleNamespace(fresh=lambda: {"x": extent - 1, "y": extent - 1}),
        output=tmp_path,
        body_id="fixture",
        body_snapshot_hash="body",
        grid={"frame_id": "map", "resolution": 1},
        recovery_centers=centers,
        boundary_centers=centers,
        boundary_pass=True,
        boundary_strategy="sequential_inner_ring",
    )
    calls = []

    def fake_goal(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": 6 if failure is True and len(calls) == 10 else 4}

    executor._run_goal = fake_goal
    if failure == "budget":
        stamps = iter([0, *([1] * 9), 361])
        monkeypatch.setattr(module.time, "monotonic", lambda: next(stamps))
    result = executor._boundary({"status": 4}, "root", 1000)
    assert result["status"] == ("SUCCEEDED" if failure is False else "FAILED")
    assert len(calls) == (
        18 if failure is False else 10 if failure is True else 9 if failure == "budget" else 0
    )
    assert "coverage_ratio" not in result
    for index, (args, kwargs) in enumerate(calls):
        assert args[:2] == ("/navigate_to_pose", "nav2_msgs/action/NavigateToPose")
        assert args[3:] == (f"root:boundary:{index}", 1000)
        pose = args[2]["pose"]["pose"]["position"]
        assert (pose["x"], pose["y"]) in centers
        assert kwargs["stage"] == "BOUNDARY_PASS" and 0 < kwargs["goal_timeout_sec"] <= 45
