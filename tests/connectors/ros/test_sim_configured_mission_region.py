"""Whole-area authorization contracts; no physical action or held-out asset."""

from copy import deepcopy

import pytest

from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor

POLYGON = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]
GRID = {
    "width": 3,
    "height": 3,
    "resolution": 1,
    "origin": [0, 0],
    "frame_id": "world",
    "accessible_cells": [0, 1, 2, 3, 6],
    "cleaning_polygon": [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
}


def executor(tmp_path, *, polygon=None, grid=None, boundary=False, centers=()):
    return RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=None,
        output=tmp_path,
        body_id="anonymous",
        body_snapshot_hash="hash",
        grid=deepcopy(grid or GRID),
        mission_polygon=deepcopy(POLYGON) if polygon is None else polygon,
        boundary_pass=boundary,
        recovery_centers=centers,
    )


def args(polygon):
    return {
        "frame_id": "world",
        "polygons": [{"points": [{"x": x, "y": y, "z": 0} for x, y in polygon]}],
    }


@pytest.mark.parametrize(
    "reverse,rotate,closed",
    [(False, 0, False), (True, 0, True), (False, 2, True), (True, 4, False)],
)
def test_entire_arbitrary_area_canonicalized_without_denominator_change(
    tmp_path, reverse, rotate, closed
):
    polygon = deepcopy(POLYGON)
    configured = executor(tmp_path, polygon=polygon)
    polygon[0][0] = 999  # Caller mutation cannot rewrite the configured allowed area.
    requested = deepcopy(POLYGON[::-1] if reverse else POLYGON)
    requested = requested[rotate:] + requested[:rotate]
    if closed:
        requested.append(requested[0])
    before = deepcopy(configured.grid)
    result = configured._coverage_goal(args(requested))
    assert result == args(POLYGON + [POLYGON[0]])
    assert (
        configured.grid == before
        and configured.grid["accessible_cells"] == GRID["accessible_cells"]
    )


@pytest.mark.parametrize(
    "fault",
    ["subset", "bbox", "crossing", "foreign_frame", "nonplanar", "nan", "extra_polygon", "empty"],
)
def test_action_cannot_change_whole_area_or_frame(tmp_path, fault):
    configured = executor(tmp_path)
    action = args(POLYGON)
    if fault == "subset":
        action = args([[0, 0], [1, 0], [1, 1], [0, 1]])
    elif fault == "bbox":
        action = args([[0, 0], [3, 0], [3, 3], [0, 3]])
    elif fault == "crossing":
        action = args([POLYGON[i] for i in [0, 2, 1, 3, 4, 5]])
    elif fault == "foreign_frame":
        action["frame_id"] = "map"
    elif fault == "nonplanar":
        action["polygons"][0]["points"][0]["z"] = 0.01
    elif fault == "nan":
        action["polygons"][0]["points"][0]["x"] = float("nan")
    elif fault == "extra_polygon":
        action["polygons"].append(deepcopy(action["polygons"][0]))
    else:
        action["polygons"][0]["points"] = []
    with pytest.raises(ValueError):
        configured._coverage_goal(action)


@pytest.mark.parametrize(
    "fault", ["outside_denominator", "self_crossing", "too_many", "huge", "rectangular_boundary"]
)
def test_invalid_configured_region_refused_without_shrinking_grid(tmp_path, fault):
    polygon, grid = deepcopy(POLYGON), deepcopy(GRID)
    boundary = False
    if fault == "outside_denominator":
        grid["accessible_cells"].append(8)
    elif fault == "self_crossing":
        polygon = [[0, 0], [3, 3], [0, 3], [3, 0]]
    elif fault == "too_many":
        polygon *= 20
    elif fault == "huge":
        polygon[0][0] = 10**400
    else:
        boundary = True
    before = deepcopy(grid)
    with pytest.raises(ValueError):
        executor(tmp_path, polygon=polygon, grid=grid, boundary=boundary)
    assert grid == before


@pytest.mark.parametrize("center", [(2.5, 2.5), (10**400, 0)])
def test_repair_centers_cannot_leave_approved_region(tmp_path, center):
    with pytest.raises(ValueError, match="repair centers"):
        executor(tmp_path, centers=[center])
