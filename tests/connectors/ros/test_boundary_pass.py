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
