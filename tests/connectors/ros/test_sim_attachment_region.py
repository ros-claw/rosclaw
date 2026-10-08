"""Explicit SIM cleaner and non-origin, nonrectangular allowed task regions."""

import copy

import pytest

from rosclaw.connectors.ros.context.sim_attachment import (
    derive_allowed_region_grid,
    validate_sim_attachment,
)
from rosclaw.connectors.ros.verification.coverage import point_in_polygon


def attachment():
    return {
        "schema_version": "rosclaw.sim_cleaning_attachment.v1",
        "evidence_domain": "SIMULATION",
        "kind": "SIMULATED_CLEANING",
        "declaration_source": "owned_fixture_policy_test",
        "cleaning_polygon": [[-0.15, -0.2], [0.25, -0.2], [0.25, 0.2], [-0.15, 0.2]],
    }


def example():
    measured = {
        "width": 24,
        "height": 24,
        "resolution": 0.1,
        "origin": [4, -3],
        "origin_orientation": [0, 0, 0, 1],
        "frame_id": "floor_frame",
        "occupancy": [0] * 576,
    }
    args = {
        "allowed_polygon": [
            (4.2, -2.8),
            (6, -2.8),
            (6, -1.8),
            (5.2, -1.8),
            (5.2, -0.8),
            (4.2, -0.8),
        ],
        "allowed_frame_id": "floor_frame",
        "attachment": attachment(),
        "physical_radius_m": 0.1,
        "start_pose": {
            "x": 4.65,
            "y": -2.35,
            "frame_id": "floor_frame",
            "observation_complete": True,
            "source": "independent_gazebo_ground_truth_subscription",
        },
    }
    return measured, args


def test_cleaner_requires_declaration_and_grants_no_real_execution():
    result = validate_sim_attachment(attachment())
    assert result["inscribed_radius_m"] == pytest.approx(0.15)
    assert result["kind"] == "SIMULATED_CLEANING"
    assert not result["usable_for_real_execution"]
    assert result["attachment_hash"]


@pytest.mark.parametrize(
    "fault", ["missing_declaration", "real", "inferred", "self_crossing", "offset", "nonfinite"]
)
def test_invalid_attachment_cannot_define_task_semantics(fault):
    declaration = attachment()
    if fault == "missing_declaration":
        declaration.pop("declaration_source")
    elif fault == "real":
        declaration["evidence_domain"] = "HARDWARE"
    elif fault == "inferred":
        declaration["kind"] = "drive_inferred_cleaner"
    elif fault == "self_crossing":
        declaration["cleaning_polygon"] = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    elif fault == "offset":
        declaration["cleaning_polygon"] = [[1, 1], [2, 1], [2, 2], [1, 2]]
    else:
        declaration["cleaning_polygon"][0][0] = float("nan")
    with pytest.raises(ValueError):
        validate_sim_attachment(declaration)


def test_l_shaped_shifted_region_uses_actual_spawn_and_preserves_input():
    measured, args = example()
    before = copy.deepcopy((measured, args))
    result = derive_allowed_region_grid(measured, **args)
    assert (measured, args) == before
    assert result == derive_allowed_region_grid(measured, **args)
    assert result["denominator_cells"] > 100
    assert result["capabilities_granted"] == [] and not result["usable_for_real_execution"]
    assert set(result["legal_center_cells"]) <= set(result["grid"]["accessible_cells"])
    for i in result["grid"]["accessible_cells"]:
        x = 4 + (i % 24 + 0.5) * 0.1
        y = -3 + (i // 24 + 0.5) * 0.1
        assert point_in_polygon(x, y, args["allowed_polygon"])
        assert not (x > 5.2 and y > -1.8)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_origin",
        "rotated_origin",
        "region_frame",
        "pose_frame",
        "incomplete_pose",
        "nav_localization",
        "outside_spawn",
        "occupied_spawn",
    ],
)
def test_unknown_coordinate_or_spawn_evidence_is_rejected(fault):
    measured, args = example()
    if fault == "unknown_origin":
        measured.pop("origin_orientation")
    elif fault == "rotated_origin":
        measured["origin_orientation"] = [0, 0, 0.7, 0.7]
    elif fault == "region_frame":
        args["allowed_frame_id"] = "other"
    elif fault == "pose_frame":
        args["start_pose"]["frame_id"] = "other"
    elif fault == "incomplete_pose":
        args["start_pose"]["observation_complete"] = False
    elif fault == "nav_localization":
        args["start_pose"]["source"] = "amcl_pose"
    elif fault == "outside_spawn":
        args["start_pose"]["x"] = 10
    else:
        measured["occupancy"] = [100] * 576
    with pytest.raises(ValueError):
        derive_allowed_region_grid(measured, **args)
