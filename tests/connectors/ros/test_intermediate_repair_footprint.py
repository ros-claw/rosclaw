"""Intermediate BT pruning does not establish requested cleaning yaw."""

import math

import pytest

from rosclaw.connectors.ros.mission import repair_optimizer as optimizer
from rosclaw.connectors.ros.verification.coverage import point_in_polygon


def test_intermediate_predictions_fit_rotated_brush_with_pruning_center_offsets():
    polygon = [[-0.275, -0.275], [0.275, -0.275], [0.275, 0.275], [-0.275, 0.275]]
    offsets = optimizer._intermediate_prediction_offsets(polygon, 0.05, 0.1)
    assert offsets and max(math.hypot(x * 0.05, y * 0.05) for x, y in offsets) <= 0.175000001
    for yaw in [i * math.pi / 12 for i in range(24)]:
        for shift_angle in [i * math.pi / 4 for i in range(8)]:
            sx, sy = 0.1 * math.cos(shift_angle), 0.1 * math.sin(shift_angle)
            for x, y in offsets:
                dx, dy = x * 0.05 - sx, y * 0.05 - sy
                assert point_in_polygon(
                    dx * math.cos(yaw) + dy * math.sin(yaw),
                    -dx * math.sin(yaw) + dy * math.cos(yaw),
                    polygon,
                )


@pytest.mark.parametrize(
    "polygon",
    [
        [[-0.04, -0.04], [0.04, -0.04], [0.04, 0.04], [-0.04, 0.04]],
        [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
    ],
)
def test_no_intermediate_prediction_when_pruning_disk_or_offset_brush_has_no_support(polygon):
    assert optimizer._intermediate_prediction_offsets(polygon, 0.05, 0.1) == ()


def test_tracking_does_not_add_an_intermediate_target_with_no_supported_gain():
    grid = {
        "width": 3,
        "height": 1,
        "resolution": 0.1,
        "origin": [0, 0],
        "accessible_cells": [0, 1, 2],
        "cleaning_polygon": [[-0.04, -0.04], [0.04, -0.04], [0.04, 0.04], [-0.04, 0.04]],
    }
    args = (
        grid,
        [(0.05, 0.05), (0.15, 0.05), (0.25, 0.05)],
        {1, 2},
        {"x": 0.05, "y": 0.05, "yaw": 0.0},
    )
    previous = optimizer.rank_repair_poses(
        *args, goal_overhead_sec=10, shared_sequence_overhead=True
    )
    conservative = optimizer.rank_repair_poses(
        *args,
        goal_overhead_sec=10,
        shared_sequence_overhead=True,
        intermediate_tracking_radius_m=0.1,
    )
    assert previous.status == conservative.status == "READY"
    assert len(previous.poses) == 2 and len(conservative.poses) == 1
    assert conservative.poses[0] == previous.poses[0]
    assert "INTERMEDIATE_INSCRIBED" in conservative.reward_model


@pytest.mark.parametrize("radius", [True, 1, 0.0, -0.1, float("nan"), float("inf"), "0.1"])
def test_intermediate_prediction_requires_explicit_finite_float_radius(radius):
    with pytest.raises(ValueError, match="intermediate tracking"):
        optimizer.rank_repair_poses(
            {},
            [],
            set(),
            {"x": 0.0, "y": 0.0, "yaw": 0.0},
            shared_sequence_overhead=True,
            intermediate_tracking_radius_m=radius,
        )


def test_intermediate_prediction_cannot_silently_enable_a_legacy_dispatch_mode():
    with pytest.raises(ValueError, match="intermediate tracking"):
        optimizer.rank_repair_poses(
            {}, [], set(), {"x": 0.0, "y": 0.0, "yaw": 0.0}, intermediate_tracking_radius_m=0.1
        )


def test_intermediate_search_keeps_supported_centers_hidden_by_nominal_footprint_deduplication():
    grid = {
        "width": 10,
        "height": 1,
        "resolution": 0.1,
        "origin": [0, 0],
        "accessible_cells": list(range(10)),
        "cleaning_polygon": [[-0.16, -0.16], [0.16, -0.16], [0.16, 0.16], [-0.16, 0.16]],
    }
    result = optimizer.rank_repair_poses(
        grid,
        [(i * 0.1 + 0.05, 0.05) for i in range(10)],
        {2, 8},
        {"x": 0.05, "y": 0.05, "yaw": 0.0},
        goal_overhead_sec=10,
        shared_sequence_overhead=True,
        intermediate_tracking_radius_m=0.1,
    )
    assert result.status == "READY" and len(result.poses) == 2
    first, final = result.poses
    assert first.center_cell in (2, 8)
    assert first.predicted_new_cells == (first.center_cell,)
    assert set(final.predicted_new_cells) - set(first.predicted_new_cells)


@pytest.mark.parametrize("robust,model", [(False, "NOMINAL"), (True, "NINE_TRANSLATIONS")])
def test_intermediate_reward_diagnostic_names_only_the_scenarios_actually_enabled(robust, model):
    result = optimizer.rank_repair_poses(
        {
            "width": 1,
            "height": 1,
            "resolution": 0.1,
            "origin": [0, 0],
            "accessible_cells": [0],
            "cleaning_polygon": [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]],
        },
        [],
        set(),
        {"x": 0.0, "y": 0.0, "yaw": 0.0},
        shared_sequence_overhead=True,
        robust_footprint=robust,
        intermediate_tracking_radius_m=0.1,
    )
    assert result.status == "NO_CANDIDATE"
    assert f"RADIUS_{model}_FINAL" in result.reward_model
