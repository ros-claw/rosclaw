"""Pure regressions for finite unit-field admission; no execution services."""

import copy

import pytest
from pydantic import ValidationError

from rosclaw.contracts.agent.pose_trajectory import (
    ContactPlaneV1,
    PoseTrajectorySpecV1,
    PoseWaypointV1,
)


def _payload():
    return {
        "schema_version": "rosclaw.pose_trajectory_spec.v1",
        "frame_id": "world",
        "tool_frame": "attachment_site",
        "contact_plane": {"normal_xyz": [0, 1, 0]},
        "waypoints": [
            {
                "position_m": [0.35, 0.3, 0.2],
                "orientation_xyzw": [1, 0, 0, 0],
                "kind": "contact",
            }
        ],
        "digest": "sha256:x",
    }


def _parse(field, value, route):
    payload = _payload()
    if field == "normal_xyz":
        data = payload["contact_plane"]
        model = ContactPlaneV1
    else:
        data = payload["waypoints"][0]
        model = PoseWaypointV1
    data[field] = list(value)
    if route == "constructor":
        return model(**data)
    if route == "model_validate":
        return model.model_validate(data)
    return PoseTrajectorySpecV1.model_validate_contract(payload)


def _assert_value(parsed, field, value, route):
    if route == "nested":
        parsed = parsed.contact_plane if field == "normal_xyz" else parsed.waypoints[0]
    actual = parsed.normal_xyz if field == "normal_xyz" else parsed.orientation_xyzw
    # Exact equality catches silent normalization/clipping of accepted values.
    assert actual == list(value)


def _reject_and_recover(field, value):
    good = [0, 1, 0] if field == "normal_xyz" else [1, 0, 0, 0]
    for route in ("constructor", "model_validate", "nested"):
        with pytest.raises(ValidationError) as caught:
            _parse(field, value, route)
        expected = (field,)
        if route == "nested":
            expected = (
                ("contact_plane", field) if field == "normal_xyz" else ("waypoints", 0, field)
            )
        assert [error["loc"] for error in caught.value.errors()] == [expected]
        # A rejection must not poison the next genuine validation.
        recovered = _parse(field, good, route)
        _assert_value(recovered, field, good, route)


def test_nonfinite_each_component_and_finite_overflow():
    for field, size in (("normal_xyz", 3), ("orientation_xyzw", 4)):
        for index in range(size):
            for invalid in (float("nan"), float("inf"), -float("inf"), 1e308, -1e308):
                value = [0.0] * size
                value[index] = invalid
                _reject_and_recover(field, value)
        # NaN must not hide behind an otherwise legitimate unit component.
        for index in range(size):
            value = [0.0] * size
            value[(index + 1) % size] = 1.0
            value[index] = float("nan")
            _reject_and_recover(field, value)


def test_old_invalid_lengths_zero_nonunit_and_tolerance():
    for field, size in (("normal_xyz", 3), ("orientation_xyzw", 4)):
        for value in (
            [],
            [1.0] + [0.0] * (size - 2),
            [1.0] + [0.0] * size,
            [0.0] * size,
            [2.0] + [0.0] * (size - 1),
            [0.998] + [0.0] * (size - 1),
            [1.002] + [0.0] * (size - 1),
        ):
            _reject_and_recover(field, value)


def test_legal_axes_signs_oblique_and_tolerance_not_normalized():
    for field, size in (("normal_xyz", 3), ("orientation_xyzw", 4)):
        values = []
        for index in range(size):
            for sign in (-1, 1):
                value = [0] * size
                value[index] = sign
                values.append(value)
        values.extend(
            [
                [0.6, 0.8] + [0.0] * (size - 2),
                [-0.6, -0.8] + [0.0] * (size - 2),
                [0.9995] + [0.0] * (size - 1),
                [1.0005] + [0.0] * (size - 1),
            ]
        )
        for value in values:
            for route in ("constructor", "model_validate", "nested"):
                _assert_value(_parse(field, value, route), field, value, route)


def test_legacy_defaults_optional_fields_and_forward_extras():
    payload = _payload()
    legacy = PoseTrajectorySpecV1.model_validate_contract(payload)
    assert legacy.contact_plane.offset_m == 0.0
    assert legacy.geometry is None
    assert legacy.plane_pose is None
    assert legacy.tool_pose_constraint is None
    assert legacy.sampling is None
    assert legacy.timing is None
    assert legacy.schema_version == "rosclaw.pose_trajectory_spec.v1"
    assert legacy.contact_plane.schema_version == "rosclaw.contact_plane.v1"
    assert legacy.waypoints[0].schema_version == "rosclaw.pose_waypoint.v1"

    payload.update(
        {
            "geometry": "shape",
            "plane_pose": {"position_m": [0, 0.3, 0], "orientation_xyzw": [1, 0, 0, 0]},
            "tool_pose_constraint": {"axis": "tool_z", "tolerance_deg": 5.0},
            "sampling": {"max_segment_m": 0.02},
            "timing": {"speed_mps": 0.05, "accel_mps2": 0.2},
            "future_spec": {"revision": 2},
        }
    )
    payload["contact_plane"]["future_plane"] = ["kept"]
    payload["contact_plane"]["offset_m"] = 0.3
    payload["waypoints"][0]["future_waypoint"] = {"kept": True}
    # All frozen waypoint kinds remain legal in a whole nested trajectory.
    waypoint = payload["waypoints"][0]
    payload["waypoints"] = []
    for kind in ("transit", "approach", "contact", "lift"):
        item = copy.deepcopy(waypoint)
        item["kind"] = kind
        payload["waypoints"].append(item)
    parsed = PoseTrajectorySpecV1.model_validate_contract(payload)
    dumped = parsed.model_dump()
    for key, value in payload.items():
        if key not in ("contact_plane", "waypoints"):
            assert dumped[key] == value
    for key, value in payload["contact_plane"].items():
        assert dumped["contact_plane"][key] == value
    for actual, expected in zip(dumped["waypoints"], payload["waypoints"], strict=False):
        for key, value in expected.items():
            assert actual[key] == value
    assert len(dumped["waypoints"]) == 4
    assert PoseTrajectorySpecV1.model_validate_contract(dumped).model_dump() == dumped
