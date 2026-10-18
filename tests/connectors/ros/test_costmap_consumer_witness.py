"""Actual grid-cell semantics, no ROS imports or physical acceptance in tests."""

import importlib.util
import math
from pathlib import Path

import pytest


@pytest.fixture
def inspect():
    path = (
        Path(__file__).resolve().parents[3]
        / "integrations/ros_probe/acceptance/costmap_consumer_witness.py"
    )
    spec = importlib.util.spec_from_file_location("grid_witness_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inspect_grid


def grid():
    return {
        "frame_id": "map",
        "info": {"width": 3, "height": 2, "resolution": 0.5, "origin": [-1.0, -2.0, 0.0]},
        "data": [0, 100, -1, 20, 30, 40],
    }


def point(x, y, frame="map"):
    return {"frame": frame, "x": x, "y": y}


def test_row_major_cells_boundaries_unknown_values_and_no_frame_guess(inspect):
    rows = inspect(
        grid(),
        [
            point(-0.25, -1.75),
            point(0.25, -1.75),
            point(-1.01, -1.75),
            point(0.5, -1.75),
            point(-0.25, -1.75, "odom"),
        ],
    )
    assert rows[0]["cell"] == [1, 0] and rows[0]["value"] == 100
    assert rows[1]["cell"] == [2, 0] and rows[1]["value"] == -1
    assert rows[2]["status"] == rows[3]["status"] == "OUTSIDE_GRID"
    assert rows[4] == {
        "point": point(-0.25, -1.75, "odom"),
        "status": "UNKNOWN_FRAME_TRANSFORM_NOT_PROVIDED",
    }


def test_rotated_grid_origin_is_not_treated_as_axis_aligned(inspect):
    original = grid()
    original["info"]["origin"] = [0.0, 0.0, math.pi / 2]
    rows = inspect(original, [point(-0.25, 0.75)])
    assert rows[0]["cell"] == [1, 0] and rows[0]["value"] == 100


@pytest.mark.parametrize(
    "kind", ["negative", "missing_cell", "range", "nan", "zero_resolution", "bool_width"]
)
def test_malformed_grid_never_fabricates_an_occupancy_result(inspect, kind):
    value = grid()
    if kind == "negative":
        value["info"].update(width=-3, height=-2)
    elif kind == "missing_cell":
        value["data"].pop()
    elif kind == "range":
        value["data"][0] = 101
    elif kind == "nan":
        value["info"]["resolution"] = math.nan
    elif kind == "zero_resolution":
        value["info"]["resolution"] = 0
    elif kind == "bool_width":
        value["info"]["width"] = True
    with pytest.raises(ValueError):
        inspect(value, [point(0.0, 0.0)])


@pytest.mark.parametrize(
    "case,node,key,value",
    [
        ("rsk_tf_future_stamp", "/unfamiliar/amcl", "transform_tolerance", 30.0),
        (
            "rsk_global_sensor_frame_unresolvable",
            "/unfamiliar/costmap",
            "obstacle_layer.scan.sensor_frame",
            "rsk_missing_sensor_frame",
        ),
    ],
)
@pytest.mark.parametrize("age_ms", [0, 600, -1000])
def test_new_capture_requires_actual_original_current_parameter(
    monkeypatch, case, node, key, value, age_ms
):
    import importlib
    from datetime import UTC, datetime, timedelta

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    capture = importlib.import_module("simulation_engineering_capture")
    now = datetime.now(UTC)
    native = {
        "captured_at": now.isoformat(),
        "observations": {
            "node_parameters": {node: {key: value}},
            "parameter_captured_at": {node: (now - timedelta(milliseconds=age_ms)).isoformat()},
        },
    }
    assert capture.counterfactual_readback_available(native, case) is (age_ms == 0)
