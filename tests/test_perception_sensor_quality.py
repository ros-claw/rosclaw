"""Native meaningful tests for the canonical pure perception quality modules.

Exercises ``rosclaw.perception.scan_quality.analyze_scan`` and
``rosclaw.perception.cloud_quality.analyze_cloud`` with plain duck-typed
objects. No ROS runtime, no DDS, no hardware: these tests run on an
ordinary Python interpreter with only ``src`` on ``sys.path``.
"""

import math
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

from rosclaw.perception import cloud_quality, scan_quality


def _scan(
    ranges,
    sec=0,
    nanosec=0,
    angle_min=0.0,
    angle_max=None,
    angle_increment=0.25,
    range_min=0.1,
    range_max=4.0,
):
    msg = types.SimpleNamespace()
    msg.header = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=sec, nanosec=nanosec))
    msg.header.frame_id = "test_frame"
    msg.angle_min = angle_min
    if angle_max is None:
        angle_max = angle_min + (len(ranges) - 1) * angle_increment if ranges else 0.0
    msg.angle_max = angle_max
    msg.angle_increment = angle_increment
    msg.range_min = range_min
    msg.range_max = range_max
    msg.ranges = list(ranges)
    return msg


def _field(name, offset, datatype=7, count=1):
    return types.SimpleNamespace(name=name, offset=offset, datatype=datatype, count=count)


def _cloud(points, bigendian=False, width=None, height=None, **overrides):
    w = width if width is not None else len(points)
    h = height if height is not None else 1
    point_step = overrides.get("point_step", 12)
    row_step = overrides.get("row_step", point_step * w)
    fields = overrides.get("fields", [_field("x", 0), _field("y", 4), _field("z", 8)])
    data = overrides.get("data")
    if data is None:
        fmt = ">" if bigendian else "<"
        buf = bytearray(row_step * h)
        for i, (x, y, z) in enumerate(points):
            row, col = divmod(i, w)
            struct.pack_into(fmt + "fff", buf, row * row_step + col * point_step, x, y, z)
        data = bytes(buf)
    return types.SimpleNamespace(
        width=w,
        height=h,
        point_step=point_step,
        row_step=row_step,
        is_bigendian=bigendian,
        is_dense=True,
        fields=fields,
        data=data,
    )


# --- scan_quality ---------------------------------------------------------


def test_scan_clear_single_valid_return():
    msg = _scan([1.0], sec=-1, nanosec=0)
    result = scan_quality.analyze_scan(msg, -1_000_000_000)
    assert result == {
        "status": "CLEAR",
        "valid_count": 1,
        "invalid_count": 0,
        "nearest_index": 0,
        "nearest_distance": 1.0,
    }


def test_scan_obstacle_and_invalid_returns_not_replaced():
    msg = _scan([0.5, math.nan, math.inf, 2.0])
    result = scan_quality.analyze_scan(msg, 0)
    assert result["status"] == "OBSTACLE"
    assert result["valid_count"] == 2
    assert result["invalid_count"] == 2
    assert result["nearest_index"] == 0
    assert result["nearest_distance"] == 0.5


def test_scan_stale_and_future_and_freshness_boundary():
    stamp_sec = 10
    stamp_ns = stamp_sec * 1_000_000_000
    msg = _scan([1.0], sec=stamp_sec)
    stale = scan_quality.analyze_scan(msg, stamp_ns + scan_quality.FRESHNESS_MAX_AGE_NS + 1)
    assert stale["status"] == "STALE"
    future = scan_quality.analyze_scan(msg, stamp_ns - scan_quality.FUTURE_TOLERANCE_NS - 1)
    assert future["status"] == "FUTURE"
    boundary = scan_quality.analyze_scan(msg, stamp_ns + scan_quality.FRESHNESS_MAX_AGE_NS)
    assert boundary["status"] == "CLEAR"


def test_scan_layout_invalid_and_signed_stamp_rules():
    bad_layout = _scan([1.0], angle_max=5.0)
    assert scan_quality.analyze_scan(bad_layout, 0)["status"] == "QUALITY_INVALID"
    bad_sec = _scan([1.0], sec=1 << 31)
    assert scan_quality.analyze_scan(bad_sec, 0)["status"] == "QUALITY_INVALID"
    bad_nanosec = _scan([1.0], nanosec=1_000_000_000)
    assert scan_quality.analyze_scan(bad_nanosec, 0)["status"] == "QUALITY_INVALID"
    with pytest.raises(ValueError):
        scan_quality.analyze_scan(_scan([1.0]), True)


def test_scan_no_valid_return():
    msg = _scan([math.nan, 99.0])
    result = scan_quality.analyze_scan(msg, 0)
    assert result["status"] == "NO_VALID_RETURN"
    assert result["valid_count"] == 0
    assert result["invalid_count"] == 2
    assert result["nearest_index"] is None
    assert result["nearest_distance"] is None


# --- cloud_quality --------------------------------------------------------


def test_cloud_valid_stats():
    msg = _cloud([(0.0, -1.0, 1.0), (1.0, 0.0, 1.5), (2.0, 0.0, 2.0)])
    result = cloud_quality.analyze_cloud(msg)
    assert result["status"] == "VALID_CLOUD"
    assert result["total_count"] == 3
    assert result["valid_count"] == 3
    assert result["invalid_count"] == 0
    assert result["min_xyz"] == [0.0, -1.0, 1.0]
    assert result["max_xyz"] == [2.0, 0.0, 2.0]
    assert result["centroid_xyz"][0] == pytest.approx(1.0)
    assert result["centroid_xyz"][1] == pytest.approx(-1.0 / 3)
    assert result["centroid_xyz"][2] == pytest.approx(1.5)


def test_cloud_bigendian_and_nan_point():
    msg = _cloud([(1.0, 2.0, 3.0), (math.nan, 0.0, 0.0)], bigendian=True)
    result = cloud_quality.analyze_cloud(msg)
    assert result["status"] == "VALID_CLOUD"
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 1
    assert result["min_xyz"] == [1.0, 2.0, 3.0]
    assert result["max_xyz"] == [1.0, 2.0, 3.0]


def test_cloud_all_nan_no_valid_return():
    msg = _cloud([(math.nan, 0.0, 0.0), (math.inf, 1.0, 1.0)])
    result = cloud_quality.analyze_cloud(msg)
    assert result["status"] == "NO_VALID_RETURN"
    assert result["total_count"] == 2
    assert result["valid_count"] == 0
    assert result["min_xyz"] is None


def test_cloud_layout_invalid_declared_count_convention():
    msg = _cloud([], width=17, height=1, point_step=16, row_step=16, data=bytes(16))
    result = cloud_quality.analyze_cloud(msg)
    assert result == {
        "status": "QUALITY_INVALID",
        "total_count": 17,
        "valid_count": 0,
        "invalid_count": 17,
        "min_xyz": None,
        "max_xyz": None,
        "centroid_xyz": None,
    }
    dup = _cloud([(1.0, 2.0, 3.0)], fields=[_field("x", 0), _field("x", 4), _field("z", 8)])
    assert cloud_quality.analyze_cloud(dup)["status"] == "QUALITY_INVALID"
    short = _cloud([(1.0, 2.0, 3.0)], data=bytes(6))
    assert cloud_quality.analyze_cloud(short)["status"] == "QUALITY_INVALID"


def test_modules_import_without_ros():
    # MCP collection installs parent-process ROS stubs. A fresh isolated child
    # tests the actual source imports without altering those shared modules.
    code = """
import sys
sys.path.insert(0, sys.argv[1])
assert "rclpy" not in sys.modules
assert "sensor_msgs" not in sys.modules
import rosclaw.perception.scan_quality
import rosclaw.perception.cloud_quality
assert "rclpy" not in sys.modules
assert "sensor_msgs" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(Path(__file__).resolve().parents[1] / "src")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
