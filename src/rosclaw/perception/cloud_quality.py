"""Canonical pure PointCloud2 quality analysis for ROSClaw perception.

Stage A relocation of the protected prior Native ``cloud_client`` diagnostic
algorithm into a canonical, ROS-free utility module. The algorithm
(functions, thresholds and literal constants, including the layout bounds)
is preserved unchanged; only module placement, docstrings and type
annotations were normalized.

The public entry point is :func:`analyze_cloud`, which accepts any
duck-typed object exposing the ``sensor_msgs/msg/PointCloud2`` fields used
by the algorithm (``width``, ``height``, ``point_step``, ``row_step``,
``fields`` with ``name``/``offset``/``datatype``/``count``, ``data``,
``is_bigendian``). No ROS runtime, DDS transport, network or hardware
access occurs anywhere in this module; it imports on an ordinary
Python 3.11 interpreter.

Resource boundary: accepted ``width`` and ``height`` are each in 1..16,
so at most 256 points are decoded. However, ``len(bytes(msg.data))``
conversion precedes the data-length rejection, and the ``fields``/``name``
traversal can be arbitrarily large for duck-typed inputs; there is no
total allocation or work guarantee for arbitrary inputs, and no real-time
safety guarantee. Bounded admission before conversion/traversal is a
Stage B concern and is not implemented here.

This is pure source only: it is not a capability, not an App, and performs
no dispatch or actuation. Stage B registration/dispatch is out of scope.
"""

import math
import struct
from typing import Any

_FLOAT32 = 7  # sensor_msgs/msg/PointField.FLOAT32
_DATATYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}


def _invalid_result(total):
    return {
        "status": "QUALITY_INVALID",
        "total_count": total,
        "valid_count": 0,
        "invalid_count": total,
        "min_xyz": None,
        "max_xyz": None,
        "centroid_xyz": None,
    }


def _exact_int(value, lo, hi):
    return isinstance(value, int) and not isinstance(value, bool) and lo <= value <= hi


def _layout_valid(msg):
    """Validate cloud layout; return (ok, total_count)."""
    width, height = msg.width, msg.height
    point_step, row_step = msg.point_step, msg.row_step
    # Declared-count convention: any pair of ints declares total = width*height
    # (a pre-CDR bool duck type declares int(True)=1), even when the values are
    # outside the allowed 1..16 envelope, so QUALITY_INVALID reports
    # total_count == invalid_count == the declared product.
    dims_are_ints = isinstance(width, int) and isinstance(height, int)
    declared_total = width * height if dims_are_ints else 0
    if not (_exact_int(width, 1, 16) and _exact_int(height, 1, 16)):
        return False, declared_total
    total = width * height
    if not (_exact_int(point_step, 1, 256) and _exact_int(row_step, 1, 4096)):
        return False, total
    if row_step < width * point_step:
        return False, total
    if len(bytes(msg.data)) != height * row_step:
        return False, total
    names = [f.name for f in msg.fields]
    if len(names) != len(set(names)):
        return False, total
    for f in msg.fields:
        if not _exact_int(f.offset, 0, 1 << 30):
            return False, total
        if not _exact_int(f.count, 1, 1 << 30):
            return False, total
        if f.datatype not in _DATATYPE_SIZE:
            return False, total
        if f.offset + _DATATYPE_SIZE[f.datatype] * f.count > point_step:
            return False, total
    for req in ("x", "y", "z"):
        matches = [f for f in msg.fields if f.name == req]
        if len(matches) != 1:
            return False, total
        f = matches[0]
        if f.datatype != _FLOAT32 or f.count != 1 or f.offset + 4 > point_step:
            return False, total
    return True, total


def analyze_cloud(message: Any) -> dict:
    """Pure PointCloud2 diagnostic; no ROS init or DDS required.

    Parameters:
        message: Duck-typed object with PointCloud2-compatible fields;
            ``message.data`` must be bytes-compatible.

    Returns:
        The exact result dict of the diagnostic schema.

    Resource note: the 1..16 bound on each of ``width`` and ``height``
    limits decoding to at most 256 points, but ``bytes(message.data)``
    conversion happens before length rejection and field/name traversal is
    unbounded for duck-typed inputs; no total allocation/work guarantee.
    """
    ok, total = _layout_valid(message)
    if not ok:
        return _invalid_result(total)
    fmt = ">" if bool(message.is_bigendian) else "<"
    offsets = {f.name: f.offset for f in message.fields if f.name in ("x", "y", "z")}
    data = bytes(message.data)
    ps, rs, w, h = message.point_step, message.row_step, message.width, message.height
    valid = 0
    mins = [math.inf] * 3
    maxs = [-math.inf] * 3
    sums = [0.0] * 3
    for row in range(h):
        base = row * rs
        for col in range(w):
            pbase = base + col * ps
            pt = []
            finite = True
            for axis, name in enumerate(("x", "y", "z")):  # noqa: B007 - AST frozen from protected prior algorithm
                (v,) = struct.unpack_from(fmt + "f", data, pbase + offsets[name])
                pt.append(v)
                if not math.isfinite(v):
                    finite = False
            if finite:
                valid += 1
                for axis in range(3):
                    v = pt[axis]
                    if v < mins[axis]:
                        mins[axis] = v
                    if v > maxs[axis]:
                        maxs[axis] = v
                    sums[axis] += v
    if valid == 0:
        return {
            "status": "NO_VALID_RETURN",
            "total_count": total,
            "valid_count": 0,
            "invalid_count": total,
            "min_xyz": None,
            "max_xyz": None,
            "centroid_xyz": None,
        }
    return {
        "status": "VALID_CLOUD",
        "total_count": total,
        "valid_count": valid,
        "invalid_count": total - valid,
        "min_xyz": mins,
        "max_xyz": maxs,
        "centroid_xyz": [s / valid for s in sums],
    }


__all__ = ["analyze_cloud"]
