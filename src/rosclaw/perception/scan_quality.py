"""Canonical pure LaserScan quality analysis for ROSClaw perception.

Stage A relocation of the protected prior Native ``scan_client`` diagnostic
algorithm into a canonical, ROS-free utility module. The algorithm
(functions, thresholds and literal constants) is preserved unchanged; only
module placement, docstrings and type annotations were normalized.

The public entry point is :func:`analyze_scan`, which accepts any duck-typed
object exposing the ``sensor_msgs/msg/LaserScan`` fields used by the
algorithm (``header.stamp.sec``, ``header.stamp.nanosec``, ``angle_min``,
``angle_max``, ``angle_increment``, ``range_min``, ``range_max``,
``ranges``) plus an explicit integer ``reference_time_ns`` clock reading.
No ROS runtime, DDS transport, network or hardware access occurs anywhere
in this module; it imports on an ordinary Python 3.11 interpreter.

Resource boundary: ``analyze_scan`` copies ``list(message.ranges)`` at
entry, so work and allocation are linear O(n) in the supplied range count.
There is no scan-size cap and no guarantee of termination for arbitrary
iterables supplied through the duck-typed ``ranges`` field. This module
makes no real-time or resource-safety guarantee for arbitrary inputs;
bounded admission before conversion is a Stage B concern and is not
implemented here.

This is pure source only: it is not a capability, not an App, and performs
no dispatch or actuation. Stage B registration/dispatch is out of scope.
"""

from typing import Any

FRESHNESS_MAX_AGE_NS = 250_000_000  # inclusive
FUTURE_TOLERANCE_NS = 20_000_000  # allowed clock lead
OBSTACLE_DISTANCE_M = 0.8
ANGLE_MAX_TOL = 1e-5

STATUS_QUALITY_INVALID = "QUALITY_INVALID"
STATUS_STALE = "STALE"
STATUS_FUTURE = "FUTURE"
STATUS_NO_VALID = "NO_VALID_RETURN"
STATUS_OBSTACLE = "OBSTACLE"
STATUS_CLEAR = "CLEAR"


def _result(status, valid_count, invalid_count, nearest_index, nearest_distance):
    return {
        "status": status,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "nearest_index": nearest_index,
        "nearest_distance": nearest_distance,
    }


def _is_finite_number(value):
    import math

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _layout_valid(msg, n_ranges):
    angle_min = msg.angle_min
    angle_max = msg.angle_max
    angle_increment = msg.angle_increment
    range_min = msg.range_min
    range_max = msg.range_max
    if not _is_finite_number(angle_min):
        return False
    if not _is_finite_number(angle_increment) or angle_increment <= 0:
        return False
    if n_ranges > 0:
        if not _is_finite_number(angle_max):
            return False
        expected = angle_min + (n_ranges - 1) * angle_increment
        if abs(angle_max - expected) > ANGLE_MAX_TOL:
            return False
    if not _is_finite_number(range_min) or range_min < 0:
        return False
    if not _is_finite_number(range_max) or range_max <= range_min:  # noqa: SIM103 - AST frozen from protected prior algorithm
        return False
    return True


def analyze_scan(message: Any, reference_time_ns: int) -> dict:
    """Pure analysis of a sensor_msgs/msg/LaserScan-like message.

    Parameters:
        message: Duck-typed object with LaserScan-compatible fields.
        reference_time_ns: Explicit reference clock reading in nanoseconds
            (signed int; bool is rejected). Freshness of the message stamp
            is judged against this caller-supplied clock, never against an
            implicit wall clock inside the function.

    Returns:
        The exact result dict of the diagnostic schema. No ROS, no I/O.

    Resource note: the function copies ``list(message.ranges)`` before any
    layout check, so its time and memory are linear in the caller-supplied
    range length. No scan-size cap or arbitrary iterator termination
    guarantee is provided.
    """
    if isinstance(reference_time_ns, bool) or not isinstance(reference_time_ns, int):
        raise ValueError("reference_time_ns must be an explicit int (bool rejected)")

    import math

    ranges = list(message.ranges)
    n = len(ranges)

    # 1) Field/layout quality first.
    if not _layout_valid(message, n):
        return _result(STATUS_QUALITY_INVALID, 0, n, None, None)

    # builtin_interfaces/msg/Time.sec is a SIGNED int32: negative stamps are
    # valid (e.g. pre-epoch or negative co-timed references). Only type and
    # int32 range are field-invalidity; freshness below decides STALE/FUTURE.
    sec = message.header.stamp.sec
    nanosec = message.header.stamp.nanosec
    if isinstance(sec, bool) or not isinstance(sec, int) or not (-(1 << 31) <= sec < (1 << 31)):
        return _result(STATUS_QUALITY_INVALID, 0, n, None, None)
    if (
        isinstance(nanosec, bool)
        or not isinstance(nanosec, int)
        or not (0 <= nanosec < 1_000_000_000)
    ):
        return _result(STATUS_QUALITY_INVALID, 0, n, None, None)

    # 2) Time freshness.
    stamp_ns = sec * 1_000_000_000 + nanosec
    age = reference_time_ns - stamp_ns
    if age > FRESHNESS_MAX_AGE_NS:
        return _result(STATUS_STALE, 0, n, None, None)
    if age < -FUTURE_TOLERANCE_NS:
        return _result(STATUS_FUTURE, 0, n, None, None)

    # 3) Range aggregation; invalid returns are never replaced.
    range_min = float(message.range_min)
    range_max = float(message.range_max)
    valid = []
    invalid_count = 0
    for idx, r in enumerate(ranges):
        rf = float(r)
        if math.isfinite(rf) and range_min <= rf <= range_max:
            valid.append((rf, idx))
        else:
            invalid_count += 1

    if not valid:
        return _result(STATUS_NO_VALID, 0, invalid_count, None, None)

    nearest_distance, nearest_index = min(valid, key=lambda t: t[0])
    status = STATUS_OBSTACLE if nearest_distance <= OBSTACLE_DISTANCE_M else STATUS_CLEAR
    return _result(status, len(valid), invalid_count, nearest_index, nearest_distance)


__all__ = [
    "ANGLE_MAX_TOL",
    "FRESHNESS_MAX_AGE_NS",
    "FUTURE_TOLERANCE_NS",
    "OBSTACLE_DISTANCE_M",
    "STATUS_CLEAR",
    "STATUS_FUTURE",
    "STATUS_NO_VALID",
    "STATUS_OBSTACLE",
    "STATUS_QUALITY_INVALID",
    "STATUS_STALE",
    "analyze_scan",
]
