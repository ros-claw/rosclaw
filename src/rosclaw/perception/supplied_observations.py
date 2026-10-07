"""Bounded supplied-record adapters for canonical perception quality analysis.

Stage B: admit only exact plain JSON-shaped containers/scalars (no subclass
instances, no properties, no custom iterators, no ``bytes``-convertible
objects) with explicit list/string/numeric bounds enforced *before* any
message-object construction, ``ranges``/``fields`` copying or ``bytes(data)``
conversion. Admitted records are converted into plain attribute namespaces
and handed to the unchanged canonical analyzers
(:func:`rosclaw.perception.scan_quality.analyze_scan` and
:func:`rosclaw.perception.cloud_quality.analyze_cloud`); the adapter result
is the canonical result dict, byte-identical.

Non-finite scan scalars are transported as the exact string tags ``NAN``,
``POS_INF`` and ``NEG_INF`` (bare JSON numbers are finite, abs <= 1e12) and
are deliberately mapped to the IEEE values only after full admission, so the
canonical algorithm's own quality rules classify them.

Malformed inputs raise ``TypeError`` or ``ValueError`` before the canonical
analyzer is invoked. No ROS, DDS, sensor, file, network or hardware access.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from rosclaw.perception import cloud_quality, scan_quality

_ABS_LIMIT = 1e12
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1
_MAX_FRAME_ID = 128
_MAX_FIELD_NAME = 64
_MAX_SEQUENCE = 4096
_MAX_FIELDS = 32
_MAX_DATA = 65536
_MAX_KEY_LENGTH = 64
_MAX_TAG_LENGTH = 16
_NONFINITE_TAGS = ("NAN", "POS_INF", "NEG_INF")
_TAG_VALUES = {"NAN": math.nan, "POS_INF": math.inf, "NEG_INF": -math.inf}

_SCAN_RECORD_KEYS = frozenset(
    {
        "header",
        "angle_min",
        "angle_max",
        "angle_increment",
        "time_increment",
        "scan_time",
        "range_min",
        "range_max",
        "ranges",
        "intensities",
    }
)
_SCAN_SCALAR_KEYS = (
    "angle_min",
    "angle_max",
    "angle_increment",
    "time_increment",
    "scan_time",
    "range_min",
    "range_max",
)
_CLOUD_RECORD_KEYS = frozenset(
    {
        "header",
        "width",
        "height",
        "point_step",
        "row_step",
        "is_bigendian",
        "is_dense",
        "fields",
        "data",
    }
)
_FIELD_KEYS = frozenset({"name", "offset", "datatype", "count"})
_HEADER_KEYS = frozenset({"stamp", "frame_id"})
_STAMP_KEYS = frozenset({"sec", "nanosec"})


def _plain_dict(value: Any, keys: frozenset, path: str) -> dict:
    if type(value) is not dict:
        # Fixed bounded reason: never echo the caller's type name (an
        # arbitrary class __name__ would amplify the diagnostic unboundedly).
        raise TypeError(f"{path} must be a plain dict; rejected non-plain value")
    # Key-count bound precedes any key-set construction/sorting/traversal so a
    # caller cannot amplify diagnostics or CPU with an arbitrary key set.
    if len(value) > len(keys):
        raise ValueError(f"{path} has too many keys (maximum {len(keys)})")
    for key in value:
        if type(key) is not str:
            raise TypeError(f"{path} keys must be plain str")
        if len(key) > _MAX_KEY_LENGTH:
            raise ValueError(f"{path} key length exceeds {_MAX_KEY_LENGTH}")
    missing = keys - value.keys()
    extra = value.keys() - keys
    if missing:
        raise ValueError(f"{path} missing keys: {sorted(missing)}")
    if extra:
        # Never echo caller-provided key names in diagnostics.
        raise ValueError(f"{path} has unexpected keys")
    return value


def _plain_list(value: Any, max_items: int, path: str) -> list:
    if type(value) is not list:
        raise TypeError(f"{path} must be a plain list; rejected non-plain value")
    if len(value) > max_items:
        raise ValueError(f"{path} has {len(value)} items, maximum is {max_items}")
    return value


def _plain_int(value: Any, lo: int, hi: int, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path} must be a plain int; rejected non-plain value")
    if not (lo <= value <= hi):
        # Static bounded message: never format the caller's integer (a huge
        # int stringification would amplify the diagnostic past any bound).
        raise ValueError(f"{path} outside admitted range [{lo}, {hi}]")
    return value


def _plain_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{path} must be a plain bool; rejected non-plain value")
    return value


def _plain_str(value: Any, max_length: int, path: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be a plain str; rejected non-plain value")
    if len(value) > max_length:
        raise ValueError(f"{path} length {len(value)} exceeds {max_length}")
    return value


def _plain_scalar(value: Any, path: str) -> float:
    """Admit a finite bounded number or an explicit nonfinite tag."""
    if type(value) is str:
        # Bound first, then classify; never echo the caller's tag text.
        if len(value) > _MAX_TAG_LENGTH:
            raise ValueError(f"{path} string tag length exceeds {_MAX_TAG_LENGTH}")
        if value not in _NONFINITE_TAGS:
            raise ValueError(f"{path} string is not an admitted tag")
        return _TAG_VALUES[value]
    if type(value) is int:
        # Exact integer comparison before float() so huge plain integers
        # (e.g. 1 << 4096) reject as ValueError instead of raising an
        # unhandled OverflowError from the float conversion.
        if abs(value) > _ABS_LIMIT:
            raise ValueError(f"{path} magnitude exceeds the admitted bound")
        return float(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite; use an explicit tag for nonfinite")
        if abs(value) > _ABS_LIMIT:
            raise ValueError(f"{path} magnitude exceeds the admitted bound")
        return value
    raise TypeError(f"{path} must be a plain number or tag; rejected non-plain value")


def _plain_float_sequence(value: Any, path: str) -> list[float]:
    items = _plain_list(value, _MAX_SEQUENCE, path)
    return [_plain_scalar(item, f"{path}[{index}]") for index, item in enumerate(items)]


def _plain_header(value: Any, path: str) -> SimpleNamespace:
    header = _plain_dict(value, _HEADER_KEYS, path)
    stamp = _plain_dict(header["stamp"], _STAMP_KEYS, f"{path}.stamp")
    sec = _plain_int(stamp["sec"], _INT32_MIN, _INT32_MAX, f"{path}.stamp.sec")
    nanosec = _plain_int(stamp["nanosec"], 0, 999_999_999, f"{path}.stamp.nanosec")
    frame_id = _plain_str(header["frame_id"], _MAX_FRAME_ID, f"{path}.frame_id")
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id)


def analyze_supplied_scan(payload: Any) -> dict:
    """Admit one bounded supplied scan record and run the canonical analyzer.

    Raises ``TypeError``/``ValueError`` for any malformed input before the
    canonical body executes. Returns the canonical result dict unchanged.
    """
    root = _plain_dict(payload, frozenset({"record", "reference_time_ns"}), "payload")
    reference_time_ns = _plain_int(
        root["reference_time_ns"], _INT64_MIN, _INT64_MAX, "payload.reference_time_ns"
    )
    record = _plain_dict(root["record"], _SCAN_RECORD_KEYS, "payload.record")
    header = _plain_header(record["header"], "payload.record.header")
    scalars = {
        key: _plain_scalar(record[key], f"payload.record.{key}") for key in _SCAN_SCALAR_KEYS
    }
    ranges = _plain_float_sequence(record["ranges"], "payload.record.ranges")
    intensities = _plain_float_sequence(record["intensities"], "payload.record.intensities")
    message = SimpleNamespace(
        header=header,
        angle_min=scalars["angle_min"],
        angle_max=scalars["angle_max"],
        angle_increment=scalars["angle_increment"],
        time_increment=scalars["time_increment"],
        scan_time=scalars["scan_time"],
        range_min=scalars["range_min"],
        range_max=scalars["range_max"],
        ranges=ranges,
        intensities=intensities,
    )
    return scan_quality.analyze_scan(message, reference_time_ns)


def analyze_supplied_cloud(payload: Any) -> dict:
    """Admit one bounded supplied cloud record and run the canonical analyzer.

    ``width``/``height`` up to 65535 are admitted as declared metadata (e.g.
    17x1 reaches the canonical analyzer and yields declared ``total_count``
    17 with ``QUALITY_INVALID``); only transport-level shape/size/type
    violations are rejected here. Raises ``TypeError``/``ValueError`` before
    the canonical body for malformed input.
    """
    root = _plain_dict(payload, frozenset({"record"}), "payload")
    record = _plain_dict(root["record"], _CLOUD_RECORD_KEYS, "payload.record")
    header = _plain_header(record["header"], "payload.record.header")
    width = _plain_int(record["width"], 0, 65535, "payload.record.width")
    height = _plain_int(record["height"], 0, 65535, "payload.record.height")
    point_step = _plain_int(record["point_step"], 0, 65535, "payload.record.point_step")
    row_step = _plain_int(record["row_step"], 0, _UINT32_MAX, "payload.record.row_step")
    is_bigendian = _plain_bool(record["is_bigendian"], "payload.record.is_bigendian")
    is_dense = _plain_bool(record["is_dense"], "payload.record.is_dense")
    raw_fields = _plain_list(record["fields"], _MAX_FIELDS, "payload.record.fields")
    fields = []
    for index, raw_field in enumerate(raw_fields):
        path = f"payload.record.fields[{index}]"
        field = _plain_dict(raw_field, _FIELD_KEYS, path)
        fields.append(
            SimpleNamespace(
                name=_plain_str(field["name"], _MAX_FIELD_NAME, f"{path}.name"),
                offset=_plain_int(field["offset"], 0, _UINT32_MAX, f"{path}.offset"),
                datatype=_plain_int(field["datatype"], 0, 255, f"{path}.datatype"),
                count=_plain_int(field["count"], 0, _UINT32_MAX, f"{path}.count"),
            )
        )
    raw_data = _plain_list(record["data"], _MAX_DATA, "payload.record.data")
    octets = [
        _plain_int(item, 0, 255, f"payload.record.data[{index}]")
        for index, item in enumerate(raw_data)
    ]
    message = SimpleNamespace(
        header=header,
        width=width,
        height=height,
        point_step=point_step,
        row_step=row_step,
        is_bigendian=is_bigendian,
        is_dense=is_dense,
        fields=fields,
        data=bytes(octets),
    )
    return cloud_quality.analyze_cloud(message)


__all__ = ["analyze_supplied_cloud", "analyze_supplied_scan"]
