"""Pure bounded validation for the optional C++ MessageInfo observer.

Shape/time validation is not source authentication or permission. Records describe
received TF messages, never complete or future authority uniqueness.
"""

import json
import re
from datetime import datetime


def parse_tf_window(raw, *, now, received_at):
    if not isinstance(raw, str) or not 0 < len(raw.encode()) <= 2_000_000:
        raise ValueError("bounded TF attribution JSON required")
    if now.tzinfo is None or received_at.tzinfo is None:
        raise ValueError("aware receiver and review times required")
    data = json.loads(
        raw, parse_constant=lambda v: (_ for _ in ()).throw(ValueError("finite JSON required"))
    )
    if (
        type(data) is not dict
        or data.get("schema_version") != "rosclaw.tf_attribution_window.v2"
        or data.get("completeness") != "NOT_ESTABLISHED"
        or data.get("execution_authority") != "NONE"
    ):
        raise ValueError("read-only incomplete attribution window required")
    for key in (
        "received_steady_ns",
        "window_ms",
        "max_edges",
        "dropped_records",
        "invalid_records",
        "discovery_errors",
        "received_transform_callbacks",
    ):
        if type(data.get(key)) is not int or not 0 <= data[key] <= 2**64 - 1:
            raise ValueError("bounded integer observer counters required")
    if (
        not 100 <= data["window_ms"] <= 5000
        or data["max_edges"] != 1024
        or type(data.get("discovery_current")) is not bool
    ):
        raise ValueError("bounded observer window required")
    captured = datetime.fromisoformat(data["captured_at"])
    if captured.tzinfo is None:
        raise ValueError("aware C++ capture time required")
    if (
        data.get("identity_kind") != "RMW_MESSAGEINFO_OPAQUE_READER_LOCAL"
        or data.get("discovery_gid_mapping") != "NOT_ESTABLISHED"
    ):
        raise ValueError("callback token identity cannot be upgraded to discovery GID")
    instance = data.get("observer_instance")
    if type(instance) is not str or not 1 <= len(instance) <= 128 or not instance.isascii():
        raise ValueError("explicit bounded observer lifetime required")
    rows = data.get("records")
    if type(rows) is not list or len(rows) > 1024:
        raise ValueError("bounded TF attribution records required")
    for row in rows:
        if type(row) is not dict or set(row) != {
            "topic",
            "parent",
            "child",
            "publisher_token",
            "reader_scope",
            "received_at",
            "received_steady_ns",
            "stamp_sec",
            "stamp_nanosec",
        }:
            raise ValueError("typed TF attribution record required")
        for key in ("topic", "parent", "child"):
            value = row[key]
            if (
                not isinstance(value, str)
                or not 0 < len(value) <= 512
                or not value.isascii()
                or any(ord(c) < 32 for c in value)
            ):
                raise ValueError("bounded ASCII TF identities required")
        if not row["topic"].startswith("/"):
            raise ValueError("explicit fully qualified TF topic required")
        if row.get("reader_scope") not in {"VOLATILE_READER", "TRANSIENT_LOCAL_READER"}:
            raise ValueError("explicit reader-local identity scope required")
        gid = row["publisher_token"]
        if gid is not None and (
            type(gid) is not str
            or not re.fullmatch(r"(?:[0-9a-f]{2}){1,64}", gid)
            or set(gid) == {"0"}
        ):
            raise ValueError("observed RMW GID or null required")
        stamp = datetime.fromisoformat(row["received_at"])
        if stamp.tzinfo is None or stamp > captured:
            raise ValueError("aware message receive time within capture required")
        if (
            type(row["received_steady_ns"]) is not int
            or not 0 <= row["received_steady_ns"] <= data["received_steady_ns"]
        ):
            raise ValueError("message receive steady time within window required")
        if (
            type(row["stamp_sec"]) is not int
            or not -(2**31) <= row["stamp_sec"] < 2**31
            or type(row["stamp_nanosec"]) is not int
            or not 0 <= row["stamp_nanosec"] < 10**9
        ):
            raise ValueError("valid ROS TF stamp fields required")
    age = (now - captured).total_seconds() * 1000
    receive_age = (now - received_at).total_seconds() * 1000
    return {
        **data,
        "status": "FRESH" if 0 <= age <= 500 and 0 <= receive_age <= 500 else "STALE_OR_FUTURE",
        "source_authentication": "NOT_ESTABLISHED_BY_PAYLOAD",
    }
