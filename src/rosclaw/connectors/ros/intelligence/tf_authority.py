"""Optional source-asserted MessageInfo records, never completeness or admission."""

import re
from datetime import datetime


def current_tf_attributions(model, *, now):
    data = model.observations.get("tf_attribution_window")
    if type(data) is not dict or data.get("status") != "FRESH":
        return []
    if (
        data.get("schema_version") != "rosclaw.tf_attribution_window.v2"
        or data.get("completeness") != "NOT_ESTABLISHED"
        or data.get("execution_authority") != "NONE"
    ):
        raise ValueError("read-only incomplete TF attribution window required")
    captured = datetime.fromisoformat(data["captured_at"])
    if captured.tzinfo is None or now.tzinfo is None:
        raise ValueError("aware TF attribution capture required")
    if not 0 <= (now - captured).total_seconds() * 1000 <= 500:
        return []
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
    result = []
    for index, row in enumerate(rows):
        if type(row) is not dict:
            raise ValueError("TF attribution record object required")
        for field in ("topic", "parent", "child"):
            value = row.get(field)
            if (
                type(value) is not str
                or not 0 < len(value) <= 512
                or not value.isascii()
                or any(ord(c) < 32 for c in value)
            ):
                raise ValueError("bounded TF attribution identity required")
        if not row["topic"].startswith("/"):
            raise ValueError("fully qualified observed TF topic required")
        if row.get("reader_scope") not in {"VOLATILE_READER", "TRANSIENT_LOCAL_READER"}:
            raise ValueError("explicit reader-local identity scope required")
        gid = row.get("publisher_token")
        if gid is not None and (
            type(gid) is not str
            or not re.fullmatch(r"(?:[0-9a-f]{2}){1,64}", gid)
            or set(gid) == {"0"}
        ):
            raise ValueError("observed RMW publisher GID or null required")
        stamp = datetime.fromisoformat(row["received_at"])
        if stamp.tzinfo is None:
            raise ValueError("aware message receive time required")
        if (
            gid is not None
            and 0 <= (now - stamp).total_seconds() * 1000 <= 500
            and stamp <= captured
        ):
            result.append((index, row, stamp))
    return result
