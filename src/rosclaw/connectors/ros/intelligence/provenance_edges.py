"""Pure JSON-pointer evidence links, retaining original documents outside the graph."""

from datetime import datetime

from rosclaw.contracts.common import content_hash

from .graph_contracts import ProvenanceEdgeV1


def pointer_value(document, pointer):
    value = document
    for component in pointer.split("/")[1:] if pointer else []:
        key = component.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def freshness(stamp, now, max_age_ms):
    if stamp.tzinfo is None or now.tzinfo is None:
        raise ValueError("aware observation and review clocks required")
    age = (now - stamp).total_seconds() * 1000
    return "FUTURE" if age < 0 else "FRESH" if age <= max_age_ms else "STALE"


def provenance_edge(document, document_ref, document_hash, pointer, stamp, now, max_age_ms):
    if not document_ref:
        raise ValueError("explicit original document reference required")
    stamp = datetime.fromisoformat(stamp) if isinstance(stamp, str) else stamp
    value = pointer_value(document, pointer)
    payload = {
        "document_ref": document_ref,
        "document_hash": document_hash,
        "pointer": pointer,
        "value_hash": content_hash("opvalue", value),
        "observed_at": stamp,
        "freshness": freshness(stamp, now, max_age_ms),
    }
    key = content_hash("opref", {**payload, "observed_at": stamp.isoformat()})
    return ProvenanceEdgeV1(id=key, **payload)
