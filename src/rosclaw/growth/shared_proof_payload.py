"""Lossless shared JSON proof payloads; pure data, no execution or promotion.

Applications own persistence and independently verify evidence/authorization.
Content hashes prove identity, not provenance, signatures or physical success.
Only one explicit dictionary payload is detached; no recursive references.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SCHEMA = "rosclaw.growth.shared_json_payload.v1"
MARKER = "$rosclaw_shared_payload"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _path(location: Any) -> tuple[str, ...]:
    if (
        not isinstance(location, (tuple, list))
        or not 1 <= len(location) <= 8
        or any(type(key) is not str or not 1 <= len(key) <= 128 for key in location)
    ):
        raise ValueError("one to eight explicit nonempty dictionary keys required")
    return tuple(location)


def _replace(document: dict[str, Any], location: tuple[str, ...], value: Any) -> dict[str, Any]:
    result = dict(document)
    original, target = document, result
    for key in location[:-1]:
        if key not in original or type(original[key]) is not dict:
            raise ValueError("explicit dictionary proof location missing")
        copied = dict(original[key])
        target[key] = copied
        original, target = original[key], copied
    if location[-1] not in original:
        raise ValueError("explicit dictionary proof location missing")
    target[location[-1]] = value
    return result


def detach_payload(
    document: dict[str, Any], location: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a new envelope and return the whole unmodified payload to store."""
    if type(document) is not dict:
        raise ValueError("complete JSON dictionary document required")
    keys = _path(location)
    payload: Any = document
    for key in keys:
        if type(payload) is not dict or key not in payload:
            raise ValueError("explicit dictionary proof location missing")
        payload = payload[key]
    if type(payload) is not dict or MARKER in payload:
        raise ValueError("complete ordinary dictionary payload required; no nested references")
    digest = canonical_hash(payload)
    envelope = {
        "schema": SCHEMA,
        "location": list(keys),
        "payload_hash": digest,
        "logical_document_hash": canonical_hash(document),
        "stripped_document": _replace(document, keys, {MARKER: digest}),
    }
    envelope["envelope_hash"] = canonical_hash(envelope)
    return envelope, payload


def restore_payload(envelope: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct all original numerical fields or reject, never grant trust."""
    if (
        type(envelope) is not dict
        or set(envelope)
        != {
            "schema",
            "location",
            "payload_hash",
            "logical_document_hash",
            "stripped_document",
            "envelope_hash",
        }
        or envelope["schema"] != SCHEMA
    ):
        raise ValueError("exact shared-payload envelope required")
    if envelope["envelope_hash"] != canonical_hash(
        {key: value for key, value in envelope.items() if key != "envelope_hash"}
    ):
        raise ValueError("shared-payload envelope seal changed")
    digest = envelope["payload_hash"]
    if type(digest) is not str or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("canonical payload identity required; not a file path")
    if type(payload) is not dict or MARKER in payload or canonical_hash(payload) != digest:
        raise ValueError("complete shared payload identity changed")
    keys = _path(envelope["location"])
    document = envelope["stripped_document"]
    if type(document) is not dict:
        raise ValueError("complete stripped dictionary document required")
    marker: Any = document
    for key in keys:
        if type(marker) is not dict or key not in marker:
            raise ValueError("shared proof marker missing")
        marker = marker[key]
    if marker != {MARKER: digest}:
        raise ValueError("exact single shared proof marker required")
    result = _replace(document, keys, payload)
    if canonical_hash(result) != envelope["logical_document_hash"]:
        raise ValueError("shared payload changed complete logical document")
    return result
