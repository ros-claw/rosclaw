"""Canonical full-document hashing with one immutable cached JSON field.

Pure persistence math, no policy or authority. Never omit the full payload from
logical hashes; avoid repeatedly converting the same large dictionary to JSON.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from rosclaw.growth.shared_proof_payload import MARKER, SCHEMA, canonical_hash


def _bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


class FrozenPayloadField:
    def __init__(self, payload: dict[str, Any]) -> None:
        if type(payload) is not dict or MARKER in payload:
            raise ValueError("complete ordinary dictionary payload required")
        self._payload_bytes = _bytes(payload)
        self.payload_hash = "sha256:" + hashlib.sha256(self._payload_bytes).hexdigest()

    def document_hash(self, fields: dict[str, Any], key: str) -> str:
        if (
            type(fields) is not dict
            or type(key) is not str
            or not 1 <= len(key) <= 128
            or key in fields
            or any(type(k) is not str for k in fields)
        ):
            raise ValueError("ordinary dictionary and one missing explicit payload key required")
        digest = hashlib.sha256()
        digest.update(b"{")
        for index, name in enumerate(sorted([*fields, key])):
            if index:
                digest.update(b",")
            digest.update(_bytes(name))
            digest.update(b":")
            digest.update(self._payload_bytes if name == key else _bytes(fields[name]))
        digest.update(b"}")
        return "sha256:" + digest.hexdigest()

    def envelope(self, fields: dict[str, Any], key: str) -> dict[str, Any]:
        logical_hash = self.document_hash(fields, key)
        # Capture independent compact fields. Caller mutation after this call
        # cannot alter an already built envelope or its seal.
        stripped = json.loads(_bytes(fields))
        stripped[key] = {MARKER: self.payload_hash}
        value = {
            "schema": SCHEMA,
            "location": [key],
            "payload_hash": self.payload_hash,
            "logical_document_hash": logical_hash,
            "stripped_document": stripped,
        }
        return {**value, "envelope_hash": canonical_hash(value)}
