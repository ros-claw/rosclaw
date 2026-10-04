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

    def restore(self, envelope: Any, key: str) -> dict[str, Any]:
        """Restore only this cached payload after complete envelope validation.

        This is persistence math, not a policy validator or trust grant.
        Returned fields and payload are freshly owned ordinary dictionaries.
        A caller must still validate any policy and its execution commitment.
        """
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
            or type(key) is not str
            or not 1 <= len(key) <= 128
            or envelope["schema"] != SCHEMA
            or envelope["location"] != [key]
            or envelope["payload_hash"] != self.payload_hash
            or self.payload_hash != "sha256:" + hashlib.sha256(self._payload_bytes).hexdigest()
            or type(envelope["stripped_document"]) is not dict
        ):
            raise ValueError("complete envelope for the immutable cached payload required")
        unsigned = {k: v for k, v in envelope.items() if k != "envelope_hash"}
        if canonical_hash(unsigned) != envelope["envelope_hash"]:
            raise ValueError("cached payload envelope seal changed")
        stripped = json.loads(_bytes(envelope["stripped_document"]))
        if stripped.get(key) != {MARKER: self.payload_hash}:
            raise ValueError("exact single cached payload marker required")
        fields = {k: v for k, v in stripped.items() if k != key}
        if self.document_hash(fields, key) != envelope["logical_document_hash"]:
            raise ValueError("complete cached logical document hash changed")
        fields[key] = json.loads(self._payload_bytes)
        return fields
