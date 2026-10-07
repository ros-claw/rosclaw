"""Complete shared-document hashes with an owned cached nested JSON payload.

Persistence math only. Hash identity proves neither physical provenance nor
authorization. No policy field is omitted, no reference is followed, and no
decoder, filesystem reader, execution callback or trust grant is provided.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from rosclaw.growth.canonical_json_snapshot import CanonicalJSONSnapshot
from rosclaw.growth.shared_proof_payload import MARKER, SCHEMA, canonical_hash


def _bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


class CachedSharedPayload:
    """Verify the existing envelope format without repeated payload serialization.

    Callers authenticate the actual payload source and independently validate
    physics/policy/authority. Returned stripped fields still contain a marker,
    NOT a usable policy. Use restore_payload for ordinary complete dictionaries.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        if type(payload) is not dict or MARKER in payload:
            raise ValueError("complete ordinary dictionary payload required")
        self._payload = CanonicalJSONSnapshot(payload)
        self._digest = self._payload.content_hash

    @property
    def payload_hash(self) -> str:
        if self._payload.verify() != self._digest:
            raise ValueError("cached complete payload identity changed")
        return self._digest

    def _hash(
        self, document: dict[str, Any], location: tuple[str, ...], omit: tuple[str, ...]
    ) -> str:
        digest = hashlib.sha256()

        def emit(fields: dict[str, Any], path: tuple[str, ...], *, root: bool) -> None:
            digest.update(b"{")
            names = [k for k in sorted(fields) if not (root and k in omit)]
            for index, name in enumerate(names):
                if index:
                    digest.update(b",")
                digest.update(_bytes(name))
                digest.update(b":")
                if name != path[0]:
                    digest.update(_bytes(fields[name]))
                elif len(path) == 1:
                    # Snapshot integrity is checked before every complete hash.
                    digest.update(self._payload._data)
                else:
                    emit(fields[name], path[1:], root=False)
            digest.update(b"}")

        emit(document, location, root=True)
        return "sha256:" + digest.hexdigest()

    def _verified(self, envelope: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
        digest = self.payload_hash
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
            or envelope["payload_hash"] != digest
        ):
            raise ValueError("exact shared envelope for the fixed complete payload required")
        owned = CanonicalJSONSnapshot(envelope).restore()
        if owned["envelope_hash"] != canonical_hash(
            {k: v for k, v in owned.items() if k != "envelope_hash"}
        ):
            raise ValueError("shared envelope seal changed")
        location = owned["location"]
        if (
            type(location) is not list
            or not 1 <= len(location) <= 8
            or any(type(k) is not str or not 1 <= len(k) <= 128 for k in location)
            or type(owned["stripped_document"]) is not dict
        ):
            raise ValueError("one to eight explicit dictionary keys required")
        marker: Any = owned["stripped_document"]
        for key in location:
            if type(marker) is not dict or key not in marker:
                raise ValueError("explicit complete payload marker location missing")
            marker = marker[key]
        if marker != {MARKER: digest}:
            raise ValueError("exact single fixed-payload marker required")
        if (
            self._hash(owned["stripped_document"], tuple(location), ())
            != owned["logical_document_hash"]
        ):
            raise ValueError("complete shared logical document changed")
        return cast(dict[str, Any], owned["stripped_document"]), tuple(location)

    def verified_fields(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Owned compact fields after checking the WHOLE logical document hash."""
        return self._verified(envelope)[0]

    def document_hash(
        self, envelope: dict[str, Any], *, omit_root_keys: tuple[str, ...] = ()
    ) -> str:
        """Hash a verified full document, optionally excluding explicit root seals.

        Intended for matching an application's existing report_hash convention.
        Never exclude the payload-containing branch. The original envelope's
        complete hash is always verified BEFORE computing the seal-free hash.
        """
        fields, location = self._verified(envelope)
        if (
            type(omit_root_keys) is not tuple
            or len(omit_root_keys) > 8
            or any(type(k) is not str or not 1 <= len(k) <= 128 for k in omit_root_keys)
            or len(set(omit_root_keys)) != len(omit_root_keys)
            or any(k not in fields or k == location[0] for k in omit_root_keys)
        ):
            raise ValueError("explicit present root seals only; never omit complete payload")
        return self._hash(fields, location, omit_root_keys)
