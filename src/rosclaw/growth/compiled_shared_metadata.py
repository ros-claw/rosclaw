"""Exact shared-document seals and optional owned full JSON restoration.

Canonical payload bytes are owned once. Each envelope still authenticates the
whole logical JSON document, optionally its root seal, and current dependencies.
Identity is not provenance, physical evidence, policy validation or authority.
Full restoration reads numerical data; it never loads or executes a policy.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rosclaw.growth import shared_proof_payload as reference


def _encode(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _chunks(document: dict[str, Any], location: tuple[str, ...], payload: bytes) -> Iterator[bytes]:
    if not location:
        yield payload
        return
    if type(document) is not dict or any(type(k) is not str for k in document):
        raise ValueError("explicit ordinary JSON object path required")
    yield b"{"
    for index, key in enumerate(sorted(document)):
        if index:
            yield b","
        yield _encode(key)
        yield b":"
        if key == location[0]:
            yield from _chunks(document[key], location[1:], payload)
        else:
            yield _encode(document[key])
    yield b"}"


def _logical_hash(document: dict[str, Any], location: tuple[str, ...], payload: bytes) -> str:
    digest = hashlib.sha256()
    for block in _chunks(document, location, payload):
        digest.update(block)
    return "sha256:" + digest.hexdigest()


class CompiledSharedMetadata:
    def __init__(self, payload: dict[str, Any]) -> None:
        if type(payload) is not dict or reference.MARKER in payload:
            raise ValueError("ordinary complete shared dictionary payload required")
        try:
            encoded = _encode(payload)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("finite canonical JSON payload required") from exc
        if not 2 <= len(encoded) <= 512 * 1024**2:
            raise ValueError("bounded shared payload required")
        if json.loads(encoded) != payload:
            raise ValueError("lossless ordinary canonical JSON payload required")
        self._payload_bytes = encoded
        self._payload_hash = "sha256:" + hashlib.sha256(encoded).hexdigest()
        self._pins = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(reference.__file__))
        }

    def verify(
        self, envelope: dict[str, Any], *, sealed_field: str | None = None
    ) -> dict[str, Any]:
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in self._pins.items()
        ):
            raise ValueError("compiled metadata dependencies changed")
        if type(envelope) is not dict:
            raise ValueError("exact shared-payload envelope required")
        value = copy.deepcopy(envelope)
        if (
            set(value)
            != {
                "schema",
                "location",
                "payload_hash",
                "logical_document_hash",
                "stripped_document",
                "envelope_hash",
            }
            or value["schema"] != reference.SCHEMA
        ):
            raise ValueError("exact shared-payload envelope required")
        try:
            if value["envelope_hash"] != reference.canonical_hash(
                {k: v for k, v in value.items() if k != "envelope_hash"}
            ):
                raise ValueError("shared-payload envelope seal changed")
            payload = self._payload_bytes
            digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            if digest != self._payload_hash or value["payload_hash"] != digest:
                raise ValueError("complete shared payload identity changed")
            location = reference._path(value["location"])
            document = value["stripped_document"]
            if type(document) is not dict:
                raise ValueError("complete stripped dictionary document required")
            marker: Any = document
            for key in location:
                if type(marker) is not dict or key not in marker:
                    raise ValueError("shared proof marker missing")
                marker = marker[key]
            if marker != {reference.MARKER: digest}:
                raise ValueError("exact single shared proof marker required")
            if _logical_hash(document, location, payload) != value["logical_document_hash"]:
                raise ValueError("shared payload changed complete logical document")
            if sealed_field is not None:
                if (
                    type(sealed_field) is not str
                    or not 1 <= len(sealed_field) <= 128
                    or sealed_field not in document
                    or sealed_field == location[0]
                ):
                    raise ValueError("explicit root seal outside the payload branch required")
                unsealed = {k: v for k, v in document.items() if k != sealed_field}
                if document[sealed_field] != _logical_hash(unsealed, location, payload):
                    raise ValueError("complete reconstructed root seal changed")
        except (TypeError, RecursionError) as exc:
            raise ValueError("finite ordinary canonical JSON metadata required") from exc
        return {
            "schema": "rosclaw.growth.verified_shared_metadata.v1",
            "metadata_with_payload_marker": document,
            "payload_hash": digest,
            "logical_document_hash": value["logical_document_hash"],
            "root_seal_verified": sealed_field is not None,
            "complete_numerical_document_returned": False,
            "identity_not_provenance_or_physical_verification": True,
            "policy_semantics_verified": False,
            "runtime_execution_authorized": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
        }

    def restore(
        self, envelope: dict[str, Any], *, sealed_field: str | None = None
    ) -> dict[str, Any]:
        """Restore all JSON fields from verified owned canonical payload bytes.

        Each call still verifies the whole envelope, payload, logical document
        and optional root seal. Each result is independently owned; callers
        must pin/recheck the actual input files and enforce evidence/authority
        separately. No compact marker is returned in place of the payload.
        """
        owned = copy.deepcopy(envelope)
        verified = self.verify(owned, sealed_field=sealed_field)
        payload = self._payload_bytes
        if (
            type(payload) is not bytes
            or "sha256:" + hashlib.sha256(payload).hexdigest() != verified["payload_hash"]
        ):
            raise ValueError("complete restoration payload changed")
        location = reference._path(owned["location"])
        encoded = b"".join(_chunks(verified["metadata_with_payload_marker"], location, payload))
        if "sha256:" + hashlib.sha256(encoded).hexdigest() != verified["logical_document_hash"]:
            raise ValueError("complete restoration document changed")
        result: dict[str, Any] = json.loads(encoded)
        return result
