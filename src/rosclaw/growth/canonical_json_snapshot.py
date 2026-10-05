"""Owned canonical JSON bytes; content identity is not execution authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True, init=False)
class CanonicalJSONSnapshot:
    """Serialize once, integrity-check bytes, return independently owned JSON.

    This is a private persistence optimization, not a policy validator. Callers
    must validate the complete document before allocation and retain their
    source/provenance checks. No references or payload fields are omitted.
    """

    _data: bytes
    content_hash: str

    def __init__(self, document: dict[str, Any]) -> None:
        if type(document) is not dict:
            raise ValueError("complete ordinary JSON dictionary required")
        data = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        # Reject JSON-coercible but lossy Python structures (integer keys,
        # tuples, etc.). A snapshot must preserve the whole logical document.
        restored = json.loads(data)
        if restored != document:
            raise ValueError("lossless ordinary JSON dictionary required")
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "content_hash", "sha256:" + hashlib.sha256(data).hexdigest())

    def verify(self) -> str:
        if (
            type(self._data) is not bytes
            or self.content_hash != "sha256:" + hashlib.sha256(self._data).hexdigest()
        ):
            raise ValueError("immutable canonical snapshot changed")
        return self.content_hash

    def restore(self) -> dict[str, Any]:
        self.verify()
        result: dict[str, Any] = json.loads(self._data)
        return result
