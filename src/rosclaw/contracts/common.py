"""Common machinery for ROSClaw contracts (ADR-0000 §2).

- ``schema_version`` strings look like ``rosclaw.<domain>.<entity>.v<N>``.
- Readers are forward-compatible: unknown *fields* are preserved, an unknown
  *major version* is rejected (fail closed).
- Every contract object has a canonical JSON form (sorted keys, compact
  separators) and a typed content hash (``<prefix>_`` + sha256 hex), computed
  over the canonical form with the hash field itself excluded.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION_RE = re.compile(r"^rosclaw\.[a-z0-9_]+(?:\.[a-z0-9_]+)*\.v(\d+)$")


class ContractError(Exception):
    """Base error for contract violations."""


class UnsupportedVersionError(ContractError):
    """Raised when a payload's schema major version is not supported."""


class ValidationError(ContractError):
    """Raised when a payload fails structural or semantic validation."""


def parse_schema_version(value: str) -> tuple[str, int]:
    """Split ``rosclaw.domain.entity.vN`` into (stem, major). Fail closed."""
    match = SCHEMA_VERSION_RE.match(value or "")
    if not match:
        raise ValidationError(f"invalid schema_version: {value!r}")
    stem = value.rsplit(".v", 1)[0]
    return stem, int(match.group(1))


def canonical_json(data: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, UTF-8 safe."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(prefix: str, data: Any) -> str:
    """Typed content hash, e.g. ``ctxb_9f2a...`` over canonical JSON."""
    digest = hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def new_id(prefix: str) -> str:
    """Fresh object id, e.g. ``mis_01H...`` (uuid4 hex suffix)."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class ContractModel(BaseModel):
    """Base class for all versioned contracts.

    Subclasses set ``SCHEMA`` (e.g. ``rosclaw.mission_session.v1``),
    ``HASH_PREFIX`` and declare ``schema_version`` as a literal field.
    Unknown fields are kept (forward compatibility); unknown major versions
    are rejected by :meth:`model_validate_contract`.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=False, populate_by_name=True)

    SCHEMA: ClassVar[str] = ""
    HASH_PREFIX: ClassVar[str] = "contract"
    #: Field name excluded from the content hash ("" = hash over everything).
    HASH_EXCLUDE_FIELD: ClassVar[str] = ""

    @classmethod
    def supported_major(cls) -> int:
        return parse_schema_version(cls.SCHEMA)[1]

    @classmethod
    def check_version(cls, payload: dict[str, Any]) -> None:
        version = payload.get("schema_version", "")
        stem, major = parse_schema_version(version)
        expected_stem = parse_schema_version(cls.SCHEMA)[0]
        if stem != expected_stem:
            raise ValidationError(f"schema mismatch: expected {expected_stem!r}, got {stem!r}")
        if major != cls.supported_major():
            raise UnsupportedVersionError(
                f"unsupported major version v{major} for {stem} "
                f"(reader supports v{cls.supported_major()})"
            )

    @classmethod
    def model_validate_contract(cls, payload: dict[str, Any]) -> ContractModel:
        """Validate with explicit version gate before pydantic parsing."""
        if not isinstance(payload, dict):
            raise ValidationError(f"contract payload must be a mapping, got {type(payload)}")
        cls.check_version(payload)
        return cls.model_validate(payload)

    def to_canonical_dict(self) -> dict[str, Any]:
        """Round-trippable dict with deterministic key handling."""
        return self.model_dump(mode="json", by_alias=True)

    def hash_payload(self) -> dict[str, Any]:
        data = self.to_canonical_dict()
        if self.HASH_EXCLUDE_FIELD:
            data.pop(self.HASH_EXCLUDE_FIELD, None)
        return data

    def canonical_hash(self) -> str:
        return content_hash(self.HASH_PREFIX, self.hash_payload())
