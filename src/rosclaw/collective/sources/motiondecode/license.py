"""Fail-closed interpretation of the custom ChingMu access terms."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_MAX_LICENSE_BYTES = 1024 * 1024
_ALLOWED_USAGE = {"research", "personal-study", "noncommercial-prototype"}


@dataclass(frozen=True)
class MotionDecodeLicenseDecision:
    license_file: str
    license_hash: str
    requested_usage: str
    permitted: bool
    commercial_use_status: str
    attribution_required: bool
    redistribution_permitted: bool
    warnings: tuple[str, ...]
    schema_version: str = "rosclaw.collective.motiondecode_license.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_motiondecode_license(
    dataset_root: Path,
    *,
    requested_usage: str,
) -> MotionDecodeLicenseDecision:
    root = dataset_root.expanduser().resolve()
    usage = requested_usage.strip().lower()
    warnings: list[str] = []
    advertised = root / "LICENSE"
    terms = root / "LICENSE.md"
    if advertised.is_file() and advertised.stat().st_size == 0:
        warnings.append("README license_link points to empty LICENSE; using non-empty LICENSE.md")
    if not terms.is_file() or terms.is_symlink():
        raise ValueError("MotionDecode LICENSE.md is missing or unsafe")
    size = terms.stat().st_size
    if not 1 <= size <= _MAX_LICENSE_BYTES:
        raise ValueError("MotionDecode LICENSE.md is empty or too large")
    payload = terms.read_bytes()
    text = payload.decode("utf-8")
    required_phrases = (
        "non-commercial",
        "Prohibited without written permission",
        "retain attribution",
    )
    if any(phrase not in text for phrase in required_phrases):
        raise ValueError("MotionDecode access terms do not match the reviewed contract")
    permitted = usage in _ALLOWED_USAGE
    return MotionDecodeLicenseDecision(
        license_file="LICENSE.md",
        license_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        requested_usage=usage,
        permitted=permitted,
        commercial_use_status="WRITTEN_PERMISSION_REQUIRED",
        attribution_required=True,
        redistribution_permitted=False,
        warnings=tuple(warnings),
    )
