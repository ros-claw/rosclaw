"""ROSClaw asset URI parser and reference types."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rosclaw.hub.errors import HubError, HubErrorCode

# Asset URI format: rosclaw://<type>/<namespace>/<name>@<version_or_range>
_REF_RE = re.compile(
    r"^rosclaw://"
    r"(?P<type>skill|provider|hardware_mcp|digital_twin|cognitive_wiki)"
    r"/(?P<namespace>[a-z0-9][a-z0-9_-]{1,63})"
    r"/(?P<name>[a-z0-9][a-z0-9_.-]{1,127})"
    r"(?:@(?P<version>[^/\s]+))?"
    r"$"
)


def normalize_name(name: str) -> str:
    """Normalize an asset name to lower-case with dashes for separators.

    This is a convenience helper used by publisher tooling; it does not
    alter valid manifest names that already conform to the schema.
    """
    return re.sub(r"[_.]+", "-", name).strip("-").lower()


@dataclass(frozen=True, slots=True)
class AssetRef:
    """Canonical reference to a ROSClaw Hub asset.

    Attributes:
        type: Asset type.
        namespace: Publisher namespace.
        name: Asset name.
        version: Optional version or range string.
    """

    type: str
    namespace: str
    name: str
    version: str | None

    def __str__(self) -> str:
        base = f"rosclaw://{self.type}/{self.namespace}/{self.name}"
        if self.version:
            base = f"{base}@{self.version}"
        return base

    def canonical(self) -> str:
        """Return the canonical fully-qualified form (alias for __str__)."""
        return str(self)

    def identity_tuple(self) -> tuple[str, str, str, str | None]:
        """Return a hashable identity tuple."""
        return (self.type, self.namespace, self.name, self.version)


def parse_ref(ref: str) -> AssetRef:
    """Parse a rosclaw:// URI into an AssetRef.

    Args:
        ref: Asset URI string.

    Returns:
        AssetRef instance.

    Raises:
        HubError: If the reference is malformed.
    """
    if not isinstance(ref, str):
        raise HubError(
            code=HubErrorCode.MANIFEST_INVALID,
            message=f"Asset reference must be a string, got {type(ref).__name__}",
            suggested_fix="Use a rosclaw://<type>/<namespace>/<name>@<version> string.",
        )

    match = _REF_RE.match(ref)
    if not match:
        raise HubError(
            code=HubErrorCode.MANIFEST_INVALID,
            message=f"Invalid asset reference: {ref!r}",
            suggested_fix=(
                "Expected format: rosclaw://<type>/<namespace>/<name>@<version> "
                "where type is one of skill, provider, hardware_mcp, digital_twin, cognitive_wiki."
            ),
        )

    parts = match.groupdict()
    return AssetRef(
        type=parts["type"],
        namespace=parts["namespace"],
        name=parts["name"],
        version=parts.get("version") or None,
    )


def parse_ref_with_version(ref: str, *, version: str | None = None) -> AssetRef:
    """Parse a ref and optionally fill in a missing version.

    Args:
        ref: Asset URI string.
        version: Optional version to use if the ref lacks one.

    Returns:
        AssetRef with version set.
    """
    parsed = parse_ref(ref)
    if version:
        return AssetRef(
            type=parsed.type,
            namespace=parsed.namespace,
            name=parsed.name,
            version=version,
        )
    return parsed


def ref_from_dict(data: dict[str, Any]) -> AssetRef:
    """Build an AssetRef from a manifest ``asset`` block."""
    return AssetRef(
        type=data["type"],
        namespace=data["namespace"],
        name=data["name"],
        version=data.get("version"),
    )
