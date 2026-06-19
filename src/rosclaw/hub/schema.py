"""Pydantic v2 manifest schema for ROSClaw Hub assets."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field, model_validator

from rosclaw.hub.errors import HubError, HubErrorCode


class AssetType(StrEnum):
    """Canonical asset type enumeration."""

    skill = "skill"
    provider = "provider"
    hardware_mcp = "hardware_mcp"
    digital_twin = "digital_twin"
    cognitive_wiki = "cognitive_wiki"


class VisibilityScope(StrEnum):
    """Visibility scope for an asset."""

    public = "public"
    private = "private"
    org = "org"
    unlisted = "unlisted"


class TrustLevel(StrEnum):
    """Publisher trust level."""

    official = "official"
    verified = "verified"
    community = "community"
    private = "private"
    unknown = "unknown"


class LifecycleStatus(StrEnum):
    """Asset lifecycle status."""

    experimental = "experimental"
    beta = "beta"
    stable = "stable"
    deprecated = "deprecated"


class AssetIdentity(BaseModel):
    """Identity section of an asset manifest."""

    type: AssetType
    namespace: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{1,127}$")
    version: str
    title: str
    summary: str
    description: str | None = None
    tags: list[str] = []


class Publisher(BaseModel):
    """Publisher metadata."""

    id: str
    display_name: str
    trust_level: TrustLevel = TrustLevel.unknown
    contact: str | None = None


class AssetManifest(BaseModel):
    """Unified ROSClaw Hub asset manifest.

    All five asset types share this top-level structure. The ``special``
    section is required to contain an entry matching ``asset.type``.
    """

    schema_version: Literal["hub.asset.v1"]
    asset: AssetIdentity
    publisher: Publisher
    visibility: dict[str, Any]
    lifecycle: dict[str, Any]
    compatibility: dict[str, Any]
    dependencies: dict[str, Any] = {}
    permissions: dict[str, Any]
    license: dict[str, Any]
    data_rights: dict[str, Any]
    security: dict[str, Any]
    artifacts: list[dict[str, Any]]
    install: dict[str, Any]
    special: dict[str, Any] = {}

    @model_validator(mode="after")
    def validate_special_matches_type(self) -> AssetManifest:
        asset_type = self.asset.type.value
        if asset_type not in self.special:
            raise ValueError(
                f"special.{asset_type} is required for asset type {asset_type}"
            )
        return self

    @model_validator(mode="after")
    def validate_visibility_scope(self) -> AssetManifest:
        scope = self.visibility.get("scope", "public")
        if scope not in VisibilityScope.__members__:
            raise ValueError(f"visibility.scope must be one of {list(VisibilityScope)}")
        return self

    @model_validator(mode="after")
    def validate_lifecycle_status(self) -> AssetManifest:
        status = self.lifecycle.get("status", "stable")
        if status not in LifecycleStatus.__members__:
            raise ValueError(f"lifecycle.status must be one of {list(LifecycleStatus)}")
        return self

    @model_validator(mode="after")
    def validate_artifacts_have_digest(self) -> AssetManifest:
        for idx, artifact in enumerate(self.artifacts):
            if "name" not in artifact:
                raise ValueError(f"artifacts[{idx}].name is required")
            if "digest" in artifact:
                digest = artifact["digest"]
                if not isinstance(digest, str) or ":" not in digest:
                    raise ValueError(
                        f"artifacts[{idx}].digest must be in the form algorithm:hash"
                    )
        return self

    def to_json_schema(self) -> dict[str, Any]:
        """Export the manifest JSON Schema."""
        return self.model_json_schema()


def load_manifest(path: str | Path) -> AssetManifest:
    """Load and validate a manifest YAML file.

    Args:
        path: Path to ``manifest.yaml``.

    Returns:
        Validated AssetManifest instance.

    Raises:
        HubError: If the file cannot be read or the manifest is invalid.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HubError(
            code=HubErrorCode.ASSET_NOT_FOUND,
            message=f"Manifest not found: {path}",
        ) from exc
    except yaml.YAMLError as exc:
        raise HubError(
            code=HubErrorCode.MANIFEST_INVALID,
            message=f"Invalid YAML in manifest: {exc}",
        ) from exc

    if not isinstance(raw, dict):
        raise HubError(
            code=HubErrorCode.MANIFEST_INVALID,
            message="Manifest must be a YAML mapping",
        )

    try:
        return AssetManifest.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - Pydantic validation can raise many types
        raise HubError(
            code=HubErrorCode.MANIFEST_INVALID,
            message=f"Manifest validation failed: {exc}",
        ) from exc


def dump_manifest_schema(*, format: Literal["json", "yaml"] = "json") -> str:
    """Return the manifest JSON Schema as a formatted string.

    Args:
        format: Output format (``json`` or ``yaml``).

    Returns:
        Formatted schema string.
    """
    schema = AssetManifest.model_json_schema()
    if format == "json":
        return json.dumps(schema, indent=2, ensure_ascii=False)
    return cast(str, yaml.safe_dump(schema, sort_keys=False, allow_unicode=True))
