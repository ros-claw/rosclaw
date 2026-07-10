"""ROSClaw extension schema for LeRobotDataset v3 sidecar metadata.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It defines the schema version and feature groups that describe how
ROSClaw-specific data is layered on top of a standard LeRobotDataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROSCLAW_EXTENSION_SCHEMA_VERSION = "rosclaw.lerobot.extension.v1"


FEATURE_GROUPS = {
    "safety",
    "intervention",
    "failure",
    "success",
    "physical_telemetry",
    "multi_camera",
    "depth",
}


@dataclass
class ExtensionSchema:
    """Description of the ROSClaw-rich dataset extension."""

    schema_version: str = ROSCLAW_EXTENSION_SCHEMA_VERSION
    dataset_format: str = "lerobot_v3"
    required_features: list[str] = field(default_factory=lambda: ["observation.state", "action"])
    optional_feature_groups: list[str] = field(default_factory=list)
    rosclaw_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_format": self.dataset_format,
            "required_features": list(self.required_features),
            "optional_feature_groups": list(self.optional_feature_groups),
            "rosclaw_fields": list(self.rosclaw_fields),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtensionSchema":
        return cls(
            schema_version=data.get("schema_version", ROSCLAW_EXTENSION_SCHEMA_VERSION),
            dataset_format=data.get("dataset_format", "lerobot_v3"),
            required_features=list(data.get("required_features", ["observation.state", "action"])),
            optional_feature_groups=[g for g in data.get("optional_feature_groups", []) if g in FEATURE_GROUPS],
            rosclaw_fields=list(data.get("rosclaw_fields", [])),
        )


def write_extension_schema(schema: ExtensionSchema, output_dir: Path | str) -> Path:
    """Write ``meta/rosclaw/schema.json`` under the dataset directory."""
    output_dir = Path(output_dir)
    sidecar_dir = output_dir / "meta" / "rosclaw"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_dir / "schema.json"
    path.write_text(json.dumps(schema.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_extension_schema(output_dir: Path | str) -> ExtensionSchema | None:
    """Read ``meta/rosclaw/schema.json`` if it exists."""
    path = Path(output_dir) / "meta" / "rosclaw" / "schema.json"
    if not path.exists():
        return None
    try:
        return ExtensionSchema.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "ROSCLAW_EXTENSION_SCHEMA_VERSION",
    "FEATURE_GROUPS",
    "ExtensionSchema",
    "write_extension_schema",
    "read_extension_schema",
]
