"""Tests for ROSClaw extension schema sidecar."""

from __future__ import annotations

from rosclaw.integrations.lerobot.dataset_extension_schema import (
    ROSCLAW_EXTENSION_SCHEMA_VERSION,
    ExtensionSchema,
    read_extension_schema,
    write_extension_schema,
)


def test_write_and_read_extension_schema(tmp_path) -> None:
    schema = ExtensionSchema(
        schema_version=ROSCLAW_EXTENSION_SCHEMA_VERSION,
        required_features=["observation.state", "action"],
        optional_feature_groups=["safety", "failure"],
        rosclaw_fields=["rosclaw.sandbox.decision"],
        dataset_format="lerobot_v3",
    )
    write_extension_schema(schema, tmp_path)
    loaded = read_extension_schema(tmp_path)
    assert loaded is not None
    assert loaded.schema_version == ROSCLAW_EXTENSION_SCHEMA_VERSION
    assert loaded.optional_feature_groups == ["safety", "failure"]
    assert "rosclaw.sandbox.decision" in loaded.rosclaw_fields


def test_read_missing_schema(tmp_path) -> None:
    assert read_extension_schema(tmp_path) is None
