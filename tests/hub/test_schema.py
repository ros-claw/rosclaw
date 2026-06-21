"""Tests for ROSClaw Hub manifest schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml
from pydantic import ValidationError

from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.schema import (
    AssetManifest,
    AssetType,
    LifecycleStatus,
    TrustLevel,
    VisibilityScope,
    dump_manifest_schema,
    load_manifest,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "hub_assets"


class TestAssetTypeEnums:
    """Tests for schema enumerations."""

    def test_asset_type_values(self) -> None:
        assert AssetType.skill.value == "skill"
        assert AssetType.provider.value == "provider"
        assert AssetType.hardware_mcp.value == "hardware_mcp"
        assert AssetType.digital_twin.value == "digital_twin"
        assert AssetType.cognitive_wiki.value == "cognitive_wiki"

    def test_visibility_scope_values(self) -> None:
        assert VisibilityScope.public.value == "public"
        assert VisibilityScope.private.value == "private"

    def test_lifecycle_status_values(self) -> None:
        assert LifecycleStatus.stable.value == "stable"
        assert LifecycleStatus.experimental.value == "experimental"

    def test_trust_level_default(self) -> None:
        assert TrustLevel.unknown.value == "unknown"


class TestLoadManifest:
    """Tests for loading manifests from fixtures."""

    @pytest.mark.parametrize(
        "asset_dir",
        [
            "skill_valid",
            "provider_valid",
            "hardware_mcp_valid",
            "digital_twin_valid",
            "cognitive_wiki_valid",
        ],
    )
    def test_valid_fixtures_load(self, asset_dir: str) -> None:
        manifest = load_manifest(FIXTURES_DIR / asset_dir / "manifest.yaml")
        assert isinstance(manifest, AssetManifest)
        assert manifest.schema_version == "hub.asset.v1"
        assert manifest.asset.type.value in {
            "skill",
            "provider",
            "hardware_mcp",
            "digital_twin",
            "cognitive_wiki",
        }

    def test_missing_file_raises_hub_error(self) -> None:
        with pytest.raises(HubError) as exc_info:
            load_manifest(FIXTURES_DIR / "nonexistent" / "manifest.yaml")
        assert exc_info.value.code == HubErrorCode.ASSET_NOT_FOUND

    def test_invalid_yaml_raises_hub_error(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("not valid yaml: [", encoding="utf-8")
        with pytest.raises(HubError) as exc_info:
            load_manifest(bad_file)
        assert exc_info.value.code == HubErrorCode.MANIFEST_INVALID


class TestManifestValidation:
    """Tests for manifest validation rules."""

    def _minimal_manifest(self, asset_type: str, special: dict) -> dict:
        return {
            "schema_version": "hub.asset.v1",
            "asset": {
                "type": asset_type,
                "namespace": "rosclaw",
                "name": "minimal-asset",
                "version": "1.0.0",
                "title": "Minimal Asset",
                "summary": "A minimal asset for testing",
            },
            "publisher": {"id": "test", "display_name": "Test Publisher"},
            "visibility": {"scope": "public"},
            "lifecycle": {"status": "stable"},
            "compatibility": {
                "rosclaw": {"min_version": "1.0.0", "max_version": None},
                "os": ["linux"],
                "arch": ["x86_64"],
                "python": {"requires": ">=3.11"},
                "ros": {"distributions": [], "required": False},
                "cuda": {"required": False, "min_version": None},
                "robot": {"eurdf_profiles": [], "body_kinds": []},
                "hardware": {"required_devices": []},
                "runtime_features": [],
            },
            "permissions": {
                "hardware": {
                    "real_robot_execution": False,
                    "actuators": [],
                    "sensors": [],
                },
                "ros": {"topics_read": [], "topics_write": [], "services": [], "actions": []},
                "mcp": {"tools": []},
                "filesystem": {"read": [], "write": []},
                "network": {"outbound": [], "inbound": []},
                "modifies": {
                    "mcp_config": False,
                    "body_yaml": False,
                    "rosclaw_yaml": False,
                    "safety_config": False,
                },
                "requires_human_approval": [],
            },
            "license": {
                "spdx": "MIT",
                "license_file": "LICENSE",
                "commercial_use": True,
                "redistribution": True,
                "attribution_required": True,
                "export_control": "none",
            },
            "data_rights": {
                "contains_training_data": False,
                "contains_robot_logs": False,
                "contains_personal_data": False,
                "allowed_usage": ["research"],
                "restrictions": [],
            },
            "security": {
                "signing": {"required": False},
                "checksums": {"algorithm": "sha256", "file": "checksums.txt"},
                "sbom": None,
                "provenance": None,
                "sandbox_required": False,
                "network_isolation_recommended": False,
            },
            "artifacts": [],
            "install": {"mode": "declarative", "entrypoints": {}, "registries": {}},
            "special": special,
        }

    def test_special_section_matches_type(self) -> None:
        manifest = AssetManifest.model_validate(
            self._minimal_manifest("skill", {"skill": {"task_domain": "test"}})
        )
        assert manifest.asset.type == AssetType.skill
        assert "skill" in manifest.special

    def test_missing_special_section_for_type_fails(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AssetManifest.model_validate(
                self._minimal_manifest("provider", {"skill": {"task_domain": "test"}})
            )
        assert "special.provider is required" in str(exc_info.value)

    def test_invalid_visibility_scope_fails(self) -> None:
        data = self._minimal_manifest("skill", {"skill": {"task_domain": "test"}})
        data["visibility"]["scope"] = "super_public"
        with pytest.raises(ValidationError) as exc_info:
            AssetManifest.model_validate(data)
        assert "visibility.scope" in str(exc_info.value)

    def test_invalid_lifecycle_status_fails(self) -> None:
        data = self._minimal_manifest("skill", {"skill": {"task_domain": "test"}})
        data["lifecycle"]["status"] = "production"
        with pytest.raises(ValidationError) as exc_info:
            AssetManifest.model_validate(data)
        assert "lifecycle.status" in str(exc_info.value)

    def test_artifact_digest_validation(self) -> None:
        data = self._minimal_manifest("skill", {"skill": {"task_domain": "test"}})
        data["artifacts"] = [
            {"name": "good", "digest": "sha256:abcd"},
            {"name": "bad", "digest": "not-colon-separated"},
        ]
        with pytest.raises(ValidationError) as exc_info:
            AssetManifest.model_validate(data)
        assert "artifacts[1].digest" in str(exc_info.value)


class TestSchemaExport:
    """Tests for JSON Schema export."""

    def test_schema_export_json(self) -> None:
        output = dump_manifest_schema(format="json")
        assert "schema_version" in output
        schema = yaml.safe_load(output)
        assert schema["title"] == "AssetManifest"

    def test_schema_export_yaml(self) -> None:
        output = dump_manifest_schema(format="yaml")
        schema = yaml.safe_load(output)
        assert schema["title"] == "AssetManifest"
        assert "AssetType" in schema.get("$defs", {})

    def test_model_json_schema_structure(self) -> None:
        schema = AssetManifest.model_json_schema()
        assert schema["$defs"]["AssetType"]["enum"] == [
            "skill",
            "provider",
            "hardware_mcp",
            "digital_twin",
            "cognitive_wiki",
        ]


class TestSchemaMetaValidation:
    """Tests validating the exported JSON Schema and fixtures against it."""

    def _exported_schema(self) -> dict[str, Any]:
        return AssetManifest.model_json_schema()

    def test_exported_schema_is_valid_json_schema(self) -> None:
        schema = self._exported_schema()
        jsonschema.Draft202012Validator.check_schema(schema)

    @pytest.mark.parametrize(
        "asset_dir",
        [
            "skill_valid",
            "provider_valid",
            "hardware_mcp_valid",
            "digital_twin_valid",
            "cognitive_wiki_valid",
        ],
    )
    def test_valid_fixtures_validate_against_schema(self, asset_dir: str) -> None:
        raw = yaml.safe_load(
            (FIXTURES_DIR / asset_dir / "manifest.yaml").read_text(encoding="utf-8")
        )
        schema = self._exported_schema()
        jsonschema.validate(raw, schema)

    def test_invalid_fixture_fails_schema_validation(self) -> None:
        schema = self._exported_schema()
        bad = {
            "schema_version": "hub.asset.v1",
            "asset": {
                "type": "not_a_real_type",
                "namespace": "rosclaw",
                "name": "minimal-asset",
                "version": "1.0.0",
                "title": "Minimal Asset",
                "summary": "A minimal asset for testing",
            },
            "publisher": {"id": "test", "display_name": "Test Publisher"},
            "visibility": {"scope": "public"},
            "lifecycle": {"status": "stable"},
            "compatibility": {},
            "permissions": {},
            "license": {},
            "data_rights": {},
            "security": {},
            "artifacts": [],
            "install": {},
            "special": {},
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
