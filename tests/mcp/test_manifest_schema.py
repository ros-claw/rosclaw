"""Unit tests for Hardware MCP manifest schema parsing and serialization."""

from __future__ import annotations

from typing import Any

from rosclaw.mcp.onboarding.schema import McpManifest


def test_unitree_manifest_loads(unitree_manifest: McpManifest) -> None:
    assert unitree_manifest.id == "io.rosclaw.hardware.unitree-g1"
    assert unitree_manifest.name == "unitree-g1"
    assert unitree_manifest.version == "1.0.0"
    assert unitree_manifest.display_name == "Unitree G1 Humanoid"
    assert unitree_manifest.server_name == "unitree-g1"


def test_realsense_manifest_loads(realsense_manifest: McpManifest) -> None:
    assert realsense_manifest.id == "io.rosclaw.hardware.realsense-d455"
    assert realsense_manifest.name == "realsense-d455"
    assert realsense_manifest.hardware is not None
    assert realsense_manifest.hardware.type == "sensor"


def test_manifest_preserves_unknown_extra_keys() -> None:
    data = {
        "id": "io.rosclaw.hardware.test",
        "name": "test",
        "version": "0.0.1",
        "displayName": "Test",
        "x_custom_field": {"nested": True},
    }
    manifest = McpManifest.from_dict(data)
    assert manifest.extra.get("x_custom_field") == {"nested": True}
    assert "x_custom_field" in manifest.to_dict()


def test_manifest_roundtrip(unitree_manifest_dict: dict[str, Any]) -> None:
    manifest = McpManifest.from_dict(unitree_manifest_dict)
    out = manifest.to_dict()
    assert out["id"] == unitree_manifest_dict["id"]
    assert out["mcp"]["serverName"] == unitree_manifest_dict["mcp"]["serverName"]
    assert out["eurdf"]["profiles"][0]["id"] == "unitree-g1"


def test_permissions_levels_parsed(unitree_manifest: McpManifest) -> None:
    assert unitree_manifest.permissions is not None
    required = {p.id: p.level for p in unitree_manifest.permissions.required}
    assert required["mcp:tools:read"] == "safe"
    assert required["mcp:prompts:read"] == "guarded"


def test_manifest_defaults_for_minimal_dict() -> None:
    data = {
        "id": "io.rosclaw.hardware.minimal",
        "name": "minimal",
        "version": "0.0.1",
    }
    manifest = McpManifest.from_dict(data)
    assert manifest.display_name == "minimal"
    assert manifest.schema_version == "1.0.0"
    assert manifest.channel == "stable"
    assert manifest.mcp is None
    assert manifest.permissions is not None
    assert manifest.permissions.required == []


def test_body_binding_write_paths(unitree_manifest: McpManifest) -> None:
    binding = unitree_manifest.body_binding
    assert binding is not None
    assert binding.binding_key == "unitree_g1"
    assert binding.write_paths.get("mcpBinding") == "mcp_bindings.unitree_g1"
    assert binding.template["mcp_bindings"]["unitree_g1"]["server"] == "rosclaw-unitree-g1"


def test_server_name_fallback_when_no_mcp() -> None:
    data = {
        "id": "io.rosclaw.hardware.no-mcp",
        "name": "no-mcp",
        "version": "1.0.0",
    }
    manifest = McpManifest.from_dict(data)
    assert manifest.server_name == "rosclaw-no-mcp"
