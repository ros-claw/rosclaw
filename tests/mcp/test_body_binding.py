"""Tests for Hardware MCP body/e-URDF binding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rosclaw.mcp.onboarding.binding import BindingResult, BodyBindingManager, _build_body_patch
from rosclaw.mcp.onboarding.errors import BodyNotLinkedError, EurdfProfileMissingError
from rosclaw.mcp.onboarding.schema import (
    BodyBindingTemplate,
    McpManifest,
)


def test_build_body_patch_with_write_paths() -> None:
    binding = BodyBindingTemplate(
        body_type="unitree-g1",
        binding_key="unitree_g1",
        write_paths={"mcpBinding": "mcp_bindings.unitree_g1"},
        template={
            "mcp_bindings": {
                "unitree_g1": {
                    "server": "rosclaw-unitree-g1",
                    "manifest_id": "io.rosclaw.hardware.unitree-g1",
                }
            }
        },
    )
    patch = _build_body_patch(binding)
    assert "mcp_bindings.unitree_g1" in patch
    assert patch["mcp_bindings.unitree_g1"]["server"] == "rosclaw-unitree-g1"


def test_build_body_patch_merge_top_level_untargeted() -> None:
    binding = BodyBindingTemplate(
        body_type="generic",
        binding_key="generic",
        write_paths={"a": "foo.bar"},
        template={"foo": {"bar": 1}, "baz": 2},
    )
    patch = _build_body_patch(binding)
    # baz is a top-level key not targeted, so it should be merged.
    assert patch["foo.bar"] == 1
    assert patch["baz"] == 2


def test_build_body_patch_skips_targeted_prefix() -> None:
    binding = BodyBindingTemplate(
        body_type="generic",
        binding_key="generic",
        write_paths={"a": "foo.bar"},
        template={"foo": {"bar": 1, "other": 2}},
    )
    patch = _build_body_patch(binding)
    assert "foo" not in patch
    assert patch["foo.bar"] == 1


def test_apply_binding_dry_run_missing_required_profile(
    unitree_manifest: McpManifest,
    fake_home: Path,
    monkeypatch_registry: Any,
) -> None:
    manager = BodyBindingManager(workspace=fake_home)
    with pytest.raises(EurdfProfileMissingError):
        manager.apply_binding(unitree_manifest, dry_run=True)


def test_apply_binding_dry_run_with_installed_profile(
    unitree_manifest: McpManifest,
    fake_home: Path,
    monkeypatch_registry: Any,
) -> None:
    monkeypatch_registry.install("unitree-g1")
    manager = BodyBindingManager(workspace=fake_home)
    result = manager.apply_binding(unitree_manifest, dry_run=True)
    assert isinstance(result, BindingResult)
    assert result.binding_key == "unitree_g1"
    assert "mcp_bindings.unitree_g1" in result.patched_paths
    assert result.eurdf_profile == "unitree-g1"
    assert result.eurdf_hash is not None


def test_apply_binding_writes_body_yaml(
    unitree_manifest: McpManifest,
    fake_home: Path,
    monkeypatch_registry: Any,
) -> None:
    monkeypatch_registry.install("unitree-g1")
    # Ensure body.yaml exists.
    body_path = fake_home / "body" / "body.yaml"
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text("schema_version: rosclaw.body.v1\n", encoding="utf-8")

    manager = BodyBindingManager(workspace=fake_home)
    result = manager.apply_binding(unitree_manifest, dry_run=False)

    data = yaml.safe_load(body_path.read_text(encoding="utf-8"))
    assert data["mcp_bindings"]["unitree_g1"]["server"] == "rosclaw-unitree-g1"
    assert result.eurdf_profile == "unitree-g1"


def test_apply_binding_raises_when_body_not_linked(
    unitree_manifest: McpManifest,
    fake_home: Path,
    monkeypatch_registry: Any,
) -> None:
    monkeypatch_registry.install("unitree-g1")
    manager = BodyBindingManager(workspace=fake_home)
    with pytest.raises(BodyNotLinkedError):
        manager.apply_binding(unitree_manifest, dry_run=False)


def test_no_binding_when_manifest_lacks_body_binding(
    unitree_manifest: McpManifest,
    fake_home: Path,
) -> None:
    manifest = unitree_manifest
    manifest.body_binding = None
    manager = BodyBindingManager(workspace=fake_home)
    result = manager.apply_binding(manifest, dry_run=True)
    assert result.binding_key == manifest.server_name
    assert result.patched_paths == []


def test_optional_profile_missing_not_error(
    realsense_manifest: McpManifest,
    fake_home: Path,
    monkeypatch_registry: Any,
) -> None:
    # realsense e-URDF profile is optional.
    manager = BodyBindingManager(workspace=fake_home)
    body_path = fake_home / "body" / "body.yaml"
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text("schema_version: rosclaw.body.v1\n", encoding="utf-8")
    result = manager.apply_binding(realsense_manifest, dry_run=False)
    assert result.eurdf_profile is None
    assert "mcp_bindings.realsense_d455" in result.patched_paths
