"""Tests for the runtime registry writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.refs import AssetRef
from rosclaw.hub.registry_writer import RegistryWriter, registry_entry_from_manifest
from rosclaw.hub.schema import AssetManifest, load_manifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hub_assets"


@pytest.fixture
def writer(tmp_path, monkeypatch) -> RegistryWriter:
    """Create a RegistryWriter in a temporary home directory."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    return RegistryWriter()


@pytest.fixture
def skill_manifest() -> AssetManifest:
    """Load the valid skill fixture manifest."""
    return load_manifest(FIXTURES / "skill_valid" / "manifest.yaml")


@pytest.fixture
def hardware_mcp_manifest() -> AssetManifest:
    """Load the valid hardware_mcp fixture manifest."""
    return load_manifest(FIXTURES / "hardware_mcp_valid" / "manifest.yaml")


def _manifest_with_entrypoint(manifest: AssetManifest, entrypoint: dict[str, Any]) -> AssetManifest:
    """Return a copy of *manifest* with the given install entrypoint."""
    manifest.install["entrypoints"] = entrypoint
    return manifest


# ---------------------------------------------------------------------------
# Entry building
# ---------------------------------------------------------------------------


def test_registry_entry_skill(skill_manifest: AssetManifest, tmp_path: Path) -> None:
    """Skill entries include ref, capabilities, components, and runtime."""
    entry = registry_entry_from_manifest(skill_manifest, tmp_path)
    assert entry["type"] == "skill"
    assert entry["ref"] == "rosclaw://skill/rosclaw/g1-pick-place@1.2.0"
    assert entry["namespace"] == "rosclaw"
    assert entry["name"] == "g1-pick-place"
    assert entry["version"] == "1.2.0"
    assert entry["asset_dir"] == str(tmp_path)
    assert "required_capabilities" not in entry  # mapped to "capabilities"
    assert entry["capabilities"] == [
        "perception.object_detection",
        "manipulation.grasp",
        "sandbox.trajectory_validation",
    ]


def test_registry_entry_hardware_mcp(hardware_mcp_manifest: AssetManifest, tmp_path: Path) -> None:
    """Hardware MCP entries parse command into program and args."""
    entry = registry_entry_from_manifest(hardware_mcp_manifest, tmp_path)
    assert entry["type"] == "hardware_mcp"
    assert entry["program"] == "python"
    assert entry["args"] == ["-m", "rosclaw_unitree_g1_mcp.server"]
    assert entry["transport"] == "stdio"
    assert entry["asset_dir"] == str(tmp_path)


def test_registry_entry_unknown_type(tmp_path: Path) -> None:
    """An unrecognized asset type raises an error."""
    fake_type = type("FakeType", (), {"value": "unknown_type"})
    fake_asset = type("FakeAsset", (), {"type": fake_type})
    fake_manifest = type("FakeManifest", (), {"asset": fake_asset})
    with pytest.raises(HubError) as exc_info:
        registry_entry_from_manifest(fake_manifest, tmp_path)  # type: ignore[arg-type]
    assert exc_info.value.code == HubErrorCode.INCOMPATIBLE_RUNTIME


# ---------------------------------------------------------------------------
# RegistryWriter operations
# ---------------------------------------------------------------------------


def test_add_asset_creates_registry_file(
    writer: RegistryWriter, skill_manifest: AssetManifest, tmp_path: Path
) -> None:
    """add_asset writes a versioned JSON registry file."""
    asset_dir = tmp_path / "installed" / "skill"
    path = writer.add_asset(skill_manifest, asset_dir)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == "1.0"
    ref = "rosclaw://skill/rosclaw/g1-pick-place@1.2.0"
    assert data["assets"][ref]["asset_dir"] == str(asset_dir)
    assert data["assets"][ref]["type"] == "skill"


def test_add_asset_replaces_existing(
    writer: RegistryWriter, skill_manifest: AssetManifest, tmp_path: Path
) -> None:
    """Adding the same asset replaces the previous entry."""
    writer.add_asset(skill_manifest, tmp_path / "a")
    writer.add_asset(skill_manifest, tmp_path / "b")
    entries = writer.list_assets("skill")
    assert len(entries) == 1
    assert entries[0]["asset_dir"] == str(tmp_path / "b")


def test_add_asset_multiple_types(
    writer: RegistryWriter,
    skill_manifest: AssetManifest,
    hardware_mcp_manifest: AssetManifest,
    tmp_path: Path,
) -> None:
    """Assets of different types are written to separate registry files."""
    writer.add_asset(skill_manifest, tmp_path / "skill")
    writer.add_asset(hardware_mcp_manifest, tmp_path / "mcp")
    assert len(writer.list_assets("skill")) == 1
    assert len(writer.list_assets("hardware_mcp")) == 1


def test_remove_asset(
    writer: RegistryWriter, skill_manifest: AssetManifest, tmp_path: Path
) -> None:
    """remove_asset deletes an entry and returns True."""
    writer.add_asset(skill_manifest, tmp_path)
    ref = AssetRef("skill", "rosclaw", "g1-pick-place", "1.2.0")
    assert writer.remove_asset(ref) is True
    assert writer.list_assets("skill") == []


def test_remove_asset_missing(writer: RegistryWriter) -> None:
    """remove_asset returns False when the registry or entry is missing."""
    ref = AssetRef("skill", "rosclaw", "missing", "1.0.0")
    assert writer.remove_asset(ref) is False


def test_registry_path_unknown_type(writer: RegistryWriter) -> None:
    """Requesting a registry path for an unknown type raises an error."""
    with pytest.raises(HubError) as exc_info:
        writer.registry_path("not_a_type")
    assert exc_info.value.code == HubErrorCode.INCOMPATIBLE_RUNTIME


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_load_corrupt_registry_file(writer: RegistryWriter, tmp_path: Path, monkeypatch) -> None:
    """A corrupt registry file raises HubError."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    reg_path = tmp_path / "runtime" / "registries" / "skills.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text("not json", encoding="utf-8")
    with pytest.raises(HubError) as exc_info:
        writer.list_assets("skill")
    assert exc_info.value.code == HubErrorCode.MANIFEST_INVALID


def test_load_registry_with_list_assets(
    writer: RegistryWriter, tmp_path: Path, monkeypatch
) -> None:
    """A registry whose assets are stored as a list is converted to a dict."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    reg_path = tmp_path / "runtime" / "registries" / "skills.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "updated_at": "",
                "assets": [{"ref": "rosclaw://skill/rosclaw/x@1.0.0", "type": "skill"}],
            }
        ),
        encoding="utf-8",
    )
    entries = writer.list_assets("skill")
    assert len(entries) == 1
    assert entries[0]["ref"] == "rosclaw://skill/rosclaw/x@1.0.0"


def test_entrypoint_env_coerced_to_strings(
    writer: RegistryWriter, hardware_mcp_manifest: AssetManifest, tmp_path: Path
) -> None:
    """Entrypoint environment values are converted to strings."""
    _manifest_with_entrypoint(
        hardware_mcp_manifest,
        {"mcp": {"command": "python -m mcp", "env": {"PORT": 8080, "DEBUG": True}}},
    )
    writer.add_asset(hardware_mcp_manifest, tmp_path)
    entry = writer.list_assets("hardware_mcp")[0]
    assert entry["command"] == "python -m mcp"
    assert entry.get("env") == {"PORT": "8080", "DEBUG": "True"}


def test_hardware_mcp_empty_command(
    writer: RegistryWriter, hardware_mcp_manifest: AssetManifest, tmp_path: Path
) -> None:
    """A hardware MCP with no command produces empty args and no program."""
    _manifest_with_entrypoint(hardware_mcp_manifest, {"mcp": {"command": ""}})
    writer.add_asset(hardware_mcp_manifest, tmp_path)
    entry = writer.list_assets("hardware_mcp")[0]
    assert entry["command"] == ""
    assert entry["args"] == []
    assert entry["program"] is None
