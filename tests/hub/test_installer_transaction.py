"""Transaction and rollback tests for the ROSClaw Hub installer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rosclaw.hub.cache import HubCache
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.installer import Installer, InstallOptions
from rosclaw.hub.mcp_merge import McpMerger
from rosclaw.hub.refs import AssetRef
from rosclaw.hub.registry_writer import RegistryWriter

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "hub_assets"
HARDWARE_MCP_FIXTURE = FIXTURES_DIR / "hardware_mcp_valid"


def _hardware_ref() -> AssetRef:
    return AssetRef(
        type="hardware_mcp",
        namespace="rosclaw",
        name="unitree-g1",
        version="1.0.0",
    )


def _install_options(**kwargs: Any) -> InstallOptions:
    defaults = {
        "verify_signature": False,
        "skip_health": True,
        "allow_real_robot": True,
    }
    defaults.update(kwargs)
    return InstallOptions(**defaults)


class TestInstallSuccess:
    """Happy-path install transaction outcomes."""

    def test_install_local_hardware_mcp(self, tmp_path: Path) -> None:
        """Installing a valid local asset writes all expected side effects."""
        home = tmp_path / "home"
        installer = Installer(home=home, project_root=home)
        options = _install_options()

        result = installer.install_local(HARDWARE_MCP_FIXTURE, options=options)

        ref = _hardware_ref()
        assert result.success is True
        assert result.ref == ref
        assert result.dry_run is False
        assert result.mcp_server_name is not None

        # Installed directory exists and contains the manifest.
        assert result.asset_dir.exists()
        assert (result.asset_dir / "manifest.yaml").exists()

        # Lockfile records the asset.
        lock = installer.assets_lock
        assert lock.is_installed(ref)
        entry = lock.get(ref)
        assert entry is not None
        assert entry.asset_dir == str(result.asset_dir.resolve())

        # Cache record exists.
        record = installer.cache.get_installed(ref)
        assert record is not None
        assert record["ref"] == str(ref)

        # Registry entry exists.
        registry = RegistryWriter(home=home)
        entries = registry.list_assets("hardware_mcp")
        assert any(e["ref"] == str(ref) for e in entries)

        # MCP config contains the managed server.
        merger = McpMerger(project_root=home, home=home)
        managed = merger.list_servers()
        assert result.mcp_server_name in managed
        assert managed[result.mcp_server_name]["rosclaw"]["ref"] == str(ref)


class TestInstallRollback:
    """Partial install failures are rolled back consistently."""

    def test_registry_failure_rolls_back_target_dir(self, tmp_path: Path) -> None:
        """If registry.add_asset fails after copy, target dir is removed."""
        home = tmp_path / "home"

        class FailingRegistry(RegistryWriter):
            def add_asset(self, manifest: Any, asset_dir: Path) -> Path:
                raise HubError(
                    code=HubErrorCode.REGISTRY_UNREACHABLE,
                    message="forced registry failure",
                )

        installer = Installer(
            home=home,
            project_root=home,
            registry=FailingRegistry(home=home),
        )
        options = _install_options()

        with pytest.raises(HubError) as exc_info:
            installer.install_local(HARDWARE_MCP_FIXTURE, options=options)
        assert exc_info.value.code == HubErrorCode.REGISTRY_UNREACHABLE

        ref = _hardware_ref()
        target_dir = installer._asset_install_dir(ref)
        assert not target_dir.exists()

        # No lockfile entry, no cache record, no mcp merge.
        assert not installer.assets_lock.is_installed(ref)
        assert installer.cache.get_installed(ref) is None
        mcp_path = home / ".mcp.json"
        assert not mcp_path.exists() or str(ref) not in mcp_path.read_text()

    def test_post_mcp_failure_rolls_back_all_side_effects(self, tmp_path: Path) -> None:
        """If a step after MCP merge fails, registry and MCP are both cleaned."""
        home = tmp_path / "home"

        class FailingCache(HubCache):
            def set_installed(self, ref: AssetRef, entry: dict[str, Any]) -> Path:
                raise RuntimeError("forced cache failure")

        cache = FailingCache(home=home)
        installer = Installer(home=home, project_root=home, cache=cache)
        options = _install_options()
        ref = _hardware_ref()

        with pytest.raises(RuntimeError, match="forced cache failure"):
            installer.install_local(HARDWARE_MCP_FIXTURE, options=options)

        target_dir = installer._asset_install_dir(ref)
        assert not target_dir.exists()

        # Registry entry removed.
        registry = RegistryWriter(home=home)
        assert not any(e["ref"] == str(ref) for e in registry.list_assets("hardware_mcp"))

        # MCP server removed.
        merger = McpMerger(project_root=home, home=home)
        assert not merger.is_managed(ref)
        assert not any(f.name.endswith(".json") for f in merger.fragments_dir.iterdir())

        # Lockfile entry removed.
        assert not installer.assets_lock.is_installed(ref)


class TestInstallIdempotency:
    """Re-installing an already-installed asset is rejected."""

    def test_install_same_asset_twice_raises(self, tmp_path: Path) -> None:
        """Second install of the same ref fails with ASSET_ALREADY_INSTALLED."""
        home = tmp_path / "home"
        installer = Installer(home=home, project_root=home)
        options = _install_options()

        result = installer.install_local(HARDWARE_MCP_FIXTURE, options=options)
        assert result.success is True

        with pytest.raises(HubError) as exc_info:
            installer.install_local(HARDWARE_MCP_FIXTURE, options=options)
        assert exc_info.value.code == HubErrorCode.ASSET_ALREADY_INSTALLED


class TestUninstall:
    """Uninstall removes every side effect created by install."""

    def test_uninstall_removes_target_dir_registry_mcp_and_lockfile(self, tmp_path: Path) -> None:
        """A full uninstall leaves no trace of the asset."""
        home = tmp_path / "home"
        installer = Installer(home=home, project_root=home)
        options = _install_options()
        ref = _hardware_ref()

        result = installer.install_local(HARDWARE_MCP_FIXTURE, options=options)
        assert result.success is True
        target_dir = result.asset_dir

        removed = installer.uninstall(ref, options=options)
        assert removed is True

        assert not target_dir.exists()
        assert not installer.assets_lock.is_installed(ref)
        assert installer.cache.get_installed(ref) is None
        assert not any(
            e["ref"] == str(ref) for e in RegistryWriter(home=home).list_assets("hardware_mcp")
        )
        assert not McpMerger(project_root=home, home=home).is_managed(ref)

    def test_uninstall_missing_asset_returns_false(self, tmp_path: Path) -> None:
        """Uninstalling a non-installed asset returns False."""
        home = tmp_path / "home"
        installer = Installer(home=home, project_root=home)
        ref = _hardware_ref()
        assert installer.uninstall(ref) is False
