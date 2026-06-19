"""End-to-end lifecycle tests using the fake local file registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.hub.cache import HubCache
from rosclaw.hub.client import FakeRegistryClient
from rosclaw.hub.index import CatalogIndex
from rosclaw.hub.installer import Installer, InstallOptions
from rosclaw.hub.lockfile import AssetsLock
from rosclaw.hub.publisher import Publisher, PublishOptions
from rosclaw.hub.refs import AssetRef
from rosclaw.hub.registry_writer import RegistryWriter
from rosclaw.hub.schema import load_manifest_from_bytes

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hub_assets"
SKILL_VALID = FIXTURES / "skill_valid"


def _ref_from_entry(entry: dict) -> AssetRef:
    asset = entry["asset"]
    return AssetRef(
        type=asset["type"],
        namespace=asset["namespace"],
        name=asset["name"],
        version=asset["version"],
    )


def _sync_catalog(client: FakeRegistryClient, index: CatalogIndex, cache: HubCache) -> None:
    """Download the catalog, index it, and cache every referenced manifest."""
    entries = client.sync()
    index.index_entries(entries)
    for entry in entries:
        ref = _ref_from_entry(entry)
        manifest_bytes = client.fetch_manifest(ref)
        load_manifest_from_bytes(manifest_bytes)
        cache.put_manifest(ref, manifest_bytes)


def test_full_lifecycle_publish_sync_search_install_list_uninstall(tmp_path: Path) -> None:
    """Exercise the full fake-registry lifecycle end-to-end."""
    home = tmp_path / "home"
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "catalog.jsonl").write_text("", encoding="utf-8")

    registry_url = f"file://{registry_dir}"
    client = FakeRegistryClient(registry_url, token="fake-valid-token")

    # 1. Publish the valid skill asset to the fake registry.
    publisher = Publisher(PublishOptions(registry=registry_url, home=home))
    publish_result = publisher.publish(SKILL_VALID, registry_client=client)
    assert publish_result.success
    assert publish_result.bundle_path is not None
    assert publish_result.bundle_path.exists()

    ref = publish_result.ref
    canonical = str(ref)

    # Registry should now contain the manifest, bundle, and catalog entry.
    manifest_dest = (
        registry_dir / "manifests" / ref.type / ref.namespace / ref.name / f"{ref.version}.yaml"
    )
    bundle_dest = (
        registry_dir / "bundles" / ref.type / ref.namespace / ref.name / f"{ref.version}.rosclaw"
    )
    assert manifest_dest.exists()
    assert bundle_dest.exists()

    catalog_lines = (registry_dir / "catalog.jsonl").read_text(encoding="utf-8").strip()
    assert catalog_lines
    catalog_entries = [line for line in catalog_lines.splitlines() if line.strip()]
    assert len(catalog_entries) == 1

    # 2. Sync the catalog into a local SQLite index and cache manifests.
    cache = HubCache(home)
    index = CatalogIndex(registry_url, cache=cache)
    _sync_catalog(client, index, cache)

    assert index.count() == 1
    catalog_id = f"{ref.type}:{ref.namespace}:{ref.name}:{ref.version}"
    assert index.get(catalog_id) is not None

    search_results = index.search("pick-place")
    assert len(search_results) == 1
    assert search_results[0]["asset"]["name"] == ref.name

    # 3. Install the asset by rosclaw:// reference.
    installer = Installer(home=home)
    options = InstallOptions(
        accept_license=True,
        allow_real_robot=True,
        skip_health=True,
        verify_signature=True,
    )
    install_result = installer.install_by_ref(canonical, options=options, registry_client=client)
    assert install_result.success
    assert install_result.ref == ref
    assert install_result.asset_dir.exists()

    installed_manifest_path = install_result.asset_dir / "manifest.yaml"
    assert installed_manifest_path.exists()

    # Lockfile should record the installed asset.
    lock = AssetsLock.load(cache.home / "assets.lock")
    assert lock.is_installed(ref)
    entry = lock.get(ref)
    assert entry is not None
    assert entry.lifecycle_status in ("healthy", "installed")

    # Runtime registry should contain the skill entry.
    registry_writer = RegistryWriter(cache=cache)
    skills = registry_writer.list_assets("skill")
    assert len(skills) == 1
    assert skills[0]["ref"] == canonical

    installed_state = cache.get_installed(ref)
    assert installed_state is not None
    assert installed_state["ref"] == canonical

    # 4. List installed assets.
    installed_refs = cache.list_installed()
    assert len(installed_refs) == 1
    assert installed_refs[0] == ref

    # 5. Uninstall the asset and verify removal.
    removed = installer.uninstall(ref)
    assert removed is True

    lock_after = AssetsLock.load(cache.home / "assets.lock")
    assert not lock_after.is_installed(ref)
    assert not install_result.asset_dir.exists()
    assert cache.get_installed(ref) is None
    assert registry_writer.list_assets("skill") == []


def test_install_by_ref_requires_cached_manifest(tmp_path: Path) -> None:
    """Installing by reference fails gracefully when the manifest is not cached."""
    home = tmp_path / "home"
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "catalog.jsonl").write_text("", encoding="utf-8")

    registry_url = f"file://{registry_dir}"
    client = FakeRegistryClient(registry_url, token="fake-valid-token")

    publisher = Publisher(PublishOptions(registry=registry_url, home=home))
    publish_result = publisher.publish(SKILL_VALID, registry_client=client)

    installer = Installer(home=home)
    from rosclaw.hub.errors import HubError, HubErrorCode

    with pytest.raises(HubError) as exc_info:
        installer.install_by_ref(
            str(publish_result.ref),
            registry_client=client,
        )
    assert exc_info.value.code == HubErrorCode.ASSET_NOT_FOUND
