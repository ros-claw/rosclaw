"""Tests for ROSClaw Hub local cache."""

from __future__ import annotations

import hashlib

import pytest

from rosclaw.hub.cache import HubCache
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.refs import AssetRef


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Create a HubCache in a temporary home directory."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    return HubCache()


def test_hub_cache_directory_layout(cache: HubCache) -> None:
    """All expected directories are created on initialization."""
    assert cache.hub_root.exists()
    assert cache.blobs_dir.exists()
    assert cache.manifests_dir.exists()
    assert cache.installed_dir.exists()
    assert cache.staging_dir.exists()
    assert cache.indexes_dir.exists()
    assert cache.backups_dir.exists()
    assert cache.config_dir.exists()
    assert cache.locks_dir.exists()


def test_manifest_path(cache: HubCache) -> None:
    """Manifest paths are structured by type, namespace, name, and version."""
    ref = AssetRef("hardware_mcp", "rosclaw", "unitree-g1", "1.0.0")
    path = cache.manifest_path(ref)
    assert path == cache.manifests_dir / "hardware_mcp" / "rosclaw" / "unitree-g1" / "1.0.0.yaml"


def test_manifest_path_requires_version(cache: HubCache) -> None:
    """A version is required to compute a manifest path."""
    ref = AssetRef("skill", "rosclaw", "g1-pick-place", None)
    with pytest.raises(HubError) as exc_info:
        cache.manifest_path(ref)
    assert exc_info.value.code == HubErrorCode.MANIFEST_INVALID


def test_put_and_get_manifest(cache: HubCache) -> None:
    """Manifests are written atomically and can be read back."""
    ref = AssetRef("skill", "rosclaw", "g1-pick-place", "1.2.0")
    content = b"name: g1-pick-place\n"
    path = cache.put_manifest(ref, content)
    assert path.exists()
    assert path.read_bytes() == content
    assert cache.get_manifest(ref) == path


def test_blob_path(cache: HubCache) -> None:
    """Blob paths use the algorithm and hexdigest."""
    path = cache.blob_path("sha256:deadbeef")
    assert path == cache.blobs_dir / "deadbeef"


def test_blob_path_rejects_unsupported_algorithm(cache: HubCache) -> None:
    """Only sha256 is supported for blobs."""
    with pytest.raises(HubError) as exc_info:
        cache.blob_path("md5:abc123")
    assert exc_info.value.code == HubErrorCode.CHECKSUM_MISMATCH


def test_put_blob_computes_digest(cache: HubCache) -> None:
    """When no digest is given, the blob digest is computed."""
    data = b"hello fake blob"
    path = cache.put_blob(data)
    expected = f"sha256:{hashlib.sha256(data).hexdigest()}"
    assert path == cache.blob_path(expected)


def test_put_blob_verifies_digest(cache: HubCache) -> None:
    """A provided digest is verified against the content."""
    data = b"hello fake blob"
    digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
    path = cache.put_blob(data, digest=digest)
    assert path == cache.blob_path(digest)


def test_put_blob_rejects_bad_digest(cache: HubCache) -> None:
    """A mismatched digest raises an error."""
    data = b"hello fake blob"
    with pytest.raises(HubError) as exc_info:
        cache.put_blob(
            data, digest="sha256:0000000000000000000000000000000000000000000000000000000000000000"
        )
    assert exc_info.value.code == HubErrorCode.CHECKSUM_MISMATCH


def test_get_missing_blob(cache: HubCache) -> None:
    """Missing blobs return None instead of raising."""
    assert (
        cache.get_blob("sha256:0000000000000000000000000000000000000000000000000000000000000000")
        is None
    )


def test_installed_state_round_trip(cache: HubCache) -> None:
    """Installed-state records are persisted and retrievable."""
    ref = AssetRef("digital_twin", "rosclaw", "g1-mujoco-basic", "0.5.0")
    record = {"source": "fake-registry", "size_bytes": 42}
    path = cache.set_installed(ref, record)
    assert path.exists()

    loaded = cache.get_installed(ref)
    assert loaded is not None
    assert loaded["ref"] == str(ref)
    assert loaded["source"] == "fake-registry"


def test_remove_installed(cache: HubCache) -> None:
    """Removing installed state cleans up empty parent directories."""
    ref = AssetRef("provider", "rosclaw", "qwen3-vl-gr00t", "0.3.1")
    cache.set_installed(ref, {})
    assert cache.remove_installed(ref) is True
    assert cache.get_installed(ref) is None
    assert not (cache.installed_dir / "provider").exists()


def test_list_installed(cache: HubCache) -> None:
    """list_installed returns all recorded asset refs."""
    ref1 = AssetRef("skill", "rosclaw", "g1-pick-place", "1.2.0")
    ref2 = AssetRef("hardware_mcp", "rosclaw", "unitree-g1", "1.0.0")
    cache.set_installed(ref1, {})
    cache.set_installed(ref2, {})
    refs = cache.list_installed()
    assert sorted(refs, key=lambda r: r.identity_tuple()) == sorted(
        [ref1, ref2], key=lambda r: r.identity_tuple()
    )


def test_staging_path(cache: HubCache) -> None:
    """Staging directories are created with a unique name."""
    stage1 = cache.staging_path("build")
    stage2 = cache.staging_path("build")
    assert stage1.exists()
    assert stage2.exists()
    assert stage1 != stage2


def test_backup_file(cache: HubCache) -> None:
    """backup_file copies an existing file into the backups directory."""
    target = cache.hub_root / "sample.txt"
    target.write_text("data")
    backup = cache.backup_file(target)
    assert backup.exists()
    assert backup.parent == cache.backups_dir


def test_backup_file_missing(cache: HubCache) -> None:
    """Backing up a missing file raises an error."""
    with pytest.raises(HubError) as exc_info:
        cache.backup_file(cache.hub_root / "missing.txt")
    assert exc_info.value.code == HubErrorCode.ASSET_NOT_FOUND


def test_clear_staging(cache: HubCache) -> None:
    """clear_staging removes all staging directories."""
    stage = cache.staging_path("test")
    (stage / "file.txt").write_text("keep me")
    cache.clear_staging()
    assert not any(cache.staging_dir.iterdir())
