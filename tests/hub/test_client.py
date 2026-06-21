"""Tests for the Hub registry client implementations."""

from __future__ import annotations

import io
import json
import tarfile
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from rosclaw.hub.client import FakeRegistryClient
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.refs import AssetRef
from rosclaw.hub.schema import load_manifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hub_assets"
FAKE_REGISTRY = Path(__file__).parent.parent / "fixtures" / "fake_registry"
SKILL_VALID = FIXTURES / "skill_valid"


def _skill_manifest() -> Any:
    return load_manifest(SKILL_VALID / "manifest.yaml")


def _tar_bytes(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(directory)))
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Local registry helpers
# ---------------------------------------------------------------------------


def test_fake_client_strips_trailing_slash() -> None:
    """Trailing slashes are normalized from the registry URL."""
    client = FakeRegistryClient(str(FAKE_REGISTRY) + "/")
    assert client.registry_url == str(FAKE_REGISTRY)


def test_fake_client_file_url() -> None:
    """file:// URLs are converted to local paths."""
    client = FakeRegistryClient(f"file://{FAKE_REGISTRY}")
    assert client._local_root == FAKE_REGISTRY


def test_fake_client_local_path() -> None:
    """Plain paths are treated as local registries."""
    client = FakeRegistryClient(str(FAKE_REGISTRY))
    assert client._local_root == FAKE_REGISTRY


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


def test_sync_returns_catalog_entries() -> None:
    """sync parses every non-empty line of catalog.jsonl."""
    client = FakeRegistryClient(str(FAKE_REGISTRY))
    entries = client.sync()
    assert len(entries) == 5
    assert all("ref" in entry for entry in entries)


def test_sync_invalid_jsonl_raises(tmp_path: Path) -> None:
    """A malformed catalog line raises INDEX_VERIFY_FAILED."""
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text('{"valid": true}\n{not json\n', encoding="utf-8")
    client = FakeRegistryClient(str(tmp_path))
    with pytest.raises(HubError) as exc_info:
        client.sync()
    assert exc_info.value.code == HubErrorCode.INDEX_VERIFY_FAILED


# ---------------------------------------------------------------------------
# fetch_manifest
# ---------------------------------------------------------------------------


def test_fetch_manifest_success() -> None:
    """fetch_manifest returns YAML bytes for a versioned reference."""
    client = FakeRegistryClient(str(FAKE_REGISTRY))
    ref = AssetRef("hardware_mcp", "rosclaw", "unitree-g1", "1.0.0")
    data = client.fetch_manifest(ref)
    manifest = yaml.safe_load(data)
    assert manifest["asset"]["name"] == "unitree-g1"


def test_fetch_manifest_requires_version() -> None:
    """fetch_manifest rejects references without a version."""
    client = FakeRegistryClient(str(FAKE_REGISTRY))
    ref = AssetRef("hardware_mcp", "rosclaw", "unitree-g1", None)
    with pytest.raises(HubError) as exc_info:
        client.fetch_manifest(ref)
    assert exc_info.value.code == HubErrorCode.MANIFEST_INVALID


# ---------------------------------------------------------------------------
# fetch_blob
# ---------------------------------------------------------------------------


def test_fetch_blob_success(tmp_path: Path) -> None:
    """fetch_blob reads a content-addressed blob from the local registry."""
    import hashlib

    blob_dir = tmp_path / "blobs" / "sha256"
    blob_dir.mkdir(parents=True)
    data = b"hello blob"
    digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
    (blob_dir / digest.split(":", 1)[1]).write_bytes(data)

    client = FakeRegistryClient(str(tmp_path))
    assert client.fetch_blob(digest) == data


def test_fetch_blob_invalid_digest_format() -> None:
    """fetch_blob rejects digests without an algorithm prefix."""
    client = FakeRegistryClient(str(FAKE_REGISTRY))
    with pytest.raises(HubError) as exc_info:
        client.fetch_blob("deadbeef")
    assert exc_info.value.code == HubErrorCode.CHECKSUM_MISMATCH


# ---------------------------------------------------------------------------
# fetch_bundle
# ---------------------------------------------------------------------------


def test_fetch_bundle_success(tmp_path: Path) -> None:
    """fetch_bundle reads a .rosclaw bundle from the local registry."""
    client = FakeRegistryClient(str(tmp_path))
    ref = AssetRef("skill", "rosclaw", "g1-pick-place", "1.2.0")
    bundle_dir = tmp_path / "bundles" / "skill" / "rosclaw" / "g1-pick-place"
    bundle_dir.mkdir(parents=True)
    bundle_path = bundle_dir / "1.2.0.rosclaw"
    bundle_path.write_bytes(b"bundle bytes")
    assert client.fetch_bundle(ref) == b"bundle bytes"


def test_fetch_bundle_requires_version() -> None:
    """fetch_bundle rejects references without a version."""
    client = FakeRegistryClient(str(FAKE_REGISTRY))
    ref = AssetRef("skill", "rosclaw", "g1-pick-place", None)
    with pytest.raises(HubError) as exc_info:
        client.fetch_bundle(ref)
    assert exc_info.value.code == HubErrorCode.MANIFEST_INVALID


# ---------------------------------------------------------------------------
# whoami
# ---------------------------------------------------------------------------


def test_whoami_success() -> None:
    """whoami returns the fake profile for the expected token."""
    client = FakeRegistryClient("http://localhost:8787", token="fake-valid-token")
    profile = client.whoami()
    assert profile["user"] == "rosclaw-tester"
    assert profile["role"] == "admin"


def test_whoami_invalid_token() -> None:
    """whoami raises AUTH_FAILED for an unexpected token."""
    client = FakeRegistryClient("http://localhost:8787", token="wrong-token")
    with pytest.raises(HubError) as exc_info:
        client.whoami()
    assert exc_info.value.code == HubErrorCode.AUTH_FAILED


# ---------------------------------------------------------------------------
# publish_bundle (local)
# ---------------------------------------------------------------------------


def test_publish_bundle_local_creates_manifest_and_catalog(tmp_path: Path) -> None:
    """Publishing a bundle to a local registry writes manifests and catalog."""
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "catalog.jsonl").write_text("", encoding="utf-8")

    manifest = _skill_manifest()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.yaml").write_bytes(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False).encode("utf-8")
    )
    bundle_bytes = _tar_bytes(prepared)
    bundle_path = tmp_path / "bundle.rosclaw"
    bundle_path.write_bytes(bundle_bytes)

    client = FakeRegistryClient(str(registry_dir), token="fake-valid-token")
    result = client.publish_bundle(bundle_path, manifest, len(bundle_bytes))

    assert result["manifest_url"] == "manifests/skill/rosclaw/g1-pick-place/1.2.0.yaml"
    manifest_dest = registry_dir / result["manifest_url"]
    assert manifest_dest.exists()

    catalog_lines = (registry_dir / "catalog.jsonl").read_text(encoding="utf-8").strip()
    assert catalog_lines
    entry = json.loads(catalog_lines.splitlines()[-1])
    assert entry["asset"]["name"] == "g1-pick-place"


def test_publish_bundle_local_missing_bundle_file(tmp_path: Path) -> None:
    """Publishing a missing bundle file raises ASSET_NOT_FOUND."""
    client = FakeRegistryClient(str(tmp_path))
    with pytest.raises(HubError) as exc_info:
        client.publish_bundle(tmp_path / "missing.rosclaw", _skill_manifest(), 0)
    assert exc_info.value.code == HubErrorCode.ASSET_NOT_FOUND


# ---------------------------------------------------------------------------
# publish_bundle (HTTP)
# ---------------------------------------------------------------------------


def _mock_urlopen(
    monkeypatch: Any, response_body: bytes | None = None, exc: Exception | None = None
) -> MagicMock:
    """Patch urllib.request.urlopen for HTTP client tests."""
    mock = MagicMock()

    if exc is not None:
        mock.side_effect = exc
    else:
        response = MagicMock()
        response.read.return_value = response_body or b""
        context_manager = MagicMock()
        context_manager.__enter__.return_value = response
        context_manager.__exit__.return_value = False
        mock.return_value = context_manager

    monkeypatch.setattr("rosclaw.hub.client.urllib.request.urlopen", mock)
    return mock


def test_publish_bundle_http_success(monkeypatch: Any, tmp_path: Path) -> None:
    """HTTP publish_bundle posts bytes and parses JSON response."""
    manifest = _skill_manifest()
    bundle_path = tmp_path / "bundle.rosclaw"
    bundle_path.write_bytes(b"data")

    response = json.dumps({"manifest_url": "http://example.com/manifest.yaml"}).encode("utf-8")
    _mock_urlopen(monkeypatch, response_body=response)

    client = FakeRegistryClient("http://registry.example", token="fake-valid-token")
    result = client.publish_bundle(bundle_path, manifest, 4)
    assert result["manifest_url"] == "http://example.com/manifest.yaml"


def test_publish_bundle_http_empty_response(monkeypatch: Any, tmp_path: Path) -> None:
    """An empty HTTP response falls back to the upload URL."""
    manifest = _skill_manifest()
    bundle_path = tmp_path / "bundle.rosclaw"
    bundle_path.write_bytes(b"data")

    _mock_urlopen(monkeypatch, response_body=b"")

    client = FakeRegistryClient("http://registry.example", token="fake-valid-token")
    result = client.publish_bundle(bundle_path, manifest, 4)
    assert "manifest_url" in result


def test_publish_bundle_http_401(monkeypatch: Any, tmp_path: Path) -> None:
    """HTTP 401 maps to AUTH_FAILED."""
    manifest = _skill_manifest()
    bundle_path = tmp_path / "bundle.rosclaw"
    bundle_path.write_bytes(b"data")

    error = urllib.error.HTTPError(
        "http://registry.example/upload/...", 401, "Unauthorized", {}, None
    )
    _mock_urlopen(monkeypatch, exc=error)

    client = FakeRegistryClient("http://registry.example", token="bad-token")
    with pytest.raises(HubError) as exc_info:
        client.publish_bundle(bundle_path, manifest, 4)
    assert exc_info.value.code == HubErrorCode.AUTH_FAILED


def test_publish_bundle_http_409(monkeypatch: Any, tmp_path: Path) -> None:
    """HTTP 409 maps to PUBLISH_REJECTED."""
    manifest = _skill_manifest()
    bundle_path = tmp_path / "bundle.rosclaw"
    bundle_path.write_bytes(b"data")

    error = urllib.error.HTTPError("http://registry.example/upload/...", 409, "Conflict", {}, None)
    _mock_urlopen(monkeypatch, exc=error)

    client = FakeRegistryClient("http://registry.example")
    with pytest.raises(HubError) as exc_info:
        client.publish_bundle(bundle_path, manifest, 4)
    assert exc_info.value.code == HubErrorCode.PUBLISH_REJECTED


def test_publish_bundle_http_500(monkeypatch: Any, tmp_path: Path) -> None:
    """Other HTTP errors map to REGISTRY_UNREACHABLE."""
    manifest = _skill_manifest()
    bundle_path = tmp_path / "bundle.rosclaw"
    bundle_path.write_bytes(b"data")

    error = urllib.error.HTTPError(
        "http://registry.example/upload/...", 500, "Server Error", {}, None
    )
    _mock_urlopen(monkeypatch, exc=error)

    client = FakeRegistryClient("http://registry.example")
    with pytest.raises(HubError) as exc_info:
        client.publish_bundle(bundle_path, manifest, 4)
    assert exc_info.value.code == HubErrorCode.REGISTRY_UNREACHABLE


def test_publish_bundle_http_url_error(monkeypatch: Any, tmp_path: Path) -> None:
    """URLError maps to REGISTRY_UNREACHABLE."""
    manifest = _skill_manifest()
    bundle_path = tmp_path / "bundle.rosclaw"
    bundle_path.write_bytes(b"data")

    _mock_urlopen(monkeypatch, exc=urllib.error.URLError("network down"))

    client = FakeRegistryClient("http://registry.example")
    with pytest.raises(HubError) as exc_info:
        client.publish_bundle(bundle_path, manifest, 4)
    assert exc_info.value.code == HubErrorCode.REGISTRY_UNREACHABLE


# ---------------------------------------------------------------------------
# HTTP GET error handling
# ---------------------------------------------------------------------------


def test_get_http_404(monkeypatch: Any) -> None:
    """HTTP 404 maps to ASSET_NOT_FOUND."""
    error = urllib.error.HTTPError(
        "http://registry.example/catalog.jsonl", 404, "Not Found", {}, None
    )
    _mock_urlopen(monkeypatch, exc=error)

    client = FakeRegistryClient("http://registry.example")
    with pytest.raises(HubError) as exc_info:
        client.sync()
    assert exc_info.value.code == HubErrorCode.ASSET_NOT_FOUND


def test_get_http_other_error(monkeypatch: Any) -> None:
    """Other HTTP GET errors map to REGISTRY_UNREACHABLE."""
    error = urllib.error.HTTPError(
        "http://registry.example/catalog.jsonl", 503, "Unavailable", {}, None
    )
    _mock_urlopen(monkeypatch, exc=error)

    client = FakeRegistryClient("http://registry.example")
    with pytest.raises(HubError) as exc_info:
        client.sync()
    assert exc_info.value.code == HubErrorCode.REGISTRY_UNREACHABLE


def test_get_http_url_error(monkeypatch: Any) -> None:
    """URLError on GET maps to REGISTRY_UNREACHABLE."""
    _mock_urlopen(monkeypatch, exc=urllib.error.URLError("name resolution failed"))

    client = FakeRegistryClient("http://registry.example")
    with pytest.raises(HubError) as exc_info:
        client.sync()
    assert exc_info.value.code == HubErrorCode.REGISTRY_UNREACHABLE
