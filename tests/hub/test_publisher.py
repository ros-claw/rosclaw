"""Tests for the ROSClaw Hub asset publisher."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import yaml

from rosclaw.hub.client import FakeRegistryClient
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.publisher import (
    Publisher,
    PublishOptions,
    scan_secrets,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hub_assets"
SKILL_VALID = FIXTURES / "skill_valid"


def _load_manifest_yaml(asset_dir: Path) -> dict:
    return yaml.safe_load((asset_dir / "manifest.yaml").read_text(encoding="utf-8"))


def test_scan_secrets_detects_bearer_token(tmp_path: Path) -> None:
    secret_file = tmp_path / "leaked.py"
    secret_file.write_text(
        'headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc.def"}\n',
        encoding="utf-8",
    )
    findings = scan_secrets(tmp_path)
    assert any("bearer_token" in finding for finding in findings)


def test_prepare_updates_digests_and_checksums(tmp_path: Path) -> None:
    home = tmp_path / "home"
    publisher = Publisher(PublishOptions(home=home))
    prepared, manifest, warnings = publisher.prepare(SKILL_VALID)

    assert prepared.exists()
    assert manifest.asset.name == "g1-pick-place"

    # Digests should have been recomputed for each artifact that exists.
    for artifact in manifest.artifacts:
        rel = artifact.get("path")
        if rel and (SKILL_VALID / rel).exists():
            assert artifact["digest"].startswith("sha256:")
            assert isinstance(artifact["size_bytes"], int)

    checksums_path = prepared / "checksums.txt"
    assert checksums_path.exists()
    content = checksums_path.read_text(encoding="utf-8")
    assert "  manifest.yaml\n" in content
    assert "  artifacts/skill/behavior_tree.xml\n" in content
    assert not warnings


def test_prepare_generates_sbom_and_provenance(tmp_path: Path) -> None:
    home = tmp_path / "home"
    publisher = Publisher(PublishOptions(home=home))
    prepared, manifest, _ = publisher.prepare(SKILL_VALID)

    sbom_path = prepared / "SBOM.spdx.json"
    assert sbom_path.exists()
    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    assert sbom["spdxVersion"] == "SPDX-2.3"
    assert manifest.asset.name in sbom["name"]

    provenance_path = prepared / "PROVENANCE.json"
    assert provenance_path.exists()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["predicateType"] == "https://slsa.dev/provenance/v1"


def test_prepare_signs_when_required(tmp_path: Path) -> None:
    home = tmp_path / "home"
    publisher = Publisher(PublishOptions(home=home))
    prepared, _manifest, _warnings = publisher.prepare(SKILL_VALID)

    cert_path = prepared / "signatures" / "cert.pem"
    sig_path = prepared / "signatures" / "signature.bin"
    assert cert_path.exists()
    assert sig_path.exists()
    assert "BEGIN CERTIFICATE" in cert_path.read_text(encoding="utf-8")
    assert len(sig_path.read_bytes()) == 32  # HMAC-SHA256


def test_bundle_creates_rosclaw_tarball(tmp_path: Path) -> None:
    home = tmp_path / "home"
    publisher = Publisher(PublishOptions(home=home))
    prepared, manifest, _ = publisher.prepare(SKILL_VALID)
    bundle_path, digest, size = publisher.bundle(prepared, manifest)

    assert bundle_path.exists()
    assert bundle_path.name.endswith(".rosclaw")
    assert len(bundle_path.read_bytes()) == size
    assert len(digest) == 64

    with tarfile.open(bundle_path, "r:gz") as tar:
        names = tar.getnames()
    assert "manifest.yaml" in names
    assert "checksums.txt" in names
    assert "SBOM.spdx.json" in names
    assert "PROVENANCE.json" in names
    assert "signatures/cert.pem" in names
    assert "signatures/signature.bin" in names
    assert "artifacts/skill/behavior_tree.xml" in names


def test_publish_dry_run_returns_no_bundle(tmp_path: Path) -> None:
    home = tmp_path / "home"
    publisher = Publisher(PublishOptions(dry_run=True, home=home))
    result = publisher.publish(SKILL_VALID)

    assert result.success
    assert result.dry_run
    assert result.bundle_path is None
    assert result.size_bytes == 0
    assert result.ref.canonical() == "rosclaw://skill/rosclaw/g1-pick-place@1.2.0"


def test_publish_local_file_registry_uploads(tmp_path: Path) -> None:
    home = tmp_path / "home"
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "catalog.jsonl").write_text("", encoding="utf-8")

    registry_url = f"file://{registry_dir}"
    client = FakeRegistryClient(registry_url, token="fake-valid-token")
    publisher = Publisher(PublishOptions(registry=registry_url, home=home))
    result = publisher.publish(SKILL_VALID, registry_client=client)

    assert result.success
    assert not result.dry_run
    assert result.bundle_path is not None
    assert result.bundle_path.exists()

    manifest_dest = (
        registry_dir / "manifests" / "skill" / "rosclaw" / "g1-pick-place" / "1.2.0.yaml"
    )
    assert manifest_dest.exists()

    catalog_lines = (registry_dir / "catalog.jsonl").read_text(encoding="utf-8").strip()
    assert catalog_lines
    entry = json.loads(catalog_lines.splitlines()[-1])
    assert entry["asset"]["name"] == "g1-pick-place"
    assert entry["size_bytes"] == result.size_bytes


def test_publish_secret_scan_fails(tmp_path: Path) -> None:
    home = tmp_path / "home"
    asset_dir = tmp_path / "leaky_asset"
    asset_dir.mkdir()
    manifest = _load_manifest_yaml(SKILL_VALID)
    manifest["security"]["signing"]["required"] = False
    (asset_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (asset_dir / "secret.py").write_text(
        'AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"\n',
        encoding="utf-8",
    )

    publisher = Publisher(PublishOptions(home=home))
    with pytest.raises(HubError) as exc_info:
        publisher.publish(asset_dir)
    assert exc_info.value.code == HubErrorCode.PUBLISH_REJECTED


# ---------------------------------------------------------------------------
# Edge cases for low-coverage publisher branches
# ---------------------------------------------------------------------------


def test_scan_secrets_skips_binary_files(tmp_path: Path) -> None:
    """Binary files are not scanned for secret-like patterns."""
    binary_file = tmp_path / "secret.bin"
    binary_file.write_bytes(b"BEGIN PRIVATE KEY\n" + b"\x00" * 32)
    findings = scan_secrets(tmp_path)
    assert not findings


def test_prepare_warns_on_missing_artifact(tmp_path: Path) -> None:
    """A declared artifact that is missing on disk produces a warning."""
    home = tmp_path / "home"
    publisher = Publisher(PublishOptions(home=home))
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    manifest = _load_manifest_yaml(SKILL_VALID)
    manifest["artifacts"].append(
        {
            "name": "missing",
            "kind": "data",
            "path": "artifacts/missing.bin",
            "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "size_bytes": 0,
        }
    )
    (asset_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    _prepared, _manifest, warnings = publisher.prepare(asset_dir)
    assert any("Declared artifact missing on disk" in w for w in warnings)


def test_publish_asset_dir_not_found(tmp_path: Path) -> None:
    """Publishing a non-existent asset directory raises ASSET_NOT_FOUND."""
    publisher = Publisher(PublishOptions(home=tmp_path / "home"))
    with pytest.raises(HubError) as exc_info:
        publisher.publish(tmp_path / "does_not_exist")
    assert exc_info.value.code == HubErrorCode.ASSET_NOT_FOUND


def test_prepare_visibility_override(tmp_path: Path) -> None:
    """options.visibility overrides the manifest visibility scope."""
    home = tmp_path / "home"
    publisher = Publisher(PublishOptions(home=home, visibility="private"))
    _prepared, manifest, _warnings = publisher.prepare(SKILL_VALID)
    assert manifest.visibility["scope"] == "private"


def test_prepare_signs_via_options(tmp_path: Path) -> None:
    """Signing is performed when options.sign is True even if not required."""
    home = tmp_path / "home"
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    manifest = _load_manifest_yaml(SKILL_VALID)
    manifest["security"]["signing"] = {"required": False}
    (asset_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    publisher = Publisher(PublishOptions(home=home, sign=True))
    prepared, _manifest, _warnings = publisher.prepare(asset_dir)
    assert (prepared / "signatures" / "cert.pem").exists()
    assert (prepared / "signatures" / "signature.bin").exists()


def test_bundle_output_directory(tmp_path: Path) -> None:
    """When output is a directory, the bundle is written inside it."""
    home = tmp_path / "home"
    output_dir = tmp_path / "bundles"
    output_dir.mkdir()
    publisher = Publisher(PublishOptions(home=home, output=output_dir))
    prepared, manifest, _ = publisher.prepare(SKILL_VALID)
    bundle_path, _digest, _size = publisher.bundle(prepared, manifest)

    assert bundle_path.parent == output_dir
    assert bundle_path.name.endswith(".rosclaw")


def test_publish_uses_registry_client_from_auth_store(tmp_path: Path) -> None:
    """Publishing to options.registry without an explicit client builds one."""
    home = tmp_path / "home"
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "catalog.jsonl").write_text("", encoding="utf-8")

    publisher = Publisher(PublishOptions(registry=str(registry_dir), home=home))
    result = publisher.publish(SKILL_VALID)

    assert result.success
    assert (
        registry_dir / "manifests" / "skill" / "rosclaw" / "g1-pick-place" / "1.2.0.yaml"
    ).exists()


def test_publish_registry_upload_failure_re_raises_hub_error(tmp_path: Path) -> None:
    """Registry upload failures that are HubError instances are re-raised."""
    home = tmp_path / "home"
    client = FakeRegistryClient("http://registry.example", token="bad-token")

    def _raise(*_args, **_kwargs):
        raise HubError(code=HubErrorCode.AUTH_FAILED, message="bad token")

    client.publish_bundle = _raise

    publisher = Publisher(PublishOptions(registry="http://registry.example", home=home))
    with pytest.raises(HubError) as exc_info:
        publisher.publish(SKILL_VALID, registry_client=client)
    assert exc_info.value.code == HubErrorCode.AUTH_FAILED


def test_publish_registry_upload_failure_wraps_generic_exception(tmp_path: Path) -> None:
    """Unexpected registry upload exceptions are wrapped as REGISTRY_UNREACHABLE."""
    home = tmp_path / "home"
    client = FakeRegistryClient("http://registry.example")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("network partition")

    client.publish_bundle = _raise

    publisher = Publisher(PublishOptions(registry="http://registry.example", home=home))
    with pytest.raises(HubError) as exc_info:
        publisher.publish(SKILL_VALID, registry_client=client)
    assert exc_info.value.code == HubErrorCode.REGISTRY_UNREACHABLE
    assert "network partition" in str(exc_info.value)
