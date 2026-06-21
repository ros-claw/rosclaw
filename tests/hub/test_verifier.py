"""Tests for the Hub asset verifier."""

from __future__ import annotations

from pathlib import Path

from rosclaw.hub.verifier import VerificationResult, verify_asset_dir

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hub_assets"


def test_verify_valid_hardware_mcp() -> None:
    result = verify_asset_dir(FIXTURES / "hardware_mcp_valid")
    assert result.ok, result.errors


def test_verify_valid_skill() -> None:
    result = verify_asset_dir(FIXTURES / "skill_valid")
    assert result.ok, result.errors


def test_verify_tampered_checksum_fails() -> None:
    result = verify_asset_dir(FIXTURES / "tampered_checksum")
    assert not result.ok
    assert any("Checksum mismatch" in e for e in result.errors)


def test_verify_tampered_signature_fails() -> None:
    result = verify_asset_dir(FIXTURES / "tampered_signature")
    assert not result.ok
    assert any("certificate" in e.lower() for e in result.errors)


def test_verify_no_signature_skips_signature_checks() -> None:
    result = verify_asset_dir(FIXTURES / "tampered_signature", require_signature=False)
    assert result.ok, result.errors


def test_verify_missing_manifest() -> None:
    result = verify_asset_dir(FIXTURES / "does_not_exist")
    assert not result.ok
    assert any("Manifest" in e for e in result.errors)


def test_verify_result_dataclass() -> None:
    result = VerificationResult()
    assert result.ok is True
    result.add_error("boom")
    assert result.ok is False
    assert result.errors == ["boom"]


# ---------------------------------------------------------------------------
# Edge cases for low-coverage verifier branches
# ---------------------------------------------------------------------------


def _write_manifest(path: Path, **overrides) -> None:
    """Write a minimal valid manifest to *path* with optional overrides."""
    import yaml

    manifest = {
        "schema_version": "hub.asset.v1",
        "asset": {
            "type": "skill",
            "namespace": "rosclaw",
            "name": "edge-case",
            "version": "1.0.0",
            "title": "Edge Case",
            "summary": "x",
            "description": "x",
            "tags": [],
        },
        "publisher": {
            "id": "rosclaw",
            "display_name": "ROSClaw",
            "trust_level": "official",
            "contact": "security@rosclaw.io",
        },
        "visibility": {"scope": "public", "allowed_orgs": [], "allowed_users": []},
        "lifecycle": {
            "status": "stable",
            "channel": "stable",
            "deprecated": False,
            "yanked": False,
            "replacement": None,
        },
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
        "dependencies": {"assets": [], "python": [], "system": [], "ros": []},
        "permissions": {
            "hardware": {"real_robot_execution": False, "actuators": [], "sensors": []},
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
            "attribution_required": False,
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
        },
        "artifacts": [],
        "install": {"mode": "declarative", "entrypoints": {}, "registries": {}},
        "special": {
            "skill": {
                "task_domain": "test",
                "task_name": "edge_case",
                "skill_api": {"inputs": [], "outputs": []},
                "required_capabilities": [],
                "supported_bodies": [],
                "components": {},
                "evaluation": {},
                "runtime": {"entrypoint": "", "rollback_safe": True},
            }
        },
    }
    manifest.update(overrides)
    # Ensure the special section matches the asset type if it was overridden.
    asset_type = manifest["asset"]["type"]
    if "special" not in manifest or asset_type not in manifest.get("special", {}):
        manifest["special"] = {asset_type: {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def test_verify_missing_sbom(tmp_path) -> None:
    """Verification fails when a declared SBOM is missing."""
    _write_manifest(
        tmp_path / "manifest.yaml",
        security={
            "signing": {"required": False},
            "checksums": {"algorithm": "sha256", "file": "checksums.txt"},
            "sbom": "SBOM.spdx.json",
        },
    )
    (tmp_path / "checksums.txt").write_text("\n", encoding="utf-8")
    result = verify_asset_dir(tmp_path)
    assert not result.ok
    assert any("SBOM file missing" in e for e in result.errors)


def test_verify_missing_provenance(tmp_path) -> None:
    """Verification fails when a declared provenance file is missing."""
    _write_manifest(
        tmp_path / "manifest.yaml",
        security={
            "signing": {"required": False},
            "checksums": {"algorithm": "sha256", "file": "checksums.txt"},
            "provenance": "PROVENANCE.json",
        },
    )
    (tmp_path / "checksums.txt").write_text("\n", encoding="utf-8")
    result = verify_asset_dir(tmp_path)
    assert not result.ok
    assert any("Provenance file missing" in e for e in result.errors)


def test_verify_missing_signing_certificate(tmp_path) -> None:
    """Verification fails when signing is required but the certificate is absent."""
    _write_manifest(
        tmp_path / "manifest.yaml",
        security={
            "signing": {
                "required": True,
                "scheme": "sigstore",
                "certificate": "signatures/cert.pem",
            },
            "checksums": {"algorithm": "sha256", "file": "checksums.txt"},
        },
    )
    (tmp_path / "checksums.txt").write_text("\n", encoding="utf-8")
    result = verify_asset_dir(tmp_path)
    assert not result.ok
    assert any("Signing certificate missing" in e for e in result.errors)


def test_verify_invalid_certificate_pem(tmp_path) -> None:
    """Verification fails when the certificate is not a valid PEM."""
    cert_path = tmp_path / "signatures" / "cert.pem"
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    cert_path.write_text("this is not a certificate", encoding="utf-8")
    _write_manifest(
        tmp_path / "manifest.yaml",
        security={
            "signing": {
                "required": True,
                "scheme": "sigstore",
                "certificate": "signatures/cert.pem",
            },
            "checksums": {"algorithm": "sha256", "file": "checksums.txt"},
        },
    )
    (tmp_path / "checksums.txt").write_text("\n", encoding="utf-8")
    result = verify_asset_dir(tmp_path)
    assert not result.ok
    assert any("not a valid PEM" in e for e in result.errors)


def test_verify_unsupported_checksum_algorithm(tmp_path) -> None:
    """Verification fails for unsupported checksum algorithms."""
    _write_manifest(
        tmp_path / "manifest.yaml",
        security={
            "signing": {"required": False},
            "checksums": {"algorithm": "md5", "file": "checksums.txt"},
        },
    )
    result = verify_asset_dir(tmp_path)
    assert not result.ok
    assert any("Unsupported checksum algorithm" in e for e in result.errors)


def test_verify_missing_checksums_file(tmp_path) -> None:
    """Verification fails when the checksums file is missing."""
    _write_manifest(tmp_path / "manifest.yaml")
    result = verify_asset_dir(tmp_path)
    assert not result.ok
    assert any("Checksums file missing" in e for e in result.errors)


def test_verify_declared_artifact_missing(tmp_path) -> None:
    """Verification fails when a declared artifact is not on disk."""
    _write_manifest(
        tmp_path / "manifest.yaml",
        artifacts=[
            {
                "name": "missing",
                "kind": "data",
                "path": "artifacts/missing.bin",
                "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "size_bytes": 0,
            }
        ],
    )
    (tmp_path / "checksums.txt").write_text("\n", encoding="utf-8")
    result = verify_asset_dir(tmp_path)
    assert not result.ok
    assert any("Declared artifact missing" in e for e in result.errors)


def test_verify_invalid_artifact_digest_format(tmp_path, monkeypatch) -> None:
    """Verification fails for artifact digests without an algorithm prefix.

    The schema validator rejects this format, so we bypass it to exercise the
    verifier's defensive check.
    """

    class _FakeManifest:
        security = {
            "checksums": {"algorithm": "sha256", "file": "checksums.txt"},
            "signing": {"required": False},
        }
        artifacts = [{"path": "artifacts/data.bin", "digest": "deadbeef"}]

    (tmp_path / "manifest.yaml").write_text("ignored", encoding="utf-8")
    (tmp_path / "checksums.txt").write_text("\n", encoding="utf-8")
    artifact_path = tmp_path / "artifacts" / "data.bin"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("hello", encoding="utf-8")

    monkeypatch.setattr("rosclaw.hub.verifier.load_manifest", lambda _path: _FakeManifest())
    result = verify_asset_dir(tmp_path)
    assert not result.ok
    assert any("Invalid artifact digest format" in e for e in result.errors)


def test_verify_unsupported_artifact_digest_algorithm(tmp_path) -> None:
    """Verification fails for artifact digest algorithms other than sha256."""
    artifact_path = tmp_path / "artifacts" / "data.bin"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("hello", encoding="utf-8")
    _write_manifest(
        tmp_path / "manifest.yaml",
        artifacts=[
            {
                "name": "data",
                "kind": "data",
                "path": "artifacts/data.bin",
                "digest": "md5:5d41402abc4b2a76b9719d911017c592",
                "size_bytes": 5,
            }
        ],
    )
    (tmp_path / "checksums.txt").write_text("\n", encoding="utf-8")
    result = verify_asset_dir(tmp_path)
    assert not result.ok
    assert any("Unsupported artifact digest algorithm" in e for e in result.errors)


def test_verify_file_in_checksums_not_declared(tmp_path) -> None:
    """A file listed in checksums but not declared as an artifact emits a warning."""
    data_path = tmp_path / "extra.txt"
    data_path.write_text("extra", encoding="utf-8")
    digest = "sha256:" + __import__("hashlib").sha256(b"extra").hexdigest()
    _write_manifest(tmp_path / "manifest.yaml")
    (tmp_path / "checksums.txt").write_text(
        f"{digest}  extra.txt\n",
        encoding="utf-8",
    )
    result = verify_asset_dir(tmp_path)
    assert result.ok
    assert any("not declared as artifact" in w for w in result.warnings)


def test_verify_empty_checksums_warning(tmp_path) -> None:
    """An empty checksums file produces a warning but does not fail."""
    _write_manifest(tmp_path / "manifest.yaml")
    (tmp_path / "checksums.txt").write_text("\n", encoding="utf-8")
    result = verify_asset_dir(tmp_path)
    assert result.ok
    assert any("Checksums file contains no entries" in w for w in result.warnings)
