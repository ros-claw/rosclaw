"""Cryptographic trust and complete-payload checks for Hub assets."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.installer import Installer, InstallOptions
from rosclaw.hub.publisher import Publisher, PublishOptions
from rosclaw.hub.verifier import verify_asset_dir

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hub_assets"
SIGNATURE_DOMAIN = b"ROSCLAW-HUB-ASSET-SIGNATURE-V1\x00"
KEY_ID = "hub-test-release-v1"


@pytest.fixture
def signing_material(tmp_path: Path) -> tuple[Ed25519PrivateKey, Path, Path]:
    private_key = Ed25519PrivateKey.generate()
    private_key_path = tmp_path / "signing-key.pem"
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust_store_path = tmp_path / "trust.json"
    trust_store_path.write_text(
        json.dumps(
            {
                "schema_version": "rosclaw.hub.trust.v1",
                "keys": {
                    KEY_ID: {
                        "algorithm": "ed25519",
                        "public_key_base64": base64.b64encode(public_key).decode("ascii"),
                        "status": "trusted",
                        "scopes": ["rosclaw://skill/rosclaw/*@*"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return private_key, private_key_path, trust_store_path


def _manifest(asset_dir: Path) -> dict:
    return yaml.safe_load((asset_dir / "manifest.yaml").read_text(encoding="utf-8"))


def _signature_file(manifest: dict) -> str:
    return str(manifest["security"]["signing"].get("file", "signatures/manifest.ed25519"))


def _rewrite_checksums(asset_dir: Path) -> bytes:
    manifest = _manifest(asset_dir)
    checksums_file = str(manifest["security"]["checksums"]["file"])
    signature_file = _signature_file(manifest)
    excluded = {checksums_file, signature_file}
    lines: list[str] = []
    for path in sorted(asset_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(asset_dir).as_posix()
        if relative in excluded:
            continue
        lines.append(f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    (asset_dir / checksums_file).write_bytes(payload)
    return payload


def _copy_signed_asset(
    tmp_path: Path,
    private_key: Ed25519PrivateKey,
    *,
    signing: dict | None = None,
) -> Path:
    asset_dir = tmp_path / "asset"
    shutil.copytree(FIXTURES / "skill_valid", asset_dir)
    shutil.rmtree(asset_dir / "signatures", ignore_errors=True)
    manifest = _manifest(asset_dir)
    manifest["security"]["signing"] = signing or {
        "required": True,
        "scheme": "ed25519",
        "key_id": KEY_ID,
        "file": "signatures/manifest.ed25519",
    }
    (asset_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    checksums = _rewrite_checksums(asset_dir)
    signature = private_key.sign(
        SIGNATURE_DOMAIN + (asset_dir / "manifest.yaml").read_bytes() + b"\x00" + checksums
    )
    signature_path = asset_dir / _signature_file(manifest)
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path.write_text(base64.b64encode(signature).decode("ascii") + "\n", encoding="ascii")
    return asset_dir


def _copy_unsigned_asset(tmp_path: Path) -> Path:
    asset_dir = tmp_path / "asset"
    shutil.copytree(FIXTURES / "skill_valid", asset_dir)
    shutil.rmtree(asset_dir / "signatures", ignore_errors=True)
    manifest = _manifest(asset_dir)
    manifest["security"]["signing"] = {"required": False}
    (asset_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    _rewrite_checksums(asset_dir)
    return asset_dir


def test_valid_ed25519_signature_is_trusted(
    tmp_path: Path,
    signing_material: tuple[Ed25519PrivateKey, Path, Path],
) -> None:
    private_key, _private_key_path, trust_store_path = signing_material
    asset_dir = _copy_signed_asset(tmp_path, private_key)

    result = verify_asset_dir(asset_dir, trust_store_path=trust_store_path)

    assert result.ok, result.errors
    assert result.signature_status == "valid"
    assert result.signature_key_id == KEY_ID
    assert result.trusted is True


def test_tampered_ed25519_signature_fails(
    tmp_path: Path,
    signing_material: tuple[Ed25519PrivateKey, Path, Path],
) -> None:
    private_key, _private_key_path, trust_store_path = signing_material
    asset_dir = _copy_signed_asset(tmp_path, private_key)
    signature_path = asset_dir / "signatures" / "manifest.ed25519"
    signature = bytearray(base64.b64decode(signature_path.read_text(encoding="ascii")))
    signature[0] ^= 0x01
    signature_path.write_text(base64.b64encode(signature).decode("ascii"), encoding="ascii")

    result = verify_asset_dir(asset_dir, trust_store_path=trust_store_path)

    assert not result.ok
    assert result.signature_status == "invalid"
    assert any("signature is invalid" in error.lower() for error in result.errors)


def test_unknown_signing_key_fails(
    tmp_path: Path,
    signing_material: tuple[Ed25519PrivateKey, Path, Path],
) -> None:
    private_key, _private_key_path, _trust_store_path = signing_material
    asset_dir = _copy_signed_asset(tmp_path, private_key)
    empty_trust_store = tmp_path / "empty-trust.json"
    empty_trust_store.write_text(
        json.dumps({"schema_version": "rosclaw.hub.trust.v1", "keys": {}}),
        encoding="utf-8",
    )

    result = verify_asset_dir(asset_dir, trust_store_path=empty_trust_store)

    assert not result.ok
    assert result.signature_status == "unknown_key"


def test_signing_key_scope_is_enforced(
    tmp_path: Path,
    signing_material: tuple[Ed25519PrivateKey, Path, Path],
) -> None:
    private_key, _private_key_path, trust_store_path = signing_material
    asset_dir = _copy_signed_asset(tmp_path, private_key)
    trust = json.loads(trust_store_path.read_text(encoding="utf-8"))
    trust["keys"][KEY_ID]["scopes"] = ["rosclaw://provider/rosclaw/*@*"]
    trust_store_path.write_text(json.dumps(trust), encoding="utf-8")

    result = verify_asset_dir(asset_dir, trust_store_path=trust_store_path)

    assert not result.ok
    assert result.signature_status == "scope_mismatch"


def test_placeholder_sigstore_material_is_rejected(tmp_path: Path) -> None:
    asset_dir = tmp_path / "asset"
    shutil.copytree(FIXTURES / "skill_valid", asset_dir)
    manifest = _manifest(asset_dir)
    manifest["security"]["signing"] = {
        "required": True,
        "scheme": "sigstore",
        "certificate": "signatures/cert.pem",
    }
    (asset_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (asset_dir / "signatures" / "cert.pem").write_text(
        "-----BEGIN CERTIFICATE-----\nplaceholder\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )
    _rewrite_checksums(asset_dir)

    result = verify_asset_dir(asset_dir)

    assert not result.ok
    assert result.signature_status == "unsupported_scheme"


def test_signature_requirement_cannot_be_disabled_by_manifest(tmp_path: Path) -> None:
    asset_dir = _copy_unsigned_asset(tmp_path)

    result = verify_asset_dir(asset_dir, require_signature=True)

    assert not result.ok
    assert result.signature_status == "missing"


def test_installer_rejects_unsigned_asset_by_default(tmp_path: Path) -> None:
    asset_dir = _copy_unsigned_asset(tmp_path)
    installer = Installer(home=tmp_path / "home")

    with pytest.raises(HubError) as exc_info:
        installer.install_local(
            asset_dir,
            options=InstallOptions(
                accept_license=True,
                skip_health=True,
                skip_mcp_merge=True,
            ),
        )

    assert exc_info.value.code == HubErrorCode.CHECKSUM_MISMATCH
    assert "trusted signature" in exc_info.value.message.lower()


def test_checksum_path_escape_is_rejected(tmp_path: Path) -> None:
    asset_dir = _copy_unsigned_asset(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    checksums = asset_dir / "checksums.txt"
    checksums.write_text(
        checksums.read_text(encoding="utf-8")
        + f"sha256:{hashlib.sha256(outside.read_bytes()).hexdigest()}  ../outside.txt\n",
        encoding="utf-8",
    )

    result = verify_asset_dir(asset_dir, require_signature=False)

    assert not result.ok
    assert any("unsafe checksums path" in error.lower() for error in result.errors)


def test_symlink_payload_is_rejected(tmp_path: Path) -> None:
    asset_dir = _copy_unsigned_asset(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (asset_dir / "linked.txt").symlink_to(outside)
    _rewrite_checksums(asset_dir)

    result = verify_asset_dir(asset_dir, require_signature=False)

    assert not result.ok
    assert any("symbolic link" in error.lower() for error in result.errors)


def test_untracked_payload_is_rejected(tmp_path: Path) -> None:
    asset_dir = _copy_unsigned_asset(tmp_path)
    (asset_dir / "untracked.py").write_text("print('not covered')\n", encoding="utf-8")

    result = verify_asset_dir(asset_dir, require_signature=False)

    assert not result.ok
    assert any("untracked payload" in error.lower() for error in result.errors)


def test_malformed_checksum_line_is_rejected(tmp_path: Path) -> None:
    asset_dir = _copy_unsigned_asset(tmp_path)
    checksums = asset_dir / "checksums.txt"
    checksums.write_text(
        checksums.read_text(encoding="utf-8") + "this line is ignored by the old verifier\n",
        encoding="utf-8",
    )

    result = verify_asset_dir(asset_dir, require_signature=False)

    assert not result.ok
    assert any("invalid checksums line" in error.lower() for error in result.errors)


def test_no_signature_mode_cannot_hide_payload_as_signature(tmp_path: Path) -> None:
    asset_dir = _copy_unsigned_asset(tmp_path)
    manifest = _manifest(asset_dir)
    manifest["security"]["signing"] = {
        "required": False,
        "scheme": "ed25519",
        "key_id": "untrusted",
        "file": "artifacts/hidden.py",
    }
    (asset_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (asset_dir / "artifacts" / "hidden.py").write_text(
        "print('untracked payload')\n", encoding="utf-8"
    )
    _rewrite_checksums(asset_dir)

    result = verify_asset_dir(asset_dir, require_signature=False)

    assert not result.ok
    assert any("signature path" in error.lower() for error in result.errors)


def test_publisher_emits_verifiable_ed25519_signature(
    tmp_path: Path,
    signing_material: tuple[Ed25519PrivateKey, Path, Path],
) -> None:
    private_key, private_key_path, trust_store_path = signing_material
    source = _copy_signed_asset(tmp_path / "source", private_key)
    publisher = Publisher(
        PublishOptions(
            home=tmp_path / "home",
            signing_key=private_key_path,
            signing_key_id=KEY_ID,
        )
    )

    prepared, _manifest_value, _warnings = publisher.prepare(source)
    result = verify_asset_dir(prepared, trust_store_path=trust_store_path)

    assert result.ok, result.errors
    assert result.signature_status == "valid"


def test_publisher_fails_closed_without_private_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ROSCLAW_HUB_SIGNING_KEY", raising=False)
    monkeypatch.delenv("ROSCLAW_HUB_SIGNING_KEY_ID", raising=False)
    source = tmp_path / "asset"
    shutil.copytree(FIXTURES / "skill_valid", source)
    manifest = _manifest(source)
    manifest["security"]["signing"] = {
        "required": True,
        "scheme": "ed25519",
        "key_id": KEY_ID,
        "file": "signatures/manifest.ed25519",
    }
    (source / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(HubError) as exc_info:
        Publisher(PublishOptions(home=tmp_path / "home")).prepare(source)

    assert exc_info.value.code == HubErrorCode.PUBLISH_REJECTED
    assert "private key" in exc_info.value.message.lower()


def test_publisher_rejects_security_control_path_collision(tmp_path: Path) -> None:
    source = tmp_path / "asset"
    shutil.copytree(FIXTURES / "skill_valid", source)
    manifest = _manifest(source)
    manifest["security"]["checksums"]["file"] = "manifest.yaml"
    (source / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(HubError) as exc_info:
        Publisher(PublishOptions(home=tmp_path / "home")).prepare(source)

    assert exc_info.value.code == HubErrorCode.PUBLISH_REJECTED
    assert "control path collision" in exc_info.value.message.lower()
