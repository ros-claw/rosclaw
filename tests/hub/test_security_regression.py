"""Security regression tests for the ROSClaw Hub install and publish paths.

These tests verify that dangerous or malformed assets are rejected at the
expected layer: verification, policy, license, or secret scanning.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.installer import Installer, InstallOptions
from rosclaw.hub.publisher import Publisher, PublishOptions
from rosclaw.hub.verifier import verify_asset_dir

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hub_assets"
SKILL_VALID = FIXTURES / "skill_valid"


def _load_manifest_yaml(asset_dir: Path) -> dict:
    return yaml.safe_load((asset_dir / "manifest.yaml").read_text(encoding="utf-8"))


def _copy_asset(source: Path, dest: Path) -> Path:
    """Copy a fixture asset directory to a temporary path."""
    shutil.copytree(source, dest)
    return dest


def _regenerate_checksums(asset_dir: Path) -> None:
    """Rewrite checksums.txt to match the current manifest and artifact files."""
    manifest = _load_manifest_yaml(asset_dir)
    lines: list[str] = []
    manifest_path = asset_dir / "manifest.yaml"
    lines.append(f"sha256:{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}  manifest.yaml")
    for artifact in manifest.get("artifacts", []):
        rel = artifact.get("path")
        if not rel:
            continue
        file_path = asset_dir / rel
        if file_path.exists():
            lines.append(f"sha256:{hashlib.sha256(file_path.read_bytes()).hexdigest()}  {rel}")
    (asset_dir / manifest["security"]["checksums"]["file"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_tampered_checksum_blocked_by_verifier() -> None:
    """A checksum mismatch is detected before any policy or install step."""
    result = verify_asset_dir(FIXTURES / "tampered_checksum")
    assert not result.ok
    assert any("Checksum mismatch" in e for e in result.errors)


def test_tampered_checksum_blocks_install(tmp_path: Path) -> None:
    """The installer refuses to install an asset with a bad checksum."""
    installer = Installer(home=tmp_path / "home")
    with pytest.raises(HubError) as exc_info:
        installer.install_local(
            FIXTURES / "tampered_checksum",
            options=InstallOptions(
                accept_license=True,
                allow_real_robot=True,
                skip_health=True,
                verify_signature=False,
            ),
        )
    assert exc_info.value.code == HubErrorCode.CHECKSUM_MISMATCH


def test_tampered_signature_blocked_by_verifier() -> None:
    """An invalid signing certificate is rejected when signatures are required."""
    result = verify_asset_dir(FIXTURES / "tampered_signature")
    assert not result.ok
    assert any("certificate" in e.lower() for e in result.errors)


def test_tampered_signature_blocks_install(tmp_path: Path) -> None:
    """The installer refuses to install an asset with an invalid certificate."""
    installer = Installer(home=tmp_path / "home")
    with pytest.raises(HubError) as exc_info:
        installer.install_local(
            FIXTURES / "tampered_signature",
            options=InstallOptions(
                accept_license=True,
                allow_real_robot=True,
                skip_health=True,
                verify_signature=True,
            ),
        )
    assert exc_info.value.code == HubErrorCode.CHECKSUM_MISMATCH


def test_missing_sbom_and_provenance_block_install(tmp_path: Path) -> None:
    """Assets that declare SBOM/provenance files but omit them are rejected."""
    asset_dir = _copy_asset(SKILL_VALID, tmp_path / "missing_sbom_asset")
    (asset_dir / "SBOM.spdx.json").unlink()
    (asset_dir / "PROVENANCE.json").unlink()

    installer = Installer(home=tmp_path / "home")
    with pytest.raises(HubError) as exc_info:
        installer.install_local(
            asset_dir,
            options=InstallOptions(
                accept_license=True,
                allow_real_robot=True,
                skip_health=True,
                verify_signature=False,
            ),
        )
    assert exc_info.value.code == HubErrorCode.CHECKSUM_MISMATCH
    assert any("SBOM" in e or "Provenance" in e for e in exc_info.value.message.split(";"))


def test_dangerous_safety_config_blocked(tmp_path: Path) -> None:
    """An asset that modifies safety configuration is rejected by default."""
    asset_dir = _copy_asset(SKILL_VALID, tmp_path / "safety_config_asset")
    manifest = _load_manifest_yaml(asset_dir)
    manifest["permissions"]["modifies"]["safety_config"] = True
    (asset_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    _regenerate_checksums(asset_dir)

    installer = Installer(home=tmp_path / "home")
    with pytest.raises(HubError) as exc_info:
        installer.install_local(
            asset_dir,
            options=InstallOptions(
                accept_license=True,
                allow_real_robot=True,
                allow_safety_config_changes=False,
                skip_health=True,
                verify_signature=False,
            ),
        )
    assert exc_info.value.code == HubErrorCode.PERMISSION_DENIED
    assert "safety" in exc_info.value.message.lower()


def test_non_local_inbound_network_blocked(tmp_path: Path) -> None:
    """Non-local inbound network access is rejected unless explicitly allowed."""
    asset_dir = _copy_asset(SKILL_VALID, tmp_path / "inbound_network_asset")
    manifest = _load_manifest_yaml(asset_dir)
    manifest["permissions"]["network"]["inbound"] = ["0.0.0.0"]
    (asset_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    _regenerate_checksums(asset_dir)

    installer = Installer(home=tmp_path / "home")
    with pytest.raises(HubError) as exc_info:
        installer.install_local(
            asset_dir,
            options=InstallOptions(
                accept_license=True,
                allow_real_robot=True,
                allow_network_inbound=False,
                skip_health=True,
                verify_signature=False,
            ),
        )
    assert exc_info.value.code == HubErrorCode.PERMISSION_DENIED
    assert "inbound" in exc_info.value.message.lower()


def test_license_denial_without_acceptance(tmp_path: Path) -> None:
    """A restricted license blocks install unless the user accepts it."""
    installer = Installer(home=tmp_path / "home")
    with pytest.raises(HubError) as exc_info:
        installer.install_local(
            FIXTURES / "license_requires_acceptance",
            options=InstallOptions(
                accept_license=False,
                allow_real_robot=True,
                skip_health=True,
                verify_signature=False,
                skip_mcp_merge=True,
            ),
        )
    assert exc_info.value.code == HubErrorCode.LICENSE_DENIED


def test_license_acceptance_allows_install(tmp_path: Path) -> None:
    """Explicit license acceptance permits installation of a restricted asset."""
    installer = Installer(home=tmp_path / "home")
    result = installer.install_local(
        FIXTURES / "license_requires_acceptance",
        options=InstallOptions(
            accept_license=True,
            allow_real_robot=True,
            skip_health=True,
            verify_signature=False,
            skip_mcp_merge=True,
        ),
    )
    assert result.success


def test_secret_scan_blocks_publish(tmp_path: Path) -> None:
    """Leaked credentials in an asset directory block publication."""
    home = tmp_path / "home"
    asset_dir = tmp_path / "leaky_asset"
    asset_dir.mkdir()
    manifest = _load_manifest_yaml(SKILL_VALID)
    # Disable signing so the fixture does not need a certificate.
    manifest["security"]["signing"]["required"] = False
    (asset_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (asset_dir / "config.py").write_text(
        'API_KEY = "sk-live-0123456789abcdef0123456789abcdef"\n',
        encoding="utf-8",
    )

    publisher = Publisher(PublishOptions(home=home))
    with pytest.raises(HubError) as exc_info:
        publisher.publish(asset_dir)
    assert exc_info.value.code == HubErrorCode.PUBLISH_REJECTED
    assert "api_key" in exc_info.value.message.lower()


def test_secret_scan_warning_does_not_block_when_configured(tmp_path: Path) -> None:
    """A publisher can be configured to warn instead of fail on secret finds."""
    home = tmp_path / "home"
    asset_dir = tmp_path / "leaky_asset"
    asset_dir.mkdir()
    manifest = _load_manifest_yaml(SKILL_VALID)
    manifest["security"]["signing"]["required"] = False
    (asset_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (asset_dir / "config.py").write_text(
        'PASSWORD = "hunter2"\n',
        encoding="utf-8",
    )

    publisher = Publisher(PublishOptions(home=home, secret_scan_fail_on_find=False))
    result = publisher.publish(asset_dir)
    assert result.success
    assert any("password" in w.lower() for w in result.warnings)
