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
