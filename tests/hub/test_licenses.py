"""Tests for the Hub license and data-rights policy checker."""

from __future__ import annotations

from pathlib import Path

from rosclaw.hub.licenses import LicenseCheckResult, check_license

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hub_assets"


def test_mit_license_accepted() -> None:
    result = check_license(
        FIXTURES / "hardware_mcp_valid" / "manifest.yaml",
        asset_dir=FIXTURES / "hardware_mcp_valid",
    )
    assert result.accepted is True
    assert result.requires_acceptance is False


def test_commercial_license_requires_acceptance() -> None:
    result = check_license(
        FIXTURES / "license_requires_acceptance" / "manifest.yaml",
        asset_dir=FIXTURES / "license_requires_acceptance",
    )
    assert result.accepted is False
    assert result.requires_acceptance is True


def test_commercial_license_accepted_with_flag() -> None:
    result = check_license(
        FIXTURES / "license_requires_acceptance" / "manifest.yaml",
        accept_license=True,
        asset_dir=FIXTURES / "license_requires_acceptance",
    )
    assert result.accepted is True
    assert result.requires_acceptance is True


def test_unknown_spdx_requires_acceptance() -> None:
    result = check_license(
        FIXTURES / "hardware_mcp_valid" / "manifest.yaml",
        approved_spdx=set(),
        asset_dir=FIXTURES / "hardware_mcp_valid",
    )
    assert result.accepted is False
    assert result.requires_acceptance is True


def test_license_result_dataclass() -> None:
    result = LicenseCheckResult()
    assert result.accepted is True
    result.reject("bad")
    assert result.accepted is False
    assert result.issues == ["bad"]
