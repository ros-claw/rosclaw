"""Tests for ROSClaw Hub CLI Phase 2 commands."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from rosclaw.hub.auth import AuthStore
from rosclaw.hub.cli import (
    cmd_hub_login,
    cmd_hub_logout,
    cmd_hub_policy_check,
    cmd_hub_search,
    cmd_hub_sync,
    cmd_hub_verify,
    cmd_hub_whoami,
)


@pytest.fixture
def fake_registry_path() -> Path:
    """Path to the fake registry fixture directory."""
    return Path(__file__).parents[1] / "fixtures" / "fake_registry"


@pytest.fixture
def store(tmp_path, monkeypatch) -> AuthStore:
    """AuthStore backed by a temporary home."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    return AuthStore()


def _namespace(**kwargs) -> argparse.Namespace:
    """Build an argparse Namespace with default values."""
    return argparse.Namespace(**kwargs)


def test_login_success(fake_registry_path, store, capsys) -> None:
    """Login stores credentials when the token is valid."""
    args = _namespace(
        registry=str(fake_registry_path),
        token="fake-valid-token",
        insecure_local=True,
    )
    rc = cmd_hub_login(args)
    assert rc == 0

    captured = capsys.readouterr()
    assert "Logged in to Hub" in captured.out
    fresh = AuthStore()
    assert fresh.get_active_profile() is not None
    assert fresh.get_active_profile()["registry"] == str(fake_registry_path)


def test_login_failure(fake_registry_path, store, capsys) -> None:
    """Login fails when the token is invalid."""
    args = _namespace(
        registry=str(fake_registry_path),
        token="wrong-token",
        insecure_local=True,
    )
    rc = cmd_hub_login(args)
    assert rc == 1
    fresh = AuthStore()
    assert fresh.get_active_profile() is None
    assert "Login failed" in capsys.readouterr().out


def test_whoami_success(fake_registry_path, store, capsys) -> None:
    """whoami shows the active profile."""
    store.login(str(fake_registry_path), "fake-valid-token", insecure_local=True)
    args = _namespace()
    rc = cmd_hub_whoami(args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "Active Hub identity" in captured.out
    assert "rosclaw-tester" in captured.out


def test_whoami_not_logged_in(store, capsys) -> None:
    """whoami fails when no profile is active."""
    args = _namespace()
    rc = cmd_hub_whoami(args)
    assert rc == 1
    assert "Not logged in" in capsys.readouterr().out


def test_logout(store, capsys) -> None:
    """logout removes the active profile."""
    store.login("http://localhost:8787", "fake-valid-token")
    args = _namespace(registry=None)
    rc = cmd_hub_logout(args)
    assert rc == 0
    fresh = AuthStore()
    assert fresh.get_active_profile() is None
    assert "Logged out" in capsys.readouterr().out


def test_logout_no_active(store, capsys) -> None:
    """logout fails when there is no active registry."""
    args = _namespace(registry=None)
    rc = cmd_hub_logout(args)
    assert rc == 1
    assert "No active registry" in capsys.readouterr().out


def test_sync_local_registry(fake_registry_path, store, capsys) -> None:
    """sync indexes assets from a local fake registry directory."""
    store.login(str(fake_registry_path), "fake-valid-token", insecure_local=True)
    args = _namespace(registry=None, clear=False)
    rc = cmd_hub_sync(args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "Synced" in captured.out
    assert "Indexed: 5" in captured.out


def test_sync_explicit_registry(fake_registry_path, store, capsys) -> None:
    """sync accepts an explicit --registry override."""
    args = _namespace(registry=str(fake_registry_path), clear=False)
    rc = cmd_hub_sync(args)
    assert rc == 0
    assert "Synced" in capsys.readouterr().out


def test_search_after_sync(fake_registry_path, store, capsys) -> None:
    """search queries the local index built by sync."""
    store.login(str(fake_registry_path), "fake-valid-token", insecure_local=True)
    cmd_hub_sync(_namespace(registry=None, clear=False))
    args = _namespace(
        query="g1",
        type=None,
        namespace=None,
        official=False,
        license=None,
        robot=None,
        compatible=False,
        limit=20,
        json=False,
    )
    rc = cmd_hub_search(args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "result(s)" in captured.out
    assert "unitree-g1" in captured.out or "g1-pick-place" in captured.out


def test_search_empty_index(store, capsys) -> None:
    """search warns when the local catalog is empty."""
    store.login("http://localhost:8787", "fake-valid-token")
    args = _namespace(
        query="g1",
        type=None,
        namespace=None,
        official=False,
        license=None,
        robot=None,
        compatible=False,
        limit=20,
        json=False,
    )
    rc = cmd_hub_search(args)
    assert rc == 1
    assert "Local catalog is empty" in capsys.readouterr().out


def test_search_json_output(fake_registry_path, store, capsys) -> None:
    """search --json emits parseable JSON."""
    store.login(str(fake_registry_path), "fake-valid-token", insecure_local=True)
    cmd_hub_sync(_namespace(registry=None, clear=False))
    capsys.readouterr()  # drain sync output
    args = _namespace(
        query="unitree",
        type=None,
        namespace=None,
        official=False,
        license=None,
        robot=None,
        compatible=False,
        limit=20,
        json=True,
    )
    rc = cmd_hub_search(args)
    assert rc == 0
    import json

    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list)
    assert any("unitree-g1" in r.get("ref", "") for r in data)


@pytest.fixture
def asset_fixture_dir() -> Path:
    """Path to the valid hardware_mcp fixture directory."""
    return Path(__file__).parents[1] / "fixtures" / "hub_assets" / "hardware_mcp_valid"


def test_verify_valid_asset(asset_fixture_dir, capsys) -> None:
    """hub verify passes for a valid asset directory."""
    args = _namespace(asset_dir=str(asset_fixture_dir), no_signature=False, json=False)
    rc = cmd_hub_verify(args)
    assert rc == 0
    assert "verification passed" in capsys.readouterr().out


def test_verify_tampered_checksum_fails(capsys) -> None:
    """hub verify fails when an artifact checksum has been tampered."""
    asset_dir = Path(__file__).parents[1] / "fixtures" / "hub_assets" / "tampered_checksum"
    args = _namespace(asset_dir=str(asset_dir), no_signature=False, json=False)
    rc = cmd_hub_verify(args)
    assert rc == 1
    assert "Checksum mismatch" in capsys.readouterr().out


def test_policy_check_valid_asset(asset_fixture_dir, capsys) -> None:
    """hub policy check passes for the valid fixture with warnings."""
    args = _namespace(
        asset_dir=str(asset_fixture_dir),
        allow_real_robot=False,
        accept_license=False,
        json=False,
    )
    rc = cmd_hub_policy_check(args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "Policy check passed" in captured.out
    assert "requires human approval" in captured.out
