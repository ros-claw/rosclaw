"""Tests for ROSClaw Hub CLI Phase 2 commands."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from rosclaw.hub.auth import AuthStore
from rosclaw.hub.cli import (
    cmd_hub_install,
    cmd_hub_list,
    cmd_hub_login,
    cmd_hub_logout,
    cmd_hub_policy_check,
    cmd_hub_publish,
    cmd_hub_ref_parse,
    cmd_hub_schema_export,
    cmd_hub_search,
    cmd_hub_sync,
    cmd_hub_uninstall,
    cmd_hub_update,
    cmd_hub_validate,
    cmd_hub_verify,
    cmd_hub_whoami,
    dispatch_hub_command,
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


def test_sync_rejects_manifest_digest_mismatch(fake_registry_path, store, tmp_path, capsys) -> None:
    registry = tmp_path / "registry"
    shutil.copytree(fake_registry_path, registry)
    manifest_path = registry / "manifests" / "skill" / "rosclaw" / "g1-pick-place" / "1.2.0.yaml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    store.login(str(registry), "fake-valid-token", insecure_local=True)

    rc = cmd_hub_sync(_namespace(registry=None, clear=False))

    assert rc == 1
    assert "digest" in capsys.readouterr().out.lower()


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


def test_policy_check_denies_real_robot_by_default(asset_fixture_dir, capsys) -> None:
    """hub policy check fails closed for real-robot permissions."""
    args = _namespace(
        asset_dir=str(asset_fixture_dir),
        allow_real_robot=False,
        accept_license=False,
        json=False,
    )
    rc = cmd_hub_policy_check(args)
    assert rc == 1
    captured = capsys.readouterr()
    assert "Policy check failed" in captured.out
    assert "denied by policy" in captured.out


def test_policy_check_allows_explicit_real_robot_opt_in(asset_fixture_dir, capsys) -> None:
    """hub policy check accepts the explicit real-robot permission flag."""
    args = _namespace(
        asset_dir=str(asset_fixture_dir),
        allow_real_robot=True,
        accept_license=False,
        json=False,
    )

    assert cmd_hub_policy_check(args) == 0
    captured = capsys.readouterr()
    assert "Policy check passed" in captured.out
    assert "requires human approval" in captured.out


# ---------------------------------------------------------------------------
# No-subcommand and dispatch routing
# ---------------------------------------------------------------------------


def test_dispatch_no_subcommand(capsys) -> None:
    """dispatch prints help when no hub subcommand is given."""
    rc = dispatch_hub_command(_namespace(hub_command=None))
    assert rc == 1
    captured = capsys.readouterr()
    assert "no subcommand given" in captured.out
    assert "validate" in captured.out
    assert "publish" in captured.out


def test_dispatch_ref_no_subcommand(capsys) -> None:
    """dispatch prints ref-specific help when ref subcommand is missing."""
    rc = dispatch_hub_command(_namespace(hub_command="ref", ref_command=None))
    assert rc == 1
    assert "hub ref: no subcommand given" in capsys.readouterr().out


def test_dispatch_schema_no_subcommand(capsys) -> None:
    """dispatch prints schema-specific help when schema subcommand is missing."""
    rc = dispatch_hub_command(_namespace(hub_command="schema", schema_command=None))
    assert rc == 1
    assert "hub schema: no subcommand given" in capsys.readouterr().out


def test_dispatch_policy_no_subcommand(capsys) -> None:
    """dispatch prints policy-specific help when policy subcommand is missing."""
    rc = dispatch_hub_command(_namespace(hub_command="policy", policy_command=None))
    assert rc == 1
    assert "hub policy: no subcommand given" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@pytest.fixture
def skill_valid_dir() -> Path:
    """Path to the valid skill fixture directory."""
    return Path(__file__).parents[1] / "fixtures" / "hub_assets" / "skill_valid"


def test_validate_valid_manifest(skill_valid_dir, capsys) -> None:
    """hub validate passes for a valid manifest."""
    args = _namespace(manifest=str(skill_valid_dir / "manifest.yaml"), json=False)
    rc = cmd_hub_validate(args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "Valid:" in captured.out
    assert "g1-pick-place" in captured.out


def test_validate_valid_manifest_json(skill_valid_dir, capsys) -> None:
    """hub validate --json emits parseable JSON."""
    args = _namespace(manifest=str(skill_valid_dir / "manifest.yaml"), json=True)
    rc = cmd_hub_validate(args)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["valid"] is True
    assert data["asset"]["name"] == "g1-pick-place"


def test_validate_invalid_manifest(tmp_path, capsys) -> None:
    """hub validate fails for a malformed manifest."""
    manifest_path = tmp_path / "bad.yaml"
    manifest_path.write_text("not: valid: yaml: [", encoding="utf-8")
    args = _namespace(manifest=str(manifest_path), json=False)
    rc = cmd_hub_validate(args)
    assert rc == 1
    assert "Manifest invalid" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ref parse
# ---------------------------------------------------------------------------


def test_ref_parse_valid(capsys) -> None:
    """hub ref parse prints a parsed reference."""
    args = _namespace(ref="rosclaw://skill/rosclaw/g1-pick-place@1.2.0", json=False)
    rc = cmd_hub_ref_parse(args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "Asset Reference" in captured.out
    assert "skill" in captured.out
    assert "g1-pick-place" in captured.out


def test_ref_parse_valid_json(capsys) -> None:
    """hub ref parse --json emits parseable JSON."""
    args = _namespace(ref="rosclaw://skill/rosclaw/g1-pick-place@1.2.0", json=True)
    rc = cmd_hub_ref_parse(args)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["type"] == "skill"
    assert data["name"] == "g1-pick-place"
    assert data["canonical"] == "rosclaw://skill/rosclaw/g1-pick-place@1.2.0"


def test_ref_parse_invalid(capsys) -> None:
    """hub ref parse fails for an invalid reference."""
    args = _namespace(ref="not-a-valid-ref", json=False)
    rc = cmd_hub_ref_parse(args)
    assert rc == 1
    assert "Invalid reference" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# schema export
# ---------------------------------------------------------------------------


def test_schema_export_json(capsys) -> None:
    """hub schema export prints JSON schema by default."""
    args = _namespace(format="json", output=None)
    rc = cmd_hub_schema_export(args)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "$defs" in data or "properties" in data


def test_schema_export_yaml(capsys) -> None:
    """hub schema export --format yaml emits YAML."""
    args = _namespace(format="yaml", output=None)
    rc = cmd_hub_schema_export(args)
    assert rc == 0
    output = capsys.readouterr().out
    assert "schema_version" in output or "hub.asset.v1" in output


def test_schema_export_to_file(tmp_path, capsys) -> None:
    """hub schema export --output writes the schema to a file."""
    out_path = tmp_path / "schema.json"
    args = _namespace(format="json", output=str(out_path))
    rc = cmd_hub_schema_export(args)
    assert rc == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "properties" in data


# ---------------------------------------------------------------------------
# install / list / uninstall / update
# ---------------------------------------------------------------------------


def _install_args(asset_dir: str | Path, **overrides) -> argparse.Namespace:
    """Build a namespace for cmd_hub_install with safe defaults."""
    defaults = {
        "asset_dir": str(asset_dir),
        "dry_run": False,
        "yes": True,
        "accept_license": False,
        "no_mcp_merge": True,
        "skip_health": True,
        "verify_signature": True,
        "allow_real_robot": True,
        "allow_safety_config_changes": False,
        "allow_network_inbound": False,
        "json": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_install_local_dry_run(skill_valid_dir, monkeypatch, tmp_path, capsys) -> None:
    """hub install --dry-run reports without writing files."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    args = _install_args(skill_valid_dir, dry_run=True)
    rc = cmd_hub_install(args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "Dry-run" in captured.out


def test_install_yes_does_not_grant_real_robot_permission(
    skill_valid_dir, monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    args = _install_args(skill_valid_dir, yes=True, allow_real_robot=False)

    rc = cmd_hub_install(args)

    assert rc == 1
    assert "real robot execution" in capsys.readouterr().out.lower()


def test_install_local_then_list_then_uninstall(
    skill_valid_dir, monkeypatch, tmp_path, capsys
) -> None:
    """Full CLI lifecycle: install, list, uninstall."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))

    install_args = _install_args(skill_valid_dir)
    rc = cmd_hub_install(install_args)
    assert rc == 0
    assert "Installed:" in capsys.readouterr().out

    list_args = _namespace(installed=True, json=False)
    rc = cmd_hub_list(list_args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "g1-pick-place" in captured.out

    ref = "rosclaw://skill/rosclaw/g1-pick-place@1.2.0"
    uninstall_args = _namespace(ref=ref, yes=True, json=False)
    rc = cmd_hub_uninstall(uninstall_args)
    assert rc == 0
    assert "Uninstalled" in capsys.readouterr().out

    # List should now be empty.
    rc = cmd_hub_list(_namespace(installed=True, json=False))
    assert rc == 0
    assert "No installed assets" in capsys.readouterr().out


def test_install_local_json(skill_valid_dir, monkeypatch, tmp_path, capsys) -> None:
    """hub install --json emits structured output."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    args = _install_args(skill_valid_dir, json=True)
    rc = cmd_hub_install(args)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["success"] is True
    assert "g1-pick-place" in data["ref"]


def test_uninstall_not_installed(capsys) -> None:
    """hub uninstall fails gracefully when the asset is not installed."""
    args = _namespace(ref="rosclaw://skill/rosclaw/not-here@1.0.0", yes=True, json=False)
    rc = cmd_hub_uninstall(args)
    assert rc == 1
    assert "not installed" in capsys.readouterr().out


def test_update_installed_asset(skill_valid_dir, monkeypatch, tmp_path, capsys) -> None:
    """hub update replaces an installed asset."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))

    rc = cmd_hub_install(_install_args(skill_valid_dir))
    assert rc == 0
    capsys.readouterr()

    ref = "rosclaw://skill/rosclaw/g1-pick-place@1.2.0"
    update_args = argparse.Namespace(
        ref=ref,
        asset_dir=str(skill_valid_dir),
        dry_run=False,
        yes=True,
        accept_license=False,
        no_mcp_merge=True,
        skip_health=True,
        verify_signature=True,
        allow_real_robot=True,
        allow_safety_config_changes=False,
        allow_network_inbound=False,
        json=False,
    )
    rc = cmd_hub_update(update_args)
    assert rc == 0
    assert "Updated:" in capsys.readouterr().out


def test_update_invalid_ref(capsys) -> None:
    """hub update fails for an invalid reference."""
    args = argparse.Namespace(
        ref="not-a-ref",
        asset_dir=".",
        dry_run=False,
        yes=True,
        accept_license=False,
        no_mcp_merge=True,
        skip_health=True,
        verify_signature=True,
        allow_real_robot=False,
        allow_safety_config_changes=False,
        allow_network_inbound=False,
        json=False,
    )
    rc = cmd_hub_update(args)
    assert rc == 1
    assert "Invalid reference" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------


def _publish_args(asset_dir: str | Path, **overrides) -> argparse.Namespace:
    """Build a namespace for cmd_hub_publish with safe defaults."""
    defaults = {
        "asset_dir": str(asset_dir),
        "dry_run": False,
        "private": False,
        "public": False,
        "sign": False,
        "registry": None,
        "output": None,
        "json": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_publish_dry_run(skill_valid_dir, monkeypatch, tmp_path, capsys) -> None:
    """hub publish --dry-run validates and scans without uploading."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    args = _publish_args(skill_valid_dir, dry_run=True)
    rc = cmd_hub_publish(args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "Dry-run" in captured.out
    assert "g1-pick-place" in captured.out


def test_publish_private_dry_run(skill_valid_dir, monkeypatch, tmp_path, capsys) -> None:
    """hub publish --private --dry-run accepts visibility flag."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    args = _publish_args(skill_valid_dir, dry_run=True, private=True)
    rc = cmd_hub_publish(args)
    assert rc == 0
    assert "Dry-run" in capsys.readouterr().out


def test_publish_private_and_public_conflict(skill_valid_dir, capsys) -> None:
    """hub publish rejects --private and --public together."""
    args = _publish_args(skill_valid_dir, private=True, public=True)
    rc = cmd_hub_publish(args)
    assert rc == 1
    assert "Cannot specify both" in capsys.readouterr().out


def test_publish_output_builds_local_bundle_without_registry(
    skill_valid_dir, monkeypatch, tmp_path, capsys
) -> None:
    """An explicit output path is a complete local publish destination."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / "home"))
    output = tmp_path / "asset.rosclaw"
    args = _publish_args(skill_valid_dir, output=output, json=True)

    assert cmd_hub_publish(args) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["success"] is True
    assert result["bundle_path"] == str(output)
    assert output.is_file()


def test_publish_requires_output_registry_or_active_profile(
    skill_valid_dir, monkeypatch, tmp_path, capsys
) -> None:
    """A non-dry-run publish cannot silently choose a destination."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / "home"))

    assert cmd_hub_publish(_publish_args(skill_valid_dir)) == 1
    assert "No publish destination" in capsys.readouterr().out


def test_publish_secret_scan_rejects(
    tmp_path, skill_valid_dir, fake_registry_path, store, monkeypatch, capsys
) -> None:
    """hub publish rejects an asset that contains a leaked secret."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    store.login(str(fake_registry_path), "fake-valid-token", insecure_local=True)
    import yaml as _yaml

    manifest = _yaml.safe_load((skill_valid_dir / "manifest.yaml").read_text(encoding="utf-8"))
    manifest["security"]["signing"]["required"] = False
    asset_dir = tmp_path / "leaky_asset"
    asset_dir.mkdir()
    (asset_dir / "manifest.yaml").write_text(
        _yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (asset_dir / "secret.py").write_text(
        'AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"\n',
        encoding="utf-8",
    )

    args = _publish_args(asset_dir)
    rc = cmd_hub_publish(args)
    assert rc == 1
    assert "Publish failed" in capsys.readouterr().out


def test_publish_json(skill_valid_dir, monkeypatch, tmp_path, capsys) -> None:
    """hub publish --json emits structured output."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    args = _publish_args(skill_valid_dir, dry_run=True, json=True)
    rc = cmd_hub_publish(args)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["success"] is True
    assert data["dry_run"] is True
    assert "g1-pick-place" in data["ref"]
