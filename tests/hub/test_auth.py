"""Tests for ROSClaw Hub authentication state management."""

from __future__ import annotations

import json
import os
import stat

import pytest

from rosclaw.hub.auth import AuthProfile, AuthStore
from rosclaw.hub.errors import HubError


def test_auth_profile_defaults() -> None:
    """AuthProfile stores registry, token, and insecure flag."""
    profile = AuthProfile(registry="http://localhost:8787", token="abc")
    assert profile.registry == "http://localhost:8787"
    assert profile.token == "abc"
    assert profile.insecure_local is False


def test_login_and_active_profile(tmp_path) -> None:
    """Logging in persists the profile and sets it active."""
    store = AuthStore(home=tmp_path)
    store.login("http://localhost:8787", "token-a", insecure_local=True)
    profile = store.get_active_profile()
    assert profile is not None
    assert profile["registry"] == "http://localhost:8787"
    assert profile["token"] == "token-a"
    assert profile["insecure_local"] is True


def test_login_normalizes_trailing_slash(tmp_path) -> None:
    """Registry URLs are stored without a trailing slash."""
    store = AuthStore(home=tmp_path)
    store.login("http://localhost:8787/", "token")
    assert store.get_token() == "token"
    assert "http://localhost:8787" in store.list_profiles()[0]["registry"]


def test_login_set_active_false(tmp_path) -> None:
    """Logging in with set_active=False does not change the active profile."""
    store = AuthStore(home=tmp_path)
    store.login("http://a.example", "token-a")
    store.login("http://b.example", "token-b", set_active=False)
    active = store.get_active_profile()
    assert active is not None
    assert active["registry"] == "http://a.example"


def test_logout_active(tmp_path) -> None:
    """Logging out removes the active profile."""
    store = AuthStore(home=tmp_path)
    store.login("http://a.example", "token-a")
    assert store.logout() is True
    assert store.get_active_profile() is None
    assert store.list_profiles() == []


def test_logout_specific_registry(tmp_path) -> None:
    """Logging out a specific registry clears active if it was active."""
    store = AuthStore(home=tmp_path)
    store.login("http://a.example", "token-a")
    store.login("http://b.example", "token-b")
    assert store.logout("http://a.example") is True
    active = store.get_active_profile()
    assert active is not None
    assert active["registry"] == "http://b.example"


def test_logout_not_logged_in(tmp_path) -> None:
    """Logging out when nothing is stored returns False."""
    store = AuthStore(home=tmp_path)
    assert store.logout() is False
    assert store.logout("http://missing.example") is False


def test_get_token_defaults_to_active(tmp_path) -> None:
    """get_token returns the active registry token when none is specified."""
    store = AuthStore(home=tmp_path)
    store.login("http://a.example", "token-a")
    assert store.get_token() == "token-a"
    assert store.get_token("http://a.example") == "token-a"


def test_get_token_no_active(tmp_path) -> None:
    """get_token returns None when there is no active registry."""
    store = AuthStore(home=tmp_path)
    assert store.get_token() is None


def test_is_insecure_local(tmp_path) -> None:
    """is_insecure_local reflects the stored flag."""
    store = AuthStore(home=tmp_path)
    store.login("http://secure.example", "token")
    store.login("http://local.example", "token", insecure_local=True)
    assert store.is_insecure_local("http://secure.example") is False
    assert store.is_insecure_local("http://local.example") is True
    assert store.is_insecure_local() is True  # active is local


def test_list_profiles(tmp_path) -> None:
    """list_profiles returns all stored profiles."""
    store = AuthStore(home=tmp_path)
    store.login("http://a.example", "token-a")
    store.login("http://b.example", "token-b")
    profiles = store.list_profiles()
    assert len(profiles) == 2
    assert {p["registry"] for p in profiles} == {"http://a.example", "http://b.example"}


def test_get_client_requires_active(tmp_path) -> None:
    """get_client raises when no registry is active."""
    store = AuthStore(home=tmp_path)
    with pytest.raises(HubError) as exc_info:
        store.get_client()
    assert "No active registry" in str(exc_info.value)


def test_get_client_uses_active(tmp_path) -> None:
    """get_client builds a FakeRegistryClient for the active registry."""
    store = AuthStore(home=tmp_path)
    store.login("http://localhost:8787", "token-x", insecure_local=True)
    client = store.get_client()
    assert client.registry_url == "http://localhost:8787"


def test_store_survives_roundtrip(tmp_path) -> None:
    """A second AuthStore instance sees the persisted data."""
    store = AuthStore(home=tmp_path)
    store.login("http://a.example", "token-a")
    fresh = AuthStore(home=tmp_path)
    assert fresh.get_token() == "token-a"
    assert fresh.get_active_profile() is not None


def test_store_handles_corrupt_file(tmp_path) -> None:
    """A corrupt auth file is treated as empty profiles."""
    auth_path = tmp_path / "config" / "hub_auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text("not json", encoding="utf-8")
    store = AuthStore(home=tmp_path)
    assert store.get_active_profile() is None
    assert store.list_profiles() == []


def test_store_handles_empty_file(tmp_path) -> None:
    """An empty auth file is treated as empty profiles."""
    auth_path = tmp_path / "config" / "hub_auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text("", encoding="utf-8")
    store = AuthStore(home=tmp_path)
    assert store.get_active_profile() is None


def test_store_saves_valid_json(tmp_path) -> None:
    """The saved auth file is valid JSON with expected keys."""
    store = AuthStore(home=tmp_path)
    store.login("http://a.example", "token-a")
    data = json.loads((tmp_path / "config" / "hub_auth.json").read_text(encoding="utf-8"))
    assert data["active"] == "http://a.example"
    assert data["profiles"]["http://a.example"]["token"] == "token-a"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission semantics")
def test_store_uses_owner_only_permissions(tmp_path) -> None:
    """Credential directories and files are private regardless of umask."""
    store = AuthStore(home=tmp_path)
    store.login("https://hub.example", "secret")

    assert stat.S_IMODE((tmp_path / "config").stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission semantics")
def test_store_tightens_existing_permissions(tmp_path) -> None:
    """An existing user-owned credential file is migrated to mode 0600."""
    auth_path = tmp_path / "config" / "hub_auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text('{"profiles": {}, "active": null}', encoding="utf-8")
    auth_path.chmod(0o666)
    auth_path.parent.chmod(0o755)

    AuthStore(home=tmp_path)

    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(auth_path.parent.stat().st_mode) == 0o700


def test_store_rejects_credential_symlink(tmp_path) -> None:
    """A pre-positioned credential symlink cannot redirect reads or writes."""
    outside = tmp_path / "outside.json"
    outside.write_text("do not overwrite", encoding="utf-8")
    auth_path = tmp_path / "config" / "hub_auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.symlink_to(outside)

    with pytest.raises(HubError, match="must be a regular file"):
        AuthStore(home=tmp_path)

    assert outside.read_text(encoding="utf-8") == "do not overwrite"


def test_store_rejects_config_directory_symlink(tmp_path) -> None:
    """The private config boundary cannot be redirected through a symlink."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "config").symlink_to(outside, target_is_directory=True)

    with pytest.raises(HubError, match="must be a real directory"):
        AuthStore(home=tmp_path)


def test_store_rejects_non_regular_credential_path(tmp_path) -> None:
    """Directories and device-like paths are not accepted as token files."""
    auth_path = tmp_path / "config" / "hub_auth.json"
    auth_path.mkdir(parents=True)

    with pytest.raises(HubError, match="must be a regular file"):
        AuthStore(home=tmp_path)
