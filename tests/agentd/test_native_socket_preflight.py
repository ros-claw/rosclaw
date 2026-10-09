"""Actual warehouse startup failure: deep evidence roots exceed pathname UDS limits."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.agentd.socket_paths import validate_native_socket_paths


def test_long_home_fails_before_native_bootstrap_or_service_mutation(tmp_path, monkeypatch, capsys):
    import rosclaw.agentd.cli as cli

    home = tmp_path / ("deep-evidence-root-" * 8)
    monkeypatch.setattr(
        cli, "_find_pi_agent_entry", lambda **_: pytest.fail("bootstrap must not run")
    )
    monkeypatch.setattr(
        cli, "AgentService", lambda *_: pytest.fail("service must not be constructed")
    )
    assert cli._chat_pi(home, SimpleNamespace()) == 2
    assert not home.exists()
    error = capsys.readouterr().err
    assert "Unix socket path" in error
    assert "shorter ROSCLAW_HOME" in error
    assert "no socket or control token was created" in error


def test_short_home_validation_has_no_filesystem_side_effect(tmp_path):
    home = Path("/tmp/rc-native-preflight")
    validate_native_socket_paths(home)


@pytest.mark.parametrize("platform,limit", [("linux", 107), ("darwin", 103)])
def test_exact_encoded_boundary(platform, limit, monkeypatch):
    monkeypatch.setattr("rosclaw.agentd.socket_paths.sys.platform", platform)
    suffix = "/run/pi-bridge.sock"
    validate_native_socket_paths(Path("/" + "a" * (limit - len(suffix) - 1)))
    with pytest.raises(ValueError, match="limit"):
        validate_native_socket_paths(Path("/" + "a" * (limit - len(suffix))))


def test_multibyte_home_uses_encoded_length(monkeypatch):
    monkeypatch.setattr("rosclaw.agentd.socket_paths.sys.platform", "linux")
    home = Path("/tmp/" + "证据" * 20)
    assert len(str(home / "run/pi-bridge.sock")) < 107
    with pytest.raises(ValueError, match="bytes"):
        validate_native_socket_paths(home)
