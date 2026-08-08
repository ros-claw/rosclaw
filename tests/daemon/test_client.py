from __future__ import annotations

from pathlib import Path

import rosclaw.daemon.client as daemon_client


def test_default_workspace_prefers_packaged_system_daemon_socket(
    tmp_path: Path,
    monkeypatch,
) -> None:
    system_socket = tmp_path / "run" / "rosclawd.sock"
    system_socket.parent.mkdir()
    system_socket.touch()
    monkeypatch.delenv("ROSCLAW_DAEMON_SOCKET", raising=False)
    monkeypatch.delenv("ROSCLAW_HOME", raising=False)
    monkeypatch.setattr(daemon_client, "SYSTEM_DAEMON_SOCKET_PATH", system_socket)

    assert daemon_client.get_daemon_socket_path() == system_socket


def test_explicit_workspace_does_not_fall_through_to_system_daemon(
    tmp_path: Path,
    monkeypatch,
) -> None:
    system_socket = tmp_path / "run" / "rosclawd.sock"
    system_socket.parent.mkdir()
    system_socket.touch()
    workspace = tmp_path / "isolated-home"
    monkeypatch.delenv("ROSCLAW_DAEMON_SOCKET", raising=False)
    monkeypatch.setenv("ROSCLAW_HOME", str(workspace))
    monkeypatch.setattr(daemon_client, "SYSTEM_DAEMON_SOCKET_PATH", system_socket)

    assert daemon_client.get_daemon_socket_path() == workspace / "run" / "rosclawd.sock"


def test_explicit_socket_overrides_system_daemon_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configured = tmp_path / "configured.sock"
    monkeypatch.delenv("ROSCLAW_HOME", raising=False)
    monkeypatch.setenv("ROSCLAW_DAEMON_SOCKET", str(configured))

    assert daemon_client.get_daemon_socket_path() == configured
