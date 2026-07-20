"""CLI tests for the lightweight ``rosclaw daemon`` dispatcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import rosclaw.daemon.cli as daemon_cli
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.cli import _parse_octal_mode, dispatch_daemon_argv
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import DaemonControlPlane


def _runtime() -> Runtime:
    return Runtime(
        RuntimeConfig(
            robot_id="cli-test",
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=False,
        )
    )


def test_daemon_status_fast_path_returns_json(
    tmp_path: Path,
    capsys,
) -> None:
    socket_path = tmp_path / "rosclawd.sock"
    daemon = RosclawDaemon(
        service=DaemonControlPlane(runtime=_runtime()),
        socket_path=socket_path,
    )
    daemon.start()
    try:
        result = dispatch_daemon_argv(["daemon", "status", "--socket", str(socket_path), "--json"])
    finally:
        daemon.stop()

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["running"] is True
    assert payload["southbound_owner"] == "rosclawd"


def test_daemon_status_missing_socket_fails_closed(
    tmp_path: Path,
    capsys,
) -> None:
    result = dispatch_daemon_argv(
        [
            "daemon",
            "status",
            "--socket",
            str(tmp_path / "missing.sock"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 2
    assert payload["ok"] is False
    assert payload["error"]["code"] == "DAEMON_UNAVAILABLE"


def test_socket_mode_parser_requires_owner_access_and_blocks_world_access() -> None:
    assert _parse_octal_mode("0600") == 0o600
    assert _parse_octal_mode("0660") == 0o660

    for mode in ("0000", "0060", "0606", "0666"):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_octal_mode(mode)


@pytest.mark.parametrize(
    ("physical_stop_observed", "expected_exit"),
    [(False, 3), (True, 0)],
)
def test_daemon_emergency_stop_exit_requires_physical_confirmation(
    physical_stop_observed: bool,
    expected_exit: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    class FakeDaemon:
        def emergency_stop(
            self,
            reason: str,
            *,
            source: str,
            timeout_sec: float,
        ) -> dict[str, object]:
            assert reason == "operator test"
            assert source == "rosclaw.daemon.cli"
            assert timeout_sec == 1.0
            return {
                "request_dispatched": True,
                "driver_acknowledged": True,
                "physical_stop_observed": physical_stop_observed,
                "stopped": physical_stop_observed,
            }

    monkeypatch.setattr(daemon_cli, "_client", lambda _args: FakeDaemon())

    result = dispatch_daemon_argv(
        [
            "daemon",
            "emergency-stop",
            "--reason",
            "operator test",
            "--json",
        ]
    )

    assert result == expected_exit
    payload = json.loads(capsys.readouterr().out)
    assert payload["physical_stop_observed"] is physical_stop_observed
