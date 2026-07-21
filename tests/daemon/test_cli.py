"""CLI tests for the lightweight ``rosclaw daemon`` dispatcher."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

import rosclaw.daemon.cli as daemon_cli
import rosclaw.daemon.service as daemon_service
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.cli import _parse_octal_mode, dispatch_daemon_argv
from rosclaw.daemon.ledger import DaemonLedger, LedgerIntegrityError
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


def test_daemon_recovery_acknowledgement_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    class FakeDaemon:
        def acknowledge_recovery(self, reason: str) -> dict[str, object]:
            assert reason == "operator reviewed evidence"
            return {
                "acknowledged": True,
                "recovery_required": False,
                "emergency_stop_latched": True,
            }

    monkeypatch.setattr(daemon_cli, "_client", lambda _args: FakeDaemon())

    result = dispatch_daemon_argv(
        [
            "daemon",
            "acknowledge-recovery",
            "--reason",
            "operator reviewed evidence",
            "--json",
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "acknowledged": True,
        "recovery_required": False,
        "emergency_stop_latched": True,
    }


def test_daemon_serve_forwards_explicit_ledger_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_daemon(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(daemon_cli, "run_daemon", fake_run_daemon)
    database = tmp_path / "ledger.sqlite3"
    key = tmp_path / "ledger.key"

    result = dispatch_daemon_argv(
        [
            "daemon",
            "serve",
            "--ledger",
            str(database),
            "--ledger-key",
            str(key),
        ]
    )

    assert result == 0
    assert captured["ledger_path"] == str(database)
    assert captured["ledger_key_path"] == str(key)


def test_tampered_ledger_blocks_daemon_before_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    with DaemonLedger(database, key_path=key) as ledger:
        ledger.append(
            "TEST_EVENT",
            entity_kind="TEST",
            entity_id="test-1",
            payload={"safe": True},
        )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE ledger_events SET payload_json = ? WHERE sequence = 2",
            ('{"safe":false}',),
        )
    runtime_built = False

    def fail_if_runtime_built(_robot_id: str):
        nonlocal runtime_built
        runtime_built = True
        raise AssertionError("runtime must not be built for a tampered ledger")

    monkeypatch.setattr(daemon_cli, "build_daemon_runtime", fail_if_runtime_built)

    with pytest.raises(LedgerIntegrityError):
        daemon_cli.run_daemon(
            socket_path=tmp_path / "run" / "rosclawd.sock",
            socket_mode=0o600,
            socket_group=None,
            robot_id="test",
            log_level="ERROR",
            ledger_path=database,
            ledger_key_path=key,
        )

    assert runtime_built is False


def test_daemon_startup_failure_stops_an_already_initialized_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeRuntime:
        def request_emergency_stop(self, *_args: object, **_kwargs: object) -> None:
            calls.append("emergency_stop")

        def stop(self) -> None:
            calls.append("stop")

    monkeypatch.setattr(daemon_cli, "build_daemon_runtime", lambda _robot_id: FakeRuntime())

    def fail_control_plane(**_kwargs: object) -> None:
        raise LedgerIntegrityError("injected semantic ledger failure")

    monkeypatch.setattr(daemon_service, "DaemonControlPlane", fail_control_plane)

    with pytest.raises(LedgerIntegrityError, match="semantic ledger"):
        daemon_cli.run_daemon(
            socket_path=tmp_path / "run" / "rosclawd.sock",
            socket_mode=0o600,
            socket_group=None,
            robot_id="test",
            log_level="ERROR",
            ledger_path=tmp_path / "state" / "ledger.sqlite3",
            ledger_key_path=tmp_path / "state" / "ledger.key",
        )

    assert calls == ["emergency_stop", "stop"]
