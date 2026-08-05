"""operatord CLI socket-selection regressions."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rosclaw.operatord import cli


def test_start_uses_configured_daemon_socket(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "system-run" / "rosclawd.sock"
    captured: dict[str, object] = {}

    class Server:
        _path = tmp_path / "operatord.sock"

    async def fake_run_operatord(**kwargs):
        captured.update(kwargs)
        return Server()

    async def stop_after_start() -> None:
        while "daemon_socket" not in captured:
            await asyncio.sleep(0)
        raise KeyboardInterrupt

    monkeypatch.setenv("ROSCLAW_DAEMON_SOCKET", str(configured))
    monkeypatch.setattr("rosclaw.operatord.server.run_operatord", fake_run_operatord)
    monkeypatch.setattr("asyncio.Event.wait", lambda _self: stop_after_start())

    assert cli.dispatch_operatord_argv(["operatord", "start", "--home", str(tmp_path)]) == 0
    assert captured["daemon_socket"] == configured
