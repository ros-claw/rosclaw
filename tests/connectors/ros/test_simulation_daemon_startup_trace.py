"""SDK-only bounded diagnostics preserve server thread failure semantics."""

import importlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_sdk_diagnostic_records_failure_and_restores_original_thread_start(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_daemon_startup_trace")
    container = tmp_path / "container"
    container.touch()
    monkeypatch.setattr(module, "Path", lambda p: container if str(p) == "/.dockerenv" else Path(p))
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")

    def unavailable(thread, *a, **k):
        raise RuntimeError("synthetic thread capacity failure")

    monkeypatch.setattr(threading.Thread, "start", unavailable)
    server = SimpleNamespace(
        _lock=threading.Lock(),
        _client_threads=set(),
        _client_slots=SimpleNamespace(_value=32),
        max_clients=32,
    )
    trace = module.DaemonStartupTrace(server, tmp_path)
    try:
        trace.sample()
        with pytest.raises(RuntimeError, match="synthetic thread capacity"):
            threading.Thread(name="rosclawd-client").start()
    finally:
        trace.close()
    assert threading.Thread.start is unavailable
    rows = [
        json.loads(line)
        for line in (tmp_path / "sdk-daemon-startup-original.jsonl").read_text().splitlines()
    ]
    assert [r["event"] for r in rows] == ["READONLY_SERVER_CAPACITY", "CLIENT_THREAD_START_FAILED"]
    assert rows[0]["client_slots_available"] == rows[0]["max_clients"] == 32
    assert rows[1]["error"] == "synthetic thread capacity failure"
    assert not any("request" in r for r in rows)
