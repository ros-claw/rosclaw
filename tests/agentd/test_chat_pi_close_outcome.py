"""Real launcher control flow, with inert deterministic external boundaries."""

import argparse
import asyncio
import io
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw import version_diag
from rosclaw.agentd import cli, onboarding, pi_entry


@pytest.fixture
def launch(monkeypatch):
    events = []
    state = SimpleNamespace(child=0, close_error=None)
    home = Path("/unit")
    close_token = object()

    class Server:
        def __init__(self, config):
            self._should_exit = False

        @property
        def should_exit(self):
            return self._should_exit

        @should_exit.setter
        def should_exit(self, value):
            assert value is True
            self._should_exit = value
            events.append("exit")

    server = Server(None)

    class Service:
        def __init__(self, config, actual_home):
            assert actual_home == home

        def create_mission(self, goal, mode):
            return SimpleNamespace(mode=SimpleNamespace(value="SIMULATION"), mission_id="unit")

        def close(self):
            assert server.should_exit
            events.append("close")
            return close_token

    class Loop:
        def stop(self):
            events.append("stop")

        def call_soon_threadsafe(self, callback):
            events.append("schedule-stop")
            callback()

    loop = Loop()

    class Event:
        def wait(self, timeout):
            assert server.should_exit
            events.append(("wait", timeout))
            return True

    class Thread:
        def __init__(self, *, target, daemon):
            assert callable(target) and daemon is True

        def start(self):
            events.append("thread-start")  # Never run target or start a thread.

        def join(self, timeout):
            events.append(("join", timeout))

    class Future:
        def result(self, timeout):
            events.append(("result", timeout))
            if state.close_error is not None:
                raise state.close_error

    def submit(token, actual_loop):
        assert token is close_token and actual_loop is loop
        events.append("submit")
        return Future()

    def child_call(argv, *, env):
        assert argv[:4] == ["unit-node", "unit-entry", "--profile", "developer"]
        assert env["ROSCLAW_HOME"] == str(home)
        events.append("child")
        if isinstance(state.child, BaseException):
            raise state.child
        return state.child

    def terminate():
        events.append("terminate")

    def popen(*args, **kwargs):
        events.append("managed-start")
        return SimpleNamespace(terminate=terminate)

    monkeypatch.setattr(cli, "_find_pi_agent_entry", lambda **kw: ("unit-node", "unit-entry"))
    monkeypatch.setattr(version_diag, "is_stamped_install", lambda: False)
    monkeypatch.setattr(cli, "load_agent_config", lambda path: object())
    monkeypatch.setattr(onboarding, "_write_retry_budget", lambda path: None)
    monkeypatch.setattr(cli, "_route_internal_diagnostics_to_log", lambda *a, **k: None)
    monkeypatch.setattr(cli, "AgentService", Service)
    monkeypatch.setattr(cli, "create_app", lambda service: object())
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(Config=lambda *a, **k: object(), Server=lambda config: server),
    )
    monkeypatch.setattr(asyncio, "new_event_loop", lambda: loop)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    monkeypatch.setattr(threading, "Event", Event)
    monkeypatch.setattr(threading, "Thread", Thread)
    monkeypatch.setattr(time, "time", lambda: 0.0)
    monkeypatch.setattr(time, "sleep", lambda duration: pytest.fail("unexpected startup wait"))
    monkeypatch.setattr(Path, "exists", lambda path: path.name != "operatord.sock")
    monkeypatch.setattr(Path, "open", lambda *a, **k: io.BytesIO())
    monkeypatch.setattr(pi_entry, "node_runtime_env", lambda env: env)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "call", child_call)

    def invoke():
        args = argparse.Namespace(mission=None, goal=None, mode="SIMULATION")
        return cli._chat_pi(home, args)

    state.invoke = invoke
    state.events = events
    return state


def assert_cleanup(events):
    assert events == [
        "thread-start",
        "managed-start",
        "child",
        "exit",
        ("wait", 10),
        "close",
        "submit",
        ("result", 15),
        "schedule-stop",
        "stop",
        ("join", 5),
        "terminate",
    ]


@pytest.mark.parametrize("cleanup", ["clean", "exception", "timeout"])
@pytest.mark.parametrize(
    "child", [0, 7, -9, "sigint"], ids=["zero", "positive", "negative", "sigint"]
)
def test_final_outcome(launch, capsys, child, cleanup):
    launch.child = KeyboardInterrupt() if child == "sigint" else child
    if cleanup != "clean":
        launch.close_error = (
            TimeoutError("bounded close") if cleanup == "timeout" else RuntimeError("close fault")
        )
    expected = 0 if child == "sigint" else child
    if expected == 0 and cleanup != "clean":
        expected = 2
    assert launch.invoke() == expected
    assert_cleanup(launch.events)
    diagnostic = capsys.readouterr().err
    if cleanup == "clean":
        assert diagnostic == ""
    else:
        assert f"{type(launch.close_error).__name__}: {launch.close_error}" in diagnostic
        assert "内核关闭异常" in diagnostic


@pytest.mark.parametrize("cleanup", ["clean", "exception", "timeout"])
@pytest.mark.parametrize("kind", [OSError, SystemExit], ids=["oserror", "systemexit"])
def test_original_exception_propagates(launch, capsys, kind, cleanup):
    original = kind("original child fault")
    launch.child = original
    if cleanup != "clean":
        launch.close_error = (
            TimeoutError("bounded close") if cleanup == "timeout" else RuntimeError("close fault")
        )
    with pytest.raises(kind) as caught:
        launch.invoke()
    assert caught.value is original
    assert_cleanup(launch.events)
    diagnostic = capsys.readouterr().err
    assert bool(diagnostic) == (cleanup != "clean")
