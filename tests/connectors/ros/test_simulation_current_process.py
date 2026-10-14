"""Independent SDK observation output contracts; no ROS or host Runtime."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_current_process")
    clock = [1_000_000_000]
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: clock[0])
    monitor = module.SdkCurrentFactsProcess(
        root=tmp_path,
        enrollment_sha256="a" * 64,
        admission={},
        binding=SimpleNamespace(body_id="synthetic"),
    )
    monitor.path.parent.mkdir()
    process = SimpleNamespace(pid=123, poll=lambda: None)
    monitor.process = process
    original = {
        "schema_version": "rosclaw.sdk_current_monitor.v1",
        "pid": 123,
        "terminal": False,
        "reason": None,
        "facts": {
            "health_review_ns": clock[0],
            "valid_until_monotonic_ns": clock[0] + 100_000_000,
            "health": {"status": "healthy"},
        },
    }

    def publish(value=original):
        monitor.path.write_text(json.dumps(value))

    publish()
    return monitor, process, clock, original, publish


def test_snapshot_copy_and_expiry_latches(fixture):
    monitor, _, clock, _, publish = fixture
    monitor.get()["health"]["status"] = "mutated"
    assert monitor.get()["health"]["status"] == "healthy"
    clock[0] += 99_999_999
    assert monitor.get()
    clock[0] += 1
    with pytest.raises(ValueError, match="window expired"):
        monitor.get()
    publish()
    clock[0] -= 1
    with pytest.raises(ValueError, match="permanently latched"):
        monitor.get()


@pytest.mark.parametrize(
    "fault", ["worker_exit", "pid", "terminal", "clock", "missing", "symlink", "extra", "bool_time"]
)
def test_invalid_observation_never_falls_back(fixture, fault):
    monitor, process, clock, original, publish = fixture
    if fault == "worker_exit":
        process.poll = lambda: 1
    elif fault == "pid":
        original["pid"] = 124
    elif fault == "terminal":
        original.update(terminal=True, reason="source lost")
    elif fault == "clock":
        clock[0] -= 1
    elif fault == "extra":
        original["approved"] = True
    elif fault == "bool_time":
        original["facts"]["health_review_ns"] = True
    publish()
    if fault == "missing":
        monitor.path.unlink()
    elif fault == "symlink":
        other = monitor.path.with_suffix(".original")
        monitor.path.rename(other)
        monitor.path.symlink_to(other)
    with pytest.raises(ValueError, match="permanently latched"):
        monitor.get()
    assert monitor.failed_reason


def test_close_terminates_owned_child_and_prevents_restart(fixture):
    monitor, process, *_ = fixture
    events = []
    process.terminate = lambda: events.append("terminate")
    process.wait = lambda **kw: events.append(("wait", kw["timeout"]))
    monitor.close()
    assert events == ["terminate", ("wait", 1)]
    with pytest.raises(ValueError, match="cannot be restarted"):
        monitor.start()
    with pytest.raises(ValueError, match="CLOSED"):
        monitor.get()


def test_cursor_barrier_waits_for_independent_observation_without_extending_expiry(
    fixture, monkeypatch
):
    monitor, _, clock, original, publish = fixture
    original["facts"]["source"] = {"last_sample": 24}
    publish()
    module = importlib.import_module("simulation_current_process")

    def advance(_):
        clock[0] += 2_000_000
        original["facts"]["source"]["last_sample"] = 25
        publish()

    monkeypatch.setattr(module.time, "sleep", advance)
    assert monitor.get_at_least_source(25)["source"]["last_sample"] == 25
    assert original["facts"]["valid_until_monotonic_ns"] == 1_100_000_000


def test_stalled_cursor_cannot_hide_expired_observation(fixture, monkeypatch):
    monitor, _, clock, original, publish = fixture
    original["facts"]["source"] = {"last_sample": 24}
    publish()
    module = importlib.import_module("simulation_current_process")

    def advance(_):
        clock[0] += 2_000_000

    monkeypatch.setattr(module.time, "sleep", advance)
    with pytest.raises(ValueError, match="window expired"):
        monitor.get_at_least_source(25)
