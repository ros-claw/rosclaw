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
        "observed_monotonic_ns": clock[0],
        "terminal": False,
        "reason": None,
        "facts": {
            "health_review_ns": clock[0],
            "completed_ns": clock[0],
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
    assert events == ["terminate", ("wait", 2)]
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


def test_atomic_replacement_race_reopens_latest_but_never_returns_expired_snapshot(
    fixture, monkeypatch
):
    monitor, _, clock, original, _ = fixture
    expired = json.dumps(original).encode()
    clock[0] += 100_000_000
    original["facts"]["health_review_ns"] = clock[0]
    original["facts"]["completed_ns"] = clock[0]
    original["observed_monotonic_ns"] = clock[0]
    original["facts"]["valid_until_monotonic_ns"] = clock[0] + 100_000_000
    fresh = json.dumps(original).encode()
    reads = iter([expired, fresh])
    module = importlib.import_module("simulation_current_process")
    monkeypatch.setattr(module, "_read", lambda *args, **kwargs: next(reads))
    result = monitor.get()
    assert result["health_review_ns"] == clock[0]
    assert monitor.failed_reason is None


def test_native_pointer_clock_is_sampled_after_concurrent_publication(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_current_facts")
    clock = [10_000_000_000]
    pointer = {
        "file": "original-native-001.json",
        "sha256": "a" * 64,
        "captured_at": "original",
        "received_monotonic_ns": 10_001_000_000,
    }

    def concurrent_read(*args, **kwargs):
        clock[0] += 2_000_000
        return json.dumps(pointer).encode()

    monkeypatch.setattr(module, "_read", concurrent_read)
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: clock[0])
    actual, observed = module.read_current_native_pointer(tmp_path)
    assert actual == pointer and observed == 10_002_000_000
    pointer["received_monotonic_ns"] = clock[0] + 3_000_000
    with pytest.raises(ValueError, match="bounded current original Native pointer"):
        module.read_current_native_pointer(tmp_path)


def test_prepared_facts_cannot_publish_health_when_daemon_start_observation_fails(
    monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_current_process")
    (tmp_path / "execution-control").mkdir()
    monkeypatch.setattr(
        module, "CurrentSimulationFacts", lambda *a, **k: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(module, "DaemonClient", lambda **k: None)

    def unavailable():
        raise ValueError("synthetic independent daemon unavailable")

    monkeypatch.setattr(
        module,
        "BoundedDaemonObservation",
        lambda **k: SimpleNamespace(start=unavailable, close=lambda: None),
    )
    with pytest.raises(ValueError, match="independent daemon unavailable"):
        module.run_worker(
            {
                "root": str(tmp_path),
                "enrollment_sha256": "a" * 64,
                "admission": {"daemon_uid": 1000},
                "body_id": "synthetic",
            }
        )
    result = json.loads((tmp_path / "execution-control/current-monitor.json").read_text())
    assert result["terminal"] is True and result["facts"] is None
    stages = [
        json.loads(line)["stage"]
        for line in (tmp_path / "sdk-current-startup-original.jsonl").read_text().splitlines()
    ]
    assert stages == [
        "CONFIGURATION_ACCEPTED",
        "CURRENT_FACTS_CONSTRUCTED",
        "DAEMON_OBSERVER_STARTING",
        "FAILED_NO_CURRENT_ADMISSION",
    ]
