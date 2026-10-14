"""Bounded independent observer contracts, without ROS or execution authority."""

import importlib
import threading
from pathlib import Path

import pytest


@pytest.fixture
def monitor_setup(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_current_monitor")
    clock = [1_000_000_000]
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: clock[0])

    class Facts(module.CurrentSimulationFacts):
        def __init__(self):
            self.failure = None

        def collect(self, *, freshness_horizon_ms):
            assert freshness_horizon_ms == 100
            if self.failure:
                raise ValueError(self.failure)
            return {
                "health_review_ns": clock[0],
                "valid_until_monotonic_ns": clock[0] + 100_000_000,
                "health": {"status": "healthy"},
            }

    facts = Facts()
    monitor = module.BoundedCurrentFactsMonitor(
        facts=facts, daemon_observer=lambda: {"generation": "original"}
    )
    return module, facts, monitor, clock


def test_snapshot_copy_and_actual_deadline(monitor_setup):
    _, _, monitor, clock = monitor_setup
    monitor._latest = monitor._collect()
    monitor.get()["health"]["status"] = "mutated"
    assert monitor.get()["health"]["status"] == "healthy"
    clock[0] += 99_999_999
    assert monitor.get()["daemon_status"]["generation"] == "original"
    clock[0] += 1
    with pytest.raises(ValueError, match="permanently latched"):
        monitor.get()
    monitor._latest = monitor._collect()
    with pytest.raises(ValueError, match="permanently latched"):
        monitor.get()


def test_clock_rollback_latches(monitor_setup):
    _, _, monitor, clock = monitor_setup
    monitor._latest = monitor._collect()
    clock[0] -= 1
    with pytest.raises(ValueError, match="CLOCK_INVALID"):
        monitor.get()


def test_worker_failure_discards_last_good(monitor_setup):
    _, facts, monitor, _ = monitor_setup
    monitor._latest = monitor._collect()
    facts.failure = "source disappeared"
    monitor._run()
    assert monitor._latest is None
    with pytest.raises(ValueError, match="source disappeared"):
        monitor.get()


def test_daemon_check_consumes_freshness_window(monitor_setup):
    _, _, monitor, clock = monitor_setup

    def delayed_daemon():
        clock[0] += 100_000_000
        return {}

    monitor._daemon_observer = delayed_daemon
    with pytest.raises(ValueError, match="expired during actual daemon check"):
        monitor._collect()


def test_close_and_restart_forbidden(monitor_setup):
    _, _, monitor, _ = monitor_setup
    monitor.start()
    monitor.close()
    assert not monitor._thread.is_alive()
    with pytest.raises(ValueError, match="CLOSED"):
        monitor.get()
    with pytest.raises(ValueError, match="cannot be restarted"):
        monitor.start()


def test_worker_stall_expires_in_consumer(monitor_setup):
    _, _, monitor, clock = monitor_setup
    monitor._latest = monitor._collect()
    blocked = threading.Event()
    released = threading.Event()
    original = monitor._collect

    def stall():
        blocked.set()
        released.wait(1)
        return original()

    monitor._collect = stall
    monitor._thread = threading.Thread(target=monitor._run, daemon=True)
    monitor._thread.start()
    assert blocked.wait(1)
    clock[0] += 100_000_000
    with pytest.raises(ValueError, match="EXPIRED"):
        monitor.get()
    released.set()
    monitor._thread.join(1)
    assert monitor._latest is None
    monitor.close()
