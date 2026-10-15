"""Synthetic independent SDK daemon observation contracts, never motion proof."""

import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.connectors.ros.test_simulation_navigation_supervision import healthy_status


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_daemon_observation")
    clock = [10_000_000_000]
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: clock[0])
    status = healthy_status()
    status["daemon_pid"] = os.getpid() + 1000
    status.update(
        daemon_peer={"pid": status["daemon_pid"], "uid": os.getuid()},
        client_peer={"pid": os.getpid()},
        process_separated=True,
    )
    client = SimpleNamespace(get_runtime_status=lambda: status)
    admission = {
        "daemon_pid": status["daemon_pid"],
        "daemon_uid": os.getuid(),
        "daemon_instance_id": status["daemon_instance_id"],
        "expires_ns": clock[0] + 1_000_000_000,
    }
    observer = module.BoundedDaemonObservation(client=client, admission=admission, body_id="mobile")
    return observer, clock, status, client


def test_expiry_and_no_last_good_after_latch(fixture):
    observer, clock, *_ = fixture
    observer._latest = observer._observe()
    observer.snapshot()["status"]["watchdog"]["healthy"] = False
    assert observer.snapshot()["status"]["watchdog"]["healthy"]
    clock[0] += 249_999_999
    assert observer.snapshot()
    clock[0] += 1
    with pytest.raises(ValueError, match="EXPIRED"):
        observer.snapshot()
    observer._latest = observer._observe()
    with pytest.raises(ValueError, match="permanently latched"):
        observer.snapshot()


@pytest.mark.parametrize("fault", ["generation", "peer", "watchdog", "ledger", "estop", "rpc"])
def test_actual_status_failure_discards_previous_observation(fixture, fault):
    observer, _, status, client = fixture
    observer._latest = observer._observe()
    if fault == "generation":
        status["daemon_instance_id"] = "new"
    elif fault == "peer":
        status["daemon_peer"]["pid"] += 1
    elif fault == "watchdog":
        status["watchdog"]["healthy"] = False
    elif fault == "ledger":
        status["ledger"]["write_failed"] = True
    elif fault == "estop":
        status["emergency_stop_latched"] = True
    else:

        def unavailable():
            raise OSError("socket unavailable")

        client.get_runtime_status = unavailable
    observer._run()
    assert observer._latest is None
    with pytest.raises(ValueError, match="permanently latched"):
        observer.snapshot()


def test_rpc_cannot_consume_observation_lifetime(fixture):
    observer, clock, status, client = fixture

    def slow():
        clock[0] += 250_000_000
        return status

    client.get_runtime_status = slow
    with pytest.raises(ValueError, match="expired during RPC"):
        observer._observe()


def test_close_is_terminal_and_owned_thread_stops(fixture):
    observer, *_ = fixture
    observer.start()
    observer.close()
    assert not observer._thread.is_alive()
    with pytest.raises(ValueError, match="CLOSED"):
        observer.snapshot()
    with pytest.raises(ValueError, match="cannot restart"):
        observer.start()
