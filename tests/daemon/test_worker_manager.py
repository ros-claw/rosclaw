"""Real subprocess fault tests for isolated Adapter workers."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.daemon.worker_manager import WorkerManager, WorkerSpec

FIXTURE = Path(__file__).parents[1] / "fixtures" / "daemon_worker.py"


def _command(mode: str, *, after: float = 0.1) -> tuple[str, ...]:
    return (
        sys.executable,
        str(FIXTURE),
        "--mode",
        mode,
        "--after",
        str(after),
    )


def _wait_for(predicate, *, timeout_sec: float = 3.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("worker condition did not become true")


def _runtime() -> Runtime:
    return Runtime(
        RuntimeConfig(
            robot_id="worker-test",
            enable_firewall=False,
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


def test_healthy_worker_reports_ready_and_fresh_heartbeat() -> None:
    manager = WorkerManager(monitor_interval_sec=0.01)
    manager.register(
        WorkerSpec(
            worker_id="healthy",
            command=_command("healthy"),
            heartbeat_timeout_sec=0.2,
        )
    )
    try:
        status = manager.start_worker("healthy")
        time.sleep(0.08)
        current = manager.get_status("healthy")
    finally:
        manager.close()

    assert status["state"] == "READY"
    assert status["generation"] == 1
    assert current["state"] == "READY"
    assert current["heartbeat_age_ms"] < 200


def test_crashed_worker_restarts_with_new_connection_generation() -> None:
    changes: list[tuple[str | None, str]] = []
    manager = WorkerManager(
        monitor_interval_sec=0.01,
        on_generation_change=lambda _worker, old, new: changes.append((old, new)),
    )
    manager.register(
        WorkerSpec(
            worker_id="crasher",
            command=_command("crash", after=0.08),
            heartbeat_timeout_sec=0.3,
            restart_limit=3,
        )
    )
    try:
        first = manager.start_worker("crasher")
        restarted = _wait_for(
            lambda: status if (status := manager.get_status("crasher"))["generation"] >= 2 else None
        )
    finally:
        manager.close()

    assert restarted["connection_id"] != first["connection_id"]
    assert changes[0][0] is None
    assert changes[1][0] == first["connection_id"]


def test_stalled_worker_is_killed_and_restart_budget_is_bounded() -> None:
    manager = WorkerManager(monitor_interval_sec=0.01)
    manager.register(
        WorkerSpec(
            worker_id="stalled",
            command=_command("stall"),
            heartbeat_timeout_sec=0.08,
            stop_timeout_sec=0.05,
            restart_limit=1,
        )
    )
    try:
        manager.start_worker("stalled")
        failed = _wait_for(
            lambda: (
                status if (status := manager.get_status("stalled"))["state"] == "FAILED" else None
            )
        )
    finally:
        manager.close()

    assert failed["generation"] == 2
    assert failed["restart_count_window"] == 1
    assert failed["last_error"] == "restart budget exhausted"
    assert failed["pid"] is None


def test_worker_generation_change_estops_and_disarms_daemon() -> None:
    service = DaemonControlPlane(runtime=_runtime())
    service.workers.register(
        WorkerSpec(
            worker_id="adapter",
            command=_command("crash", after=0.08),
            heartbeat_timeout_sec=0.3,
            restart_limit=2,
        )
    )
    peer = PeerCredentials(pid=11, uid=os.geteuid(), gid=os.getegid())
    service.start()
    try:
        service.arm_runtime("test preflight complete", peer)
        first = service.control_worker("start", "adapter", peer)["worker"]
        status = _wait_for(
            lambda: (
                value
                if (
                    (value := service.get_runtime_status(peer))["workers"]["workers"][0][
                        "generation"
                    ]
                    >= 2
                    and value["emergency_stop_latched"] is True
                )
                else None
            )
        )
    finally:
        service.close()

    worker = status["workers"]["workers"][0]
    assert worker["connection_id"] != first["connection_id"]
    assert status["supervision_state"] == "ESTOPPED"


def test_generation_callback_runs_without_worker_manager_lock() -> None:
    callback_results: list[bool] = []
    manager: WorkerManager

    def callback(_worker: str, _old: str | None, _new: str) -> None:
        inspected = threading.Event()

        def inspect_status() -> None:
            manager.status()
            inspected.set()

        thread = threading.Thread(target=inspect_status)
        thread.start()
        callback_results.append(inspected.wait(timeout=0.5))
        thread.join(timeout=1.0)

    manager = WorkerManager(
        monitor_interval_sec=0.01,
        on_generation_change=callback,
    )
    manager.register(WorkerSpec(worker_id="callback", command=_command("healthy")))
    try:
        manager.start_worker("callback")
    finally:
        manager.close()

    assert callback_results == [True]


@pytest.mark.parametrize("mode", ["invalid", "oversize"])
def test_invalid_worker_protocol_is_terminated_without_unbounded_restart(mode: str) -> None:
    manager = WorkerManager(monitor_interval_sec=0.01)
    manager.register(
        WorkerSpec(
            worker_id=mode,
            command=_command(mode),
            heartbeat_timeout_sec=0.08,
            stop_timeout_sec=0.05,
            restart_limit=0,
        )
    )
    try:
        manager.start_worker(mode, wait_ready=False)
        failed = _wait_for(
            lambda: status if (status := manager.get_status(mode))["state"] == "FAILED" else None
        )
    finally:
        manager.close()

    assert failed["generation"] == 1
    assert failed["pid"] is None
    assert failed["desired_running"] is False
