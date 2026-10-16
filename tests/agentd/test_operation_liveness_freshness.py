"""Source-only liveness regressions: real SQLite/events, no worker processes.

Self-contained fixtures allow execution with --noconftest against either source.
Only the clock is replaced; sweep and all SQL/event producers are real.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

import rosclaw.task_kernel.operation_manager as operation_module
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.operation_manager import OPERATION_TERMINAL, OperationManager

BASE = datetime(2026, 10, 16, tzinfo=UTC)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    class Clock(datetime):
        seconds = 100

        @classmethod
        def now(cls, tz=None):
            return (BASE + timedelta(seconds=cls.seconds)).astimezone(tz)

    monkeypatch.setattr(operation_module, "datetime", Clock)
    conn = sqlite3.connect(tmp_path / "liveness.db")
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    conn.execute(
        "INSERT INTO tasks (task_id, mission_id, root_goal, mode, body_id, "
        "state, active_revision, workspace_path, created_at, updated_at) "
        "VALUES ('t', 'm', 'liveness fixture', 'SIMULATION', '', 'ACTIVE', 1, '', ?, ?)",
        (BASE.isoformat(), BASE.isoformat()),
    )
    conn.commit()
    try:
        yield conn, OperationManager(None, conn), Clock
    finally:
        conn.close()


def seed(conn, state="RUNNING", failure_code="", heartbeat=None):
    conn.execute(
        "INSERT INTO operations (operation_id, task_id, attempt_id, kind, state, "
        "resumable, started_at, heartbeat_at, failure_code, cancel_reason) "
        "VALUES ('op', 't', '', 'process', ?, 0, ?, ?, ?, 'owner-reason')",
        (state, BASE.isoformat(), heartbeat or BASE.isoformat(), failure_code),
    )
    conn.commit()


def sweep(manager):
    return asyncio.run(manager.sweep_liveness(stale_after_s=30))


def events(manager):
    return manager.events_since("t", 0)


@pytest.mark.parametrize("state", ["QUEUED", "ADMITTED", "RUNNING"])
def test_repeated_quiet_sweeps_never_manufacture_recovery(ledger, state):
    conn, manager, clock = ledger
    seed(conn, state)
    assert sweep(manager) == {"degraded": 1, "resumed": 0}
    assert manager.get("op")["heartbeat_at"] == BASE.isoformat()
    first_events = events(manager)
    assert [e["event_type"] for e in first_events] == ["operation.degraded"]
    assert first_events[0]["payload"] == {"operation_id": "op", "stale_after_s": 30}
    for second in (101, 102, 131, 200):
        clock.seconds = second
        assert sweep(manager) == {"degraded": 0, "resumed": 0}
        assert manager.get("op")["state"] == "DEGRADED"
        assert manager.get("op")["heartbeat_at"] == BASE.isoformat()
        assert events(manager) == first_events
    assert not manager._procs and not manager._drivers and not manager._actions


@pytest.mark.parametrize("producer", ["heartbeat", "progress", "output"])
def test_real_observation_resumes_without_extending_its_freshness(ledger, producer):
    conn, manager, clock = ledger
    seed(conn)
    assert sweep(manager) == {"degraded": 1, "resumed": 0}
    clock.seconds = 110
    observed_at = clock.now(UTC).isoformat()
    if producer == "heartbeat":
        manager._touch("op", "t")
    elif producer == "progress":
        manager.report_progress("op", {"fraction": 0.5})
        assert json.loads(manager.get("op")["progress_json"]) == {"fraction": 0.5}
    else:
        previous = {"offset": 0, "pending_hex": "", "finalized": False}
        conn.execute(
            "UPDATE operations SET output_checkpoint_json = ? WHERE operation_id = 'op'",
            (json.dumps(previous),),
        )
        checkpoint = {**previous, "offset": 5}
        manager._checkpoint_output(manager.get("op"), checkpoint, "alive", previous)
        assert json.loads(manager.get("op")["output_checkpoint_json"]) == checkpoint
    assert manager.get("op")["heartbeat_at"] == observed_at
    assert manager.get("op")["state"] == "DEGRADED"
    clock.seconds = 120
    assert sweep(manager) == {"degraded": 0, "resumed": 1}
    assert manager.get("op")["heartbeat_at"] == observed_at
    assert manager.get("op")["state"] == "RUNNING"
    assert sweep(manager) == {"degraded": 0, "resumed": 0}
    # At exactly the threshold the observation is still fresh (strict >).
    clock.seconds = 140
    assert sweep(manager) == {"degraded": 0, "resumed": 0}
    clock.seconds = 141
    assert sweep(manager) == {"degraded": 1, "resumed": 0}
    assert manager.get("op")["heartbeat_at"] == observed_at
    expected = ["operation.degraded"]
    if producer != "heartbeat":
        expected.append(f"operation.{producer}")
    expected += ["operation.resumed", "operation.degraded"]
    stream = events(manager)
    assert [e["event_type"] for e in stream] == expected
    assert [e["seq"] for e in stream] == sorted({e["seq"] for e in stream})
    assert manager.events_since("t", stream[1]["seq"]) == stream[2:]
    assert all(e["operation_id"] == "op" for e in stream)


@pytest.mark.parametrize("state", ["RUNNING", "DEGRADED"])
@pytest.mark.parametrize("code", ["PROCESS_IDENTITY_UNVERIFIED", "OUTPUT_OBSERVATION_UNRESOLVED"])
@pytest.mark.parametrize("fresh", [False, True])
def test_unresolved_codes_are_skipped_even_with_fresh_observations(ledger, state, code, fresh):
    conn, manager, clock = ledger
    seed(conn, state, code, clock.now(UTC).isoformat() if fresh else None)
    before = manager.get("op")
    assert sweep(manager) == {"degraded": 0, "resumed": 0}
    assert manager.get("op") == before
    assert events(manager) == []


@pytest.mark.parametrize("state", ["CANCELING", *sorted(OPERATION_TERMINAL)])
def test_sweep_and_ordinary_transition_preserve_ledger_ownership(ledger, state):
    conn, manager, _ = ledger
    seed(conn, state, "CANCEL_STOP_UNCONFIRMED")
    before = manager.get("op")
    assert sweep(manager) == {"degraded": 0, "resumed": 0}
    manager._transition("op", "RUNNING", event="operation.resumed")
    assert manager.get("op") == before
    assert events(manager) == []


def test_non_sweep_transitions_keep_refreshing_heartbeat(ledger):
    conn, manager, clock = ledger
    seed(conn, "QUEUED")
    for seconds, state, event in (
        (100, "ADMITTED", "operation.admitted"),
        (110, "RUNNING", None),
        (120, "DEGRADED", "operation.reattached"),
    ):
        clock.seconds = seconds
        manager._transition("op", state, event=event, payload={"source": "non-sweep"})
        assert manager.get("op")["state"] == state
        assert manager.get("op")["heartbeat_at"] == clock.now(UTC).isoformat()
    assert [e["event_type"] for e in events(manager)] == [
        "operation.admitted",
        "operation.reattached",
    ]
