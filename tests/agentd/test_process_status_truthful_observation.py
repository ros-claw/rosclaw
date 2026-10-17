"""Source-only status regressions; not executed by the source author.

The dispatcher seam uses the real read-only OperationManager.get against an
isolated SQLite ledger. Owned-process fixtures execute only when requested by
pytest; cleanup uses only that fixture's Popen handle.
"""

import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher, ToolBridgeError
from rosclaw.task_kernel.operation_manager import OperationManager
from rosclaw.task_kernel.process_identity import ProcessIdentity


@pytest.fixture
def identity():
    return ProcessIdentity(
        pid=12345,
        start_ticks=9876,
        pgid=12345,
        sid=12345,
        uid=os.geteuid(),
        boot_id="fixture-boot",
        cwd="/private/fixture",
        command_sha256=hashlib.sha256(b"private command").hexdigest(),
    )


@pytest.fixture
def ledger(identity):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Only get() is under test: avoid unrelated lifecycle/background machinery.
    conn.executescript("""
        CREATE TABLE operations (
            operation_id TEXT, task_id TEXT, revision INTEGER, state TEXT,
            failure_code TEXT, provider TEXT, pid INTEGER,
            process_identity_json TEXT, heartbeat_at TEXT,
            progress_json TEXT, output_checkpoint_json TEXT, cancel_reason TEXT
        );
        CREATE TABLE task_events (seq INTEGER, event_type TEXT, payload TEXT);
    """)
    conn.execute(
        "INSERT INTO operations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "op",
            "task",
            7,
            "DEGRADED",
            "",
            "process",
            identity.pid,
            identity.to_json(),
            "2000-01-01T00:00:00+00:00",
            '{"fraction":0.25}',
            '{"offset":17,"pending_hex":"","finalized":false}',
            "owner-reason",
        ),
    )
    conn.execute("INSERT INTO task_events VALUES (1, 'operation.output', 'original')")
    conn.commit()
    manager = OperationManager(None, conn)
    dispatcher = object.__new__(PiToolDispatcher)
    dispatcher._service = SimpleNamespace(_operation_manager=manager)
    try:
        yield conn, dispatcher
    finally:
        conn.close()


def status(ledger):
    return asyncio.run(
        ledger[1]._process_status(
            SimpleNamespace(
                arguments={"operation_id": "op"},
                request_id="status-request",
            )
        )
    )


def update(ledger, **values):
    conn = ledger[0]
    for key, value in values.items():
        conn.execute(f"UPDATE operations SET {key} = ?", (value,))
    conn.commit()


def snapshot(ledger):
    conn = ledger[0]
    return (
        list(conn.iterdump()),
        conn.total_changes,
        [tuple(row) for row in conn.execute("SELECT * FROM operations")],
        [tuple(row) for row in conn.execute("SELECT * FROM task_events")],
    )


def capture_as(monkeypatch, value):
    calls = []

    def capture(cls, pid):
        calls.append(pid)
        return value

    monkeypatch.setattr(ProcessIdentity, "capture", classmethod(capture))
    return calls


def assert_agreement(result):
    owner = result.operation
    obs = owner["observation"]
    assert result.ok is True
    assert result.status == owner["state"]
    assert owner["operation_id"] == "op"
    assert owner["task_id"] == "task"
    assert owner["revision"] == 7
    assert owner["state"] in result.summary
    assert obs["process_identity"]["status"] in result.summary
    assert obs["process_identity"]["reason"] in result.summary
    assert obs["observed_at"] in result.summary
    freshness = obs["producer_freshness"]
    assert (freshness["heartbeat_at"] or "UNKNOWN") in result.summary
    assert (
        str(freshness["age_s"] if freshness["age_s"] is not None else "UNKNOWN") in result.summary
    )
    assert all(warning in result.summary for warning in obs["warnings"])
    assert obs["limits"] in result.summary
    if owner["failure_code"]:
        assert owner["failure_code"] in result.summary
    return obs


def test_exact_quiet_identity_does_not_refresh_or_resume(ledger, identity, monkeypatch):
    calls = capture_as(monkeypatch, identity)
    before = snapshot(ledger)
    for _ in range(3):
        result = status(ledger)
        obs = assert_agreement(result)
        assert result.status == "DEGRADED"
        assert obs["process_identity"] == {
            "status": "OBSERVED_IDENTITY_PRESENT",
            "reason": "EXACT_IDENTITY_MATCH",
        }
        assert obs["producer_freshness"]["age_s"] > 0
        assert "Cause is not inferred from heartbeat age" in result.summary
        assert snapshot(ledger) == before
    assert calls == [identity.pid] * 3
    serialized = json.dumps(result.model_dump())
    for private in (identity.cwd, identity.boot_id, identity.command_sha256, "private command"):
        assert private not in serialized


@pytest.mark.parametrize("raw", [None, "", "{", "[]", "null", "{}", '{"pid":true}', "[" * 2000])
def test_invalid_identity_never_captures_or_claims_absence(ledger, raw, monkeypatch):
    calls = capture_as(monkeypatch, None)
    update(ledger, process_identity_json=raw)
    before = snapshot(ledger)
    obs = assert_agreement(status(ledger))
    assert obs["process_identity"] == {
        "status": "UNKNOWN",
        "reason": "STORED_IDENTITY_INVALID_OR_ABSENT",
    }
    assert calls == []
    assert snapshot(ledger) == before


def test_legacy_empty_command_marker_is_unverified(ledger, identity, monkeypatch):
    calls = capture_as(monkeypatch, identity)
    update(
        ledger,
        process_identity_json=replace(
            identity,
            command_sha256=hashlib.sha256(b"").hexdigest(),
        ).to_json(),
    )
    assert assert_agreement(status(ledger))["process_identity"]["status"] == "UNKNOWN"
    assert calls == []


@pytest.mark.parametrize("pid", [None, 1, "untrusted"])
def test_stored_pid_conflict_never_probes_foreign_pid(ledger, pid, monkeypatch):
    calls = capture_as(monkeypatch, None)
    update(ledger, pid=pid)
    obs = assert_agreement(status(ledger))
    assert obs["process_identity"]["reason"] == "STORED_PID_IDENTITY_CONFLICT"
    assert obs["process_identity"]["status"] == "UNKNOWN"
    assert calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_ticks", 9877),
        ("boot_id", "foreign-boot"),
        ("uid", 1001),
        ("pid", 12346),
        ("pgid", 12346),
        ("sid", 12346),
        ("cwd", "/changed"),
        ("command_sha256", "changed-command"),
    ],
)
def test_reused_or_changed_identity_not_adopted(ledger, identity, monkeypatch, field, value):
    if field == "uid":
        value = max(os.getuid(), os.geteuid()) + 1
    capture_as(monkeypatch, replace(identity, **{field: value}))
    before = snapshot(ledger)
    obs = assert_agreement(status(ledger))
    assert obs["process_identity"] == {
        "status": "UNKNOWN",
        "reason": "CURRENT_IDENTITY_MISMATCH",
    }
    assert snapshot(ledger) == before


@pytest.mark.parametrize("failure", [None, PermissionError, ProcessLookupError, ValueError])
def test_unreadable_or_racing_capture_is_unknown(ledger, monkeypatch, failure):
    def capture(cls, pid):
        if failure:
            raise failure("private diagnostic must not leak")
        return None

    monkeypatch.setattr(ProcessIdentity, "capture", classmethod(capture))
    result = status(ledger)
    assert assert_agreement(result)["process_identity"] == {
        "status": "UNKNOWN",
        "reason": "CURRENT_IDENTITY_UNAVAILABLE",
    }
    assert "private diagnostic" not in result.summary


@pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "CANCELLED", "LOST", "CANCELING"])
@pytest.mark.parametrize("present", [True, False])
def test_terminal_and_canceling_ledger_not_reclassified(
    ledger, identity, monkeypatch, state, present
):
    capture_as(monkeypatch, identity if present else None)
    update(ledger, state=state, failure_code="CANCEL_STOP_UNCONFIRMED")
    before = snapshot(ledger)
    result = status(ledger)
    assert_agreement(result)
    assert result.status == state
    assert "Requested stop remains unconfirmed" in result.summary
    assert snapshot(ledger) == before


@pytest.mark.parametrize(
    "code,warning",
    [
        ("PROCESS_IDENTITY_UNVERIFIED", "Stored identity verification remains unresolved"),
        ("OUTPUT_OBSERVATION_UNRESOLVED", "Output observation remains unresolved"),
        ("CANCEL_STOP_UNCONFIRMED", "Requested stop remains unconfirmed"),
        ("OTHER_FAILURE", "OTHER_FAILURE"),
    ],
)
def test_failure_code_survives_exact_match(ledger, identity, monkeypatch, code, warning):
    capture_as(monkeypatch, identity)
    update(ledger, failure_code=code)
    result = status(ledger)
    assert_agreement(result)
    assert result.operation["failure_code"] == code
    assert warning in result.summary


@pytest.mark.parametrize(
    "heartbeat", [None, "", "invalid", "2026-01-01T00:00:00", "9999-01-01T00:00:00+00:00"]
)
def test_unknown_or_future_freshness_is_not_invented(ledger, monkeypatch, heartbeat):
    capture_as(monkeypatch, None)
    update(ledger, heartbeat_at=heartbeat)
    obs = assert_agreement(status(ledger))
    assert obs["producer_freshness"]["age_s"] is None
    if heartbeat != "9999-01-01T00:00:00+00:00":
        assert obs["producer_freshness"]["heartbeat_at"] is None


def test_nonprocess_provider_does_not_probe(ledger, identity, monkeypatch):
    calls = capture_as(monkeypatch, identity)
    update(ledger, provider="action")
    obs = assert_agreement(status(ledger))
    assert obs["process_identity"]["reason"] == "NOT_PROCESS_PROVIDER"
    assert calls == []


def test_unknown_operation_keeps_not_found_contract(ledger):
    ledger[0].execute("DELETE FROM operations")
    with pytest.raises(ToolBridgeError) as error:
        status(ledger)
    assert error.value.code == "NOT_FOUND"


@pytest.fixture
def owned_quiet_process(tmp_path):
    # A one-byte handshake ensures exec completed before capture; after that
    # the owned child is quiet and waits for EOF, without sleeps or polling.
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; sys.stdout.write('R'); sys.stdout.flush(); sys.stdin.buffer.read()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        start_new_session=True,
    )
    try:
        assert child.stdout.read(1) == b"R"
        assert child.poll() is None
        identity = ProcessIdentity.capture(child.pid)
        assert identity is not None
        assert ProcessIdentity.parse(identity.to_json()) == identity
        yield child, identity
    finally:
        child.stdin.close()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()  # only this fixture's owned Popen handle
            child.wait(timeout=5)
        child.stdout.close()
        child.stderr.close()


def test_owned_quiet_process_with_stale_heartbeat(ledger, owned_quiet_process):
    child, identity = owned_quiet_process
    update(ledger, pid=identity.pid, process_identity_json=identity.to_json())
    before = snapshot(ledger)
    for _ in range(3):
        result = status(ledger)
        obs = assert_agreement(result)
        assert obs["process_identity"]["status"] == "OBSERVED_IDENTITY_PRESENT"
        assert result.status == "DEGRADED"
        assert child.poll() is None
        assert snapshot(ledger) == before
    # EOF is directed solely to the owned child; no PID/group scanning/signals.
    child.stdin.close()
    assert child.wait(timeout=5) == 0
    obs = assert_agreement(status(ledger))
    assert obs["process_identity"]["status"] == "UNKNOWN"
    assert obs["process_identity"]["reason"] == "CURRENT_IDENTITY_UNAVAILABLE"
    assert snapshot(ledger) == before


@pytest.mark.parametrize(
    "code,warning",
    [
        ("PROCESS_IDENTITY_UNVERIFIED", "Stored identity verification remains unresolved"),
        ("OUTPUT_OBSERVATION_UNRESOLVED", "Output observation remains unresolved"),
    ],
)
@pytest.mark.parametrize("present", [True, False])
def test_fresh_heartbeat_does_not_explain_or_clear_degraded(
    ledger,
    identity,
    monkeypatch,
    code,
    warning,
    present,
):
    fixed = datetime.now(UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz)

    monkeypatch.setattr(sys.modules[PiToolDispatcher.__module__], "datetime", Clock)
    capture_as(monkeypatch, identity if present else None)
    update(ledger, heartbeat_at=fixed.isoformat(), failure_code=code)
    before = snapshot(ledger)
    for _ in range(2):
        result = status(ledger)
        obs = assert_agreement(result)
        assert obs["producer_freshness"]["age_s"] == 0
        assert result.status == "DEGRADED"
        assert result.operation["failure_code"] == code
        assert warning in result.summary
        assert "DEGRADED means stale" not in result.summary
        assert "Cause is not inferred from heartbeat age" in result.summary
        assert obs["process_identity"]["status"] == (
            "OBSERVED_IDENTITY_PRESENT" if present else "UNKNOWN"
        )
        assert snapshot(ledger) == before


@pytest.mark.parametrize("code", ["", "OTHER_FAILURE"])
def test_degraded_unknown_cause_is_not_resolved_by_timestamp(
    ledger,
    identity,
    monkeypatch,
    code,
):
    capture_as(monkeypatch, identity)
    update(ledger, failure_code=code)
    before = snapshot(ledger)
    result = status(ledger)
    obs = assert_agreement(result)
    assert obs["producer_freshness"]["age_s"] > 0
    assert result.operation["failure_code"] == code
    assert "DEGRADED is a ledger state, not a unique cause" in result.summary
    assert "DEGRADED means stale" not in result.summary
    assert snapshot(ledger) == before


def test_matching_foreign_stored_and_current_identity_is_not_local_evidence(
    ledger,
    identity,
    monkeypatch,
):
    foreign = replace(identity, uid=max(os.getuid(), os.geteuid()) + 1)
    calls = capture_as(monkeypatch, foreign)
    update(ledger, process_identity_json=foreign.to_json())
    before = snapshot(ledger)
    # The capture double would return exactly the stored foreign identity.
    # Reject before probing: parse/equality alone is not local ownership.
    for _ in range(2):
        result = status(ledger)
        obs = assert_agreement(result)
        assert obs["process_identity"] == {
            "status": "UNKNOWN",
            "reason": "STORED_IDENTITY_OUTSIDE_LOCAL_UID_BOUNDARY",
        }
        assert "OBSERVED_IDENTITY_PRESENT" not in result.summary
        assert calls == []
        assert snapshot(ledger) == before


@pytest.mark.parametrize("boundary", ["getuid", "geteuid"])
def test_exact_identity_at_actual_local_uid_boundary(ledger, identity, monkeypatch, boundary):
    local = replace(identity, uid=getattr(os, boundary)())
    capture_as(monkeypatch, local)
    update(ledger, process_identity_json=local.to_json())
    before = snapshot(ledger)
    obs = assert_agreement(status(ledger))
    assert obs["process_identity"] == {
        "status": "OBSERVED_IDENTITY_PRESENT",
        "reason": "EXACT_IDENTITY_MATCH",
    }
    assert snapshot(ledger) == before
