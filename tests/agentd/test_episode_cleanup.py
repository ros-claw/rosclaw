"""Real private ledger/process tests, no model/ROS/hardware."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest

from benchmarks.harnessbench import episode_cleanup as cleanup
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.operation_manager import OperationManager
from rosclaw.task_kernel.process_identity import ProcessIdentity
from rosclaw.task_kernel.service import TaskKernel


def ledger(home):
    (home / "agentd").mkdir(parents=True)
    conn = sqlite3.connect(home / "agentd/missions.db")
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    conn.commit()
    return conn


@pytest.mark.asyncio
async def test_zero_operations_real_schema_close_and_alias(tmp_path):
    home = tmp_path / "home"
    conn = ledger(home)
    conn.close()
    alias = tmp_path / "socket-safe"
    alias.symlink_to(home, target_is_directory=True)
    result = await cleanup.cleanup_owned_episode(alias, declared_home=home)
    assert result["status"] == "NO_OPERATIONS" and result["operations"] == []


@pytest.mark.asyncio
async def test_protected_or_different_home_rejected_before_connect(tmp_path, monkeypatch):
    def forbidden(_):
        raise AssertionError("connection must not be opened")

    monkeypatch.setattr(cleanup, "_connect", forbidden)
    for home, declared, protected in [
        (tmp_path, tmp_path / "other", ()),
        (tmp_path, tmp_path, (tmp_path,)),
        (Path.home() / ".rosclaw", Path.home() / ".rosclaw", ()),
    ]:
        with pytest.raises(ValueError, match="MAIN_HOME_FORBIDDEN"):
            await cleanup.cleanup_owned_episode(
                home,
                declared_home=declared,
                protected_homes=protected,
            )


@pytest.mark.asyncio
async def test_actual_active_process_cancel_close_and_fresh_durable_read(tmp_path):
    home = tmp_path / "private"
    conn = ledger(home)
    kernel = TaskKernel(conn, home)
    task = kernel.bind_message(
        mission_id="m",
        session_ref="s",
        backend_native_id="n",
        message_id="m1",
        text="private finite fixture",
        cwd=str(home),
    )["task_id"]
    manager = OperationManager(kernel, conn)
    op = await manager.start(
        task_id=task,
        attempt_id="",
        kind="process",
        argv=[sys.executable, "-c", "import time;time.sleep(2)"],
        cwd=str(home),
    )
    conn.commit()
    identity = ProcessIdentity.parse(op["process_identity_json"])
    assert identity and identity.matches()
    try:
        result = await cleanup.cleanup_owned_episode(home, declared_home=home)
        assert result["status"] == "CONFIRMED_CAPTURED_PROCESS_STOP"
        record = result["operations"][0]
        assert record["persisted_state"] == "CANCELLED"
        assert not record["remaining_captured_births"] and not identity.matches()
        await asyncio.wait_for(manager._drivers[op["operation_id"]], timeout=3)
        conn.commit()
        assert cleanup._persisted(home / "agentd/missions.db", op["operation_id"]) == "CANCELLED"
    finally:
        if identity.matches():
            await manager.cancel(op["operation_id"], reason="test_cleanup")
            conn.commit()
        await manager.close()
        conn.close()


@pytest.mark.asyncio
async def test_terminal_finite_process_is_not_recancelled(tmp_path, monkeypatch):
    home = tmp_path / "private"
    conn = ledger(home)
    kernel = TaskKernel(conn, home)
    task = kernel.bind_message(
        mission_id="m",
        session_ref="s",
        backend_native_id="n",
        message_id="m1",
        text="finite fixture",
        cwd=str(home),
    )["task_id"]
    manager = OperationManager(kernel, conn)
    op = await manager.start(
        task_id=task,
        attempt_id="",
        kind="process",
        argv=[sys.executable, "-c", "print('finite')"],
        cwd=str(home),
    )
    await manager.wait(op["operation_id"], timeout=3)
    conn.commit()
    await manager.close()
    conn.close()

    async def no_cancel(*_args, **_kwargs):
        raise AssertionError("already succeeded")

    monkeypatch.setattr(OperationManager, "cancel", no_cancel)
    result = await cleanup.cleanup_owned_episode(home, declared_home=home)
    assert result["status"] == "DURABLE_TERMINAL_PHYSICAL_NOT_PROVEN"
    assert result["operations"][0]["persisted_state"] == "SUCCEEDED"


@pytest.mark.asyncio
async def test_commit_failure_keeps_stop_and_durable_state_separate(tmp_path, monkeypatch):
    home = tmp_path / "private"
    conn = ledger(home)
    conn.execute(
        "INSERT INTO operations (operation_id,task_id,attempt_id,kind,state,started_at) "
        "VALUES ('op','t','','process','SUCCEEDED','2026-10-05')"
    )
    conn.commit()
    conn.close()
    real_connect = cleanup._connect

    class CommitFailure:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def commit(self):
            raise sqlite3.OperationalError("synthetic durability failure")

    monkeypatch.setattr(cleanup, "_connect", lambda path: CommitFailure(real_connect(path)))
    result = await cleanup.cleanup_owned_episode(home, declared_home=home)
    assert result["status"] == "STOP_UNCONFIRMED"
    row = result["operations"][0]
    assert row["error_class"] == "OperationalError" and row["persisted_state"] == "SUCCEEDED"


@pytest.mark.asyncio
async def test_no_ledger_is_not_created(tmp_path):
    result = await cleanup.cleanup_owned_episode(tmp_path, declared_home=tmp_path)
    assert result["status"] == "NO_PRIVATE_LEDGER"
    assert not (tmp_path / "agentd").exists()


@pytest.mark.asyncio
async def test_unknown_process_identity_keeps_pending_cancel_truth(tmp_path):
    home = tmp_path / "private"
    conn = ledger(home)
    conn.execute(
        "INSERT INTO operations (operation_id,task_id,attempt_id,kind,state,started_at) "
        "VALUES ('op','t','','process','RUNNING','2026-10-05')"
    )
    conn.commit()
    conn.close()
    result = await cleanup.cleanup_owned_episode(home, declared_home=home)
    assert result["status"] == "STOP_UNCONFIRMED"
    assert result["operations"][0]["persisted_state"] == "CANCELING"
    assert result["operations"][0]["error_class"] == "OperationCancellationUnresolvedError"


@pytest.mark.asyncio
async def test_nonprocess_active_provider_is_not_cancelled(tmp_path, monkeypatch):
    home = tmp_path / "private"
    conn = ledger(home)
    conn.execute(
        "INSERT INTO operations (operation_id,task_id,attempt_id,kind,state,started_at,provider) "
        "VALUES ('op','t','','action','RUNNING','2026-10-05','ros2_action')"
    )
    conn.commit()
    conn.close()

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("no ROS action cancellation authority")

    monkeypatch.setattr(OperationManager, "cancel", forbidden)
    result = await cleanup.cleanup_owned_episode(home, declared_home=home)
    assert result["status"] == "STOP_UNCONFIRMED"
    assert result["operations"][0]["persisted_state"] == "RUNNING"


@pytest.mark.asyncio
async def test_ledger_symlink_to_protected_home_rejected_before_connect(tmp_path, monkeypatch):
    protected = tmp_path / "protected"
    conn = ledger(protected)
    conn.close()
    original = (protected / "agentd/missions.db").read_bytes()
    private = tmp_path / "private"
    (private / "agentd").mkdir(parents=True)
    (private / "agentd/missions.db").symlink_to(protected / "agentd/missions.db")

    def forbidden(_):
        raise AssertionError("symlink target must not be opened")

    monkeypatch.setattr(cleanup, "_connect", forbidden)
    with pytest.raises(ValueError, match="PRIVATE_LEDGER_PATH_SCOPE_REQUIRED"):
        await cleanup.cleanup_owned_episode(
            private, declared_home=private, protected_homes=(protected,)
        )
    assert (protected / "agentd/missions.db").read_bytes() == original


@pytest.mark.asyncio
async def test_terminal_without_identity_is_ledger_only_not_physical_stop(tmp_path):
    home = tmp_path / "private"
    conn = ledger(home)
    conn.execute(
        "INSERT INTO operations (operation_id,task_id,attempt_id,kind,state,started_at) "
        "VALUES ('op','t','','process','SUCCEEDED','2026-10-05')"
    )
    conn.commit()
    conn.close()
    result = await cleanup.cleanup_owned_episode(home, declared_home=home)
    assert result["status"] == "DURABLE_TERMINAL_PHYSICAL_NOT_PROVEN"
    row = result["operations"][0]
    assert row["durable_ledger_status"] == "TERMINAL"
    assert row["process_birth_observation"] == "NOT_CAPTURED"
    assert row["physical_stop"] == "NOT_PROVEN"


@pytest.mark.asyncio
async def test_live_identity_mismatch_never_stop_proof(tmp_path):
    import subprocess
    from dataclasses import replace

    home = tmp_path / "private"
    conn = ledger(home)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(2)"], start_new_session=True
    )
    identity = ProcessIdentity.capture(child.pid)
    assert identity
    mismatch = replace(identity, start_ticks=identity.start_ticks + 1)
    conn.execute(
        "INSERT INTO operations (operation_id,task_id,attempt_id,kind,state,started_at,"
        "pid,process_identity_json) VALUES ('op','t','','process','RUNNING',"
        "'2026-10-05',?,?)",
        (child.pid, mismatch.to_json()),
    )
    conn.commit()
    conn.close()
    try:
        result = await cleanup.cleanup_owned_episode(home, declared_home=home)
        assert result["status"] == "STOP_UNCONFIRMED"
        row = result["operations"][0]
        assert row["process_birth_observation"] == "LIVE_IDENTITY_MISMATCH"
        assert row["physical_stop"] == "NOT_PROVEN" and identity.matches()
    finally:
        child.terminate()
        child.wait(timeout=3)


@pytest.mark.asyncio
async def test_terminal_ledger_with_proved_live_child_is_unconfirmed_not_killed(tmp_path):
    import subprocess

    home = tmp_path / "private"
    conn = ledger(home)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(2)"], start_new_session=True
    )
    identity = ProcessIdentity.capture(child.pid)
    assert identity
    conn.execute(
        "INSERT INTO operations (operation_id,task_id,attempt_id,kind,state,started_at,"
        "pid,process_identity_json) VALUES ('op','t','','process','SUCCEEDED',"
        "'2026-10-05',?,?)",
        (child.pid, identity.to_json()),
    )
    conn.commit()
    conn.close()
    try:
        result = await cleanup.cleanup_owned_episode(home, declared_home=home)
        row = result["operations"][0]
        assert result["status"] == "STOP_UNCONFIRMED"
        assert row["durable_ledger_status"] == "TERMINAL"
        assert row["remaining_captured_births"] == [child.pid]
        assert row["physical_stop"] == "NOT_PROVEN" and identity.matches()
    finally:
        child.terminate()
        child.wait(timeout=3)


@pytest.mark.asyncio
async def test_commit_failure_after_actual_stop_keeps_both_observations(tmp_path, monkeypatch):
    import subprocess

    home = tmp_path / "private"
    conn = ledger(home)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(2)"], start_new_session=True
    )
    identity = ProcessIdentity.capture(child.pid)
    assert identity
    conn.execute(
        "INSERT INTO operations (operation_id,task_id,attempt_id,kind,state,started_at,"
        "pid,process_identity_json) VALUES ('op','t','','process','RUNNING',"
        "'2026-10-05',?,?)",
        (child.pid, identity.to_json()),
    )
    conn.commit()
    conn.close()
    real_connect = cleanup._connect

    class CommitFailure:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def commit(self):
            raise sqlite3.OperationalError("synthetic commit failure after stop")

    monkeypatch.setattr(cleanup, "_connect", lambda path: CommitFailure(real_connect(path)))
    try:
        result = await cleanup.cleanup_owned_episode(home, declared_home=home)
        row = result["operations"][0]
        assert result["status"] == "STOP_UNCONFIRMED"
        assert row["persisted_state"] == "RUNNING"
        assert row["physical_stop"] == "CAPTURED_MEMBERS_GONE_ONLY"
        assert not identity.matches()
    finally:
        if identity.matches():
            child.terminate()
        child.wait(timeout=3)
