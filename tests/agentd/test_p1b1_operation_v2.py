"""P1-B1 红测试（0824 总纲 §12/P1-B）：OperationManager V2。

复现的真实事故/缺口：
1. 状态机只有 RUNNING/SUCCEEDED/FAILED/CANCELLED——无 QUEUED/
   ADMITTED/CANCELING/DEGRADED/LOST，ROS 2 Action 的异步取消握手
   与失联/重启语义无法表达；
2. heartbeat 写了没人读——静默进程与卡死进程不可区分（无
   DEGRADED 标记；也绝不能退化成 wall-clock kill）；
3. agentd 重启后 RUNNING 行变僵尸（无 reattach、无诚实 LOST）；
4. rosclaw_wait_operation 轮询工具仍在 wire 上（已退模型面但没删）。

断言 V2：
- start → QUEUED→ADMITTED→RUNNING 事件链完整；
- cancel → CANCELING（带 reason）→ CANCELLED；迟到完成不覆盖；
- liveness sweep：heartbeat 过期 → DEGRADED（不杀进程）；恢复 →
  RUNNING；长任务（2h 语义）永不因时长被杀；
- 重启 sweep：pid 活 → reattach（非 LOST）；pid 死+exitcode →
  应用终态；pid 死+无 exitcode → 诚实 LOST；
- _wait_operation 从 dispatch/effect_resolver/TS 面删除。
"""

from __future__ import annotations

import asyncio
import fcntl
import inspect
import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from rosclaw.task_kernel.operation_manager import (
    OPERATION_TERMINAL,
    OperationManager,
)
from rosclaw.task_kernel.process_identity import (
    ProcessIdentity,
    session_members,
    signal_owned_members,
)


async def _evidence_until(predicate, description, timeout=5):
    """Bounded prerequisite acquisition, never retry recovery/asserted outcomes."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"prerequisite unavailable: {description}")
        await asyncio.sleep(0.01)


def _spool_closed(mgr, op_id):
    stream, _ = mgr._open_spool(mgr.get(op_id))
    with stream:
        return mgr._writers_closed(stream.fileno())


async def _dead_spool_prerequisites(mgr, op, proc):
    await asyncio.wait_for(proc.wait(), 5)
    assert not mgr._pid_alive(proc.pid)
    # wait() reaps only the shell. Killed descendants can still own the
    # inherited locked open-file description until their exit teardown runs.
    await _evidence_until(
        lambda: _spool_closed(mgr, op["operation_id"]), "all spool writers closed"
    )
    assert _spool_closed(mgr, op["operation_id"])


def _conn(tmp_path: Path) -> sqlite3.Connection:
    from rosclaw.storage.migrations import MigrationRunner

    conn = sqlite3.connect(tmp_path / "missions.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return conn


def _task(conn: sqlite3.Connection, task_id: str = "task_1") -> None:
    now = "2026-08-25T00:00:00+00:00"
    conn.execute(
        "INSERT INTO tasks (task_id, mission_id, root_goal, mode, body_id, "
        "state, active_revision, workspace_path, created_at, updated_at) "
        "VALUES (?, 'm1', 'goal', 'SIMULATION', '', 'ACTIVE', 1, '', ?, ?)",
        (task_id, now, now),
    )
    conn.commit()


def _events(conn: sqlite3.Connection, task_id: str = "task_1") -> list[dict]:
    rows = conn.execute(
        "SELECT event_type, payload_json FROM task_events WHERE task_id = ? ORDER BY seq",
        (task_id,),
    ).fetchall()
    return [{"type": r["event_type"], "payload": json.loads(r["payload_json"])} for r in rows]


class TestStateMachine:
    def test_start_emits_queued_admitted_running(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        mgr = OperationManager(None, conn)

        async def run():
            op = await mgr.start(
                task_id="task_1",
                attempt_id="",
                kind="process",
                argv=["sh", "-c", "sleep 30"],
            )
            await mgr.cancel(op["operation_id"], reason="test-done")
            return op

        op = asyncio.run(run())
        types = [e["type"] for e in _events(conn)]
        assert "operation.queued" in types
        assert "operation.admitted" in types
        assert op["state"] in ("ADMITTED", "RUNNING")
        assert op["goal_id"], "缺 goal_id"
        assert op["provider"] == "process"
        assert op["pid"] and int(op["pid"]) > 0

    def test_cancel_goes_through_canceling_with_reason(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        mgr = OperationManager(None, conn)

        async def run():
            op = await mgr.start(
                task_id="task_1",
                attempt_id="",
                kind="process",
                argv=["sh", "-c", "sleep 30"],
            )
            await mgr.cancel(op["operation_id"], reason="user-request")
            return op["operation_id"]

        op_id = asyncio.run(run())
        row = mgr.get(op_id)
        assert row["state"] == "CANCELLED"
        assert row["cancel_reason"] == "user-request"
        types = [e["type"] for e in _events(conn)]
        assert "operation.canceling" in types
        assert "operation.cancelled" in types

    def test_late_completion_never_overwrites_cancelled(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        mgr = OperationManager(None, conn)

        async def run():
            op = await mgr.start(
                task_id="task_1",
                attempt_id="",
                kind="process",
                argv=["sh", "-c", "sleep 0.2"],
            )
            await mgr.cancel(op["operation_id"], reason="race")
            await asyncio.sleep(0.5)  # 进程自然退出窗口
            return op["operation_id"]

        op_id = asyncio.run(run())
        assert mgr.get(op_id)["state"] == "CANCELLED"

    def test_late_result_never_overwrites_canceling(self, tmp_path: Path) -> None:
        """CI 实证（p1b3 flake 根治）：CANCELING 窗内到达的迟到
        action_result(SUCCEEDED) 曾覆盖 CANCELING（_write_terminal 只查
        OPERATION_TERMINAL）——goal 永远停在 SUCCEEDED，grace 的
        CANCELLED 被终态拒绝。取消流程持有账本：CANCELING 下只接受
        CANCELLED（服务端 CANCELED 确认），SUCCEEDED/FAILED 拒。"""
        conn = _conn(tmp_path)
        _task(conn)
        mgr = OperationManager(None, conn)

        async def run():
            op = await mgr.start(
                task_id="task_1",
                attempt_id="",
                kind="process",
                argv=["sh", "-c", "sleep 30"],
            )
            return op["operation_id"]

        op_id = asyncio.run(run())
        # 直接置于 CANCELING（action 取消窗——listener 回调与
        # _write_terminal 同入口）。
        conn.execute(
            "UPDATE operations SET state = 'CANCELING', cancel_reason = 'r' WHERE operation_id = ?",
            (op_id,),
        )
        # 取消窗内的迟到完成：SUCCEEDED/FAILED 拒（取消流程持有账本）。
        mgr._write_terminal(op_id, "SUCCEEDED")
        assert mgr.get(op_id)["state"] == "CANCELING", (
            "迟到 SUCCEEDED 覆盖了 CANCELING——取消握手被破坏"
        )
        mgr._write_terminal(op_id, "FAILED")
        assert mgr.get(op_id)["state"] == "CANCELING"
        # 服务端 CANCELED 确认正常落地（握手完成）。
        mgr._write_terminal(op_id, "CANCELLED")
        assert mgr.get(op_id)["state"] == "CANCELLED"


class TestLiveness:
    def test_stale_heartbeat_marks_degraded_never_kills(self, tmp_path: Path) -> None:
        """静默但活着的进程 → DEGRADED（不杀）；恢复输出 → RUNNING。"""
        conn = _conn(tmp_path)
        _task(conn)
        mgr = OperationManager(None, conn)

        async def run():
            op = await mgr.start(
                task_id="task_1",
                attempt_id="",
                kind="process",
                argv=["sh", "-c", "sleep 1.2; echo alive-again; sleep 30"],
            )
            op_id = op["operation_id"]
            # 人为把 heartbeat 拨到过去（等价于长时间静默）。
            old = "2020-01-01T00:00:00+00:00"
            conn.execute(
                "UPDATE operations SET heartbeat_at = ? WHERE operation_id = ?",
                (old, op_id),
            )
            conn.commit()
            await mgr.sweep_liveness(stale_after_s=1.0)
            row = mgr.get(op_id)
            assert row["state"] == "DEGRADED"
            pid = int(row["pid"])
            os.kill(pid, 0)  # 进程必须还活着——liveness 永不 kill
            # 进程恢复输出 → 下一 sweep 回 RUNNING。
            await asyncio.sleep(1.5)
            await mgr.sweep_liveness(stale_after_s=1.0)
            assert mgr.get(op_id)["state"] == "RUNNING"
            types = [e["type"] for e in _events(conn)]
            assert "operation.degraded" in types
            await mgr.cancel(op_id, reason="done")

        asyncio.run(run())

    def test_long_task_never_killed_by_duration(self, tmp_path: Path) -> None:
        """2h 语义：持续 heartbeat 的 operation 时长再久也不被 kill。"""
        conn = _conn(tmp_path)
        _task(conn)
        mgr = OperationManager(None, conn)

        async def run():
            op = await mgr.start(
                task_id="task_1",
                attempt_id="",
                kind="process",
                argv=["sh", "-c", "for i in 1 2 3 4 5 6; do echo hb; sleep 0.3; done; sleep 30"],
            )
            op_id = op["operation_id"]
            for _ in range(6):
                await asyncio.sleep(0.4)
                await mgr.sweep_liveness(stale_after_s=1.0)
                row = mgr.get(op_id)
                assert row["state"] in ("RUNNING", "ADMITTED", "DEGRADED")
                os.kill(int(row["pid"]), 0)  # 仍在运行——无 wall-clock kill
            await mgr.cancel(op_id, reason="done")

        asyncio.run(run())


class TestRestartRecovery:
    @staticmethod
    async def _start_op(conn, argv):
        mgr = OperationManager(None, conn)
        op = await mgr.start(task_id="task_1", attempt_id="", kind="process", argv=argv)
        proc = mgr._procs[op["operation_id"]]
        driver = mgr._drivers[op["operation_id"]]
        await mgr.close()  # preserve-only handoff, not a producer stop
        assert driver.done()
        return mgr, op, proc

    @staticmethod
    async def _cleanup(proc, mgr2):
        try:
            if proc.returncode is None:
                row = mgr2.get(proc.operation_id)
                identity = ProcessIdentity.parse(row["process_identity_json"])
                assert identity is not None and identity.matches()
                members = session_members(identity)
                assert signal_owned_members(members, signal.SIGKILL)
            await asyncio.wait_for(proc.wait(), 5)
        finally:
            await mgr2.close()

    def test_reattach_when_pid_alive(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)

        async def run():
            _, op, proc = await self._start_op(conn, ["sh", "-c", "sleep 30"])
            mgr2 = OperationManager(None, conn)
            try:
                identity = ProcessIdentity.parse(op["process_identity_json"])
                assert identity is not None and identity.matches()
                assert mgr2._pid_alive(proc.pid)
                assert not _spool_closed(mgr2, op["operation_id"])
                report = await mgr2.recover_on_boot()
                assert mgr2.get(op["operation_id"])["state"] != "LOST"
                assert report["reattached"] >= 1
                await mgr2.cancel(op["operation_id"])
            finally:
                await self._cleanup(proc, mgr2)

        asyncio.run(run())

    def test_dead_pid_with_exitcode_applies_terminal(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)

        async def run():
            _, op, proc = await self._start_op(conn, ["sh", "-c", "exit 3"])
            mgr2 = OperationManager(None, conn)
            try:
                await _dead_spool_prerequisites(mgr2, op, proc)
                assert mgr2._read_exitcode(op["exitcode_path"]) == 3
                await mgr2.recover_on_boot()
                row = mgr2.get(op["operation_id"])
                assert row["state"] == "FAILED"
                assert "exit_3" in (row["failure_code"] or "")
            finally:
                await self._cleanup(proc, mgr2)

        asyncio.run(run())

    def test_dead_pid_without_exitcode_is_honest_lost(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)

        async def run():
            ready = tmp_path / "producer-ready"
            _, op, proc = await self._start_op(
                conn,
                [
                    sys.executable,
                    "-c",
                    "import pathlib,signal,sys; pathlib.Path(sys.argv[1]).touch(); signal.pause()",
                    str(ready),
                ],
            )
            mgr2 = OperationManager(None, conn)
            try:
                await _evidence_until(ready.exists, "producer entered no-fork wait")
                identity = ProcessIdentity.parse(op["process_identity_json"])
                assert identity is not None and identity.matches()
                members = session_members(identity)
                assert len(members) == 2  # wrapper plus the coordinated producer
                assert signal_owned_members(members, signal.SIGKILL)
                await _dead_spool_prerequisites(mgr2, op, proc)
                # No deletion of exit evidence: the killed wrapper never wrote it.
                assert not Path(op["exitcode_path"]).exists()
                assert mgr2._read_exitcode(op["exitcode_path"]) is None
                report = await mgr2.recover_on_boot()
                assert mgr2.get(op["operation_id"])["state"] == "LOST"
                assert report["lost"] >= 1
                assert "operation.lost" in [e["type"] for e in _events(conn)]
                assert "LOST" in OPERATION_TERMINAL
            finally:
                await self._cleanup(proc, mgr2)

        asyncio.run(run())


class TestRestartRecoveryEvidence:
    @pytest.mark.parametrize("exit_code", [None, 3])
    def test_dead_wrapper_does_not_close_another_writer(self, tmp_path, exit_code):
        """A controlled independent writer forbids dead-PID/exit-file shortcuts."""
        conn = _conn(tmp_path)
        _task(conn)

        async def run():
            producer_ready = tmp_path / "producer-ready"
            release = tmp_path / "producer-release"
            argv = [
                sys.executable,
                "-c",
                "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch()\n"
                "while not pathlib.Path(sys.argv[2]).exists(): time.sleep(0.01)\n"
                "raise SystemExit(3)",
                str(producer_ready),
                str(release),
            ]
            _, op, proc = await TestRestartRecovery._start_op(conn, argv)
            mgr = OperationManager(None, conn)
            writer = None
            try:
                await _evidence_until(producer_ready.exists, "producer ready")
                assert mgr.get(op["operation_id"])["state"] not in OPERATION_TERMINAL
                if exit_code is None:
                    identity = ProcessIdentity.parse(op["process_identity_json"])
                    assert identity is not None and identity.matches()
                    members = session_members(identity)
                    assert len(members) == 2
                    assert signal_owned_members(members, signal.SIGKILL)
                else:
                    release.touch()
                await _dead_spool_prerequisites(mgr, op, proc)
                assert mgr._read_exitcode(op["exitcode_path"]) == exit_code
                # A separately owned child holds the same kind of flock as an
                # inherited descendant. No session/stop authority is conferred.
                ready = tmp_path / "writer-ready"
                fd = os.open(op["output_path"], os.O_WRONLY | os.O_APPEND)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    writer = subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            "import os,pathlib,sys; os.write(1,b'head\\xe2'); "
                            "pathlib.Path(sys.argv[1]).touch(); "
                            "assert sys.stdin.buffer.read(1)==b'x'; "
                            "os.write(1,b'\\x82\\xac-tail')",
                            str(ready),
                        ],
                        stdin=subprocess.PIPE,
                        stdout=fd,
                        stderr=fd,
                        start_new_session=True,
                    )
                finally:
                    os.close(fd)
                await _evidence_until(ready.exists, "writer prefix committed")
                assert writer.poll() is None
                assert not mgr._pid_alive(proc.pid)
                assert not _spool_closed(mgr, op["operation_id"])
                report = await mgr.recover_on_boot()
                assert report["reattached"] == 1
                assert report["lost"] == report["terminated"] == 0
                assert mgr.get(op["operation_id"])["state"] not in OPERATION_TERMINAL

                def prefix_checkpointed():
                    cp = json.loads(mgr.get(op["operation_id"])["output_checkpoint_json"])
                    return cp["offset"] == 5 and cp["pending_hex"] == "e2"

                await _evidence_until(prefix_checkpointed, "partial UTF-8 checkpoint")
                assert not json.loads(mgr.get(op["operation_id"])["output_checkpoint_json"])[
                    "finalized"
                ]
                # Handoff neither stops this writer nor flushes incomplete UTF-8.
                await mgr.close()
                assert writer.poll() is None
                assert not _spool_closed(mgr, op["operation_id"])
                writer.stdin.write(b"x")
                writer.stdin.flush()
                writer.stdin.close()
                await _evidence_until(lambda: writer.poll() is not None, "writer exited")
                assert writer.wait(timeout=1) == 0
                assert _spool_closed(mgr, op["operation_id"])
                mgr = OperationManager(None, conn)
                report = await mgr.recover_on_boot()
                expected = "LOST" if exit_code is None else "FAILED"
                row = mgr.get(op["operation_id"])
                assert row["state"] == expected
                assert report["lost" if exit_code is None else "terminated"] == 1
                if exit_code is not None:
                    assert row["failure_code"] == "exit_3"
                checkpoint = json.loads(row["output_checkpoint_json"])
                assert checkpoint["finalized"] and checkpoint["pending_hex"] == ""
                assert checkpoint["offset"] == Path(op["output_path"]).stat().st_size
                output = "".join(
                    e["payload"]["text"] for e in _events(conn) if e["type"] == "operation.output"
                )
                assert output == "head€-tail"
                terminal_event = "operation.lost" if exit_code is None else "operation.failed"
                assert [e["type"] for e in _events(conn)].count(terminal_event) == 1
                before = _events(conn)
                await mgr.recover_on_boot()
                mgr._write_terminal(op["operation_id"], "SUCCEEDED")
                assert mgr.get(op["operation_id"])["state"] == expected
                assert _events(conn) == before
            finally:
                if writer is not None:
                    if writer.poll() is None:
                        writer.kill()  # exact retained Popen child, never a PGID
                    writer.wait(timeout=5)
                    if writer.stdin is not None:
                        writer.stdin.close()
                await TestRestartRecovery._cleanup(proc, mgr)
                conn.close()

        asyncio.run(run())

    @pytest.mark.parametrize("bad_identity", ["", "reused_birth"])
    def test_live_unverified_identity_is_not_adopted(self, tmp_path, bad_identity):
        conn = _conn(tmp_path)
        _task(conn)

        async def run():
            _, op, proc = await TestRestartRecovery._start_op(conn, ["sh", "-c", "sleep 30"])
            mgr = OperationManager(None, conn)
            original = op["process_identity_json"]
            try:
                identity = ProcessIdentity.parse(original)
                assert identity is not None and identity.matches()
                raw = bad_identity
                if bad_identity == "reused_birth":
                    data = json.loads(original)
                    data["start_ticks"] += 1
                    raw = json.dumps(data)
                conn.execute(
                    "UPDATE operations SET process_identity_json = ? WHERE operation_id = ?",
                    (raw, op["operation_id"]),
                )
                report = await mgr.recover_on_boot()
                assert report["unresolved"] == 1
                assert report["reattached"] == report["lost"] == 0
                row = mgr.get(op["operation_id"])
                assert row["state"] == "DEGRADED"
                assert row["failure_code"] == "PROCESS_IDENTITY_UNVERIFIED"
                assert row["process_identity_json"] == raw
                assert op["operation_id"] not in mgr._drivers
                await mgr.sweep_liveness(stale_after_s=999999999)
                assert mgr.get(op["operation_id"])["state"] == "DEGRADED"
                assert identity.matches()  # live producer was not stopped
                assert "operation.recovery_unresolved" in [e["type"] for e in _events(conn)]
            finally:
                # Restore only the test-corrupted row for owned cleanup; never
                # refresh identity from the live numeric PID.
                conn.execute(
                    "UPDATE operations SET process_identity_json = ? WHERE operation_id = ?",
                    (original, op["operation_id"]),
                )
                await TestRestartRecovery._cleanup(proc, mgr)
                conn.close()

        asyncio.run(run())


class TestWaitOperationRemoved:
    def test_dispatch_has_no_wait_operation(self) -> None:
        from rosclaw.agentd.pi_bridge import tool_dispatch

        source = inspect.getsource(tool_dispatch)
        assert "_wait_operation" not in source
        assert "rosclaw_wait_operation" not in source

    def test_effect_resolver_has_no_wait_operation(self) -> None:
        from rosclaw.agentd.tooling import effect_resolver

        assert "rosclaw_wait_operation" not in inspect.getsource(effect_resolver)

    def test_ts_surface_has_no_wait_operation(self) -> None:
        ts = (
            Path(__file__).resolve().parents[2]
            / "packages"
            / "rosclaw-agent"
            / "src"
            / "tools"
            / "embodiment-exec.ts"
        )
        assert "rosclaw_wait_operation" not in ts.read_text(encoding="utf-8")
