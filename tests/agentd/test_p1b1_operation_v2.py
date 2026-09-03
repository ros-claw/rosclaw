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
import inspect
import json
import os
import sqlite3
import time
from pathlib import Path

from rosclaw.task_kernel.operation_manager import (
    OPERATION_TERMINAL,
    OperationManager,
)


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
        "SELECT event_type, payload_json FROM task_events WHERE task_id = ? "
        "ORDER BY seq",
        (task_id,),
    ).fetchall()
    return [
        {"type": r["event_type"], "payload": json.loads(r["payload_json"])}
        for r in rows
    ]


class TestStateMachine:
    def test_start_emits_queued_admitted_running(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        mgr = OperationManager(None, conn)

        async def run():
            op = await mgr.start(
                task_id="task_1", attempt_id="", kind="process",
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
                task_id="task_1", attempt_id="", kind="process",
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
                task_id="task_1", attempt_id="", kind="process",
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
                task_id="task_1", attempt_id="", kind="process",
                argv=["sh", "-c", "sleep 30"],
            )
            return op["operation_id"]

        op_id = asyncio.run(run())
        # 直接置于 CANCELING（action 取消窗——listener 回调与
        # _write_terminal 同入口）。
        conn.execute(
            "UPDATE operations SET state = 'CANCELING', cancel_reason = 'r' "
            "WHERE operation_id = ?",
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
                task_id="task_1", attempt_id="", kind="process",
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
                task_id="task_1", attempt_id="", kind="process",
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
    def _start_op(self, conn: sqlite3.Connection, argv: list[str]) -> dict:
        mgr = OperationManager(None, conn)
        return asyncio.run(
            mgr.start(task_id="task_1", attempt_id="", kind="process", argv=argv)
        )

    def test_reattach_when_pid_alive(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        op = self._start_op(conn, ["sh", "-c", "sleep 30"])
        # 模拟重启：新 manager（内存驱动全丢），sweep 恢复。
        mgr2 = OperationManager(None, conn)
        report = asyncio.run(mgr2.recover_on_boot())
        row = mgr2.get(op["operation_id"])
        assert row["state"] != "LOST", "活 pid 被误判 LOST"
        assert report["reattached"] >= 1
        os.kill(int(row["pid"]), 15)  # 清理

    def test_dead_pid_with_exitcode_applies_terminal(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        op = self._start_op(conn, ["sh", "-c", "exit 3"])
        time.sleep(0.5)  # 进程已退出（exitcode 文件应由 wrapper 写）
        mgr2 = OperationManager(None, conn)
        asyncio.run(mgr2.recover_on_boot())
        row = mgr2.get(op["operation_id"])
        assert row["state"] == "FAILED"
        assert "exit_3" in (row["failure_code"] or "")

    def test_dead_pid_without_exitcode_is_honest_lost(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        op = self._start_op(conn, ["sh", "-c", "sleep 30"])
        op_id = op["operation_id"]
        pid = int(op["pid"])
        os.kill(pid, 9)
        time.sleep(0.3)
        # 删掉 exitcode（模拟 agentd 被杀时进程也被杀、无退出记录）。
        for candidate in tmp_path.rglob(f"*{op_id}*exit*"):
            candidate.unlink()
        mgr2 = OperationManager(None, conn)
        report = asyncio.run(mgr2.recover_on_boot())
        row = mgr2.get(op_id)
        assert row["state"] == "LOST"
        assert report["lost"] >= 1
        types = [e["type"] for e in _events(conn)]
        assert "operation.lost" in types
        assert "LOST" in OPERATION_TERMINAL


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
            / "packages" / "rosclaw-agent" / "src" / "tools" / "embodiment-exec.ts"
        )
        assert "rosclaw_wait_operation" not in ts.read_text(encoding="utf-8")
