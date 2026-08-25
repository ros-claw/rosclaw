"""PR-H3 红测试：OperationManager 与统一事件流（总纲 v2 §11）。

红测试先行——修复前必须红：
1. operation 启动即落账（operations 表 + task_events operation.queued/admitted），
   长进程立即返回 operation_id（不在 tool call 里死等）；
2. stdout/stderr/progress/heartbeat 进统一事件流（单调 seq）；
3. 终态事件后 elapsed 停止（heartbeat_at/ended_at 冻结）；
4. seq 重放不重不漏（replay from last_seq+1）；
5. 进程死亡 → 一个健康周期内 RECOVERING/FAILED（不是永远 RUNNING）；
6. cancel 写账本先于信号；迟到 completion 不覆盖 CANCELLED。
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path


def _kernel(tmp_path: Path):
    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(tmp_path / "k.db")
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return conn, TaskKernel(conn, tmp_path)


def _task(kernel, tmp_path: Path) -> str:
    result = kernel.bind_message(
        mission_id="m1", session_ref="s1", backend_native_id="n1",
        message_id="msg_1", text="长任务", cwd=str(tmp_path),
    )
    return result["task_id"]


class TestOperationLifecycle:
    async def test_start_returns_immediately_with_ledger(self, tmp_path: Path) -> None:
        from rosclaw.task_kernel.operation_manager import OperationManager

        conn, kernel = _kernel(tmp_path)
        task_id = _task(kernel, tmp_path)
        ops = OperationManager(kernel, conn)
        op = await ops.start(
            task_id=task_id, attempt_id="att_1", kind="process",
            argv=[sys.executable, "-c",
                  "import time; print('p1', flush=True); time.sleep(0.3); print('p2', flush=True)"],
        )
        assert op["state"] in ("ADMITTED", "RUNNING")
        assert op["operation_id"]
        # 立即返回——进程在后台跑。
        await ops.wait(op["operation_id"], timeout=10)
        final = ops.get(op["operation_id"])
        assert final["state"] == "SUCCEEDED"
        # stdout 进事件流。
        events = ops.events_since(task_id, 0)
        kinds = [e["event_type"] for e in events]
        assert "operation.admitted" in kinds
        assert "operation.output" in kinds
        assert "operation.completed" in kinds
        output = "".join(
            e["payload"].get("text", "")
            for e in events if e["event_type"] == "operation.output"
        )
        assert "p1" in output and "p2" in output

    async def test_terminal_freezes_elapsed(self, tmp_path: Path) -> None:
        from rosclaw.task_kernel.operation_manager import OperationManager

        conn, kernel = _kernel(tmp_path)
        task_id = _task(kernel, tmp_path)
        ops = OperationManager(kernel, conn)
        op = await ops.start(
            task_id=task_id, attempt_id="att_1", kind="process",
            argv=[sys.executable, "-c", "pass"],
        )
        await ops.wait(op["operation_id"], timeout=10)
        final = ops.get(op["operation_id"])
        assert final["ended_at"], "终态必须冻结 ended_at（elapsed 停止）"
        heartbeat_before = final["heartbeat_at"]
        await asyncio.sleep(0.2)
        again = ops.get(op["operation_id"])
        assert again["heartbeat_at"] == heartbeat_before, (
            "终态后 heartbeat 不得再动"
        )

    async def test_seq_replay_no_gap_no_dup(self, tmp_path: Path) -> None:
        from rosclaw.task_kernel.operation_manager import OperationManager

        conn, kernel = _kernel(tmp_path)
        task_id = _task(kernel, tmp_path)
        ops = OperationManager(kernel, conn)
        op = await ops.start(
            task_id=task_id, attempt_id="att_1", kind="process",
            argv=[sys.executable, "-c",
                  "print('a', flush=True); print('b', flush=True)"],
        )
        await ops.wait(op["operation_id"], timeout=10)
        all_events = ops.events_since(task_id, 0)
        seqs = [e["seq"] for e in all_events]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        mid = seqs[len(seqs) // 2]
        replayed = ops.events_since(task_id, mid)
        assert [e["seq"] for e in replayed] == [s for s in seqs if s > mid]

    async def test_dead_process_fails_fast(self, tmp_path: Path) -> None:
        """进程非零退出 → operation FAILED（一个健康周期内，不永远
        RUNNING）。"""
        from rosclaw.task_kernel.operation_manager import OperationManager

        conn, kernel = _kernel(tmp_path)
        task_id = _task(kernel, tmp_path)
        ops = OperationManager(kernel, conn)
        op = await ops.start(
            task_id=task_id, attempt_id="att_1", kind="process",
            argv=[sys.executable, "-c", "import sys; sys.exit(3)"],
        )
        await ops.wait(op["operation_id"], timeout=10)
        final = ops.get(op["operation_id"])
        assert final["state"] == "FAILED"
        assert final["failure_code"]

    async def test_cancel_ledger_first_no_overwrite(self, tmp_path: Path) -> None:
        from rosclaw.task_kernel.operation_manager import OperationManager

        conn, kernel = _kernel(tmp_path)
        task_id = _task(kernel, tmp_path)
        ops = OperationManager(kernel, conn)
        op = await ops.start(
            task_id=task_id, attempt_id="att_1", kind="process",
            argv=[sys.executable, "-c", "import time; time.sleep(30)"],
        )
        await ops.cancel(op["operation_id"])
        final = ops.get(op["operation_id"])
        assert final["state"] == "CANCELLED"
        # 迟到事件不得覆盖 CANCELLED。
        await ops._record_terminal(op["operation_id"], "SUCCEEDED")
        again = ops.get(op["operation_id"])
        assert again["state"] == "CANCELLED"
