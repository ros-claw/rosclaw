"""OperationManager（PR-H3，总纲 v2 §11）——长任务/进程的统一运营层。

Operation ≠ Worker：无模型的确定性执行过程（仿真 rollout、渲染、
数据处理、长测试、编译）。核心不变量：

- start() 立即返回 operation_id（不在调用方死等）；
- stdout/stderr/progress/heartbeat 全部进 task_events（单调 seq，
  断线从 last_seq+1 重放，不重不漏）；
- 终态即冻结 ended_at/heartbeat_at（elapsed 必须停止）；
- cancel 先写账本再发信号；迟到 completion 不得覆盖 CANCELLED；
- 进程死亡 → FAILED（非零退出）/SUCCEEDED——一个驱动周期内收敛，
  不允许 DB 显示 RUNNING 但无进程。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from rosclaw.contracts.common import new_id

OPERATION_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})

#: 单行输出事件的最大字节（防爆 task_events）。
_MAX_OUTPUT_CHUNK = 4000


class OperationManager:
    """operations 表 + task_events 事件流的唯一写者。"""

    def __init__(self, kernel, conn: sqlite3.Connection) -> None:
        self._kernel = kernel
        self._conn = conn
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._drivers: dict[str, asyncio.Task] = {}

    # --------------------------------------------------------------
    # 生命周期
    # --------------------------------------------------------------
    async def start(
        self,
        *,
        task_id: str,
        attempt_id: str,
        kind: str,
        argv: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        resumable: bool = False,
    ) -> dict[str, Any]:
        """启动后台进程 operation——立即返回（调用方不死等）。"""
        operation_id = new_id("op")
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT INTO operations (operation_id, task_id, attempt_id, kind, "
            "state, resumable, started_at, heartbeat_at) "
            "VALUES (?, ?, ?, ?, 'RUNNING', ?, ?, ?)",
            (operation_id, task_id, attempt_id, kind,
             1 if resumable else 0, now, now),
        )
        self._emit(task_id, "operation.started",
                   {"operation_id": operation_id, "kind": kind,
                    "argv": argv[:5]},
                   operation_id=operation_id, attempt_id=attempt_id)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        self._procs[operation_id] = proc
        self._drivers[operation_id] = asyncio.create_task(
            self._drive(operation_id, task_id, attempt_id, proc)
        )
        return self.get(operation_id)

    async def _drive(
        self, operation_id: str, task_id: str, attempt_id: str,
        proc: asyncio.subprocess.Process,
    ) -> None:
        """后台驱动：读 stdout → output 事件 + heartbeat；进程退出 →
        终态（账本优先于一切）。"""
        assert proc.stdout is not None
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode(errors="replace")[:_MAX_OUTPUT_CHUNK]
                self._touch(operation_id, task_id)
                self._emit(task_id, "operation.output", {"text": text},
                           operation_id=operation_id, attempt_id=attempt_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 读失败是数据
            self._emit(task_id, "operation.failed",
                       {"error": f"reader: {exc}"[:300]},
                       operation_id=operation_id, attempt_id=attempt_id)
        returncode = await proc.wait()
        self._procs.pop(operation_id, None)
        if self.get(operation_id)["state"] == "CANCELLED":
            return  # 取消已落账——迟到完成不覆盖
        await self._record_terminal(
            operation_id,
            "SUCCEEDED" if returncode == 0 else "FAILED",
            failure_code="" if returncode == 0 else f"exit_{returncode}",
        )

    async def _record_terminal(
        self, operation_id: str, state: str, *, failure_code: str = ""
    ) -> None:
        """终态落账（终态不可逆——CANCELLED 不被迟到完成覆盖）。"""
        row = self.get(operation_id)
        if row is None or row["state"] in OPERATION_TERMINAL:
            return
        now = datetime.now(UTC).isoformat()
        # ended_at + heartbeat_at 同时冻结（elapsed 停止）。
        self._conn.execute(
            "UPDATE operations SET state = ?, ended_at = ?, heartbeat_at = ?, "
            "failure_code = ? WHERE operation_id = ?",
            (state, now, now, failure_code, operation_id),
        )
        event_type = ("operation.completed" if state == "SUCCEEDED"
                      else "operation.failed")
        self._emit(row["task_id"], event_type,
                   {"operation_id": operation_id, "state": state,
                    "failure_code": failure_code},
                   operation_id=operation_id)

    async def wait(self, operation_id: str, *, timeout: float = 60.0) -> dict:
        """等终态（测试/短操作同步点——不是轮询生产路径）。"""
        driver = self._drivers.get(operation_id)
        if driver is not None:
            await asyncio.wait_for(asyncio.shield(driver), timeout=timeout)
        return self.get(operation_id)

    async def cancel(self, operation_id: str, *, reason: str = "user") -> None:
        """先写账本再发信号（迟到完成不覆盖 CANCELLED）。"""
        await self._record_terminal(operation_id, "CANCELLED",
                                    failure_code=reason)
        proc = self._procs.pop(operation_id, None)
        if proc is not None and proc.returncode is None:
            import contextlib
            import signal

            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()

    # --------------------------------------------------------------
    # 查询/事件流
    # --------------------------------------------------------------
    def get(self, operation_id: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        return dict(row) if row else {}

    def events_since(self, task_id: str, last_seq: int) -> list[dict]:
        """seq 重放（断线从 last_seq+1——不重不漏）。"""
        rows = self._conn.execute(
            "SELECT seq, session_ref, attempt_id, operation_id, event_type, "
            "payload_json, created_at FROM task_events "
            "WHERE task_id = ? AND seq > ? ORDER BY seq",
            (task_id, last_seq),
        ).fetchall()
        return [
            {
                "seq": r["seq"],
                "session_ref": r["session_ref"],
                "attempt_id": r["attempt_id"],
                "operation_id": r["operation_id"],
                "event_type": r["event_type"],
                "payload": json.loads(r["payload_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def _touch(self, operation_id: str, task_id: str) -> None:
        """heartbeat——仅非终态（终态后心跳冻结）。"""
        row = self.get(operation_id)
        if not row or row["state"] in OPERATION_TERMINAL:
            return
        self._conn.execute(
            "UPDATE operations SET heartbeat_at = ? WHERE operation_id = ?",
            (datetime.now(UTC).isoformat(), operation_id),
        )

    def _emit(self, task_id: str, event_type: str, payload: dict,
              *, operation_id: str = "", attempt_id: str = "") -> None:
        self._conn.execute(
            "INSERT INTO task_events (task_id, attempt_id, operation_id, "
            "event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, attempt_id or None, operation_id or None, event_type,
             json.dumps(payload, ensure_ascii=False),
             datetime.now(UTC).isoformat()),
        )
