"""OperationManager V2（P1-B1，0824 总纲 §12）——长任务统一运营层。

Operation ≠ Worker：无模型的确定性执行过程（仿真 rollout、渲染、
数据处理、长测试、编译、ROS 2 Action）。核心不变量：

- 状态机：QUEUED → ADMITTED → RUNNING → SUCCEEDED/FAILED；
  cancel 经 CANCELING（带 reason）→ CANCELLED；失联 DEGRADED；
  重启不可证实 LOST。终态不可逆——迟到完成绝不覆盖。
- start() 立即返回 operation_id（不在调用方死等）；
- stdout/progress/heartbeat 全部进 task_events（单调 seq，断线从
  last_seq+1 重放，不重不漏）；
- **没有默认 wall-clock kill**：deadline 是任务语义、lease 是控制权、
  liveness timeout 只标 DEGRADED（§12.2）——sweep 永不杀进程；
- 重启恢复（reattach-or-LOST）：pid 活 → reattach（DEGRADED +
  pid-watcher）；pid 死 + exitcode 文件 → 应用真实终态；否则诚实
  LOST（绝不把僵尸行留在 RUNNING）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.contracts.common import new_id

#: 终态集合（不可逆）。LOST：重启后无法证实结局的诚实终态。
OPERATION_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "LOST"})

_LOG = logging.getLogger("rosclaw.operation_manager")

#: 单行输出事件的最大字节（防爆 task_events）。
_MAX_OUTPUT_CHUNK = 4000

#: pid-watcher 轮询间隔（reattach 后等进程消失）。
_WATCH_POLL_S = 2.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


class OperationManager:
    """operations 表 + task_events 事件流的唯一写者。"""

    def __init__(self, kernel, conn: sqlite3.Connection) -> None:
        self._kernel = kernel
        self._conn = conn
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._drivers: dict[str, asyncio.Task] = {}
        # operation_id → (client, goal_id)——ROS 2 Action 操作（P1-B3）。
        self._actions: dict[str, tuple[object, str]] = {}

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
        goal_id: str = "",
        provider: str = "process",
    ) -> dict[str, Any]:
        """启动后台 operation——QUEUED→ADMITTED，立即返回。"""
        operation_id = new_id("op")
        goal_id = goal_id or new_id("goal")
        now = _now()
        task_row = self._conn.execute(
            "SELECT active_revision FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        revision = int(task_row["active_revision"]) if task_row else None
        exitcode_path = str(self._operations_dir() / f"{operation_id}.exitcode")
        self._conn.execute(
            "INSERT INTO operations (operation_id, task_id, attempt_id, kind, "
            "state, resumable, started_at, heartbeat_at, revision, goal_id, "
            "provider, exitcode_path) "
            "VALUES (?, ?, ?, ?, 'QUEUED', ?, ?, ?, ?, ?, ?, ?)",
            (operation_id, task_id, attempt_id, kind,
             1 if resumable else 0, now, now, revision, goal_id, provider,
             exitcode_path),
        )
        self._emit(task_id, "operation.queued",
                   {"operation_id": operation_id, "goal_id": goal_id,
                    "provider": provider, "kind": kind},
                   operation_id=operation_id, attempt_id=attempt_id)
        proc = await self._spawn(operation_id, argv, cwd, env, exitcode_path)
        self._procs[operation_id] = proc
        self._transition(operation_id, "ADMITTED",
                         event="operation.admitted",
                         payload={"pid": proc.pid, "argv": argv[:5]})
        self._conn.execute(
            "UPDATE operations SET pid = ? WHERE operation_id = ?",
            (proc.pid, operation_id),
        )
        self._drivers[operation_id] = asyncio.create_task(
            self._drive(operation_id, task_id, attempt_id, proc)
        )
        return self.get(operation_id)

    async def _spawn(
        self,
        operation_id: str,
        argv: list[str],
        cwd: str | None,
        env: dict[str, str] | None,
        exitcode_path: str,
    ) -> asyncio.subprocess.Process:
        """exitcode wrapper：进程退出码落盘——agentd 死后重启 sweep
        仍能恢复真实终态（不靠运气）。"""
        spawn_env = dict(os.environ if env is None else env)
        spawn_env["OP_EXITCODE_FILE"] = exitcode_path
        wrapped = [
            "sh", "-c",
            '"$@"; rc=$?; printf %s "$rc" > "$OP_EXITCODE_FILE"',
            "op-wrap", *argv,
        ]
        return await asyncio.create_subprocess_exec(
            *wrapped,
            cwd=cwd,
            env=spawn_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )

    async def _drive(
        self, operation_id: str, task_id: str, attempt_id: str,
        proc: asyncio.subprocess.Process,
    ) -> None:
        """后台驱动：置 RUNNING → 读 stdout（output 事件 + heartbeat）
        → 进程退出 → 终态（账本优先于一切）。"""
        self._transition(operation_id, "RUNNING", event=None)
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
        current = self.get(operation_id)["state"]
        if current in ("CANCELING", "CANCELLED"):
            return  # 取消流程持有账本——迟到完成不覆盖
        # 权威退出码在 exitcode 文件（wrapper 自身的 rc 是 printf 的
        # 0——不能拿来判成败）；无文件（wrapper 被信号杀）回落 returncode。
        row = self.get(operation_id)
        file_rc = self._read_exitcode(str(row.get("exitcode_path") or ""))
        effective = file_rc if file_rc is not None else returncode
        await self._record_terminal(
            operation_id,
            "SUCCEEDED" if effective == 0 else "FAILED",
            failure_code="" if effective == 0 else f"exit_{effective}",
        )

    async def _record_terminal(
        self, operation_id: str, state: str, *, failure_code: str = "",
        result_ref: str = "",
    ) -> None:
        self._write_terminal(
            operation_id, state, failure_code=failure_code,
            result_ref=result_ref,
        )

    def _write_terminal(
        self, operation_id: str, state: str, *, failure_code: str = "",
        result_ref: str = "",
    ) -> None:
        """终态落账（终态不可逆——CANCELLED/LOST 不被迟到事件覆盖）。
        同步核心：Action result 回调（listener 线程）也走这里。

        CI 实证（p1b3 flake 根治）：CANCELING 也必须被保护——
        action_result(SUCCEEDED) 在取消宽限窗内到达时曾覆盖
        CANCELING（goal2 永远停在 SUCCEEDED）。取消流程持有账本：
        CANCELING 下只接受 CANCELLED（服务端 CANCELED 确认——
        握手正常完成）；SUCCEEDED/FAILED 是迟到完成，拒。"""
        row = self.get(operation_id)
        if row is None or row["state"] in OPERATION_TERMINAL:
            return
        if row["state"] == "CANCELING" and state != "CANCELLED":
            return  # 取消流程持有账本——迟到完成不覆盖 CANCELING
        now = _now()
        self._conn.execute(
            "UPDATE operations SET state = ?, ended_at = ?, heartbeat_at = ?, "
            "failure_code = ?, result_ref = ? WHERE operation_id = ?",
            (state, now, now, failure_code,
             result_ref or row.get("result_ref") or "", operation_id),
        )
        event_type = {
            "SUCCEEDED": "operation.completed",
            "FAILED": "operation.failed",
            "CANCELLED": "operation.cancelled",
            "LOST": "operation.lost",
        }[state]
        self._emit(row["task_id"], event_type,
                   {"operation_id": operation_id, "state": state,
                    "failure_code": failure_code},
                   operation_id=operation_id)

    # --------------------------------------------------------------
    # ROS 2 Action provider（P1-B3，0824 总纲 §12/P1-B）
    # --------------------------------------------------------------
    async def start_action(
        self,
        *,
        task_id: str,
        attempt_id: str,
        action: str,
        action_type: str,
        args: dict,
        client,
        goal_id: str = "",
    ) -> dict[str, Any]:
        """ROS 2 Action → 同一 Operation 契约（QUEUED→ADMITTED→
        RUNNING；feedback→progress；result→终态）。"""
        from rosclaw.connectors.ros.action_client import (
            STATUS_CANCELED,
            STATUS_SUCCEEDED,
        )

        operation_id = new_id("op")
        goal_id = goal_id or new_id("goal")
        now = _now()
        task_row = self._conn.execute(
            "SELECT active_revision FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        revision = int(task_row["active_revision"]) if task_row else None
        self._conn.execute(
            "INSERT INTO operations (operation_id, task_id, attempt_id, kind, "
            "state, resumable, started_at, heartbeat_at, revision, goal_id, "
            "provider, exitcode_path) "
            "VALUES (?, ?, ?, ?, 'QUEUED', 0, ?, ?, ?, ?, 'ros2_action', '')",
            (operation_id, task_id, attempt_id, "action", now, now, revision,
             goal_id),
        )
        self._emit(task_id, "operation.queued",
                   {"operation_id": operation_id, "goal_id": goal_id,
                    "provider": "ros2_action", "action": action,
                    "action_type": action_type},
                   operation_id=operation_id, attempt_id=attempt_id)

        loop = asyncio.get_running_loop()

        def _marshal(fn):
            def _wrapped(*cb_args):
                try:
                    loop.call_soon_threadsafe(fn, *cb_args)
                except RuntimeError:
                    # 事件循环已关（测试相位）——同步执行；连接为
                    # check_same_thread=False 时安全。生产路径 loop 恒活。
                    fn(*cb_args)

            return _wrapped

        def _on_feedback(values: dict) -> None:
            self.report_progress(operation_id, values)

        def _on_result(status: int, values: dict) -> None:
            result_ref = json.dumps(values, ensure_ascii=False)[:500]
            if status == STATUS_SUCCEEDED:
                state, code = "SUCCEEDED", ""
            elif status == STATUS_CANCELED:
                state = "CANCELLED"
                code = str(self.get(operation_id).get("cancel_reason") or "action_canceled")
            else:
                state, code = "FAILED", f"action_status_{status}"
            self._actions.pop(operation_id, None)
            self._write_terminal(
                operation_id, state,
                failure_code=code, result_ref=result_ref,
            )

        client.send_goal(
            action=action, action_type=action_type, args=args,
            goal_id=goal_id,
            on_feedback=_marshal(_on_feedback),
            on_result=_marshal(_on_result),
        )
        self._actions[operation_id] = (client, goal_id)
        self._transition(operation_id, "ADMITTED",
                         event="operation.admitted",
                         payload={"action": action, "goal_id": goal_id})
        self._transition(operation_id, "RUNNING", event=None)
        return self.get(operation_id)

    async def wait(self, operation_id: str, *, timeout: float = 60.0) -> dict:
        """等终态（测试/短操作同步点——不是轮询生产路径）。"""
        driver = self._drivers.get(operation_id)
        if driver is not None:
            await asyncio.wait_for(asyncio.shield(driver), timeout=timeout)
        return self.get(operation_id)

    async def cancel(self, operation_id: str, *, reason: str = "user") -> None:
        """CANCELING（账本+原因）→ 信号 → CANCELLED。

        迟到完成在 CANCELING/CANCELLED 下都不覆盖（§12.1 取消握手）。
        """
        row = self.get(operation_id)
        if not row or row["state"] in OPERATION_TERMINAL:
            return
        now = _now()
        self._conn.execute(
            "UPDATE operations SET state = 'CANCELING', cancel_reason = ?, "
            "heartbeat_at = ? WHERE operation_id = ?",
            (reason, now, operation_id),
        )
        self._emit(row["task_id"], "operation.canceling",
                   {"operation_id": operation_id, "reason": reason},
                   operation_id=operation_id)
        action_ref = self._actions.get(operation_id)
        if action_ref is not None:
            # ROS 2 Action：cancel_goal 请求——终态由 action_result
            # (CANCELED) 确认；宽限后服务端无响应也落 CANCELLED
            # （诚实：请求已发，结局不可考）。
            # 0902 复核 L1：cancel_goal 抛异常时 grace 任务必须先
            # 建好——否则 operation 永卡 CANCELING（sweep_liveness
            # 不管 CANCELING，重启才收 LOST）。
            client, goal_id = action_ref
            self._drivers[operation_id] = asyncio.create_task(
                self._cancel_grace(operation_id, reason)
            )
            try:
                client.cancel_goal(goal_id)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - 发送失败由 grace 落 CANCELLED
                _LOG.warning(
                    "cancel_goal 发送失败（%s）——grace 落 CANCELLED",
                    operation_id, exc_info=True,
                )
            return
        proc = self._procs.pop(operation_id, None)
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
        await self._record_terminal(operation_id, "CANCELLED",
                                    failure_code=reason)

    async def _cancel_grace(
        self, operation_id: str, reason: str, grace_s: float = 5.0
    ) -> None:
        """Action 取消宽限：action_result 未在宽限内到达也落
        CANCELLED（账本不再悬空）。"""
        await asyncio.sleep(grace_s)
        await self._record_terminal(operation_id, "CANCELLED",
                                    failure_code=reason)

    # --------------------------------------------------------------
    # liveness（§12.2：只标 DEGRADED，永不 kill）
    # --------------------------------------------------------------
    async def sweep_liveness(self, *, stale_after_s: float = 30.0) -> dict:
        """heartbeat 过期 → DEGRADED；恢复 → RUNNING。绝不杀进程。"""
        degraded = resumed = 0
        now = datetime.now(UTC).timestamp()
        rows = self._conn.execute(
            "SELECT operation_id, task_id, state, heartbeat_at FROM operations "
            "WHERE state IN ('QUEUED', 'ADMITTED', 'RUNNING', 'DEGRADED')",
        ).fetchall()
        for row in rows:
            heartbeat = datetime.fromisoformat(str(row["heartbeat_at"])).timestamp()
            stale = (now - heartbeat) > stale_after_s
            if stale and row["state"] != "DEGRADED":
                self._transition(
                    row["operation_id"], "DEGRADED",
                    event="operation.degraded",
                    payload={"stale_after_s": stale_after_s},
                )
                degraded += 1
            elif not stale and row["state"] == "DEGRADED":
                self._transition(row["operation_id"], "RUNNING",
                                 event="operation.resumed", payload={})
                resumed += 1
        return {"degraded": degraded, "resumed": resumed}

    # --------------------------------------------------------------
    # 重启恢复（reattach-or-LOST）
    # --------------------------------------------------------------
    async def recover_on_boot(self) -> dict:
        """agentd 重启后的 operation 对账。

        - pid 活 → reattach（DEGRADED + pid-watcher 等退出）；
        - pid 死 + exitcode 文件 → 应用真实终态；
        - 否则 → 诚实 LOST（事件留痕，绝不留僵尸 RUNNING）。
        """
        report = {"reattached": 0, "terminated": 0, "lost": 0}
        rows = self._conn.execute(
            "SELECT operation_id, task_id, pid, exitcode_path FROM operations "
            "WHERE state IN ('QUEUED', 'ADMITTED', 'RUNNING', 'DEGRADED', "
            "'CANCELING')",
        ).fetchall()
        for row in rows:
            op_id = str(row["operation_id"])
            pid = int(row["pid"] or 0)
            exitcode = self._read_exitcode(str(row["exitcode_path"] or ""))
            if exitcode is not None:
                await self._record_terminal(
                    op_id, "SUCCEEDED" if exitcode == 0 else "FAILED",
                    failure_code="" if exitcode == 0 else f"exit_{exitcode}",
                )
                report["terminated"] += 1
                continue
            if pid > 0 and self._pid_alive(pid):
                self._transition(op_id, "DEGRADED",
                                 event="operation.reattached",
                                 payload={"pid": pid})
                self._drivers[op_id] = asyncio.create_task(
                    self._watch_pid(op_id, str(row["task_id"]), pid,
                                    str(row["exitcode_path"] or ""))
                )
                report["reattached"] += 1
                continue
            await self._record_terminal(
                op_id, "LOST",
                failure_code="restart_unverifiable",
            )
            report["lost"] += 1
        return report

    async def _watch_pid(
        self, operation_id: str, task_id: str, pid: int, exitcode_path: str
    ) -> None:
        """reattach 后的 pid-watcher：等进程消失 → exitcode 定终态
        （无 exitcode = 诚实 LOST——退出码不可考）。"""
        while self._pid_alive(pid):
            exitcode = self._read_exitcode(exitcode_path)
            if exitcode is not None:
                break
            await asyncio.sleep(_WATCH_POLL_S)
        current = self.get(operation_id)["state"]
        if current in OPERATION_TERMINAL or current == "CANCELING":
            return
        exitcode = self._read_exitcode(exitcode_path)
        if exitcode is not None:
            await self._record_terminal(
                operation_id, "SUCCEEDED" if exitcode == 0 else "FAILED",
                failure_code="" if exitcode == 0 else f"exit_{exitcode}",
            )
        else:
            await self._record_terminal(
                operation_id, "LOST", failure_code="exit_unverifiable",
            )

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        # kill(pid,0) 对 zombie 也成功——/proc stat 'Z' 必须判死
        # （CI 实证：kill -9 后无人 reap → 僵尸被误判存活 → 假 reattach）。
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            if stat.rpartition(")")[2].split()[0] == "Z":
                return False
        except OSError:
            return False
        return True

    @staticmethod
    def _read_exitcode(path: str) -> int | None:
        if not path:
            return None
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    # --------------------------------------------------------------
    # progress / result（§12.3：provider feedback → operation.progress）
    # --------------------------------------------------------------
    def report_progress(self, operation_id: str, progress: dict) -> None:
        """provider 反馈 → progress_json + operation.progress 事件
        （UI 按 operation_id upsert；不进模型上下文）。"""
        row = self.get(operation_id)
        if not row or row["state"] in OPERATION_TERMINAL:
            return
        self._touch(operation_id, str(row["task_id"]))
        self._conn.execute(
            "UPDATE operations SET progress_json = ? WHERE operation_id = ?",
            (json.dumps(progress, ensure_ascii=False), operation_id),
        )
        self._emit(str(row["task_id"]), "operation.progress",
                   {"operation_id": operation_id, "progress": progress},
                   operation_id=operation_id)

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

    # --------------------------------------------------------------
    # 内部
    # --------------------------------------------------------------
    def _operations_dir(self) -> Path:
        db_file = self._conn.execute(
            "PRAGMA database_list"
        ).fetchone()["file"]
        directory = Path(str(db_file)).parent / "operations"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _transition(
        self, operation_id: str, state: str, *,
        event: str | None, payload: dict | None = None,
    ) -> None:
        row = self.get(operation_id)
        # 终态不可逆；CANCELING 同样不可被普通迁移覆盖（取消流程持有
        # 账本——driver 迟到置 RUNNING 不得踩掉 CANCELING）。
        if not row or row["state"] in OPERATION_TERMINAL or (
            row["state"] == "CANCELING" and state != "CANCELLED"
        ):
            return
        self._conn.execute(
            "UPDATE operations SET state = ?, heartbeat_at = ? "
            "WHERE operation_id = ?",
            (state, _now(), operation_id),
        )
        if event:
            self._emit(str(row["task_id"]), event,
                       {"operation_id": operation_id, **(payload or {})},
                       operation_id=operation_id)

    def _touch(self, operation_id: str, task_id: str) -> None:
        """heartbeat——仅非终态（终态后心跳冻结）。"""
        row = self.get(operation_id)
        if not row or row["state"] in OPERATION_TERMINAL:
            return
        self._conn.execute(
            "UPDATE operations SET heartbeat_at = ? WHERE operation_id = ?",
            (_now(), operation_id),
        )

    def _emit(self, task_id: str, event_type: str, payload: dict,
              *, operation_id: str = "", attempt_id: str = "") -> None:
        self._conn.execute(
            "INSERT INTO task_events (task_id, attempt_id, operation_id, "
            "event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, attempt_id or None, operation_id or None, event_type,
             json.dumps(payload, ensure_ascii=False), _now()),
        )
