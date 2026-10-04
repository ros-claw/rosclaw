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
- 重启恢复：持久身份匹配的活 pid → reattach；旧/未知身份保留
  DEGRADED + 未确认事件，不接管任意同号 PID；pid 死 + exitcode
  文件 → 应用真实终态，否则诚实 LOST。
"""

from __future__ import annotations

import asyncio
import codecs
import fcntl
import json
import logging
import os
import signal
import sqlite3
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.contracts.common import new_id
from rosclaw.task_kernel.process_identity import (
    ProcessIdentity,
    session_members,
    signal_owned_members,
)

#: 终态集合（不可逆）。LOST：重启后无法证实结局的诚实终态。
OPERATION_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "LOST"})

_LOG = logging.getLogger("rosclaw.operation_manager")

#: 每次输出读取的最大字节（事件有界，不依赖生产者换行）。
_MAX_OUTPUT_CHUNK = 4000

#: pid-watcher 轮询间隔（reattach 后等进程消失）。
_WATCH_POLL_S = 2.0
_CANCEL_GRACE_S = 5.0


class _OutputObserverSupersededError(RuntimeError):
    """Another observer already committed this byte range."""


class _PreservedProcess:
    """No asyncio transport: loop shutdown/GC must never kill the producer."""

    stdout = None

    def __init__(self, process: subprocess.Popen, operation_id: str) -> None:
        self._process = process
        self.operation_id = operation_id
        self.pid = process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.poll()

    async def wait(self) -> int:
        while self.returncode is None:
            await asyncio.sleep(0.02)
        _REAPABLE_CHILDREN.pop(self.operation_id, None)
        return self.returncode


# Retain only owned Popen handles for natural reaping after observer handoff;
# retaining no output FDs, threads, event loops or database connections.
_REAPABLE_CHILDREN: dict[str, _PreservedProcess] = {}


def _reap_finished_children() -> None:
    for op_id, child in list(_REAPABLE_CHILDREN.items()):
        if child.returncode is not None:
            _REAPABLE_CHILDREN.pop(op_id, None)


class OperationHandoffUnresolvedError(RuntimeError):
    """No safe preserve-only shutdown exists for these active operations."""

    code = "OPERATION_HANDOFF_UNRESOLVED"

    def __init__(self, operations: list[dict]) -> None:
        self.operations = operations
        super().__init__(self.code)


class OperationCancellationUnresolvedError(RuntimeError):
    """Cancellation was requested, but owned process stop is not confirmed."""

    code = "CANCEL_STOP_UNCONFIRMED"

    def __init__(self, operation_id: str) -> None:
        self.operation_id = operation_id
        super().__init__(self.code)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class OperationManager:
    """operations 表 + task_events 事件流的唯一写者。"""

    def __init__(self, kernel, conn: sqlite3.Connection) -> None:
        self._kernel = kernel
        self._conn = conn
        self._procs: dict[str, asyncio.subprocess.Process | _PreservedProcess] = {}
        _reap_finished_children()
        self._drivers: dict[str, asyncio.Task] = {}
        # operation_id → (client, goal_id)——ROS 2 Action 操作（P1-B3）。
        self._actions: dict[str, tuple[object, str]] = {}
        self._lifecycle_lock = asyncio.Lock()
        self._closing = False

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
        async with self._lifecycle_lock:
            if self._closing:
                raise RuntimeError("operation manager is closed")
            return await self._start(
                task_id=task_id,
                attempt_id=attempt_id,
                kind=kind,
                argv=argv,
                cwd=cwd,
                env=env,
                resumable=resumable,
                goal_id=goal_id,
                provider=provider,
            )

    async def _start(
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
            (
                operation_id,
                task_id,
                attempt_id,
                kind,
                1 if resumable else 0,
                now,
                now,
                revision,
                goal_id,
                provider,
                exitcode_path,
            ),
        )
        self._emit(
            task_id,
            "operation.queued",
            {"operation_id": operation_id, "goal_id": goal_id, "provider": provider, "kind": kind},
            operation_id=operation_id,
            attempt_id=attempt_id,
        )
        proc = await self._spawn(operation_id, argv, cwd, env, exitcode_path)
        self._procs[operation_id] = proc
        self._transition(
            operation_id,
            "ADMITTED",
            event="operation.admitted",
            payload={"pid": proc.pid, "argv": argv[:5]},
        )
        identity = ProcessIdentity.capture(proc.pid)
        self._conn.execute(
            "UPDATE operations SET pid = ?, process_identity_json = ? WHERE operation_id = ?",
            (proc.pid, identity.to_json() if identity else "", operation_id),
        )
        self._drivers[operation_id] = asyncio.create_task(
            self._drive(operation_id, task_id, attempt_id, proc)
        )
        # Start observation before returning admission; unlike asyncio's spawn
        # transport, Popen itself contains no event-loop scheduling boundary.
        await asyncio.sleep(0.01)
        return self.get(operation_id)

    async def _spawn(
        self,
        operation_id: str,
        argv: list[str],
        cwd: str | None,
        env: dict[str, str] | None,
        exitcode_path: str,
    ) -> _PreservedProcess:
        """exitcode wrapper：进程退出码落盘——agentd 死后重启 sweep
        仍能恢复真实终态（不靠运气）。"""
        spawn_env = dict(os.environ if env is None else env)
        spawn_env["OP_EXITCODE_FILE"] = exitcode_path
        output_path = self._operations_dir() / f"{operation_id}.stdout"
        fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            # The open-file-description lock survives dup/fork/exec: EOF is not
            # declared while a descendant still owns an inherited output FD.
            fcntl.flock(fd, fcntl.LOCK_EX)
            info = os.fstat(fd)
            checkpoint = {
                "device": info.st_dev,
                "inode": info.st_ino,
                "offset": 0,
                "pending_hex": "",
                "finalized": False,
            }
            self._conn.execute(
                "UPDATE operations SET output_path = ?, output_checkpoint_json = ? "
                "WHERE operation_id = ?",
                (str(output_path), json.dumps(checkpoint), operation_id),
            )
            wrapped = [
                "sh",
                "-c",
                '"$@"; rc=$?; umask 077; '
                'printf %s "$rc" > "$OP_EXITCODE_FILE.tmp" && '
                'mv -f "$OP_EXITCODE_FILE.tmp" "$OP_EXITCODE_FILE"; exit "$rc"',
                "op-wrap",
                *argv,
            ]
            _reap_finished_children()
            child = _PreservedProcess(
                subprocess.Popen(
                    wrapped,
                    cwd=cwd,
                    env=spawn_env,
                    stdout=fd,
                    stderr=fd,
                    start_new_session=True,
                ),
                operation_id,
            )
            _REAPABLE_CHILDREN[operation_id] = child
            return child
        finally:
            os.close(fd)

    def _open_spool(self, row: dict):
        path = Path(str(row["output_path"]))
        if path != self._operations_dir() / f"{row['operation_id']}.stdout":
            raise ValueError("unexpected spool path")
        checkpoint = json.loads(row["output_checkpoint_json"])
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_dev != checkpoint["device"]
                or info.st_ino != checkpoint["inode"]
            ):
                raise ValueError("spool file identity or permissions changed")
            offset = checkpoint["offset"]
            pending = bytes.fromhex(checkpoint["pending_hex"])
            if (
                type(offset) is not int
                or offset < 0
                or offset > info.st_size
                or len(pending) > 3
                or len(pending) > offset
                or type(checkpoint["finalized"]) is not bool
                or (checkpoint["finalized"] and offset != info.st_size)
            ):
                raise ValueError("invalid output checkpoint")
            if pending and os.pread(fd, len(pending), offset - len(pending)) != pending:
                raise ValueError("pending UTF-8 bytes disagree with spool")
            validator = codecs.getincrementaldecoder("utf-8")(errors="replace")
            if (
                validator.decode(pending, final=False)
                or validator.getstate()[0] != pending
                or (checkpoint["finalized"] and pending)
            ):
                raise ValueError("invalid UTF-8 decoder checkpoint")
            return os.fdopen(fd, "rb", buffering=0), checkpoint
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _writers_closed(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True

    def _checkpoint_output(self, row: dict, checkpoint: dict, text: str, previous: dict) -> None:
        # In autocommit production SQLite this is one durable transaction; in
        # caller-owned transactions both the event and cursor commit together.
        self._conn.execute("SAVEPOINT operation_output")
        try:
            current = self._conn.execute(
                "SELECT output_checkpoint_json FROM operations WHERE operation_id = ?",
                (row["operation_id"],),
            ).fetchone()
            if current is None or json.loads(current["output_checkpoint_json"]) != previous:
                raise _OutputObserverSupersededError("output cursor already advanced")
            if text:
                self._emit(
                    row["task_id"],
                    "operation.output",
                    {"text": text},
                    operation_id=row["operation_id"],
                    attempt_id=str(row.get("attempt_id") or ""),
                )
            self._conn.execute(
                "UPDATE operations SET output_checkpoint_json = ?, heartbeat_at = CASE "
                "WHEN state IN ('SUCCEEDED','FAILED','CANCELLED','LOST') THEN heartbeat_at "
                "ELSE ? END WHERE operation_id = ?",
                (json.dumps(checkpoint), _now(), row["operation_id"]),
            )
            self._conn.execute("RELEASE operation_output")
        except BaseException:
            self._conn.execute("ROLLBACK TO operation_output")
            self._conn.execute("RELEASE operation_output")
            raise

    def preflight_handoff(self) -> None:
        unresolved = []
        for row in self._conn.execute(
            "SELECT * FROM operations WHERE state NOT IN ('SUCCEEDED','FAILED','CANCELLED','LOST')"
        ).fetchall():
            row = dict(row)
            op_id = row["operation_id"]
            if row.get("provider") != "process":
                unresolved.append({"operation_id": op_id, "reason": "active_action_no_handoff"})
            elif row.get("output_path"):
                try:
                    stream, _ = self._open_spool(row)
                    stream.close()
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    unresolved.append({"operation_id": op_id, "reason": f"spool: {exc}"})
            elif op_id in self._procs and self._procs[op_id].stdout is not None:
                driver = self._drivers.get(op_id)
                if driver is not None and not driver.done():
                    unresolved.append({"operation_id": op_id, "reason": "legacy_active_PIPE"})
            elif self._pid_alive(int(row.get("pid") or 0)):
                unresolved.append(
                    {"operation_id": op_id, "reason": "legacy_output_not_recoverable"}
                )
        # A legacy terminal ledger does not prove its PIPE/transport or pending
        # Action callbacks are safe to destroy. Inspect local ownership too.
        listed = {item["operation_id"] for item in unresolved}
        for op_id, proc in self._procs.items():
            if (
                proc.stdout is not None
                and op_id not in listed
                and (proc.returncode is None or not proc.stdout.at_eof())
            ):
                unresolved.append({"operation_id": op_id, "reason": "legacy_active_PIPE"})
                listed.add(op_id)
        for op_id in self._actions:
            if op_id not in listed:
                unresolved.append({"operation_id": op_id, "reason": "active_action_no_handoff"})
        if unresolved:
            raise OperationHandoffUnresolvedError(unresolved)

    async def close(self) -> None:
        """Join a shared preserve-only handoff; caller cancellation is local."""
        task = getattr(self, "_handoff_task", None)
        if task is None:
            task = asyncio.create_task(self._close_observers())
            self._handoff_task = task
        try:
            await asyncio.shield(task)
        except Exception:
            if getattr(self, "_handoff_task", None) is task:
                self._handoff_task = None
            raise

    async def _close_observers(self) -> None:
        """Preserve operations; settle only safe observers before DB closure."""
        async with self._lifecycle_lock:
            self.preflight_handoff()  # no observer cancellation on a failed preflight
            self._closing = True
            drivers = list(self._drivers.values())
            for driver in drivers:
                if not driver.done() and not driver.cancelling():
                    driver.cancel()
            if drivers:
                await asyncio.gather(*drivers, return_exceptions=True)
            for row in self._conn.execute(
                "SELECT * FROM operations WHERE output_path != '' "
                "AND state NOT IN ('SUCCEEDED','FAILED','CANCELLED','LOST')"
            ).fetchall():
                checkpoint = json.loads(row["output_checkpoint_json"])
                self._emit(
                    row["task_id"],
                    "operation.output_handoff",
                    {
                        "operation_id": row["operation_id"],
                        "output_cursor_bytes": checkpoint["offset"],
                        "close_requested_stop": False,
                    },
                    operation_id=row["operation_id"],
                )
            self._conn.commit()

    async def _drive_spool(
        self, operation_id: str, proc: asyncio.subprocess.Process | _PreservedProcess | None = None
    ) -> None:
        row = self.get(operation_id)
        self._transition(operation_id, "RUNNING", event=None)
        try:
            stream, checkpoint = self._open_spool(row)
            with stream:
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                decoder.setstate((bytes.fromhex(checkpoint["pending_hex"]), 0))
                while not checkpoint["finalized"]:
                    if checkpoint["offset"] > os.fstat(stream.fileno()).st_size:
                        raise ValueError("spool truncated below cursor")
                    stream.seek(checkpoint["offset"])
                    chunk = stream.read(_MAX_OUTPUT_CHUNK - len(decoder.getstate()[0]))
                    file_rc = self._read_exitcode(str(row.get("exitcode_path") or ""))
                    final = (
                        not chunk
                        and self._writers_closed(stream.fileno())
                        and (
                            file_rc is not None
                            or (proc is not None and proc.returncode is not None)
                            or row["state"] in OPERATION_TERMINAL
                            or not self._pid_alive(int(row.get("pid") or 0))
                        )
                    )
                    if chunk or final:
                        text = decoder.decode(chunk, final=final)
                        previous = checkpoint
                        checkpoint = {
                            **checkpoint,
                            "offset": checkpoint["offset"] + len(chunk),
                            "pending_hex": decoder.getstate()[0].hex(),
                            "finalized": final,
                        }
                        self._checkpoint_output(row, checkpoint, text, previous)
                        await asyncio.sleep(0)  # bounded tail work yields to close/cancel
                    else:
                        await asyncio.sleep(0.02)
            if proc is not None:
                rc = await proc.wait()
                self._procs.pop(operation_id, None)
            else:
                rc = None
            file_rc = self._read_exitcode(str(row.get("exitcode_path") or ""))
            effective = file_rc if file_rc is not None else rc
            await self._record_terminal(
                operation_id,
                "LOST" if effective is None else "SUCCEEDED" if effective == 0 else "FAILED",
                failure_code="exit_unverifiable"
                if effective is None
                else ""
                if effective == 0
                else f"exit_{effective}",
            )
        except _OutputObserverSupersededError:
            return  # another durable observer owns this range; never duplicate it
        except asyncio.CancelledError:
            raise  # regular output FD remains owned by producer; no kill/PIPE close
        except Exception as exc:  # observation failure must never become success
            if self.get(operation_id)["state"] in OPERATION_TERMINAL:
                self._emit(
                    row["task_id"],
                    "operation.output_unresolved",
                    {"error": str(exc)[:300]},
                    operation_id=operation_id,
                )
            else:
                self._conn.execute(
                    "UPDATE operations SET failure_code = ? WHERE operation_id = ?",
                    ("OUTPUT_OBSERVATION_UNRESOLVED", operation_id),
                )
                self._transition(
                    operation_id,
                    "DEGRADED",
                    event="operation.output_unresolved",
                    payload={"error": str(exc)[:300]},
                )

    async def _drive(
        self,
        operation_id: str,
        task_id: str,
        attempt_id: str,
        proc: asyncio.subprocess.Process | _PreservedProcess,
    ) -> None:
        """后台驱动：置 RUNNING → 读 stdout（output 事件 + heartbeat）
        → 进程退出 → 终态（账本优先于一切）。"""
        if self.get(operation_id).get("output_path"):
            await self._drive_spool(operation_id, proc)
            return
        self._transition(operation_id, "RUNNING", event=None)
        assert proc.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                chunk = await proc.stdout.read(_MAX_OUTPUT_CHUNK)
                text = decoder.decode(chunk, final=not chunk)
                if chunk:
                    self._touch(operation_id, task_id)
                if text:
                    self._emit(
                        task_id,
                        "operation.output",
                        {"text": text},
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                    )
                if not chunk:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # legacy PIPE read failure cannot imply success
            self._conn.execute(
                "UPDATE operations SET failure_code = ? WHERE operation_id = ?",
                ("OUTPUT_OBSERVATION_UNRESOLVED", operation_id),
            )
            self._transition(
                operation_id,
                "DEGRADED",
                event="operation.output_unresolved",
                payload={"error": f"legacy reader: {exc}"[:300]},
            )
            return  # keep the PIPE/process handle; close preflight remains honest
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
        self,
        operation_id: str,
        state: str,
        *,
        failure_code: str = "",
        result_ref: str = "",
    ) -> None:
        self._write_terminal(
            operation_id,
            state,
            failure_code=failure_code,
            result_ref=result_ref,
        )

    def _write_terminal(
        self,
        operation_id: str,
        state: str,
        *,
        failure_code: str = "",
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
            (
                state,
                now,
                now,
                failure_code,
                result_ref or row.get("result_ref") or "",
                operation_id,
            ),
        )
        event_type = {
            "SUCCEEDED": "operation.completed",
            "FAILED": "operation.failed",
            "CANCELLED": "operation.cancelled",
            "LOST": "operation.lost",
        }[state]
        self._emit(
            row["task_id"],
            event_type,
            {"operation_id": operation_id, "state": state, "failure_code": failure_code},
            operation_id=operation_id,
        )

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
        if self._closing:
            raise RuntimeError("operation manager is closed")
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
            (operation_id, task_id, attempt_id, "action", now, now, revision, goal_id),
        )
        self._emit(
            task_id,
            "operation.queued",
            {
                "operation_id": operation_id,
                "goal_id": goal_id,
                "provider": "ros2_action",
                "action": action,
                "action_type": action_type,
            },
            operation_id=operation_id,
            attempt_id=attempt_id,
        )

        loop = asyncio.get_running_loop()

        def _marshal(fn):
            def _apply(*cb_args):
                # A listener may have queued this before preserve-only close.
                # Recheck on the database-owning loop immediately before use.
                if self._closing:
                    return
                fn(*cb_args)

            def _wrapped(*cb_args):
                # Closed managers have no authority to apply late updates.
                if self._closing:
                    return
                try:
                    loop.call_soon_threadsafe(_apply, *cb_args)
                except RuntimeError:
                    # Loop shutdown is not permission to write SQLite from a
                    # listener thread, even for check_same_thread=False stores.
                    _LOG.debug(
                        "Ignoring Action callback after owning loop closed (%s)", operation_id
                    )

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
                operation_id,
                state,
                failure_code=code,
                result_ref=result_ref,
            )

        client.send_goal(
            action=action,
            action_type=action_type,
            args=args,
            goal_id=goal_id,
            on_feedback=_marshal(_on_feedback),
            on_result=_marshal(_on_result),
        )
        self._actions[operation_id] = (client, goal_id)
        self._transition(
            operation_id,
            "ADMITTED",
            event="operation.admitted",
            payload={"action": action, "goal_id": goal_id},
        )
        self._transition(operation_id, "RUNNING", event=None)
        return self.get(operation_id)

    async def wait(self, operation_id: str, *, timeout: float = 60.0) -> dict:
        """等终态（测试/短操作同步点——不是轮询生产路径）。"""
        row = self.get(operation_id)
        if row.get("state") in OPERATION_TERMINAL:
            # A confirmed owned-session stop does not imply EOF from writers
            # outside that session. Terminal status never waits for their logs.
            return row
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
            if row and self.stop_confirmation_missing(row):
                raise OperationCancellationUnresolvedError(operation_id)
            return
        now = _now()
        self._conn.execute(
            "UPDATE operations SET state = 'CANCELING', cancel_reason = ?, "
            "heartbeat_at = ? WHERE operation_id = ?",
            (reason, now, operation_id),
        )
        self._emit(
            row["task_id"],
            "operation.canceling",
            {"operation_id": operation_id, "reason": reason},
            operation_id=operation_id,
        )
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
                    operation_id,
                    exc_info=True,
                )
            return
        proc = self._procs.get(operation_id)
        identity = ProcessIdentity.parse(str(row.get("process_identity_json") or ""))
        # The current process handle can confirm an already-reaped short child.
        # Across restart, a numeric PID alone never authorizes signaling.
        if proc is not None and proc.returncode is not None:
            try:
                stopped = identity is not None and not session_members(identity)
            except (OSError, ValueError):
                stopped = False
        else:
            stopped = await self._stop_owned_session(identity, int(row.get("pid") or 0))
        if not stopped:
            self._conn.execute(
                "UPDATE operations SET failure_code = ? WHERE operation_id = ?",
                (OperationCancellationUnresolvedError.code, operation_id),
            )
            self._emit(
                row["task_id"],
                "operation.cancel_unresolved",
                {
                    "operation_id": operation_id,
                    "code": OperationCancellationUnresolvedError.code,
                    "stop_confirmed": False,
                },
                operation_id=operation_id,
            )
            raise OperationCancellationUnresolvedError(operation_id)
        if proc is not None:
            await proc.wait()
        self._procs.pop(operation_id, None)
        self._emit(
            row["task_id"],
            "operation.process_stopped",
            {
                "operation_id": operation_id,
                "stop_confirmed": True,
                "pid": int(row.get("pid") or 0),
                "scope": "owned_session",
                "sid": identity.sid,
            },
            operation_id=operation_id,
        )
        await self._record_terminal(operation_id, "CANCELLED", failure_code=reason)

    async def _stop_owned_session(self, identity: ProcessIdentity | None, pid: int) -> bool:
        if identity is None or identity.pid != pid or not identity.matches():
            return False
        try:
            members = session_members(identity)
            if not identity.matches():
                return False
            if not signal_owned_members(members, signal.SIGTERM):
                return False
            for phase in range(2):
                deadline = asyncio.get_running_loop().time() + _CANCEL_GRACE_S
                while session_members(identity):
                    if asyncio.get_running_loop().time() >= deadline:
                        break
                    await asyncio.sleep(0.02)
                else:
                    return True
                if phase == 0:
                    # TERM may kill the wrapper while a child ignores it. Only
                    # exact captured kernel birth identities allow escalation;
                    # an uncaptured/reused PID never authorizes signaling.
                    if any(
                        not any(known.same_birth(member) for known in members)
                        for member in session_members(identity)
                    ):
                        return False
                    if not signal_owned_members(members, signal.SIGKILL):
                        return False
            return False
        except (OSError, ValueError, NotImplementedError):
            return False

    async def cancel_many(self, operation_ids: list[str], *, reason: str = "user") -> dict:
        """Typed partial cancellation report; never count unresolved stops."""
        cancelled = 0
        unresolved = []
        for operation_id in operation_ids:
            before = self.get(operation_id)
            try:
                await self.cancel(operation_id, reason=reason)
            except OperationCancellationUnresolvedError:
                unresolved.append(operation_id)
            else:
                cancelled += (
                    before.get("state") not in OPERATION_TERMINAL
                    and self.get(operation_id).get("state") == "CANCELLED"
                )
        return {
            "ok": not unresolved,
            "operations_cancelled": cancelled,
            "operations_unresolved": unresolved,
            "code": OperationCancellationUnresolvedError.code if unresolved else "",
        }

    def stop_confirmation_missing(self, row: dict) -> bool:
        """Legacy CANCELLED is a ledger fact, not retrospective stop evidence."""
        if (
            row.get("state") != "CANCELLED"
            or row.get("provider") != "process"
            or not row.get("pid")
        ):
            return False
        return (
            self._conn.execute(
                "SELECT 1 FROM task_events WHERE operation_id = ? "
                "AND event_type = 'operation.process_stopped' LIMIT 1",
                (row["operation_id"],),
            ).fetchone()
            is None
        )

    async def _cancel_grace(self, operation_id: str, reason: str, grace_s: float = 5.0) -> None:
        """Action 取消宽限：action_result 未在宽限内到达也落
        CANCELLED（账本不再悬空）。"""
        await asyncio.sleep(grace_s)
        await self._record_terminal(operation_id, "CANCELLED", failure_code=reason)

    # --------------------------------------------------------------
    # liveness（§12.2：只标 DEGRADED，永不 kill）
    # --------------------------------------------------------------
    async def sweep_liveness(self, *, stale_after_s: float = 30.0) -> dict:
        """heartbeat 过期 → DEGRADED；恢复 → RUNNING。绝不杀进程。"""
        degraded = resumed = 0
        now = datetime.now(UTC).timestamp()
        rows = self._conn.execute(
            "SELECT operation_id, task_id, state, heartbeat_at, failure_code FROM operations "
            "WHERE state IN ('QUEUED', 'ADMITTED', 'RUNNING', 'DEGRADED')",
        ).fetchall()
        for row in rows:
            if row["failure_code"] in (
                "PROCESS_IDENTITY_UNVERIFIED",
                "OUTPUT_OBSERVATION_UNRESOLVED",
            ):
                continue  # Unverified recovery is not a heartbeat-based resume.
            heartbeat = datetime.fromisoformat(str(row["heartbeat_at"])).timestamp()
            stale = (now - heartbeat) > stale_after_s
            if stale and row["state"] != "DEGRADED":
                self._transition(
                    row["operation_id"],
                    "DEGRADED",
                    event="operation.degraded",
                    payload={"stale_after_s": stale_after_s},
                )
                degraded += 1
            elif not stale and row["state"] == "DEGRADED":
                self._transition(
                    row["operation_id"], "RUNNING", event="operation.resumed", payload={}
                )
                resumed += 1
        return {"degraded": degraded, "resumed": resumed}

    # --------------------------------------------------------------
    # 重启恢复（reattach-or-LOST）
    # --------------------------------------------------------------
    async def recover_on_boot(self) -> dict:
        """agentd 重启后的 operation 对账。

        - pid 活且持久身份一致 → reattach；无法证明归属则 unresolved；
        - pid 死 + exitcode 文件 → 应用真实终态；
        - 否则 → 诚实 LOST（事件留痕，绝不留僵尸 RUNNING）。
        """
        report = {
            "reattached": 0,
            "terminated": 0,
            "lost": 0,
            "unresolved": 0,
            "terminal_output_drained": 0,
            "terminal_output_pending": 0,
        }
        rows = self._conn.execute(
            "SELECT * FROM operations "
            "WHERE state IN ('QUEUED', 'ADMITTED', 'RUNNING', 'DEGRADED', "
            "'CANCELING') OR output_path != ''",
        ).fetchall()
        for row in rows:
            op_id = str(row["operation_id"])
            pid = int(row["pid"] or 0)
            if row["output_path"]:
                if row["state"] in OPERATION_TERMINAL:
                    try:
                        if json.loads(row["output_checkpoint_json"]).get("finalized") is True:
                            continue
                        stream, _ = self._open_spool(dict(row))
                        with stream:
                            writers_closed = self._writers_closed(stream.fileno())
                        if not writers_closed:
                            self._emit(
                                str(row["task_id"]),
                                "operation.output_pending",
                                {
                                    "scope": "spool_observation",
                                    "terminal_state": row["state"],
                                    "output_finalized": False,
                                    "additional_stop_authority": False,
                                },
                                operation_id=op_id,
                            )
                            self._drivers[op_id] = asyncio.create_task(self._drive_spool(op_id))
                            report["terminal_output_pending"] += 1
                            continue
                        await self._drive_spool(op_id)
                        finished = json.loads(self.get(op_id)["output_checkpoint_json"])[
                            "finalized"
                        ]
                        report["terminal_output_drained" if finished else "unresolved"] += 1
                    except (OSError, ValueError, KeyError, TypeError):
                        report["unresolved"] += 1
                    continue
                identity = ProcessIdentity.parse(str(row["process_identity_json"] or ""))
                if self._pid_alive(pid) and (
                    identity is None or identity.pid != pid or not identity.matches()
                ):
                    if row["state"] != "CANCELING":
                        self._conn.execute(
                            "UPDATE operations SET failure_code = ? WHERE operation_id = ?",
                            ("PROCESS_IDENTITY_UNVERIFIED", op_id),
                        )
                        self._transition(op_id, "DEGRADED", event=None)
                    self._emit(
                        str(row["task_id"]),
                        "operation.recovery_unresolved",
                        {
                            "code": "PROCESS_IDENTITY_UNVERIFIED",
                            "pid": pid,
                            "stop_confirmed": False,
                        },
                        operation_id=op_id,
                    )
                    report["unresolved"] += 1
                    continue
                if row["state"] == "CANCELING" and self._pid_alive(pid):
                    try:
                        await self.cancel(op_id, reason=str(row["cancel_reason"] or "user"))
                    except OperationCancellationUnresolvedError:
                        report["unresolved"] += 1
                    else:
                        report["terminated"] += 1
                    continue
                try:
                    stream, _ = self._open_spool(dict(row))
                    with stream:
                        writers_closed = self._writers_closed(stream.fileno())
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self._conn.execute(
                        "UPDATE operations SET failure_code = ? WHERE operation_id = ?",
                        ("OUTPUT_OBSERVATION_UNRESOLVED", op_id),
                    )
                    self._transition(op_id, "DEGRADED", event=None)
                    self._emit(
                        str(row["task_id"]),
                        "operation.output_unresolved",
                        {"error": str(exc)[:300]},
                        operation_id=op_id,
                    )
                    report["unresolved"] += 1
                    continue
                if writers_closed and not self._pid_alive(pid):
                    # Finished producers are drained before boot recovery returns,
                    # preserving the existing terminal reconciliation contract.
                    await self._drive_spool(op_id)
                    state = self.get(op_id)["state"]
                    report[
                        "lost"
                        if state == "LOST"
                        else "terminated"
                        if state in OPERATION_TERMINAL
                        else "unresolved"
                    ] += 1
                else:
                    self._emit(
                        str(row["task_id"]),
                        "operation.reattached",
                        {
                            "pid": pid,
                            "scope": "spool_observation",
                            "process_identity_verified": bool(identity and identity.matches()),
                            "output_replay": True,
                        },
                        operation_id=op_id,
                    )
                    self._drivers[op_id] = asyncio.create_task(self._drive_spool(op_id))
                    report["reattached"] += 1
                continue
            exitcode = self._read_exitcode(str(row["exitcode_path"] or ""))
            if exitcode is not None:
                await self._record_terminal(
                    op_id,
                    "SUCCEEDED" if exitcode == 0 else "FAILED",
                    failure_code="" if exitcode == 0 else f"exit_{exitcode}",
                )
                report["terminated"] += 1
                continue
            if pid > 0 and self._pid_alive(pid):
                identity = ProcessIdentity.parse(str(row["process_identity_json"] or ""))
                if identity is None or identity.pid != pid or not identity.matches():
                    if row["state"] != "CANCELING":
                        self._conn.execute(
                            "UPDATE operations SET failure_code = 'PROCESS_IDENTITY_UNVERIFIED' "
                            "WHERE operation_id = ?",
                            (op_id,),
                        )
                        self._transition(op_id, "DEGRADED", event=None)
                    # Existing CANCELING and its unresolved-stop code remain
                    # intact. The general transition guard also protects this;
                    # recovery must never reclassify it as heartbeat-resumable.
                    self._emit(
                        str(row["task_id"]),
                        "operation.recovery_unresolved",
                        {
                            "pid": pid,
                            "stop_confirmed": False,
                            "code": "PROCESS_IDENTITY_UNVERIFIED",
                        },
                        operation_id=op_id,
                    )
                    report["unresolved"] += 1
                    continue
                if row["state"] == "CANCELING":
                    try:
                        await self.cancel(op_id, reason=str(row["cancel_reason"] or "user"))
                    except OperationCancellationUnresolvedError:
                        report["unresolved"] += 1
                    else:
                        report["terminated"] += 1
                    continue
                self._transition(
                    op_id, "DEGRADED", event="operation.reattached", payload={"pid": pid}
                )
                self._drivers[op_id] = asyncio.create_task(
                    self._watch_pid(
                        op_id, str(row["task_id"]), pid, str(row["exitcode_path"] or "")
                    )
                )
                report["reattached"] += 1
                continue
            await self._record_terminal(
                op_id,
                "LOST",
                failure_code="restart_unverifiable",
            )
            report["lost"] += 1
        return report

    async def _watch_pid(
        self, operation_id: str, task_id: str, pid: int, exitcode_path: str
    ) -> None:
        """reattach 后的 pid-watcher：等进程消失 → exitcode 定终态
        （无 exitcode = 诚实 LOST——退出码不可考）。"""
        identity = ProcessIdentity.parse(
            str(self.get(operation_id).get("process_identity_json") or "")
        )
        while identity is not None and identity.pid == pid and identity.matches():
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
                operation_id,
                "SUCCEEDED" if exitcode == 0 else "FAILED",
                failure_code="" if exitcode == 0 else f"exit_{exitcode}",
            )
        else:
            await self._record_terminal(
                operation_id,
                "LOST",
                failure_code="exit_unverifiable",
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
        self._emit(
            str(row["task_id"]),
            "operation.progress",
            {"operation_id": operation_id, "progress": progress},
            operation_id=operation_id,
        )

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
        db_file = self._conn.execute("PRAGMA database_list").fetchone()["file"]
        directory = Path(str(db_file)).parent / "operations"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        return directory

    def _transition(
        self,
        operation_id: str,
        state: str,
        *,
        event: str | None,
        payload: dict | None = None,
    ) -> None:
        row = self.get(operation_id)
        # 终态不可逆；CANCELING 同样不可被普通迁移覆盖（取消流程持有
        # 账本——driver 迟到置 RUNNING 不得踩掉 CANCELING）。
        if (
            not row
            or row["state"] in OPERATION_TERMINAL
            or (row["state"] == "CANCELING" and state != "CANCELLED")
        ):
            return
        self._conn.execute(
            "UPDATE operations SET state = ?, heartbeat_at = ? WHERE operation_id = ?",
            (state, _now(), operation_id),
        )
        if event:
            self._emit(
                str(row["task_id"]),
                event,
                {"operation_id": operation_id, **(payload or {})},
                operation_id=operation_id,
            )

    def _touch(self, operation_id: str, task_id: str) -> None:
        """heartbeat——仅非终态（终态后心跳冻结）。"""
        row = self.get(operation_id)
        if not row or row["state"] in OPERATION_TERMINAL:
            return
        self._conn.execute(
            "UPDATE operations SET heartbeat_at = ? WHERE operation_id = ?",
            (_now(), operation_id),
        )

    def _emit(
        self,
        task_id: str,
        event_type: str,
        payload: dict,
        *,
        operation_id: str = "",
        attempt_id: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT INTO task_events (task_id, attempt_id, operation_id, "
            "event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                task_id,
                attempt_id or None,
                operation_id or None,
                event_type,
                json.dumps(payload, ensure_ascii=False),
                _now(),
            ),
        )
