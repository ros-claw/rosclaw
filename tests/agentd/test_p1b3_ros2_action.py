"""P1-B3 红测试（0824 总纲 §12/P1-B）：ROS 2 Action provider adapter。

真实缺口：ROS 连接只有同步 request/response——没有 Action client
（send_goal/feedback/result/cancel），长 Action 无法纳入 Operation
统一运营层（goal 生命周期/progress/cancel 握手/终态）。

断言（Operation 同一契约的 Action provider）：
1. send_goal → operation QUEUED→ADMITTED→RUNNING，goal_id 落账；
2. feedback → operation.progress（progress_json 更新 + 事件）；
3. result(SUCCEEDED=4) → operation SUCCEEDED + result_ref；
4. result(ABORTED=6/其他) → FAILED；
5. cancel → CANCELING → cancel_goal → result(CANCELED=5) → CANCELLED；
6. 2h 假 Action（持续 feedback、无 result）→ sweep 永不 kill；
7. 重启：ros2_action provider 无 pid → 诚实 LOST（不可证实）。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.operation_manager import (
    OPERATION_TERMINAL,
    OperationManager,
)


def _conn(tmp_path: Path) -> sqlite3.Connection:
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


def _events(conn: sqlite3.Connection, task_id: str = "task_1") -> list[str]:
    rows = conn.execute(
        "SELECT event_type FROM task_events WHERE task_id = ? ORDER BY seq",
        (task_id,),
    ).fetchall()
    return [r["event_type"] for r in rows]


class FakeActionClient:
    """协议级假 Action client（与 Ros2ActionClient 同接口）。"""

    def __init__(self) -> None:
        self.sent_goals: list[dict] = []
        self.cancelled: list[str] = []
        self._feedback_cb = None
        self._result_cb = None

    def send_goal(
        self, *, action: str, action_type: str, args: dict, goal_id: str,
        on_feedback, on_result,
    ) -> None:
        self.sent_goals.append(
            {"action": action, "action_type": action_type,
             "args": args, "goal_id": goal_id}
        )
        self._feedback_cb = on_feedback
        self._result_cb = on_result

    def cancel_goal(self, goal_id: str) -> None:
        self.cancelled.append(goal_id)

    # -- 测试驱动 --
    def emit_feedback(self, values: dict) -> None:
        self._feedback_cb(values)

    def emit_result(self, status: int, values: dict) -> None:
        self._result_cb(status, values)


SUCCEEDED, CANCELED, ABORTED = 4, 5, 6


class TestActionOperationLifecycle:
    def _start(self, conn, client):
        mgr = OperationManager(None, conn)

        async def run():
            return await mgr.start_action(
                task_id="task_1", attempt_id="",
                action="/fibonacci",
                action_type="action_tutorials_interfaces/action/Fibonacci",
                args={"order": 5},
                client=client,
            )

        return mgr, asyncio.run(run())

    def test_send_goal_registers_operation(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        client = FakeActionClient()
        mgr, op = self._start(conn, client)
        assert len(client.sent_goals) == 1
        goal = client.sent_goals[0]
        assert goal["action"] == "/fibonacci"
        assert op["goal_id"] == goal["goal_id"]
        assert op["provider"] == "ros2_action"
        assert op["state"] in ("ADMITTED", "RUNNING")
        types = _events(conn)
        assert "operation.queued" in types
        assert "operation.admitted" in types

    def test_feedback_becomes_progress(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        client = FakeActionClient()
        mgr, op = self._start(conn, client)
        client.emit_feedback({"sequence": [0, 1, 1]})
        row = mgr.get(op["operation_id"])
        progress = json.loads(row["progress_json"])
        assert progress, "feedback 未落成 progress"
        assert "operation.progress" in _events(conn)

    def test_result_succeeded_completes(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        client = FakeActionClient()
        mgr, op = self._start(conn, client)
        client.emit_result(SUCCEEDED, {"sequence": [0, 1, 1, 2, 3]})
        row = mgr.get(op["operation_id"])
        assert row["state"] == "SUCCEEDED"
        assert row["result_ref"], "缺 result_ref"
        assert "operation.completed" in _events(conn)

    def test_result_aborted_fails(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        client = FakeActionClient()
        mgr, op = self._start(conn, client)
        client.emit_result(ABORTED, {})
        row = mgr.get(op["operation_id"])
        assert row["state"] == "FAILED"

    def test_cancel_handshake(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        client = FakeActionClient()
        mgr, op = self._start(conn, client)

        async def cancel():
            await mgr.cancel(op["operation_id"], reason="user-stop")

        asyncio.run(cancel())
        assert client.cancelled == [op["goal_id"]], "cancel_goal 未发"
        # result(CANCELED) 到达 → CANCELLED 终态。
        client.emit_result(CANCELED, {})
        row = mgr.get(op["operation_id"])
        assert row["state"] == "CANCELLED"
        assert row["cancel_reason"] == "user-stop"
        types = _events(conn)
        assert "operation.canceling" in types
        assert "operation.cancelled" in types

    def test_two_hour_action_never_killed(self, tmp_path: Path) -> None:
        """验收：持续 feedback 的长 Action——sweep 永不 kill（无默认
        wall-clock kill）。"""
        conn = _conn(tmp_path)
        _task(conn)
        client = FakeActionClient()
        mgr, op = self._start(conn, client)

        async def run():
            for _ in range(5):
                client.emit_feedback({"sequence": [0, 1]})
                await mgr.sweep_liveness(stale_after_s=1.0)
                row = mgr.get(op["operation_id"])
                assert row["state"] in ("RUNNING", "ADMITTED")
            return op["operation_id"]

        op_id = asyncio.run(run())
        assert mgr.get(op_id)["state"] not in OPERATION_TERMINAL

    def test_restart_marks_action_lost_honestly(self, tmp_path: Path) -> None:
        """ros2_action 无 pid——重启不可证实 → LOST（不假装存活）。"""
        conn = _conn(tmp_path)
        _task(conn)
        client = FakeActionClient()
        mgr, op = self._start(conn, client)
        mgr2 = OperationManager(None, conn)
        report = asyncio.run(mgr2.recover_on_boot())
        row = mgr2.get(op["operation_id"])
        assert row["state"] == "LOST"
        assert report["lost"] >= 1
        assert "operation.lost" in _events(conn)
