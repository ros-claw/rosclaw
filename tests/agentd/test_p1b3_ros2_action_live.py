"""P1-B3 live 旅程（0824 总纲 §12/P1-B）：ROS 2 Action 全链路。

协议级假 rosbridge action server（websockets——忠实 rosbridge action
协议：send_goal → action_feedback 推送 → action_result；cancel_goal
→ action_result(CANCELED)）+ 真实 RosbridgeTransport + 真实
Ros2ActionClient + 真实 OperationManager（SQLite 账本）：

1. goal → QUEUED→ADMITTED→RUNNING；goal_id 落账；
2. feedback → operation.progress（progress_json + 事件）；
3. result(SUCCEEDED) → operation SUCCEEDED + result_ref；
4. 第二个 goal cancel → CANCELING → action_result(CANCELED) →
   CANCELLED（终态不被迟到事件覆盖）。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

import websockets

from rosclaw.connectors.ros.action_client import Ros2ActionClient
from rosclaw.connectors.ros.transport.base import RosbridgeEndpoint
from rosclaw.connectors.ros.transport.rosbridge import RosbridgeTransport
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.operation_manager import OperationManager

SUCCEEDED, CANCELED = 4, 5


class FakeRosbridgeActionServer:
    """协议级假 rosbridge action server（Fibonacci 语义）。"""

    def __init__(self) -> None:
        self.goals: list[dict] = []
        self.cancelled: list[str] = []
        self._loop = asyncio.new_event_loop()
        self._port = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._port and time.monotonic() < deadline:
            time.sleep(0.05)

    @property
    def port(self) -> int:
        return self._port

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        running: dict[str, asyncio.Task] = {}

        async def fibonacci(ws, goal_id: str, order: int) -> None:
            sequence = [0, 1]
            for _ in range(max(1, min(order, 8))):
                sequence.append(sequence[-1] + sequence[-2])
                await ws.send(json.dumps({
                    "op": "action_feedback", "id": goal_id,
                    "values": {"sequence": list(sequence)},
                }))
                await asyncio.sleep(0.4)
            await ws.send(json.dumps({
                "op": "action_result", "id": goal_id,
                "values": {"status": SUCCEEDED,
                           "result": {"sequence": sequence}},
            }))

        async def handler(ws) -> None:
            # 并发语义（与真 rosbridge 一致）：goal 执行与消息读取并行——
            # cancel 可在执行中到达。
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("op") == "send_goal":
                    self.goals.append(msg)
                    goal_id = msg["id"]
                    order = int(msg.get("args", {}).get("order", 3))
                    running[goal_id] = asyncio.create_task(
                        fibonacci(ws, goal_id, order)
                    )
                elif msg.get("op") == "cancel_goal":
                    goal_id = str(msg.get("id", ""))
                    self.cancelled.append(goal_id)
                    task = running.pop(goal_id, None)
                    if task is not None:
                        task.cancel()
                    await ws.send(json.dumps({
                        "op": "action_result", "id": goal_id,
                        "values": {"status": CANCELED, "result": {}},
                    }))

        self._server = await websockets.serve(handler, "127.0.0.1", 0)
        self._port = self._server.sockets[0].getsockname()[1]
        await self._server.wait_closed()

    def close(self) -> None:
        server = getattr(self, "_server", None)
        if server is not None:
            self._loop.call_soon_threadsafe(server.close)
            deadline = time.monotonic() + 5
            while self._thread.is_alive() and time.monotonic() < deadline:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=0.5)
        else:
            self._loop.call_soon_threadsafe(self._loop.stop)


def _wait(predicate, timeout: float = 10.0, label: str = "") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"timeout waiting: {label}")


class TestRos2ActionLiveJourney:
    def test_full_action_lifecycle(self, tmp_path: Path) -> None:
        server = FakeRosbridgeActionServer()
        conn = sqlite3.connect(tmp_path / "missions.db", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, "sqlite")
        now = "2026-08-25T00:00:00+00:00"
        conn.execute(
            "INSERT INTO tasks (task_id, mission_id, root_goal, mode, body_id, "
            "state, active_revision, workspace_path, created_at, updated_at) "
            "VALUES ('task_1', 'm1', 'goal', 'SIMULATION', '', 'ACTIVE', 1, "
            "'', ?, ?)",
            (now, now),
        )
        conn.commit()
        transport = RosbridgeTransport(
            RosbridgeEndpoint(host="127.0.0.1", port=server.port)
        )
        client = Ros2ActionClient(transport)
        mgr = OperationManager(None, conn)
        try:
            async def run() -> dict:
                op = await mgr.start_action(
                    task_id="task_1", attempt_id="",
                    action="/fibonacci",
                    action_type="action_tutorials_interfaces/action/Fibonacci",
                    args={"order": 4},
                    client=client,
                )
                # cancel 链路（第二个 goal）。
                op2 = await mgr.start_action(
                    task_id="task_1", attempt_id="",
                    action="/fibonacci",
                    action_type="action_tutorials_interfaces/action/Fibonacci",
                    args={"order": 8},
                    client=client,
                )
                await mgr.cancel(op2["operation_id"], reason="journey-stop")
                return op

            op = asyncio.run(run())
            op_id = op["operation_id"]

            _wait(lambda: mgr.get(op_id)["state"] == "SUCCEEDED",
                  label="goal1 SUCCEEDED")
            row = mgr.get(op_id)
            assert row["goal_id"], "缺 goal_id"
            assert row["provider"] == "ros2_action"
            assert row["result_ref"], "缺 result_ref"
            progress = json.loads(row["progress_json"])
            assert progress.get("sequence"), f"feedback 未落成 progress: {row}"

            rows = conn.execute(
                "SELECT operation_id, state, cancel_reason FROM operations "
                "WHERE operation_id != ?",
                (op_id,),
            ).fetchall()
            assert len(rows) == 1
            _wait(lambda: conn.execute(
                "SELECT state FROM operations WHERE operation_id = ?",
                (rows[0]["operation_id"],),
            ).fetchone()["state"] == "CANCELLED", label="goal2 CANCELLED")
            final = conn.execute(
                "SELECT state, cancel_reason FROM operations "
                "WHERE operation_id = ?",
                (rows[0]["operation_id"],),
            ).fetchone()
            assert final["state"] == "CANCELLED"
            assert final["cancel_reason"] == "journey-stop"
            assert server.cancelled, "cancel_goal 未到服务端"

            types = [
                r["event_type"]
                for r in conn.execute(
                    "SELECT event_type FROM task_events ORDER BY seq"
                ).fetchall()
            ]
            for expected in (
                "operation.queued", "operation.admitted", "operation.progress",
                "operation.completed", "operation.canceling",
                "operation.cancelled",
            ):
                assert expected in types, f"事件链缺 {expected}"
        finally:
            client.close()
            server.close()
