"""G-4（0916 三审 B-2）：取消传播闭环。

三审稿（纠正后）：取消能力不缺工具（process_stop 早已
model-visible），缺的是**传播**——Esc/Ctrl-C/自然语言停止到
TaskController→operation+子进程+模型回合全停；迟到成功不得
翻转 CANCELLED（账本防护已在位——operation_manager._write_terminal
终态拒写 + CANCELING 拒非 CANCELLED 写，本文件不再重复钉）。

断点（代码地图实证）：
1. 杀进程只打 sh 包装——孙子进程（渲染/仿真/xvfb-run）取消后
   成孤儿（start_new_session 的 pgid 本来为此隔离却没人 killpg）；
2. pi.task.cancel 只落 task 账——在途 operation 照跑；
3. 自然语言"停下来/别做了"零拦截——指望模型志愿调 process_stop；
4. Esc 只停 pi 模型回合（vendored 内部 abort），后台 operation
   无感（TS 侧级联见 packages/rosclaw-agent extension）。

闭环断言：进程组全灭 / 级联取消 / NL-stop 确定性拦截。
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.server import _match_stop_intent
from rosclaw.task_kernel.operation_manager import OperationManager
from tests.agentd.test_p1b1_operation_v2 import _conn, _task


class TestProcessGroupKill:
    def test_cancel_kills_grandchildren(self, tmp_path: Path) -> None:
        """取消必须灭整个进程组——孙子进程不得成孤儿（killpg 实证）。"""
        conn = _conn(tmp_path)
        _task(conn)
        mgr = OperationManager(None, conn)
        pid_file = tmp_path / "grandchild.pid"

        async def run():
            op = await mgr.start(
                task_id="task_1", attempt_id="", kind="process",
                argv=[
                    "sh", "-c",
                    f"sleep 300 & echo $! > {pid_file}; wait",
                ],
            )
            # 等孙子进程落 pid 文件。
            for _ in range(100):
                if pid_file.exists():
                    break
                await asyncio.sleep(0.05)
            await mgr.cancel(op["operation_id"], reason="user")
            return op["operation_id"]

        op_id = asyncio.run(run())
        assert mgr.get(op_id)["state"] == "CANCELLED"
        grandchild = int(pid_file.read_text().strip())
        time.sleep(0.2)
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild, 0)  # 孙子必须死——不得孤儿存活


class TestStopIntentMatcher:
    @pytest.mark.parametrize("text", [
        "停下来", "停止", "别做了", "别做", "不要再做了", "取消", "取消吧",
        "取消任务", "stop", "cancel", "Cancel", "停下来，不要做了",
    ])
    def test_stop_commands_match(self, text: str) -> None:
        assert _match_stop_intent(text), text

    @pytest.mark.parametrize("text", [
        "帮我写一个取消订单的功能",  # 任务内容含"取消"——不是指令
        "画一个五角星", "继续", "取消订单接口怎么设计比较合理？",  # 长文本
        "渲染一个视频",
    ])
    def test_task_content_never_matches(self, text: str) -> None:
        assert not _match_stop_intent(text), text


class TestNlStopCascade:
    async def test_nl_stop_cancels_operations_and_suppresses(
        self, tmp_path: Path,
    ) -> None:
        """"别做了"→确定性级联：在途 operation 落 CANCELLED +
        suppress 模型回合 + cancel_report 回声（不进 Pi 再绕一圈）。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        conn = service._store.connection
        # 造一个在途 operation（归属本 mission 的 task）。
        task_id = "task_nl_1"
        conn.execute(
"INSERT OR IGNORE INTO tasks (task_id, mission_id, root_goal, mode, "
            "workspace_path, state, active_revision, locale, created_at, "
            "updated_at) VALUES (?, ?, 't', 'SIMULATION', '/tmp/ws', "
            "'EXECUTING', 1, 'zh', 'now', 'now')",
            (task_id, mission.mission_id),
        )
        conn.execute(
            "INSERT INTO operations (operation_id, task_id, attempt_id, kind, "
            "state, resumable, revision, started_at) "
            "VALUES ('op_nl_1', ?, '', 'process', 'RUNNING', 0, 1, 'now')",
            (task_id,),
        )
        conn.commit()

        async def _persist(text: str):
            return await bridge._dispatch(
                "user:local:1000", 1, "pi.input.persist",
                {
                    "token": service.control_token,
                    "mission_id": mission.mission_id,
                    "session_ref": "pi_1",
                    "message_id": "msg_nl_stop",
                    "text": text,
                },
            )

        result = await _persist("别做了")
        assert result.get("ok"), result
        disposition = result["turn_disposition"]
        assert disposition["suppress_model_turn"] is True
        assert disposition["owner"] == "TASK_ROUTER"
        report = disposition["cancel_report"]
        assert report["operations_cancelled"] == 1
        row = conn.execute(
            "SELECT state FROM operations WHERE operation_id = 'op_nl_1'",
        ).fetchone()
        assert row["state"] == "CANCELLED"

    async def test_task_content_with_cancel_word_not_intercepted(
        self, tmp_path: Path,
    ) -> None:
        """任务内容里的"取消"（长文本）不得触发拦截——归 Pi。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.input.persist",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "session_ref": "pi_1",
                "message_id": "msg_nl_task",
                "text": "帮我写一个取消订单的功能",
            },
        )
        disposition = result["turn_disposition"]
        assert disposition["owner"] == "PI_CONVERSATION"
        assert disposition["suppress_model_turn"] is False


class TestTaskCancelCascade:
    async def test_pi_task_cancel_cascades_operations(
        self, tmp_path: Path,
    ) -> None:
        """pi.task.cancel 落 task CANCELLED 且级联停在途 operation
        （此前 operation 照跑成孤儿）。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        conn = service._store.connection
        task_id = "task_cascade_1"
        conn.execute(
"INSERT INTO tasks (task_id, mission_id, root_goal, mode, "
            "workspace_path, state, active_revision, locale, created_at, "
            "updated_at) VALUES (?, ?, 't', 'SIMULATION', '/tmp/ws', "
            "'EXECUTING', 1, 'zh', 'now', 'now')",
            (task_id, mission.mission_id),
        )
        conn.execute(
            "INSERT INTO operations (operation_id, task_id, attempt_id, kind, "
            "state, resumable, revision, started_at) "
            "VALUES ('op_cascade_1', ?, '', 'process', 'RUNNING', 0, 1, 'now')",
            (task_id,),
        )
        conn.commit()
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.task.cancel",
            {"token": service.control_token, "task_id": task_id},
        )
        assert result.get("ok"), result
        assert result["operations_cancelled"] == 1
        row = conn.execute(
            "SELECT state FROM operations WHERE operation_id = 'op_cascade_1'",
        ).fetchone()
        assert row["state"] == "CANCELLED"
