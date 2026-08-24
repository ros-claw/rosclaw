"""P0-C 红测试（0824 总纲 §19.P0-C）：Conversation/Task 分离与自动 admission。

红测试先行——persist_input/ensure_task_for_effect 不存在时必须红。

验收（文档原文）：
- `hello` 后 tasks=0；
- 提问状态仍 tasks=0；
- 第一次 SIM action 前 task 已持久化；
- 输入丢失/重复均不会重复执行。

§6.3 Revision 规则：活跃任务中的 steer/follow-up 同 task 新
revision；同一输入的多个 effectful call 不重复 bump。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _kernel(home: Path):
    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, home), conn


def _task_count(conn) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"])


class TestPersistOnlyInputs:
    def test_hello_creates_no_task(self, tmp_path: Path) -> None:
        kernel, conn = _kernel(tmp_path)
        kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_1", text="hello",
        )
        assert _task_count(conn) == 0, "问候创建了 Task——职责错位"

    def test_status_query_creates_no_task(self, tmp_path: Path) -> None:
        kernel, conn = _kernel(tmp_path)
        kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_1", text="现在进度如何？",
        )
        kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_2", text="能力列表给我看看",
        )
        assert _task_count(conn) == 0

    def test_persist_input_idempotent(self, tmp_path: Path) -> None:
        """同一 message_id 重复 persist（输入重发/重放）只落一次——
        输入丢失/重复均不会重复执行。"""
        kernel, conn = _kernel(tmp_path)
        first = kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_1", text="画五角星",
        )
        second = kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_1", text="画五角星",
        )
        assert first["input_id"] == second["input_id"]
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM user_inputs WHERE message_id='msg_1'"
        ).fetchone()
        assert int(rows["n"]) == 1


class TestEnsureTaskForEffect:
    def test_first_effectful_call_creates_task(self, tmp_path: Path) -> None:
        """首个 effectful call 前 task 已持久化（root_goal 来自动机
        输入）。"""
        kernel, conn = _kernel(tmp_path)
        kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_1", text="帮我用机械臂画一个五角星",
        )
        result = kernel.ensure_task_for_effect(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            cwd=str(tmp_path),
        )
        assert result["task_id"], "effectful call 未原子建 task"
        task = kernel.get_task(result["task_id"])
        assert "五角星" in str(task["root_goal"])
        assert int(task["active_revision"]) == 1

    def test_effectful_calls_same_input_no_revision_bump(
        self, tmp_path: Path
    ) -> None:
        """同一动机输入的连续 effectful call（plan→simulate→render）
        共享 task revision——不得每次调用 bump。"""
        kernel, _conn = _kernel(tmp_path)
        kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_1", text="画五角星",
        )
        first = kernel.ensure_task_for_effect(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            cwd=str(tmp_path),
        )
        second = kernel.ensure_task_for_effect(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            cwd=str(tmp_path),
        )
        assert first["task_id"] == second["task_id"]
        assert first["revision"] == second["revision"] == 1, (
            "同一输入的 effectful 链被重复 bump revision"
        )

    def test_new_input_during_active_task_revises(
        self, tmp_path: Path
    ) -> None:
        """活跃任务中的新输入（steer/follow-up）→ 同 task 新 revision。"""
        kernel, _conn = _kernel(tmp_path)
        kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_1", text="画五角星",
        )
        first = kernel.ensure_task_for_effect(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            cwd=str(tmp_path),
        )
        kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_2", text="画大一点",
        )
        second = kernel.ensure_task_for_effect(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            cwd=str(tmp_path),
        )
        assert second["task_id"] == first["task_id"]
        assert second["revision"] == 2

    def test_ensure_without_input_fails_honest(self, tmp_path: Path) -> None:
        """无任何持久化输入时的 effectful call——诚实拒绝（无动机
        输入不猜目标）。"""
        kernel, _conn = _kernel(tmp_path)
        with pytest.raises(ValueError, match="INPUT"):
            kernel.ensure_task_for_effect(
                mission_id="mis_1", session_ref="s1", backend_native_id="s1",
                cwd=str(tmp_path),
            )

    def test_explicit_goal_begin(self, tmp_path: Path) -> None:
        """/goal 提前创建 TaskSpec——不等首个 effectful call。"""
        kernel, _conn = _kernel(tmp_path)
        kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_1", text="/goal 画一个五角星",
        )
        result = kernel.ensure_task_for_effect(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            cwd=str(tmp_path), explicit_goal="画一个五角星",
        )
        task = kernel.get_task(result["task_id"])
        assert task is not None
