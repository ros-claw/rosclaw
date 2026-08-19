"""PR-H2 红测试：TaskKernel 输入事务（总纲 v2 §9.3 + Gate B）。

红测试先行——修复前必须红：
1. 首条用户消息 → 创建 root task + 固定 workspace + revision 1 +
   active primary binding；
2. 同一活跃 task 的后续消息 → revision+1（不是新 task）；
3. message_id 重放 → 幂等（不重复建 task/revision）；
4. /done 终态后的新消息 → 新 task；
5. 一个 task 同时只能有一个 active primary session（DB 唯一索引）；
6. 终态不可被覆盖（CANCELLED 不被迟到事件涂改）；
7. 不用字符串相似度决定 task 身份（身份只来自输入事务）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _kernel(tmp_path: Path):
    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    db_path = tmp_path / "k.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return conn, TaskKernel(conn, tmp_path)


class TestInputTransaction:
    def test_first_message_creates_root_task(self, tmp_path: Path) -> None:
        conn, kernel = _kernel(tmp_path)
        result = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_1", text="画一个五角星", cwd=str(tmp_path),
        )
        assert result["created_task"] and result["revision"] == 1
        workspace = Path(result["workspace_path"])
        assert workspace.exists()
        assert (workspace.parent / "manifest.json").exists()
        task = kernel.get_task(result["task_id"])
        assert task["state"] == "RUNNING"
        assert task["root_goal"] == "画一个五角星"

    def test_followup_message_is_revision_not_new_task(self, tmp_path: Path) -> None:
        """用户纠错/追问 → 同一 task 的 revision（工作区/会话不变）。"""
        conn, kernel = _kernel(tmp_path)
        first = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_1", text="画五角星", cwd=str(tmp_path),
        )
        second = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_2", text="视频里看不到机械臂，修好",
            cwd=str(tmp_path),
        )
        assert second["task_id"] == first["task_id"]
        assert second["revision"] == 2
        assert not second["created_task"]
        assert second["workspace_path"] == first["workspace_path"]
        rows = conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()
        assert rows["c"] == 1, "纠错不得裂变 root task"

    def test_message_replay_idempotent(self, tmp_path: Path) -> None:
        conn, kernel = _kernel(tmp_path)
        first = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_1", text="画五角星", cwd=str(tmp_path),
        )
        replay = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_1", text="画五角星", cwd=str(tmp_path),
        )
        assert replay["replayed"] and replay["task_id"] == first["task_id"]
        assert conn.execute(
            "SELECT COUNT(*) AS c FROM task_revisions"
        ).fetchone()["c"] == 1

    def test_new_task_after_done(self, tmp_path: Path) -> None:
        conn, kernel = _kernel(tmp_path)
        first = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_1", text="画五角星", cwd=str(tmp_path),
        )
        kernel.transition(first["task_id"], "SUCCEEDED", reason="verified")
        second = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_2", text="再画一个圆", cwd=str(tmp_path),
        )
        assert second["created_task"] and second["task_id"] != first["task_id"]

    def test_force_new_creates_new_task(self, tmp_path: Path) -> None:
        """/new 显式开新任务（旧 binding 失活——一个 task 一个活跃
        primary 的唯一索引不破）。"""
        conn, kernel = _kernel(tmp_path)
        first = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_1", text="任务A", cwd=str(tmp_path),
        )
        second = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_2", text="任务B", cwd=str(tmp_path),
            force_new=True,
        )
        assert second["created_task"] and second["task_id"] != first["task_id"]
        active = conn.execute(
            "SELECT COUNT(*) AS c FROM task_session_bindings "
            "WHERE active = 1 AND role = 'primary'"
        ).fetchone()["c"]
        assert active == 1

    def test_one_active_primary_enforced(self, tmp_path: Path) -> None:
        """DB 唯一索引：直接插第二个 active primary binding 必须失败。"""
        conn, kernel = _kernel(tmp_path)
        result = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_1", text="任务", cwd=str(tmp_path),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO task_session_bindings (task_id, session_ref, "
                "role, active, created_at) VALUES (?, 's2', 'primary', 1, 'now')",
                (result["task_id"],),
            )

    def test_terminal_not_overwritten(self, tmp_path: Path) -> None:
        conn, kernel = _kernel(tmp_path)
        result = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_1", text="任务", cwd=str(tmp_path),
        )
        kernel.transition(result["task_id"], "CANCELLED", reason="用户取消")
        kernel.transition(result["task_id"], "SUCCEEDED", reason="迟到完成")
        task = kernel.get_task(result["task_id"])
        assert task["state"] == "CANCELLED", "迟到完成不得覆盖 CANCELLED"
