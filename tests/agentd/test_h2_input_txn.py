"""PR-H2 产品 Gate（总纲 v2 §21 Gate B）：输入与 Root Task 一致性。

PTY `rosclaw chat` 真实输入：
1. 首条任务消息 → root task 创建（revision 1）+ 消息可见（不消失）；
2. 追问 → revision 2（同一 task，不裂变）；
3. /new 后的新目标 → 新 task；
4. work_orders/task_executions 恒为 0（Worker 不参与默认链）。
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from tests.agentd.test_h1_native_work import (
    _FakeServer,
    _prepare_home,
)
from tests.agentd.test_product_journey import PtySession

REPO = Path(__file__).resolve().parents[2]

import pytest  # noqa: E402

from rosclaw.agentd.pi_entry import find_pi_agent_entry  # noqa: E402

pytestmark = pytest.mark.skipif(
    not find_pi_agent_entry(),
    reason="无 Node/dist（CI 全回归 job 未构建）——诚实 skip",
)


class TestInputRootTaskGate:
    def test_input_binds_single_root_task(self, tmp_path: Path) -> None:
        fake = _FakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env, log_path=tmp_path / "pty-h2.log", cwd=workspace,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("画一个五角星\r")
            session.expect("已在同一会话直接完成".encode(), timeout=180)
            db = sqlite3.connect(home / "agentd" / "missions.db")
            tasks = db.execute(
                "SELECT task_id, state, active_revision FROM tasks"
            ).fetchall()
            db.close()
            assert len(tasks) == 1, f"首个目标必须只建一个 root task: {tasks}"
            task_id = tasks[0][0]
            assert tasks[0][2] == 1
            # 消息可见（不消失）：session JSONL 含用户文本。
            sessions = list((home / "agent" / "sessions").glob("*.jsonl"))
            assert sessions, "session JSONL 不存在"
            blob = sessions[0].read_text(encoding="utf-8", errors="replace")
            assert "画一个五角星" in blob, "用户消息幽灵消失"

            # 追问 → revision 2（同一 task）。
            marker = len(session.clean)
            session.send("改成画圆形\r")
            session.expect("已在同一会话直接完成".encode(), timeout=180,
                           after=marker)
            db = sqlite3.connect(home / "agentd" / "missions.db")
            tasks = db.execute(
                "SELECT task_id, active_revision FROM tasks"
            ).fetchall()
            revisions = db.execute(
                "SELECT COUNT(*) FROM task_revisions WHERE task_id = ?",
                (task_id,),
            ).fetchone()[0]
            db.close()
            assert len(tasks) == 1, f"追问不得裂变: {tasks}"
            assert tasks[0][1] == 2 and revisions == 2

            # /new 后新目标 → 新 task。
            session.send("/newtask\r")
            import time

            time.sleep(2.0)
            marker = len(session.clean)
            session.send("完全不同的新任务\r")
            session.expect("已在同一会话直接完成".encode(), timeout=180,
                           after=marker)
            db = sqlite3.connect(home / "agentd" / "missions.db")
            count = db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            orders = db.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            execs = db.execute(
                "SELECT COUNT(*) FROM task_executions"
            ).fetchone()[0]
            db.close()
            assert count == 2, f"/new 必须开新 task: {count}"
            assert orders == 0 and execs == 0
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
