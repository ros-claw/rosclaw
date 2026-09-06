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
        """大道至简 R0-1/R0-2b：输入 → Pi 模型回合 → 首个 effectful
        工具调用（write hello.txt）由 admission 建 root task；追问
         bump revision（不裂变）；/newtask 后新目标开新 task。
        与本测试创立时同一意图，但驱动面从「确定性链抢答」换成
        「Pi 工作」——h1 假模型对所有输入走 write→bash→回答流。
        降级环境（无 bwrap）：bash 的授权卡拒绝（fail closed）。
        """
        fake = _FakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env, log_path=tmp_path / "pty-h2.log", cwd=workspace,
        )

        def _drive_one_input(text: str) -> None:
            """一条输入的完整驱动（journey A 同款授权卡处理——
            无 bwrap 主机弹 ui.select，标题"本机无 OS 沙箱"，
            第一项"允许一次"，Enter 首帧竞态需每秒重试直到对话
            框消失；有沙箱主机无卡直跑）。模型最终回答落屏。"""
            import time as _time

            # 竞态防线：只认本次输入之后的输出（上一腿的同款最终
            # 回答还留在屏上——不锚定 after 会瞬时误匹配）。
            marker = len(session.clean)
            session.send(text + "\r")
            deadline = _time.time() + 240
            answered = False
            while _time.time() < deadline and not answered:
                if "本机无 OS 沙箱".encode() in session.clean[-4000:]:
                    for _ in range(15):
                        _time.sleep(1.0)
                        session.send("\r")
                        _time.sleep(0.5)
                        if "本机无 OS 沙箱".encode() not in session.clean[-4000:]:
                            break
                    session.expect("已批准".encode(), timeout=30)
                try:
                    session.expect(
                        "已在同一会话直接完成".encode(), timeout=30,
                        after=marker,
                    )
                    answered = True
                except AssertionError:
                    continue
            if not answered:
                raise AssertionError(f"输入 {text!r} 未驱动到最终回答")


        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            _drive_one_input("画一个五角星")
            db = sqlite3.connect(home / "agentd" / "missions.db")
            tasks = db.execute(
                "SELECT task_id, state, active_revision FROM tasks"
            ).fetchall()
            db.close()
            assert len(tasks) == 1, (
                f"首个 effectful 调用必须只建一个 root task: {tasks}"
            )
            task_id = tasks[0][0]
            assert tasks[0][2] == 1
            assert "画一个五角星" in session.clean.decode(
                "utf-8", errors="replace"
            ), "屏幕无指令回声"
            db = sqlite3.connect(home / "agentd" / "missions.db")
            input_row = db.execute(
                "SELECT text FROM user_inputs WHERE text LIKE '%画一个五角星%'"
            ).fetchone()
            db.close()
            assert input_row, "user_inputs 权威账本缺输入（幽灵执行防线）"

            # 追问 → revision 2（同一 task，不裂变）。
            _drive_one_input("改成画圆形")
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

            # /newtask 后新目标 → 新 task。
            session.send("/newtask\r")
            import time

            time.sleep(2.0)
            _drive_one_input("完全不同的新任务")
            db = sqlite3.connect(home / "agentd" / "missions.db")
            count = db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            orders = db.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            execs = db.execute(
                "SELECT COUNT(*) FROM task_executions"
            ).fetchone()[0]
            db.close()
            assert count == 2, f"/newtask 必须开新 task: {count}"
            assert orders == 0 and execs == 0
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
