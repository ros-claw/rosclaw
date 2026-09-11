"""PR-H4 产品 Gate（总纲 v2 §12）：验收闭环 + 内核静默终态。

PTY `rosclaw chat` + 假模型编排 + 真实内核。当前契约（大道至简
R0-2b，#515 之后的真实行为）：
1. 模型 write → deliver → 自己向用户汇报（叙述权在模型——Kernel
   不在 turn_end 自动宣布"任务完成：验收"，R0-2b 已删除该播报）；
2. Kernel Coordinator 在 turn_end **静默**自动收尾：tasks=SUCCEEDED、
   verifications=PASS、accepted_at 落账、task_outcomes 六维齐全——
   全程零模型调用（收尾不唤醒 Agent）；
3. 零 WorkOrder/零 execution；TUI 不出现 Kernel 验收播报。

0911 验证实证：本测试曾等待 R0-2b 已删除的 TUI 播报而腐烂
（#440 后未更新；CI 无 dist 即 skip，零覆盖）。现改断新契约：
DB 终态（轮询等待）+ 内核静默（PTY 无播报）+ 零模型调用收尾。
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from rosclaw.agentd.pi_entry import find_pi_agent_entry
from tests.agentd.test_h1_native_work import (
    _FakeServer,
    _Handler,
    _prepare_home,
)
from tests.agentd.test_product_journey import (
    PtySession,
    _chunk,
    _sse,
    _tool_call_frames,
)

pytestmark = pytest.mark.skipif(
    not find_pi_agent_entry(),
    reason="无 Node/dist（CI 全回归 job 未构建）——诚实 skip",
)


class _VerifierFake:
    """编排（P0-D）：write → deliver → 汇报（不收尾——Coordinator
    自动收尾）。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def answer(self, body: dict) -> bytes:
        self.requests.append(body)
        messages = body.get("messages", [])

        if not body.get("stream"):
            return json.dumps({
                "id": "c", "object": "chat.completion", "created": 1,
                "model": "fake-k3",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }).encode()
        has_tool_result = bool(messages) and messages[-1].get("role") == "tool"
        if has_tool_result:
            call_id = str(messages[-1].get("tool_call_id", ""))
            if call_id == "call_write":
                # P0-D：幂等 deliver（模型面唯一交付入口）。
                frames = _tool_call_frames(
                    "call_deliver", "rosclaw_deliver",
                    json.dumps({"path": "hello.txt"}),
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if call_id == "call_deliver":
                # 直接回答——不调 task_finish（模型面已删除；
                # Coordinator 在 turn_end 自动收尾）。
                frames = [_sse(_chunk("hello.txt 已写入并交付。")),
                          _sse(_chunk("", "stop")), b"data: [DONE]\n\n"]
                return b"".join(frames)
        # 用户回合 → write。
        frames = _tool_call_frames(
            "call_write", "write",
            json.dumps({"path": "hello.txt", "content": "hello-verified\n"}),
        )
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


class _VerifierFakeServer(_FakeServer):
    def __init__(self) -> None:
        self.fake = _VerifierFake()
        handler = type("H", (_Handler,), {"fake": self.fake})
        import threading as _threading
        from http.server import ThreadingHTTPServer

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        _threading.Thread(target=self.server.serve_forever, daemon=True).start()


def _await_task_terminal(db_path: Path, timeout: float = 180.0) -> tuple[list, list]:
    """轮询内核账本直到任务终态（Coordinator 自动收尾是异步的——
    不等 TUI 播报，DB 是唯一权威）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        db = sqlite3.connect(db_path)
        tasks = db.execute(
            "SELECT task_id, state, accepted_at FROM tasks"
        ).fetchall()
        verifications = db.execute("SELECT status FROM verifications").fetchall()
        db.close()
        if tasks and tasks[0][1] == "SUCCEEDED" and tasks[0][2]:
            return tasks, verifications
        time.sleep(2)
    return tasks, verifications


class TestVerifierClosedLoop:
    def test_coordinator_auto_finish(self, tmp_path: Path) -> None:
        fake = _VerifierFakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env, log_path=tmp_path / "pty-h4.log", cwd=workspace,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("写一个 hello.txt 并交付\r")
            # 模型叙述权：模型自己汇报已交付（R0-2b——Kernel 不替
            # 模型宣布验收）。
            session.expect("hello.txt 已写入并交付".encode(), timeout=240)
            requests_at_final_answer = len(fake.fake.requests)
            # 内核静默自动收尾：轮询 DB 直到 SUCCEEDED + accepted_at。
            tasks, verifications = _await_task_terminal(
                home / "agentd" / "missions.db"
            )
            db = sqlite3.connect(home / "agentd" / "missions.db")
            artifacts = db.execute("SELECT path, sha256 FROM artifacts").fetchall()
            orders = db.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            # TaskOutcomeV2 落库且六维齐全。
            outcome_rows = db.execute(
                "SELECT outcome_json FROM task_outcomes"
            ).fetchall()
            db.close()
            assert len(tasks) == 1 and tasks[0][1] == "SUCCEEDED", (
                f"Coordinator 未在 180s 内自动收尾: {tasks}"
            )
            assert tasks[0][2], "accepted_at 必须落账"
            # 零模型调用收尾：最终回答到账本终态之间不得有新请求。
            assert len(fake.fake.requests) == requests_at_final_answer, (
                f"收尾过程唤醒了模型（{requests_at_final_answer} → "
                f"{len(fake.fake.requests)} 请求）——终态后零模型回合被违反"
            )
            # P0-D：模型全程未调用 task_finish（收尾仪式已删除）。
            tool_frames = json.dumps(
                [m for req in fake.fake.requests for m in req.get("messages", [])]
            )
            assert "rosclaw_task_finish" not in tool_frames, (
                "模型仍在手动 task_finish——收尾仪式未删除"
            )
            assert outcome_rows, "task_outcomes 未落库"
            outcome = json.loads(outcome_rows[0][0])
            assert outcome["lifecycle"] == "COMPLETED"
            assert outcome["verification"] == "PASS"
            assert outcome["delivery"] == "DELIVERED"
            assert verifications and verifications[0][0] == "PASS", verifications
            assert artifacts, "artifact 未登记"
            assert orders == 0
            # R0-2b 内核静默契约：PTY 全程不得出现 Kernel 验收播报
            #（用户看到的是模型自己的叙述，不是 Kernel 的终态宣布）。
            pty_log = (tmp_path / "pty-h4.log").read_bytes()
            assert "任务完成：验收".encode() not in pty_log, (
                "Kernel 仍在自动宣布验收——R0-2b 静默契约被违反"
            )
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
