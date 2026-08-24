"""PR-H4 产品 Gate（总纲 v2 §12）：验收闭环 + TurnGuard。

PTY `rosclaw chat` + 假模型编排 + 真实内核：
1. 模型 write 后直接回答（不收尾）→ TurnGuard 注入一次结构化提醒；
2. 模型被提醒后 artifact_register + task_finish → verifier 真跑 →
   SUCCEEDED + verifications 行 + accepted_at；
3. 零 WorkOrder/零 execution；task 终态与 TUI 一致。
"""

from __future__ import annotations

import json
import sqlite3
import sys
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

        def _text(m: dict) -> str:
            content = m.get("content", "")
            if isinstance(content, list):
                return " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
            return str(content)

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
            # 模型回答已交付 → turn_end Coordinator 自动验收 →
            # 任务完成通知（零模型调用收尾）。
            session.expect("hello.txt 已写入并交付".encode(), timeout=240)
            session.expect("任务完成：验收".encode(), timeout=120)
            db = sqlite3.connect(home / "agentd" / "missions.db")
            tasks = db.execute(
                "SELECT task_id, state, accepted_at FROM tasks"
            ).fetchall()
            verifications = db.execute(
                "SELECT status FROM verifications"
            ).fetchall()
            artifacts = db.execute("SELECT path, sha256 FROM artifacts").fetchall()
            orders = db.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            # TaskOutcomeV2 落库且六维齐全。
            outcome_rows = db.execute(
                "SELECT outcome_json FROM task_outcomes"
            ).fetchall()
            db.close()
            assert len(tasks) == 1 and tasks[0][1] == "SUCCEEDED", tasks
            # P0-D：模型全程未调用 task_finish（收尾仪式已删除）。
            import json as _json

            tool_frames = _json.dumps(
                [m for req in fake.fake.requests for m in req.get("messages", [])]
            )
            assert "rosclaw_task_finish" not in tool_frames, (
                "模型仍在手动 task_finish——收尾仪式未删除"
            )
            assert outcome_rows, "task_outcomes 未落库"
            outcome = _json.loads(outcome_rows[0][0])
            assert outcome["lifecycle"] == "COMPLETED"
            assert outcome["verification"] == "PASS"
            assert outcome["delivery"] == "DELIVERED"
            assert tasks[0][2], "accepted_at 必须落账"
            assert verifications and verifications[0][0] == "PASS", verifications
            assert artifacts, "artifact 未登记"
            assert orders == 0
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
