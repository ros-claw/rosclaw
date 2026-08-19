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
    """编排：write →（TurnGuard 提醒）→ register → finish → 汇报。"""

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
        last_text = _text(messages[-1]) if messages else ""
        if has_tool_result:
            call_id = str(messages[-1].get("tool_call_id", ""))
            if call_id == "call_write":
                # 故意不收尾（没有 finish）——TurnGuard 应提醒。
                frames = [_sse(_chunk("写完了。")), _sse(_chunk("", "stop")),
                          b"data: [DONE]\n\n"]
                return b"".join(frames)
            if call_id == "call_register":
                import re as _re

                content = str(messages[-1].get("content", ""))
                match = _re.search(r"artifact_id=(art_[a-f0-9]+)", content)
                frames = _tool_call_frames(
                    "call_finish", "rosclaw_task_finish",
                    json.dumps({"summary": "hello.txt 已写入",
                                "artifact_ids": [match.group(1) if match else ""]}),
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            if call_id == "call_finish":
                frames = [_sse(_chunk("验收通过，任务完成。")),
                          _sse(_chunk("", "stop")), b"data: [DONE]\n\n"]
                return b"".join(frames)
        if "没有收尾" in last_text or "rosclaw_task_finish" in last_text:
            # TurnGuard 提醒回合 → 登记 artifact。
            frames = _tool_call_frames(
                "call_register", "rosclaw_artifact_register",
                json.dumps({"path": "hello.txt"}),
            )
            frames.append(b"data: [DONE]\n\n")
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
    def test_turnguard_to_verified_finish(self, tmp_path: Path) -> None:
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
            session.send("写一个 hello.txt 并收尾\r")
            # TurnGuard 提醒（write 后未收尾）→ 登记 → finish → 验收通过。
            session.expect("验收通过，任务完成".encode(), timeout=240)
            db = sqlite3.connect(home / "agentd" / "missions.db")
            tasks = db.execute(
                "SELECT task_id, state, accepted_at FROM tasks"
            ).fetchall()
            verifications = db.execute(
                "SELECT status FROM verifications"
            ).fetchall()
            artifacts = db.execute("SELECT path, sha256 FROM artifacts").fetchall()
            orders = db.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            db.close()
            assert len(tasks) == 1 and tasks[0][1] == "SUCCEEDED", tasks
            assert tasks[0][2], "accepted_at 必须落账"
            assert verifications and verifications[0][0] == "PASS", verifications
            assert artifacts, "artifact 未登记"
            assert orders == 0
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
