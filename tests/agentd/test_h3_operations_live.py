"""PR-H3 产品 Gate（总纲 v2 §21 Gate E 子集）：长 Operation 闭环。

PTY `rosclaw chat` + 假模型编排 + 真实 OperationManager：
1. 模型 process_start → 立即返回 operation_id（回合不死等）；
2. 后台进程真实跑完 → 终态一次性 followUp 注入同一 session；
3. 模型收到通知回合 → 用 process_output 读输出 → 汇报；
4. 账本：operations SUCCEEDED + task_events 完整链（started/output/
   completed）；零 WorkOrder；progress 不经 LLM（通知只一次）。
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


class _OperationFake:
    """编排：process_start → 首回合结束；终态 followUp → 读输出汇报。"""

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

        has_tool_result = bool(messages) and messages[-1].get("role") == "tool"
        last_text = _text(messages[-1]) if messages else ""
        if not body.get("stream"):
            return json.dumps({
                "id": "c", "object": "chat.completion", "created": 1,
                "model": "fake-k3",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }).encode()
        if has_tool_result:
            last = messages[-1]
            call_id = str(last.get("tool_call_id", ""))
            if call_id == "call_opstart":
                # 首回合结束——不死等（operation 后台跑）。
                frames = [_sse(_chunk("已在后台启动 operation，完成会通知。")),
                          _sse(_chunk("", "stop")), b"data: [DONE]\n\n"]
                return b"".join(frames)
            if call_id == "call_opoutput":
                frames = [_sse(_chunk("后台 Operation 输出已读到并完成汇报。")),
                          _sse(_chunk("", "stop")), b"data: [DONE]\n\n"]
                return b"".join(frames)
        if "已终止" in last_text:
            # 终态 followUp 回合 → 读输出。
            import re as _re

            match = _re.search(r"op_[a-f0-9]+", last_text)
            frames = _tool_call_frames(
                "call_opoutput", "process_output",
                json.dumps({"operation_id": match.group(0) if match else ""}),
            )
            frames.append(b"data: [DONE]\n\n")
            return b"".join(frames)
        # 用户回合 → process_start。
        frames = _tool_call_frames(
            "call_opstart", "process_start",
            json.dumps({"command": "echo op-output-line && sleep 0.5"}),
        )
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


class _OpFakeServer(_FakeServer):
    def __init__(self) -> None:
        self.fake = _OperationFake()
        handler = type("H", (_Handler,), {"fake": self.fake})
        import threading as _threading
        from http.server import ThreadingHTTPServer

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        _threading.Thread(target=self.server.serve_forever, daemon=True).start()


class TestOperationClosedLoop:
    def test_process_operation_full_loop(self, tmp_path: Path) -> None:
        fake = _OpFakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env, log_path=tmp_path / "pty-h3.log", cwd=workspace,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("后台跑一个长命令并告诉我结果\r")
            session.expect("已在后台启动 operation".encode(), timeout=180)
            # 终态 followUp → 读输出 → 汇报（同一 session，用户零追问）。
            session.expect("后台 Operation 输出已读到并完成汇报".encode(),
                           timeout=180)
            db = sqlite3.connect(home / "agentd" / "missions.db")
            ops = db.execute(
                "SELECT operation_id, state, ended_at FROM operations"
            ).fetchall()
            events = db.execute(
                "SELECT event_type FROM task_events ORDER BY seq"
            ).fetchall()
            orders = db.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            db.close()
            assert len(ops) == 1, f"一个 operation: {ops}"
            assert ops[0][1] == "SUCCEEDED", ops
            assert ops[0][2], "终态必须冻结 ended_at"
            kinds = [e[0] for e in events]
            assert "operation.admitted" in kinds
            assert "operation.output" in kinds
            assert "operation.completed" in kinds
            assert orders == 0
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
