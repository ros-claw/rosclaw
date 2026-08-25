"""P1-B2 产品旅程（0824 总纲 §12.3）：operation progress 流式进 TUI。

PTY `rosclaw chat` + 假模型：process_start 启动一个分阶段输出的长
命令（progress-step-1..3，间隔 1.2s）→ 断言 operation **运行期间**
TUI 已按 operation_id 原位渲染最新输出行（widget——用户不需要
/logs 或模型轮询）→ 终态 followUp 闭环照旧（WP-1 语义不回归）。

红证据：B2 前 TUI 只有 Working… 占位——运行中看不到任何输出行。
"""

from __future__ import annotations

import json
import re
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


class _ProgressFake:
    """process_start（分阶段长命令）→ 首回合结束；终态 followUp → 汇报。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def answer(self, body: dict) -> bytes:
        self.requests.append(body)
        messages = body.get("messages", [])

        def _text(m: dict) -> str:
            content = m.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    str(b.get("text", "")) for b in content if isinstance(b, dict)
                )
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
            frames = [_sse(_chunk("已在后台启动 operation，完成会通知。")),
                      _sse(_chunk("", "stop")), b"data: [DONE]\n\n"]
            return b"".join(frames)
        if "已终止" in last_text:
            frames = [_sse(_chunk("后台 Operation 已完成并汇报。")),
                      _sse(_chunk("", "stop")), b"data: [DONE]\n\n"]
            return b"".join(frames)
        frames = _tool_call_frames(
            "call_opstart", "process_start",
            json.dumps({
                "command": "for i in 1 2 3; do echo progress-step-$i; sleep 2.5; done",
            }),
        )
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


class _ProgressFakeServer(_FakeServer):
    def __init__(self) -> None:
        self.fake = _ProgressFake()
        handler = type("H", (_Handler,), {"fake": self.fake})
        import threading as _threading
        from http.server import ThreadingHTTPServer

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        _threading.Thread(target=self.server.serve_forever, daemon=True).start()


class TestOperationProgressWidget:
    def test_progress_visible_during_run(self, tmp_path: Path) -> None:
        fake = _ProgressFakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env, log_path=tmp_path / "pty-b2.log", cwd=workspace,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("后台跑分阶段命令\r")
            session.expect("已在后台启动 operation".encode(), timeout=180)
            # operation 运行期间（~3.6s 总时长内）widget 原位渲染输出
            # 行——progress-step-1 或 2 必须先于完成通知可见。
            session.expect(b"progress-step-", timeout=60)
            seen = bytes(session.output)
            match = re.search(rb"progress-step-(\d)", seen)
            assert match, "运行期间未见任何输出行（widget 未渲染）"
            assert int(match.group(1)) < 3, (
                "只在完成后才看到输出——不是流式（看到了最后一步才出现）"
            )
            # 终态 followUp 闭环不回归。
            session.expect("后台 Operation 已完成并汇报".encode(), timeout=180)
            db = sqlite3.connect(home / "agentd" / "missions.db")
            state = db.execute(
                "SELECT state FROM operations ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            db.close()
            assert state and state[0] == "SUCCEEDED", state
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
