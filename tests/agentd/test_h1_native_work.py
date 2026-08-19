"""PR-H1 产品冒烟（总纲 v2 §20 PR-H1 DoD）：

    coding smoke：同 session 修改文件、运行命令、验证结果；
    Worker/WorkOrder 写入为 0。

真实链路：PTY `rosclaw chat`（仓库 pinned 入口）→ 假模型编排（SSE
tool_call）→ 主会话真实执行 write/bash 工具 → 断言落盘 + 账本零裂变。
与十六审 Gate 的分界：这里假模型只编排"调用什么工具"，工具执行是
主会话真实 Pi 工具（写文件/跑命令都是真的）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tests.agentd.test_product_journey import (
    PtySession,
    _chunk,
    _sse,
    _tool_call_frames,
)

REPO = Path(__file__).resolve().parents[2]

import pytest  # noqa: E402

from rosclaw.agentd.pi_entry import find_pi_agent_entry  # noqa: E402

pytestmark = pytest.mark.skipif(
    not find_pi_agent_entry(),
    reason="无 Node/dist（CI 全回归 job 未构建）——诚实 skip",
)


class _DirectWorkFake:
    """编排主会话直接干活的假模型：write → bash → 最终回答。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def answer(self, body: dict) -> bytes:
        self.requests.append(body)
        messages = body.get("messages", [])
        has_tool_result = bool(messages) and messages[-1].get("role") == "tool"
        if not body.get("stream"):
            return json.dumps(
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fake-k3",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                },
            ).encode()
        if has_tool_result:
            last = messages[-1]
            call_id = str(last.get("tool_call_id", ""))
            if call_id == "call_write":
                frames = _tool_call_frames(
                    "call_bash", "bash",
                    json.dumps({"command": "cat hello.txt"}),
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            # bash 结果已回 → 最终回答
            frames = [
                _sse(_chunk("已在同一会话直接完成：写入 hello.txt 并验证内容。")),
                _sse(_chunk("", "stop")),
                b"data: [DONE]\n\n",
            ]
            return b"".join(frames)
        # 用户回合 → write 工具调用
        frames = _tool_call_frames(
            "call_write", "write",
            json.dumps({"path": "hello.txt", "content": "hello-rosclaw\n"}),
        )
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


class _Handler(BaseHTTPRequestHandler):
    fake: _DirectWorkFake

    def log_message(self, *args) -> None:  # 静默
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        payload = self.fake.answer(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _FakeServer:
    def __init__(self) -> None:
        self.fake = _DirectWorkFake()
        handler = type("H", (_Handler,), {"fake": self.fake})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self.server.shutdown()


def _prepare_home(tmp_path: Path, base_url: str) -> tuple[Path, dict[str, str]]:
    home = tmp_path / "rh"
    (home / "run").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "agent:\n  enabled: true\n  default_profile: embodied_default\n"
        "models:\n  backend: legacy\n  profiles:\n    embodied_default:\n"
        "      provider: kimi_code\n      model: fake-k3\n"
        f"      base_url: {base_url}\n"
        "      api_key_ref: env:FAKE_JOURNEY_KEY\n"
        "      capabilities: [llm.chat, llm.structured_decision, llm.tool_use]\n",
        encoding="utf-8",
    )
    (home / "agent").mkdir(parents=True, exist_ok=True)
    (home / "agent" / "settings.json").write_text(
        json.dumps({"defaultProvider": "journey-fake", "defaultModel": "fake-k3"}),
        encoding="utf-8",
    )
    (home / "agent" / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "journey-fake": {
                        "name": "Journey Fake",
                        "baseUrl": base_url,
                        "api": "openai-completions",
                        "apiKey": "$FAKE_JOURNEY_KEY",
                        "models": [{
                            "id": "fake-k3",
                            "name": "Fake K3",
                            "contextWindow": 8192,
                            "maxTokens": 4096,
                        }],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    env = dict(
        os.environ,
        ROSCLAW_HOME=str(home),
        TERM="xterm",
        FAKE_JOURNEY_KEY="sk-fake-journey",
        ROSCLAW_UI_LOCALE="en-US",
    )
    return home, env


class TestNativeWorksDirectly:
    def test_coding_smoke_same_session_zero_workers(self, tmp_path: Path) -> None:
        """主会话直接 write+bash 完成编码任务——零 WorkOrder/零 execution/
        零第二个 session。"""
        fake = _FakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env, log_path=tmp_path / "pty-h1.log", cwd=workspace,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("写一个 hello.txt 并验证内容\r")
            session.expect(
                "已在同一会话直接完成".encode(), timeout=180
            )
            # 工具真实执行：文件落盘在工作区。
            produced = list(tmp_path.rglob("hello.txt"))
            assert produced, "write 工具未真实落盘"
            assert produced[0].read_text().strip() == "hello-rosclaw"
            # 零裂变：没有 WorkOrder、没有 task execution、没有第二个
            # Pi session。
            db_path = home / "agentd" / "missions.db"
            if db_path.exists():
                db = sqlite3.connect(db_path)
                orders = db.execute(
                    "SELECT COUNT(*) FROM work_orders"
                ).fetchone()[0]
                execs = db.execute(
                    "SELECT COUNT(*) FROM task_executions"
                ).fetchone()[0]
                db.close()
                assert orders == 0, f"默认旅程不得创建 WorkOrder: {orders}"
                assert execs == 0, f"直接工作不得创建 execution: {execs}"
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
