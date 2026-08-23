"""WP-1 红测试（0823 审计 §四.WP-1 验收测试）：终态后幽灵执行。

审计验收原文：
  1. Task 完成后注入延迟 operation 事件。
  2. 等待一分钟。
  3. 断言没有任何模型请求、工具调用或新文件。

真实链路：PTY `rosclaw chat`（假模型：process_start 慢操作 → write →
artifact_register → task_finish → 最终回答）→ task SUCCEEDED 后慢
operation 才终止 → 断言：最终回答之后模型请求数恒为 0、工作区无
新文件。
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rosclaw.agentd.pi_entry import find_pi_agent_entry
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


class _GhostFake:
    """process_start（慢命令）→ write → artifact_register →
    task_finish → 最终回答。回答后再来任何请求都是幽灵回合。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.final_answered_at: float | None = None

    def answer(self, body: dict) -> bytes:
        self.requests.append(body)
        if not body.get("stream"):
            return json.dumps({
                "id": "c", "object": "chat.completion", "created": 1,
                "model": "fake-k3",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }).encode()
        messages = body.get("messages", [])
        if messages and messages[-1].get("role") == "tool":
            call_id = str(messages[-1].get("tool_call_id", ""))
            nxt = {
                "call_op": ("call_write", "write",
                            {"path": "note.txt", "content": "wp1\n"}),
                "call_write": ("call_art", "rosclaw_artifact_register",
                               {"path": "note.txt"}),
                "call_art": ("call_fin", "rosclaw_task_finish",
                             {"summary": "完成"}),
            }.get(call_id)
            if nxt is not None:
                frames = _tool_call_frames(nxt[0], nxt[1], json.dumps(nxt[2]))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            self.final_answered_at = time.time()
            frames = [_sse(_chunk("任务完成。")), _sse(_chunk("", "stop")),
                      b"data: [DONE]\n\n"]
            return b"".join(frames)
        frames = _tool_call_frames(
            "call_op", "process_start",
            json.dumps({"command": "sleep 5 && echo wp1-late-marker"}),
        )
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


class _Handler(BaseHTTPRequestHandler):
    fake: _GhostFake

    def log_message(self, *args) -> None:
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
        self.fake = _GhostFake()
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
        json.dumps({
            "providers": {
                "journey-fake": {
                    "name": "Journey Fake", "baseUrl": base_url,
                    "api": "openai-completions", "apiKey": "$FAKE_JOURNEY_KEY",
                    "models": [{"id": "fake-k3", "name": "Fake K3",
                                "contextWindow": 8192, "maxTokens": 4096}],
                }
            }
        }),
        encoding="utf-8",
    )
    env = dict(
        os.environ, ROSCLAW_HOME=str(home), TERM="xterm",
        FAKE_JOURNEY_KEY="sk-fake-journey", ROSCLAW_UI_LOCALE="zh-CN",
    )
    return home, env


class TestTerminalGhost:
    def test_no_model_calls_after_task_terminal(self, tmp_path: Path) -> None:
        fake = _FakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workdir = tmp_path / "wp1-work"
        workdir.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat",
             "--workspace", str(workdir)],
            env, log_path=tmp_path / "pty-wp1.log", cwd=workdir,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("写 note.txt 并交付\r")
            session.expect("任务完成。".encode(), timeout=240)
            # 任务已终态；慢 operation（5s）随后终止——审计验收：终态
            # 后模型调用恒为 0。
            assert fake.fake.final_answered_at is not None
            answered_count = len(fake.fake.requests)
            files_before = set(workdir.iterdir())
            time.sleep(12)  # 慢 operation (5s) 终止 + watcher 周期
            after_count = len(fake.fake.requests)
            assert after_count == answered_count, (
                f"终态后出现 {after_count - answered_count} 次幽灵模型请求"
                f"（延迟 operation 触发了新回合）"
            )
            files_after = set(workdir.iterdir())
            assert files_after == files_before, (
                f"终态后工作区出现新文件: {files_after - files_before}"
            )
            # 账本：任务 SUCCEEDED；迟到事件只进 task_events 存档。
            db = sqlite3.connect(home / "agentd" / "missions.db")
            try:
                task = db.execute(
                    "SELECT state FROM tasks ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                assert task and task[0] == "SUCCEEDED", task
            finally:
                db.close()
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
