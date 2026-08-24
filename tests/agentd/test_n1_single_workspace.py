"""PR-N1 PTY 验证：启动路径与工作区一致性（N 总纲 §5.4 退出条件）。

真实链路：PTY `rosclaw chat` 从非 git 目录启动 → header 显示真实
workspace（不谎称 Project）；任务产物登记与 bash 在同一根。
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
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


class _WriteAndDoneFake:
    """write → artifact_register → task_finish → 最终回答。"""

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
            nxt = {
                "call_write": ("call_art", "rosclaw_deliver", {"path": "note.txt"}),
                "call_art": ("call_fin", "rosclaw_task_finish", {"summary": "完成"}),
            }.get(call_id)
            if nxt is not None:
                frames = _tool_call_frames(nxt[0], nxt[1], json.dumps(nxt[2]))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            frames = [
                _sse(_chunk("任务完成。")),
                _sse(_chunk("", "stop")),
                b"data: [DONE]\n\n",
            ]
            return b"".join(frames)
        frames = _tool_call_frames(
            "call_write", "write",
            json.dumps({"path": "note.txt", "content": "n1-marker\n"}),
        )
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


class _Handler(BaseHTTPRequestHandler):
    fake: _WriteAndDoneFake

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
        self.fake = _WriteAndDoneFake()
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
                    "models": [{"id": "fake-k3", "name": "Fake K3", "contextWindow": 8192, "maxTokens": 4096}],
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


class TestSingleWorkspace:
    def test_plain_dir_start_honest_header_and_same_root(self, tmp_path: Path) -> None:
        """非 git 目录启动：header 显示真实 workspace（不谎称项目）；
        write/登记/账本全在同一根。"""
        fake = _FakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workdir = tmp_path / "plain-work"
        workdir.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat",
             "--workspace", str(workdir)],
            env, log_path=tmp_path / "pty-n1.log", cwd=workdir,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            # header 显示真实 workspace 名（显式 --workspace 的 basename）。
            session.expect(b"plain-work", timeout=60)
            session.send("写一个 note.txt 并交付\r")
            session.expect("任务完成。".encode(), timeout=240)
            # write 真实落在 workspace（同一根）。
            assert (workdir / "note.txt").read_text().strip() == "n1-marker"
            # 账本：任务 SUCCEEDED + 产物行指向同一根。
            db = sqlite3.connect(home / "agentd" / "missions.db")
            try:
                task = db.execute(
                    "SELECT state, workspace_path FROM tasks "
                    "ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                assert task and task[0] == "SUCCEEDED", task
                assert str(task[1]) == str(workdir), (
                    f"任务 workspace 与实际工作目录分裂: {task[1]} != {workdir}"
                )
                art = db.execute(
                    "SELECT path FROM artifacts ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                assert art and str(art[0]).startswith(str(workdir)), art
            finally:
                db.close()
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
