"""PR-HP1 PTY 产品腿：input.persisted 先于任何模型请求（方案硬不变量）。

真实链路：PTY `rosclaw chat`（假模型记录首个请求到达时刻）→ 用户
发消息 → journal 里 input.persisted 的时间戳必须 ≤ 假模型首个请求
到达时刻；input.dispatched 在 persisted 之后。
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
from tests.agentd.test_product_journey import PtySession, _chunk, _sse

pytestmark = pytest.mark.skipif(
    not find_pi_agent_entry(),
    reason="无 Node/dist（CI 全回归 job 未构建）——诚实 skip",
)


class _TimedFake:
    def __init__(self) -> None:
        self.first_request_at: float | None = None

    def answer(self, body: dict) -> bytes:
        if self.first_request_at is None:
            self.first_request_at = time.time()
        frames = [_sse(_chunk("收到。")), _sse(_chunk("", "stop")), b"data: [DONE]\n\n"]
        return b"".join(frames)


class _Handler(BaseHTTPRequestHandler):
    fake: _TimedFake

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
        self.fake = _TimedFake()
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


class TestInputPersistedGate:
    def test_persisted_precedes_first_model_request(self, tmp_path: Path) -> None:
        fake = _FakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workdir = tmp_path / "hp1-work"
        workdir.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat",
             "--workspace", str(workdir)],
            env, log_path=tmp_path / "pty-hp1.log", cwd=workdir,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("你好\r")
            session.expect("收到。".encode(), timeout=120)
            assert fake.fake.first_request_at is not None
            # journal：input.persisted 存在且先于首个模型请求；
            # input.dispatched 在其后。
            db = sqlite3.connect(home / "agentd" / "missions.db")
            try:
                rows = db.execute(
                    "SELECT type, timestamp, session_id, task_id, revision, "
                    "model_visible FROM agent_events "
                    "WHERE type IN ('input.persisted', 'input.dispatched') "
                    "ORDER BY sequence"
                ).fetchall()
            finally:
                db.close()
            types = [r[0] for r in rows]
            assert "input.persisted" in types, f"journal 缺 input.persisted: {types}"
            assert "input.dispatched" in types, f"journal 缺 input.dispatched: {types}"
            assert types.index("input.persisted") < types.index("input.dispatched")
            persisted = rows[types.index("input.persisted")]
            # ISO 时间戳 → epoch 比较（允许同时）。
            from datetime import datetime

            persisted_at = datetime.fromisoformat(
                str(persisted[1])
            ).timestamp()
            assert persisted_at <= fake.fake.first_request_at + 0.001, (
                f"input.persisted({persisted_at}) 晚于首个模型请求"
                f"({fake.fake.first_request_at})——输入门被破坏"
            )
            # 一等列携带：session/model_visible。P0-C（0824 总纲
            # §6.1）：persist 不再逐条 bind task——input.persisted
            # 不再携带 task_id（任务在首个 effectful call 才附着，
            # 可经 user_inputs.task_id 追溯）；纯对话输入（你好）
            # 必须保持 tasks=0。
            assert persisted[2], "input.persisted 缺 session_id 列"
            assert persisted[5] == 1, "input.persisted 缺 model_visible 列"
            db2 = sqlite3.connect(home / "agentd" / "missions.db")
            try:
                task_count = db2.execute(
                    "SELECT COUNT(*) FROM tasks"
                ).fetchone()[0]
            finally:
                db2.close()
            assert task_count == 0, (
                "P0-C 契约破坏：纯对话输入创建了 Task（问候/解释/"
                "只读查询必须 tasks=0）"
            )
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
