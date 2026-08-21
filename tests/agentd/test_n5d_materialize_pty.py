"""PR-N5D PTY 产品验证：物化工具真实到达模型并可调用（调整方案 §三.N5D）。

真实链路：PTY `rosclaw chat`（假模型）→ 模型工具面必须包含物化的
强类型能力工具（如 sim_get_state），且 rosclaw_compute /
rosclaw_execute 不在模型面；模型直接用精确参数调用物化工具 →
内核验证链执行 → canonical 结果回到模型。
"""

from __future__ import annotations

import json
import os
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

#: N5D：通用入口退出模型面。
_BANNED = {"rosclaw_compute", "rosclaw_execute"}


class _MaterializeFake:
    """工具面检查 + 物化工具调用：sim_get_state 在面内即调用。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.banned_seen: list[str] = []
        self.materialized_seen = False
        self.called = False

    def answer(self, body: dict) -> bytes:
        self.requests.append(body)
        if not body.get("stream"):
            return json.dumps({
                "id": "c", "object": "chat.completion", "created": 1,
                "model": "fake-k3",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }).encode()
        tool_names = {
            t.get("function", {}).get("name", "")
            for t in body.get("tools", []) or []
        }
        self.banned_seen.extend(sorted(tool_names & _BANNED))
        if "sim_get_state" in tool_names:
            self.materialized_seen = True
        messages = body.get("messages", [])
        has_tool_result = bool(messages) and messages[-1].get("role") == "tool"
        if has_tool_result:
            frames = [
                _sse(_chunk("状态已读取。")),
                _sse(_chunk("", "stop")),
                b"data: [DONE]\n\n",
            ]
            return b"".join(frames)
        if self.materialized_seen and not self.called:
            self.called = True
            # 精确参数直接调用物化工具（不经过通用入口）。
            frames = _tool_call_frames(
                "call_state", "sim_get_state", json.dumps({"verbose": False}),
            )
            frames.append(b"data: [DONE]\n\n")
            return b"".join(frames)
        frames = [
            _sse(_chunk("等待工具面。")),
            _sse(_chunk("", "stop")),
            b"data: [DONE]\n\n",
        ]
        return b"".join(frames)


class _Handler(BaseHTTPRequestHandler):
    fake: _MaterializeFake

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
        self.fake = _MaterializeFake()
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


class TestMaterializedToolJourney:
    def test_model_sees_and_calls_materialized_tool(self, tmp_path: Path) -> None:
        fake = _FakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workdir = tmp_path / "n5d-work"
        workdir.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat",
             "--workspace", str(workdir)],
            env, log_path=tmp_path / "pty-n5d.log", cwd=workdir,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("读取机器人状态\r")
            # 第一回合若物化尚未生效（注册发生在 before_agent_start），
            # 模型会先说"等待工具面"——再驱动一个回合。
            try:
                session.expect("状态已读取。".encode(), timeout=120)
            except AssertionError:
                session.send("继续\r")
                session.expect("状态已读取。".encode(), timeout=120)
            assert fake.fake.materialized_seen, (
                "物化工具 sim_get_state 从未出现在模型工具面"
            )
            assert fake.fake.called, "模型未调用物化工具"
            assert not fake.fake.banned_seen, (
                f"通用入口仍在模型面: {fake.fake.banned_seen}"
            )
            # 结果真实回到模型（canonical value 投影含 body_id）。
            tool_msgs = [
                m for req in fake.fake.requests for m in req.get("messages", [])
                if m.get("role") == "tool"
            ]
            assert tool_msgs, "无工具结果回传"
            assert "body_id" in str(tool_msgs[-1].get("content", ""))
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
