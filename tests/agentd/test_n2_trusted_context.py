"""PR-N2 PTY 验证：可信上下文与内置 Skill 真实到达模型。

真实链路：workspace 里放 CLAUDE.md（含唯一 marker）→ PTY chat →
假模型收到的 system/上下文必须含 marker（项目认知恢复）+
rosclaw-embodied Skill 在可用列表；robot profile 则全不出现。
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
from tests.agentd.test_product_journey import PtySession, _chunk, _sse

pytestmark = pytest.mark.skipif(
    not find_pi_agent_entry(),
    reason="无 Node/dist（CI 全回归 job 未构建）——诚实 skip",
)

MARKER = "N2-CTX-MARKER-7f3a9b"


class _EchoFake:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def answer(self, body: dict) -> bytes:
        self.requests.append(body)
        if not body.get("stream"):
            return json.dumps({
                "id": "c", "object": "chat.completion", "created": 1,
                "model": "fake-k3",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }).encode()
        frames = [
            _sse(_chunk("已收到。")),
            _sse(_chunk("", "stop")),
            b"data: [DONE]\n\n",
        ]
        return b"".join(frames)


class _Handler(BaseHTTPRequestHandler):
    fake: _EchoFake

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
        self.fake = _EchoFake()
        handler = type("H", (_Handler,), {"fake": self.fake})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self.server.shutdown()


def _prepare(tmp_path: Path, base_url: str) -> tuple[Path, Path, dict[str, str]]:
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
    workdir = tmp_path / "ws"
    workdir.mkdir()
    (workdir / ".git").mkdir()  # git root → workspace=workdir（N1 规则）
    (workdir / "CLAUDE.md").write_text(
        f"# 项目说明\n\n本项目的唯一标识：{MARKER}\n", encoding="utf-8",
    )
    env = dict(
        os.environ, ROSCLAW_HOME=str(home), TERM="xterm",
        FAKE_JOURNEY_KEY="sk-fake-journey", ROSCLAW_UI_LOCALE="zh-CN",
    )
    return home, workdir, env


class TestTrustedContext:
    def test_context_file_and_bundled_skill_reach_model(self, tmp_path: Path) -> None:
        fake = _FakeServer()
        home, workdir, env = _prepare(tmp_path, fake.base_url)
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env, log_path=tmp_path / "pty-n2.log", cwd=workdir,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("你好\r")
            session.expect("已收到。".encode(), timeout=180)
            assert fake.fake.requests, "假模型未收到请求"
            all_text = json.dumps(fake.fake.requests, ensure_ascii=False)
            # 项目上下文恢复：CLAUDE.md marker 到达模型。
            assert MARKER in all_text, (
                "CLAUDE.md 内容未到达模型（可信上下文未恢复）"
            )
            # 内置签名 Skill 在可用技能面。
            assert "rosclaw-embodied" in all_text, (
                "内置 Skill 未出现在模型可见面"
            )
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()


class _InspectFake:
    """第一轮 → rosclaw_inspect(robot ur5e)；结果回 → 最终回答。"""

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
            frames = [
                _sse(_chunk("UR5e 权威资产已定位。")),
                _sse(_chunk("", "stop")),
                b"data: [DONE]\n\n",
            ]
            return b"".join(frames)
        from tests.agentd.test_product_journey import _tool_call_frames

        frames = _tool_call_frames(
            "call_inspect", "rosclaw_inspect",
            json.dumps({"kind": "robot", "query": "ur5e"}),
        )
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


class TestInspectTool:
    def test_inspect_robot_returns_canonical_chain_to_model(self, tmp_path: Path) -> None:
        """模型调 rosclaw_inspect → 一次调用拿到权威资产链（不再全盘
        搜索同名文件）。"""
        fake = _FakeServer()
        fake.fake = _InspectFake()  # type: ignore[assignment]
        handler = type("H", (_Handler,), {"fake": fake.fake})
        fake.server.RequestHandlerClass = handler
        # _FakeServer 构造时已绑定 handler 类——直接替换 fake 对象即可
        # （handler 通过类属性读 fake.fake 的引用——直接换）。
        home, workdir, env = _prepare(tmp_path, fake.base_url)
        (workdir / ".git").mkdir(exist_ok=True)
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env, log_path=tmp_path / "pty-n2-inspect.log", cwd=workdir,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("查一下 ur5e 的权威模型资产\r")
            session.expect("UR5e 权威资产已定位。".encode(), timeout=240)
            # 假模型收到的 tool 结果必须含权威链。
            all_text = json.dumps(fake.fake.requests, ensure_ascii=False)
            assert "robot.mjcf.xml" in all_text, "权威 MJCF 未到达模型"
            assert "e-urdf-zoo" in all_text
            assert "canonical" in all_text
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
