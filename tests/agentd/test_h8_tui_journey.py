"""PR-H8 红测试：Task Activity/Logs/Artifacts 产品旅程（总纲 v2 §20 PR-H8）。

真实链路：PTY `rosclaw chat` → 假模型编排（write → process_start →
rosclaw_artifact_register → rosclaw_task_finish）→ 主会话真实执行 →
用户用 /activity /logs /artifacts 三个命令查看任务进度——数据全部
来自 TaskKernel 事件流/产物账本（不经 LLM 总结）。

红测试先行：/activity /logs /artifacts 命令实现前必须红。
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


class _FullJourneyFake:
    """write → process_start → artifact_register → task_finish → 回答。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def answer(self, body: dict) -> bytes:
        self.requests.append(body)
        messages = body.get("messages", [])
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
        has_tool_result = bool(messages) and messages[-1].get("role") == "tool"
        if has_tool_result:
            call_id = str(messages[-1].get("tool_call_id", ""))
            nxt = {
                "call_write": (
                    "call_op", "process_start",
                    {"command": "echo h8-operation-marker"},
                ),
                "call_op": (
                    "call_art", "rosclaw_artifact_register",
                    {"path": "report.txt", "media_type": "text/plain"},
                ),
                "call_art": (
                    "call_fin", "rosclaw_task_finish",
                    {"summary": "报告已生成并验收"},
                ),
            }.get(call_id)
            if nxt is not None:
                frames = _tool_call_frames(nxt[0], nxt[1], json.dumps(nxt[2]))
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            # task_finish 后（及 operation followUp 回合）→ 最终回答
            frames = [
                _sse(_chunk("任务完成：report.txt 已交付。")),
                _sse(_chunk("", "stop")),
                b"data: [DONE]\n\n",
            ]
            return b"".join(frames)
        frames = _tool_call_frames(
            "call_write", "write",
            json.dumps({"path": "report.txt", "content": "h8-report-body\n"}),
        )
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


class _Handler(BaseHTTPRequestHandler):
    fake: _FullJourneyFake

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
        self.fake = _FullJourneyFake()
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
        ROSCLAW_UI_LOCALE="zh-CN",
    )
    return home, env


class TestTaskActivityJourney:
    def test_activity_logs_artifacts_commands(self, tmp_path: Path) -> None:
        """一条任务走完后，/activity /logs /artifacts 都给出账本数据。"""
        fake = _FakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env, log_path=tmp_path / "pty-h8.log", cwd=workspace,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("生成 report.txt 并交付\r")
            session.expect("任务完成：report.txt 已交付".encode(), timeout=240)
            produced = list(tmp_path.rglob("report.txt"))
            assert produced, "write 工具未真实落盘"

            # /activity：阶段行来自 task_events（任务开始/进程/产物/验收）。
            session.send("/activity\r")
            session.expect("任务开始".encode(), timeout=30)
            session.expect("验收".encode(), timeout=30)

            # /logs：operation 输出（echo 的 marker）。
            session.send("/logs\r")
            session.expect(b"h8-operation-marker", timeout=30)

            # /artifacts：产物账本（report.txt + sha 短码）。
            session.send("/artifacts\r")
            session.expect(b"report.txt", timeout=30)

            # 账本侧对账：任务 SUCCEEDED + 产物行存在。
            db_path = home / "agentd" / "missions.db"
            assert db_path.exists()
            db = sqlite3.connect(db_path)
            try:
                state = db.execute(
                    "SELECT state FROM tasks ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                assert state and state[0] == "SUCCEEDED", f"任务终态异常: {state}"
                arts = db.execute(
                    "SELECT COUNT(*) FROM artifacts"
                ).fetchone()[0]
                assert arts >= 1, "产物账本为空"
            finally:
                db.close()

            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()
