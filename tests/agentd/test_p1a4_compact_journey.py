"""P1-A4 PTY 旅程（0824 总纲 P1-A）：/compact 后 TaskRefs 不丢。

真实链路：PTY `rosclaw chat` → 假模型完成一条任务（write → deliver）
→ 用户 /compact（Pi 内置压缩，fake 给出的摘要**故意不含**任何
task/artifact refs）→ 用户再发一轮 → 断言下一轮请求的消息里出现
rosclaw.task_anchor 锚（task_id + 产物路径——来自内核权威账本，
不是模型记忆），且会话 JSONL 落盘该锚。

红测试先行：锚机制实现前，下一轮请求里找不到这些 refs。
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


class _CompactJourneyFake:
    """write → deliver → 最终回答；之后：summary 请求 → 无 refs 摘要；
    普通回合 → 普通回答。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.task_done = False
        self.debug: list[str] = []

    @staticmethod
    def _text_answer(text: str) -> bytes:
        frames = [_sse(_chunk(text)), _sse(_chunk("", "stop")), b"data: [DONE]\n\n"]
        return b"".join(frames)

    def answer(self, body: dict) -> bytes:
        self.requests.append(body)
        messages = body.get("messages", [])
        last = messages[-1] if messages else {}
        self.debug.append(
            f"tools={bool(body.get('tools'))} "
            f"last_role={last.get('role')} "
            f"last_tcid={last.get('tool_call_id', '')} "
            f"n_msg={len(messages)}"
        )
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
        # trusted context 注入在 typed input 之后也是 user 角色——
        # 不能只看最后一条；全部 user 文本合并判定。
        user_texts = []
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content")
                user_texts.append(
                    content
                    if isinstance(content, str)
                    else json.dumps(content, ensure_ascii=False)
                )
        last_user = "\n".join(user_texts)
        # compaction 摘要请求（generateSummary 不带 tools——正常回合
        # 都带工具面；trusted context 里可能有 "summary" 字样，不能按
        # 文本猜）→ 故意不给 refs。
        if not body.get("tools"):
            return self._text_answer("摘要：用户让助手生成了一个文件并交付。")
        if last.get("role") == "tool":
            call_id = str(last.get("tool_call_id", ""))
            if call_id == "call_write":
                frames = _tool_call_frames(
                    "call_art", "rosclaw_deliver",
                    json.dumps({"path": "report.txt", "media_type": "text/plain"}),
                )
                frames.append(b"data: [DONE]\n\n")
                return b"".join(frames)
            self.task_done = True
            return self._text_answer("任务完成：report.txt 已交付。")
        # 填充回合 → 长回答（把会话撑过可压缩下限）；尾部标记=
        # 流结束——expect 它即等回合真正收束。
        if "填充" in last_user:
            return self._text_answer(
                "填充内容：具身智能任务上下文。" * 150 + "==FILLER-END=="
            )
        # 首个用户回合 → write；compact 后的用户回合 → 普通回答。
        if "report" in last_user and not self.task_done:
            frames = _tool_call_frames(
                "call_write", "write",
                json.dumps({"path": "report.txt", "content": "a4-report-body\n"}),
            )
            frames.append(b"data: [DONE]\n\n")
            return b"".join(frames)
        return self._text_answer("好的，继续。")


class _Handler(BaseHTTPRequestHandler):
    fake: _CompactJourneyFake

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
        self.fake = _CompactJourneyFake()
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
        json.dumps({
            "defaultProvider": "journey-fake",
            "defaultModel": "fake-k3",
            # 旅程确定性：Pi 默认 keepRecentTokens=20000，小会话 /compact
            # 会被拒（"Nothing to compact"）——旅程内调小，不改产品默认。
            "compaction": {"keepRecentTokens": 2000},
        }),
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
                            "contextWindow": 262144,
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


def _wait_summary_request(fake: _CompactJourneyFake, timeout: float = 240.0) -> None:
    """等 Pi compaction 的摘要请求到达 fake（= /compact 已被处理）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # compaction 的 generateSummary 调用不带 tools——与正常回合区分。
        if any(not body.get("tools") for body in fake.requests):
            return
        time.sleep(1.0)
    raise AssertionError("compaction 摘要请求未到达（/compact 未生效）")


class TestCompactTaskRefsJourney:
    def test_compact_then_next_turn_carries_task_anchor(self, tmp_path: Path) -> None:
        fake = _FakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env, log_path=tmp_path / "pty-a4.log", cwd=workspace,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("生成 report.txt 并交付\r")
            session.expect("任务完成：report.txt 已交付".encode(), timeout=240)
            # 账本侧先拿到权威 task_id。
            db_path = home / "agentd" / "missions.db"
            assert db_path.exists()
            db = sqlite3.connect(db_path)
            try:
                row = db.execute(
                    "SELECT task_id FROM tasks ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                assert row, "任务未创建"
                task_id = row[0]
            finally:
                db.close()

            # 会话太小 Pi 拒绝压缩（"Nothing to compact"）——先三轮
            # 填充撑过下限。
            for i in range(3):
                session.send(f"第{i}轮填充\r")
                session.expect(b"==FILLER-END==", timeout=120)
                time.sleep(1.5)

            # /compact——fake 的摘要故意不含 refs。
            session.send("/compact\r")
            _wait_summary_request(fake.fake)

            # compact 后的下一回合：锚必须随上下文到达模型。
            session.send("继续\r")
            # convertToLlm 把 custom 消息转为 role=user 的纯内容——请求
            # 里没有 customType 字面量；锚的判别标记是内容里的
            # "TaskRefs 锚"（compaction 后工具结果已被摘要替换，task_id
            # /report.txt 只能来自锚或每轮 trusted context）。
            deadline = time.monotonic() + 240
            seen_anchor = ""
            while time.monotonic() < deadline:
                last = fake.fake.requests[-1] if fake.fake.requests else {}
                texts = json.dumps(last.get("messages", []), ensure_ascii=False)
                if "TaskRefs" in texts:
                    seen_anchor = texts
                    break
                time.sleep(1.0)
            assert seen_anchor, (
                "compact 后下一回合上下文无 TaskRefs 锚——"
                "TaskRefs 已随压缩丢失"
            )
            assert task_id in seen_anchor, "锚中缺权威 task_id"
            assert "report.txt" in seen_anchor, "锚中缺产物 ref"

            # 锚落盘会话历史（后续 compact 的 summarizer 也能看到）。
            sessions_dir = home / "agent" / "sessions"
            deadline = time.monotonic() + 60
            anchored = False
            while time.monotonic() < deadline:
                for jsonl in sessions_dir.rglob("*.jsonl"):
                    if "rosclaw.task_anchor" in jsonl.read_text(
                        encoding="utf-8", errors="replace"
                    ):
                        anchored = True
                        break
                if anchored:
                    break
                time.sleep(1.0)
            assert anchored, "锚未落盘会话历史"

            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            print("FAKE DEBUG:", *fake.fake.debug[-15:], sep="\n  ")
            session.stop()
            fake.close()
