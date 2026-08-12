"""NINE-1 红测试（九审 §0.1/§25.1）：输入吞噬的最小复现（PTY 级）。

红测试先行——九审实测顺序：hello（正常）→ 自然语言五角星（输入
消失，后台出现任务结果，模型不知情）。本测试固定：自然语言任务
输入必须进入 Pi session JSONL 的 user message——TUI、模型消息、
持久化账本三者一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.agentd.test_product_journey import (
    FakeModelServer,
    PtySession,
    _build_and_install,
    _hidden_source_checkout,
    _prepare_installed_chat,
)


@pytest.mark.slow
class TestInputNeverLost:
    def test_nl_task_input_lands_in_session_jsonl(self, tmp_path: Path) -> None:
        fake = FakeModelServer(log_path=tmp_path / "fake-requests.jsonl")
        prefix, _root = _build_and_install(tmp_path)
        home, env, rosclaw = _prepare_installed_chat(tmp_path, fake, prefix)
        try:
            with _hidden_source_checkout():
                session = PtySession(
                    [str(rosclaw), "chat"], env,
                    log_path=tmp_path / "pty-input.log",
                )
                try:
                    session.expect(b"ROSClaw Native Agent", timeout=60)
                    session.send("你好\r")
                    session.expect("你好，我是 ROSClaw".encode(), timeout=90)
                    # 九审复现步骤：正常回合后输入自然语言任务。
                    session.send("我想用机械臂画一个五角星\r")
                    # 等任务结果或模型回应（任一路径）——关键是输入落账。
                    import time

                    time.sleep(15.0)
                    session.send("/quit\r")
                    session.expect(b"rosclaw continue", timeout=30)
                    session.proc.wait(timeout=30)
                finally:
                    session.stop()
        finally:
            fake.close()
        # 核心断言：用户输入作为 user message 存在于 session JSONL。
        sessions_dir = home / "agent" / "sessions"
        found = False
        for session_file in sessions_dir.glob("*.jsonl"):
            for line in session_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "message":
                    continue
                message = entry.get("message", {})
                if message.get("role") != "user":
                    continue
                content = message.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        str(b.get("text", "")) for b in content if isinstance(b, dict)
                    )
                if "画一个五角星" in str(content):
                    found = True
                    break
        assert found, (
            "用户输入'我想用机械臂画一个五角星'未进入 session JSONL——"
            "P0-INPUT-LOSS（幽灵执行）"
        )
