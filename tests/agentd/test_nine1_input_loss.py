"""NINE-1（九审 §0.1/§25.1）：输入吞噬的最小复现（PTY 级）。

大道至简 R0-1 后语义反转：所有自然语言无条件进 Pi——用户输入
必须作为 user message 落在 session transcript（Pi 看得见），
且不再有任何"确定性链接管"回声卡（路由已退役）。输入吞噬防线
不变：输入在 transcript 必须可见。

红测试先行的原始场景：hello（正常）→ 自然语言五角星（旧时代
输入消失、后台出任务结果、模型不知情）。
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
                    # 等模型回应——关键是输入落账且进模型（唯一大脑）。
                    import time

                    time.sleep(15.0)
                    session.send("/quit\r")
                    session.expect(b"rosclaw continue", timeout=30)
                    session.proc.wait(timeout=30)
                finally:
                    session.stop()
        finally:
            fake.close()
        # 核心断言（大道至简 R0-1）：用户输入必须作为 user message
        # 落在会话 transcript（Pi 唯一大脑看得见）——且没有任何
        # 确定性链回声卡（路由退役，无第二通道）。
        sessions_dir = home / "agent" / "sessions"
        found_user = False
        found_echo = False
        for session_file in sessions_dir.glob("*.jsonl"):
            for line in session_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "message":
                    message = entry.get("message", {})
                    if message.get("role") == "user":
                        content = message.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                str(b.get("text", ""))
                                for b in content
                                if isinstance(b, dict)
                            )
                        if "画一个五角星" in str(content):
                            found_user = True
                if (
                    entry.get("type") == "custom_message"
                    and entry.get("customType") == "rosclaw.user_directive"
                ):
                    found_echo = True
        assert found_user, (
            "用户输入'画一个五角星'未作为 user message 进入 session "
            "JSONL——输入在 transcript 不可见（HP1 会话证据缺失）"
        )
        assert not found_echo, (
            "竟出现确定性链接管回声卡——聊天自动路由未退役干净"
        )
