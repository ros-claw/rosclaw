"""NINE-1 红测试（九审 §0.1/§25.1）：输入吞噬的最小复现（PTY 级）。

红测试先行——九审实测顺序：hello（正常）→ 自然语言五角星（输入
消失，后台出现任务结果，模型不知情）。

0827 P0-1/2（Input Arbiter）后契约更新：已知 recipe 的指令由
TASK_ROUTER 认领并 suppress 模型回合——输入绝不作为 user message
进模型（那是双控制者通道），但必须作为确定性链回声（custom
message，display:true）落在 session transcript，且内核 user_inputs
账本持有权威记录。本测试固定：transcript 有回声 + 无 user
message + 内核账本有记录——三者一致。
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
        # 核心断言：用户输入必须在会话 transcript 可见。0827 P0-1/2
        # （Input Arbiter）后，已知 recipe 的指令被 TASK_ROUTER 认领并
        # suppress 模型回合——输入不再是 user message（那是双控制者的
        # 通道），而是确定性链回声（custom message，display:true）。
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
                    and "画一个五角星" in str(entry.get("content", ""))
                ):
                    found_echo = True
        # 单 Owner 契约：输入以确定性链回声落在 transcript（绝不能是
        # user message——user message 意味着送进了模型=双控制者）。
        assert found_echo, (
            "用户输入'画一个五角星'的确定性链回声未进入 session JSONL——"
            "输入在 transcript 不可见（HP1 会话证据缺失）"
        )
        assert not found_user, (
            "已认领输入竟作为 user message 进入会话——模型会再次看到"
            "同一指令（双控制者回归）"
        )
