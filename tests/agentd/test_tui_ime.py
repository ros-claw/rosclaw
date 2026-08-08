"""T-TUI/IME（二次审计 Gate E）：安装产物 PTY 驱动的输入法/终端矩阵。

- 中文输入：编辑器回显完整 CJK 字符（不是乱码/替换符）；
- CJK 退格：整字删除（不留半个宽字符残骸）；
- bracketed paste：多行中文粘贴不逐行提交；
- resize：TIOCSWINSZ 后 TUI 重绘不崩、进程存活、可正常退出。

复用 test_product_journey 的构建/安装/PTY 设施与 test_start_exit_soak
的 fake home 配置。
"""

from __future__ import annotations

import fcntl
import os
import struct
import termios
import time
from pathlib import Path

import pytest

from tests.agentd.test_product_journey import (
    FakeModelServer,
    PtySession,
    _build_and_install,
)
from tests.agentd.test_start_exit_soak import _write_fake_home


def _set_winsize(master_fd: int, rows: int, cols: int) -> None:
    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


@pytest.mark.slow
class TestTuiImeMatrix:
    def test_cjk_paste_resize(self, tmp_path: Path) -> None:
        fake = FakeModelServer(log_path=tmp_path / "fake-requests.jsonl")
        prefix, _root = _build_and_install(tmp_path)
        home = tmp_path / "rh"
        _write_fake_home(home, fake.base_url)
        rosclaw = prefix / "bin" / "rosclaw"
        env = dict(
            os.environ,
            ROSCLAW_HOME=str(home),
            TERM="xterm",
            FAKE_JOURNEY_KEY="sk-fake-journey",
            KIMI_API_KEY="sk-fake-journey",
            PATH=f"{prefix / 'bin'}:{os.environ['PATH']}",
        )
        session = PtySession([str(rosclaw), "chat", "--engine", "pi"], env)
        try:
            session.expect(b"ROSClaw Native Agent", timeout=90)

            # 1. 中文输入：编辑器回显完整字符。
            probe = "机械臂巡检"
            session.send(probe)
            time.sleep(1.5)
            assert probe.encode() in session.output, (
                f"中文输入未回显；尾部: {session.output[-400:]!r}"
            )

            # 2. CJK 退格：删两个字后剩余前缀回显，无乱码替换符。
            for _ in range(2):
                session.send("\x7f")
                time.sleep(0.4)
            time.sleep(1.0)
            tail = session.output[-2000:]
            assert "机械臂".encode() in tail, f"退格后前缀丢失；尾部: {tail!r}"
            assert b"\xef\xbf\xbd" not in tail, f"出现 U+FFFD 替换符（半个 CJK 残骸）: {tail!r}"

            # 清空输入行（继续退格到空），避免影响后续步骤。
            for _ in range(3):
                session.send("\x7f")
                time.sleep(0.3)

            # 3. bracketed paste：多行内容作为一次粘贴，不逐行提交
            #    （若逐行提交会触发模型请求——fake 会记录）。
            requests_before = len(fake.fake.requests)
            session.send("\x1b[200~第一行：检查电池\n第二行：检查关节\x1b[201~")
            time.sleep(1.5)
            assert len(fake.fake.requests) == requests_before, (
                "bracketed paste 被当成逐行提交（产生了模型请求）"
            )
            assert "第二行：检查关节".encode() in session.output, (
                f"粘贴内容未完整回显；尾部: {session.output[-400:]!r}"
            )
            # 清空粘贴内容（Ctrl+U 类 kill-line 不行就逐字退格）。
            session.send("\x15")  # Ctrl+U
            time.sleep(0.5)

            # 4. resize：两次改变窗口尺寸，TUI 必须存活且能交互。
            _set_winsize(session.master, 30, 100)
            time.sleep(1.0)
            _set_winsize(session.master, 50, 160)
            time.sleep(1.0)
            assert session.proc.poll() is None, "resize 后进程退出"
            # resize 后仍能提交对话（fake 会答固定问候）。
            session.send("你好\r")
            session.expect("你好，我是 ROSClaw".encode(), timeout=90)

            # 干净退出。
            session.send("/quit\r")
            session.proc.wait(timeout=30)
            assert session.proc.returncode == 0, session.output[-300:]
        finally:
            session.stop()
            fake.close()
