"""T-TMUX（二次审计 Gate E：IME/resize/tmux/SSH 环境腿）。

真实 tmux 会话里跑安装产物的 chat：品牌 header 渲染、中文对话、
/quit 干净退出、tmux 窗格无乱码。SSH 腿与 tmux 等价（都是
"远程/复用终端里的全屏 TUI"），本机无 sshd 时以 tmux 为准并
在报告里如实标注。
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from tests.agentd.test_product_journey import FakeModelServer, _build_and_install
from tests.agentd.test_start_exit_soak import _write_fake_home

TMUX = "/usr/bin/tmux"
SESSION = "rosclaw-gate-e"


def _tmux(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [TMUX, *args], capture_output=True, text=True, timeout=30, env=env
    )


def _capture() -> str:
    result = _tmux("capture-pane", "-t", SESSION, "-p")
    return result.stdout


@pytest.mark.slow
class TestTmuxEnvironment:
    def test_chat_inside_tmux(self, tmp_path: Path) -> None:
        fake = FakeModelServer(log_path=tmp_path / "fake-requests.jsonl")
        prefix, _root = _build_and_install(tmp_path)
        home = tmp_path / "rh"
        _write_fake_home(home, fake.base_url)
        rosclaw = prefix / "bin" / "rosclaw"
        env = dict(
            os.environ,
            ROSCLAW_HOME=str(home),
            TERM="screen-256color",
            FAKE_JOURNEY_KEY="sk-fake-journey",
            KIMI_API_KEY="sk-fake-journey",
            PATH=f"{prefix / 'bin'}:{os.environ['PATH']}",
        )
        _tmux("kill-session", "-t", SESSION)
        try:
            # tmux server 有自己的环境——变量必须内联进 shell 命令。
            shell_cmd = (
                f"ROSCLAW_HOME={home} TERM=screen-256color "
                f"FAKE_JOURNEY_KEY=sk-fake-journey KIMI_API_KEY=sk-fake-journey "
                f"PATH={prefix / 'bin'}:$PATH "
                f"{rosclaw} chat --engine pi"
            )
            result = _tmux(
                "new-session", "-d", "-s", SESSION, "-x", "120", "-y", "40",
                shell_cmd,
                env=env,
            )
            assert result.returncode == 0, result.stderr
            # 进程退出后保留窗格最后一帧——否则 /quit 后 pane 立即销毁，
            # resume 提示根本抓不到。
            _tmux("set-option", "-t", SESSION, "remain-on-exit", "on")
            # 1. 品牌 header。
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and "ROSClaw Native Agent" not in _capture():
                time.sleep(1.0)
            pane = _capture()
            assert "ROSClaw Native Agent" in pane, f"tmux 里 header 未渲染: {pane[-400:]}"
            assert "engine=pi" not in pane

            # 2. 中文对话（tmux send-keys 逐字送入）。
            _tmux("send-keys", "-t", SESSION, "你好", "Enter")
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and "你好，我是 ROSClaw" not in _capture():
                time.sleep(1.0)
            pane = _capture()
            assert "你好，我是 ROSClaw" in pane, f"tmux 里中文回合未完成: {pane[-400:]}"
            # 静止帧判定：连续两次抓屏一致且非空——重绘中途的瞬时帧
            # 可能是空屏或带半个宽字符（tmux 在边缘以占位符渲染）。
            settled = ""
            for _ in range(40):
                first = _capture()
                time.sleep(0.3)
                second = _capture()
                if first and first == second:
                    settled = second
                    break
                time.sleep(0.3)
            assert settled, "tmux 画面 24s 未静止"
            assert "\ufffd" not in settled, f"tmux 静止帧出现乱码替换符: {settled[-400:]}"

            # 3. /quit 干净退出——tmux 窗格回到 shell 或会话结束。
            _tmux("send-keys", "-t", SESSION, "/quit", "Enter")
            deadline = time.monotonic() + 30
            exited = False
            last_pane = ""
            while time.monotonic() < deadline:
                probe = _tmux("list-panes", "-t", SESSION, "-F", "#{pane_dead}")
                pane = _capture()
                if pane:
                    # 进程退出后窗格可能关闭——留住最后有内容的一帧。
                    last_pane = pane
                if probe.returncode != 0 or "1" in probe.stdout:
                    exited = True
                    break
                if "rosclaw chat --resume" in pane:
                    exited = True
                    break
                time.sleep(1.0)
            assert exited, f"/quit 后会话未结束: {last_pane[-400:]}"
            assert "rosclaw chat --resume" in last_pane, (
                f"退出未见 ROSClaw resume 提示: {last_pane[-400:]}"
            )
            assert "pi --session" not in last_pane
        finally:
            _tmux("kill-session", "-t", SESSION)
            fake.close()
