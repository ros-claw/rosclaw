"""G-3（0916 三审 B-6）：chat 启动自动登录流。

三审稿：chat 缺配置/凭据不应把用户推出去跑 setup（绕路）——
缺配置直接进配置流。PR-1 后 /login 是 chat 内唯一交互登录
机制（OAuth/API key 原生可用），所以自动登录流 = TTY 确认
→ 写内置默认配置 → 直接进 chat（会话内 /login）。

闭环断言：
- TTY 答 Y：写默认配置（configure_model kimi-code）且进入 chat
  （_chat_pi 被调用）；
- TTY 答 n：rc=2，不写配置，不进 chat；
- 非 TTY：rc=2 + 指引（脚本化诚实，不交互悬挂）；
- 已配置：不打扰直接进 chat。
"""
from __future__ import annotations

import io
from pathlib import Path

from rosclaw.agentd import cli as agentd_cli


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _args(home: Path):
    return type("A", (), {"home": str(home), "legacy": False, "engine": None})()


class TestChatAutoLogin:
    def test_tty_yes_configures_and_enters(self, tmp_path, monkeypatch) -> None:
        entered = []
        monkeypatch.setattr(agentd_cli, "_chat_pi", lambda h, a: entered.append(h) or 0)
        monkeypatch.setattr("sys.stdin", _Tty("y\n"))
        monkeypatch.setattr("sys.stdout", _Tty())
        rc = agentd_cli.cmd_chat(_args(tmp_path))
        assert rc == 0
        assert entered == [tmp_path], "应答 Y 必须直接进入 chat"
        from rosclaw.agentd.pi_config import pi_model_configured

        assert pi_model_configured(tmp_path), "必须写入默认配置"

    def test_tty_no_aborts_without_writing(self, tmp_path, monkeypatch) -> None:
        entered = []
        monkeypatch.setattr(agentd_cli, "_chat_pi", lambda h, a: entered.append(h) or 0)
        monkeypatch.setattr("sys.stdin", _Tty("n\n"))
        monkeypatch.setattr("sys.stdout", _Tty())
        rc = agentd_cli.cmd_chat(_args(tmp_path))
        assert rc == 2
        assert not entered
        from rosclaw.agentd.pi_config import pi_model_configured

        assert not pi_model_configured(tmp_path), "答 n 不得写配置"

    def test_non_tty_honest_error(self, tmp_path, monkeypatch) -> None:
        entered = []
        monkeypatch.setattr(agentd_cli, "_chat_pi", lambda h, a: entered.append(h) or 0)
        monkeypatch.setattr("sys.stdin", io.StringIO(""))  # 非 TTY
        monkeypatch.setattr("sys.stdout", io.StringIO())
        rc = agentd_cli.cmd_chat(_args(tmp_path))
        assert rc == 2
        assert not entered

    def test_configured_home_not_disturbed(self, tmp_path, monkeypatch) -> None:
        from rosclaw.agentd.onboarding import configure_model

        configure_model(tmp_path, "kimi-code")
        entered = []
        monkeypatch.setattr(agentd_cli, "_chat_pi", lambda h, a: entered.append(h) or 0)
        # 无 TTY 也必须直接进（已配置无需交互）。
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        monkeypatch.setattr("sys.stdout", io.StringIO())
        rc = agentd_cli.cmd_chat(_args(tmp_path))
        assert rc == 0
        assert entered == [tmp_path]
