"""G-2（0916 三审 B-5）：setup status 默认本地-only。

三审稿：「rosclaw setup status 不应默认发起联网模型探测」——
状态页是上手第一步，每次看状态都烧一次模型 API 调用（还受
限流/配额影响把"看状态"变成"报错"）。

契约：
- 默认本地：配置存在性（settings/models 单源）+ 凭据存在性
  （env/auth.json——只看存在，不验证）→ READY/NEEDS_SETUP；
  detail 明示"本地检查——未联网验证"；
- --probe 显式联网：走 doctor 全探测（状态格不变）；
- 默认路径绝不调用 doctor/probe（联网只能用户显式发起）。
"""
from __future__ import annotations

from pathlib import Path

from rosclaw import setup_cli


def _configured_home(home: Path, *, with_key: bool, monkeypatch) -> None:
    from rosclaw.agentd.onboarding import configure_model

    configure_model(home, "kimi-code")
    if with_key:
        monkeypatch.setenv("KIMI_API_KEY", "sk-test-local-only")


def _no_doctor(monkeypatch) -> None:
    from rosclaw.agentd import onboarding

    def _forbidden(home, *, deep=False):
        raise AssertionError("默认 status 竟发起联网探测（doctor 被调用）")

    monkeypatch.setattr(onboarding, "doctor", _forbidden)


class TestStatusLocalOnly:
    def test_default_status_never_calls_doctor(
        self, tmp_path, monkeypatch, capsys,
    ) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        _configured_home(tmp_path, with_key=True, monkeypatch=monkeypatch)
        _no_doctor(monkeypatch)
        args = type("A", (), {"json": False, "probe": False})()
        rc = setup_cli._cmd_status(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "未完成" not in out, out

    def test_default_marks_unverified(
        self, tmp_path, monkeypatch, capsys,
    ) -> None:
        """本地 READY 必须明示未联网验证——READY 语义不夸大。"""
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        _configured_home(tmp_path, with_key=True, monkeypatch=monkeypatch)
        _no_doctor(monkeypatch)
        args = type("A", (), {"json": True, "probe": False})()
        setup_cli._cmd_status(args)
        import json

        doc = json.loads(capsys.readouterr().out)
        assert doc["model"]["state"] == "READY"
        assert "未联网" in doc["model"]["detail"]

    def test_local_missing_credential_is_needs_setup(
        self, tmp_path, monkeypatch, capsys,
    ) -> None:
        """配置了 provider 但无凭据（env/auth.json 皆无）——本地
        即可判 NEEDS_SETUP（不用联网也知道缺 key）。"""
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        _configured_home(tmp_path, with_key=False, monkeypatch=monkeypatch)
        for key in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "ROSCLAW_KIMI_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        _no_doctor(monkeypatch)
        args = type("A", (), {"json": True, "probe": False})()
        setup_cli._cmd_status(args)
        import json

        doc = json.loads(capsys.readouterr().out)
        assert doc["model"]["state"] == "NEEDS_SETUP"

    def test_probe_flag_uses_network_path(
        self, tmp_path, monkeypatch, capsys,
    ) -> None:
        """--probe 显式联网：doctor 路径被调用（探测分级保留）。"""
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        _configured_home(tmp_path, with_key=True, monkeypatch=monkeypatch)
        from rosclaw.agentd import onboarding

        called = []

        def _spy(home, *, deep=False):
            called.append(home)
            return {
                "status": "CHAT_READY",
                "model": {"provider": "kimi-code", "model": "k3"},
            }

        monkeypatch.setattr(onboarding, "doctor", _spy)
        args = type("A", (), {"json": True, "probe": True})()
        setup_cli._cmd_status(args)
        import json

        doc = json.loads(capsys.readouterr().out)
        assert called, "--probe 必须走联网探测"
        assert doc["model"]["state"] == "READY"
        assert doc["model"]["detail"] == "CHAT_READY"
