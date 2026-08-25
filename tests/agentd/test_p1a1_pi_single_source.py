"""P1-A1 红测试（0824 总纲 §10.1/P1-A）：模型配置与探测单源化。

复现的真实事故：setup 把模型配置写进 ``~/.rosclaw/config.yaml``，
而 chat 实际消费 ``~/.rosclaw/agent/{settings,models}.json``（Pi
ModelRuntime）——setup 后 chat 看到的是另一套（可能空的）配置，
doctor 的 probe 走 Python 自带 HTTP chat 栈（OpenAICompatGateway）
而不是 Pi ModelRuntime，同一台机器三个"真相"。

本文件断言单一事实源：

1. ``configure_model`` 写 Pi 配置（settings.json defaultProvider/
   defaultModel + models.json provider 条目），apiKey 只写 ``$ENV``
   引用——绝不写 config.yaml 模型段、绝不落任何原始 key；
2. chat 准入门槛读 Pi 配置（不再读 config.yaml profiles）；
3. ``probe_home``/doctor 的 probe 经 Pi engine（node main.js --probe），
   不再构造 OpenAICompatGateway；engine 缺失时诚实报错；
4. doctor 的就绪判定基于 Pi 配置 + Pi probe。
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from rosclaw.agentd import onboarding
from rosclaw.agentd.onboarding import configure_model, doctor


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestConfigureModelSingleSource:
    def test_writes_pi_settings_and_models(self, tmp_path: Path) -> None:
        summary = configure_model(tmp_path, "kimi-code")
        assert summary["configured"] is True
        settings = _read(tmp_path / "agent" / "settings.json")
        assert settings["defaultProvider"] == "kimi-code"
        assert settings["defaultModel"]  # 非空（模板 k3）
        models = _read(tmp_path / "agent" / "models.json")
        provider = models["providers"]["kimi-code"]
        assert provider["baseUrl"] == summary["base_url"]
        assert provider["api"]  # openai-completions 族
        # key 只写 $ENV 引用——与生产 home（tests live gate）同构。
        assert provider["apiKey"] == "$ROSCLAW_KIMI_API_KEY"
        entry_ids = [m["id"] for m in provider["models"]]
        assert settings["defaultModel"] in entry_ids

    def test_never_writes_raw_key_or_config_yaml_model_section(self, tmp_path: Path) -> None:
        configure_model(tmp_path, "kimi-code")
        for path in (
            tmp_path / "agent" / "models.json",
            tmp_path / "agent" / "settings.json",
        ):
            text = path.read_text(encoding="utf-8")
            assert "sk-" not in text, f"{path} 含原始 key 材料"
        config = tmp_path / "config.yaml"
        if config.exists():
            assert "profiles:" not in config.read_text(encoding="utf-8"), (
                "模型 profiles 仍写进 config.yaml——配置双源"
            )

    def test_refuses_raw_key_material(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="env:"):
            configure_model(
                tmp_path,
                "openai-compat",
                base_url="https://example.invalid/v1",
                model="m",
                api_key_ref="sk-rawkeymaterial",
            )

    def test_custom_provider_roundtrip(self, tmp_path: Path) -> None:
        summary = configure_model(
            tmp_path,
            "openai-compat",
            base_url="https://example.invalid/v1",
            model="my-model",
            api_key_ref="env:MY_PROVIDER_KEY",
        )
        assert summary["configured"] is True
        models = _read(tmp_path / "agent" / "models.json")
        provider = models["providers"]["openai-compat"]
        assert provider["baseUrl"] == "https://example.invalid/v1"
        assert provider["apiKey"] == "$MY_PROVIDER_KEY"
        settings = _read(tmp_path / "agent" / "settings.json")
        assert settings["defaultModel"] == "my-model"

    def test_preserves_existing_settings_keys(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "settings.json").write_text(
            json.dumps({"hideThinkingBlock": True}), encoding="utf-8"
        )
        configure_model(tmp_path, "kimi-code")
        settings = _read(agent_dir / "settings.json")
        assert settings["hideThinkingBlock"] is True
        assert settings["defaultProvider"] == "kimi-code"


class TestChatGateSingleSource:
    def test_chat_gate_accepts_pi_config_without_config_yaml(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """chat 准入读 Pi 配置——只有 Pi 文件（无 config.yaml）必须放行。"""
        from rosclaw.agentd import cli as agentd_cli

        configure_model(tmp_path, "kimi-code")
        monkeypatch.setattr(agentd_cli, "_chat_pi", lambda home, args: 0)
        rc = agentd_cli.main(["--home", str(tmp_path), "chat"])
        assert rc == 0

    def test_chat_gate_rejects_legacy_only_config(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        """只有 legacy config.yaml（旧事故形态）→ 诚实拒绝并指向 setup。"""
        from rosclaw.agentd import cli as agentd_cli
        from rosclaw.agentd.cli import main as agentd_main

        def _forbidden_chat(home, args):  # pragma: no cover - 红期防空转
            raise AssertionError("legacy-only 配置不应进入 _chat_pi")

        monkeypatch.setattr(agentd_cli, "_chat_pi", _forbidden_chat)

        (tmp_path / "config.yaml").write_text(
            "agent:\n  enabled: true\n  default_profile: p\n"
            "models:\n  backend: legacy\n  profiles:\n    p:\n"
            "      provider: kimi_code\n      model: k3\n"
            "      base_url: https://api.kimi.com/coding/v1\n"
            "      api_key_ref: env:ROSCLAW_KIMI_API_KEY\n",
            encoding="utf-8",
        )
        rc = agentd_main(["--home", str(tmp_path), "chat"])
        assert rc == 2
        assert "setup model" in capsys.readouterr().err


class TestProbeViaPiEngine:
    def test_onboarding_has_no_legacy_gateway_probe(self) -> None:
        """结构上根除：onboarding 不再引用 OpenAICompatGateway。"""
        source = inspect.getsource(onboarding)
        assert "OpenAICompatGateway" not in source

    def test_probe_home_invokes_pi_engine(self, tmp_path: Path, monkeypatch) -> None:
        """probe_home 经 node main.js --probe（同一 ModelRuntime）。"""
        import asyncio

        from rosclaw.agentd import pi_entry

        configure_model(tmp_path, "kimi-code")
        # CI 无已构建 dist——engine 发现打桩（engine-missing 有独立用例）。
        monkeypatch.setattr(pi_entry, "find_pi_agent_entry", lambda: ("node", "/fake/main.js"))
        calls: list[list[str]] = []

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                payload = json.dumps(
                    {
                        "engine": "pi",
                        "reachable": True,
                        "auth_configured": True,
                        "models_visible": ["k3"],
                        "expected_model_present": True,
                        "chat_ok": True,
                        "tool_call_ok": True,
                        "provider": "kimi-code",
                        "model": "k3",
                    }
                ).encode()
                return payload, b""

        async def fake_exec(*cmd, **kwargs):
            calls.append([str(c) for c in cmd])
            return _FakeProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        result = asyncio.run(onboarding.probe_home(tmp_path))
        assert result.reachable is True
        assert result.chat_ok is True and result.tool_call_ok is True
        assert result.expected_model_present is True
        assert calls, "probe 未发起子进程"
        cmdline = " ".join(calls[0])
        assert "main.js" in cmdline and "--probe" in cmdline

    def test_probe_home_engine_missing_is_honest(self, tmp_path: Path, monkeypatch) -> None:
        import asyncio

        from rosclaw.agentd import pi_entry

        configure_model(tmp_path, "kimi-code")
        monkeypatch.setattr(pi_entry, "find_pi_agent_entry", lambda: None)
        monkeypatch.setattr(onboarding, "_find_pi_agent_entry", lambda: None, raising=False)
        result = asyncio.run(onboarding.probe_home(tmp_path))
        assert result.reachable is False
        assert "PI_ENGINE" in (result.error or "")

    def test_probe_home_failure_payload_honest(self, tmp_path: Path, monkeypatch) -> None:
        """Pi probe 报告失败（如 401）→ 原样透传，不粉饰。"""
        import asyncio

        from rosclaw.agentd import pi_entry

        configure_model(tmp_path, "kimi-code")
        monkeypatch.setattr(pi_entry, "find_pi_agent_entry", lambda: ("node", "/fake/main.js"))

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                payload = json.dumps(
                    {
                        "engine": "pi",
                        "reachable": False,
                        "auth_configured": True,
                        "models_visible": [],
                        "expected_model_present": False,
                        "chat_ok": False,
                        "tool_call_ok": False,
                        "error": "AUTH_FAILED: 401",
                    }
                ).encode()
                return payload, b""

        async def fake_exec(*cmd, **kwargs):
            return _FakeProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        result = asyncio.run(onboarding.probe_home(tmp_path))
        assert result.reachable is False
        assert "401" in (result.error or "")


class TestDoctorSingleSource:
    def _write_pi_config(self, home: Path) -> None:
        configure_model(home, "kimi-code")

    def test_doctor_ready_with_pi_config_and_probe(self, tmp_path: Path, monkeypatch) -> None:

        from rosclaw.agentd.models.gateway import ModelProbeResult

        self._write_pi_config(tmp_path)
        monkeypatch.setenv("ROSCLAW_KIMI_API_KEY", "sk-kimi-doctor-test")
        monkeypatch.setattr(
            onboarding,
            "probe_home",
            lambda home: _async_result(
                ModelProbeResult(
                    reachable=True,
                    models_visible=("k3",),
                    expected_model_present=True,
                    chat_ok=True,
                    tool_call_ok=True,
                )
            ),
        )
        report = doctor(tmp_path)
        assert report["status"] == "READY"
        assert report["probe"]["chat_ok"] is True
        assert report["probe"]["tool_call_ok"] is True
        assert report["api_key_present"] is True

    def test_doctor_not_ready_without_pi_config(self, tmp_path: Path) -> None:
        report = doctor(tmp_path)
        assert report["status"] == "MODEL_NOT_READY"
        assert "setup model" in report["reason"]


async def _async_result(value):
    return value
