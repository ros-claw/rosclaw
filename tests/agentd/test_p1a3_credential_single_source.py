"""P1-A3 红测试（0824 总纲 §10.1/P1-A）：credential 单源。

复现的真实事故：模型凭据有三个来源——进程 env、Pi
``agent/auth.json``、以及 legacy ``agentd/credentials.json``（经
ModelCredentialBroker 在 chat/doctor/init 前一次性注入进程 env）。
同一 key 多处存放/多条解析路径，setup 后仍可能"另一套凭据"。

本文件断言：

1. ``agentd/credentials.py``（AgentCredentialStore/ModelCredentialBroker）
   与 ``_load_stored_credentials``/``agentd credential`` 子命令全部
   移除——凭据来源只有 env 与 Pi auth.json；
2. legacy credentials.json 即使存在也**绝不注入**进程 env（doctor/
   chat gate 路径都不读值）；
3. doctor 的 credential_sources 只报告 env / auth.json 来源；legacy
   文件只报"已停用"事实（env 名可列，值绝不出现）；
4. ACP serve 同样不再 inject legacy store。
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

from rosclaw.agentd import cli as agentd_cli
from rosclaw.agentd import onboarding
from rosclaw.agentd.cli import main as agentd_main

_LEGACY_STORE = json.dumps({"environment": {"ROSCLAW_FAKE_KEY": "sk-fake-legacy"}})


def _write_legacy_store(home: Path) -> None:
    store_dir = home / "agentd"
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / "credentials.json").write_text(_LEGACY_STORE, encoding="utf-8")


class TestLegacySurfaceRemoved:
    def test_credentials_module_deleted(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            import importlib

            importlib.import_module("rosclaw.agentd.credentials")

    def test_cli_has_no_injection_step_or_credential_command(self) -> None:
        source = inspect.getsource(agentd_cli)
        assert "_load_stored_credentials" not in source
        assert "AgentCredentialStore" not in source
        assert "ModelCredentialBroker" not in source
        assert "cmd_credential" not in source

    def test_credential_subcommand_rejected(self, tmp_path: Path, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            agentd_main(
                ["--home", str(tmp_path), "credential", "status", "--provider", "kimi-code"]
            )
        assert excinfo.value.code == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_acp_serve_does_not_inject_legacy_store(self, tmp_path: Path) -> None:
        from rosclaw.adapters.acp import cli as acp_cli

        source = inspect.getsource(acp_cli)
        assert "AgentCredentialStore" not in source


class TestLegacyStoreNeverInjected:
    def test_chat_gate_does_not_inject(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """legacy store 存在 + 未配置模型 → chat gate 拒绝；且 FAKE_KEY
        绝不进入进程 env。"""
        _write_legacy_store(tmp_path)
        monkeypatch.delenv("ROSCLAW_FAKE_KEY", raising=False)
        monkeypatch.setattr(agentd_cli, "_chat_pi", lambda home, args: 0)
        rc = agentd_main(["--home", str(tmp_path), "chat"])
        assert rc == 2  # 未配置模型——legacy store 不算配置
        assert "ROSCLAW_FAKE_KEY" not in os.environ

    def test_doctor_does_not_inject(self, tmp_path: Path, monkeypatch) -> None:
        _write_legacy_store(tmp_path)
        monkeypatch.delenv("ROSCLAW_FAKE_KEY", raising=False)
        report = onboarding.doctor(tmp_path)
        assert "ROSCLAW_FAKE_KEY" not in os.environ
        # 报告全文绝不含 legacy 值。
        assert "sk-fake-legacy" not in json.dumps(report, ensure_ascii=False)


class TestDoctorCredentialSources:
    def test_sources_cover_env_and_pi_auth_only(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ROSCLAW_KIMI_API_KEY", "sk-kimi-source-test")
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "auth.json").write_text(
            json.dumps({"kimi-code": {"type": "api_key", "key": "sk-stored"}}),
            encoding="utf-8",
        )
        report = onboarding.doctor(tmp_path)
        sources = report["credential_sources"]
        text = json.dumps(sources, ensure_ascii=False)
        assert "sk-stored" not in text, "auth.json 的值被打印"
        kinds = {entry.get("source") for entry in sources}
        assert kinds <= {"env", "pi-auth-file", "none", "legacy-disabled"}, kinds
        kimi = next(
            e for e in sources if e.get("env_name") in ("ROSCLAW_KIMI_API_KEY", "KIMI_API_KEY")
        )
        assert kimi["source"] == "env"
        assert kimi.get("fingerprint")

    def test_legacy_file_reported_as_disabled(self, tmp_path: Path, monkeypatch) -> None:
        _write_legacy_store(tmp_path)
        monkeypatch.delenv("ROSCLAW_FAKE_KEY", raising=False)
        report = onboarding.doctor(tmp_path)
        legacy = [e for e in report["credential_sources"] if e.get("source") == "legacy-disabled"]
        assert legacy, "legacy 文件未被如实标注"
        assert "ROSCLAW_FAKE_KEY" in legacy[0].get("env_names", [])
        assert "sk-fake-legacy" not in json.dumps(legacy)
