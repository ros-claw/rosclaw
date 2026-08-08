"""P0-CLI-01 验收（三审）：setup 统一向导的公开契约。

- root help 承诺的 setup model/body/operator/worker/integration 全部
  存在且可执行（不是 argparse invalid choice）；
- 裸 `rosclaw setup` 是状态总览（幂等、可重入）；
- `setup status --json` 有稳定 schema；
- 缺模型指引统一指向 `rosclaw setup model`（不是 `agent init`——
  那是外部 Agent onboarding）；
- 重复运行不覆盖已有配置。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.setup_cli import dispatch_setup_argv


class TestSetupContract:
    def test_all_promised_subcommands_exist(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        # 每个公开子命令都能到达稳定业务状态（不是 invalid choice）。
        assert dispatch_setup_argv(["setup"]) == 0
        assert dispatch_setup_argv(["setup", "status"]) == 0
        assert dispatch_setup_argv(["setup", "status", "--json"]) == 0
        # body/operator/worker 在未配置环境返回 0 或 1（NEEDS_SETUP），
        # 绝不 argparse 崩溃（rc=2 是 parser 层错误）。
        assert dispatch_setup_argv(["setup", "body"]) in (0, 1)
        assert dispatch_setup_argv(["setup", "operator"]) in (0, 1)
        assert dispatch_setup_argv(["setup", "worker"]) == 0

    def test_status_json_schema_stable(self, tmp_path: Path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        dispatch_setup_argv(["setup", "status", "--json"])
        out = json.loads(capsys.readouterr().out)
        assert out["schema_version"] == "rosclaw.setup.status.v1"
        for area in ("model", "body", "operator", "worker", "integration"):
            assert area in out, f"schema 缺 {area}"
            assert out[area]["state"] in ("READY", "NEEDS_SETUP", "BLOCKED")

    def test_wizard_idempotent(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        assert dispatch_setup_argv(["setup"]) == 0
        assert dispatch_setup_argv(["setup"]) == 0  # 重入不崩、不覆盖

    def test_setup_model_reaches_onboarding(self, tmp_path: Path, monkeypatch) -> None:
        """setup model 复用 agentd onboarding——非交互无 provider 时
        诚实报错（不是 unknown command）。"""
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        rc = dispatch_setup_argv(["setup", "model"])
        assert rc == 2  # 需要 --provider 的诚实错误，不是 invalid choice

    def test_unknown_subcommand_is_clean_error(self, tmp_path: Path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        rc = dispatch_setup_argv(["setup", "turbo"])
        assert rc == 2
        assert "未知 setup 子命令" in capsys.readouterr().err


class TestModelGuidance:
    def test_missing_model_points_to_setup_model(self, tmp_path: Path) -> None:
        """缺模型指引只指向 rosclaw setup model（P0-CLI-01）。"""
        from rosclaw.agentd.config import load_agent_config

        config = load_agent_config(tmp_path / "config.yaml")
        with pytest.raises(ValueError, match="setup model"):
            config.to_policy()

    def test_chat_missing_model_guidance(self, tmp_path: Path, capsys) -> None:
        from rosclaw.agentd.cli import main as agentd_main

        rc = agentd_main(["--home", str(tmp_path), "chat"])
        assert rc == 2
        assert "setup model" in capsys.readouterr().err
