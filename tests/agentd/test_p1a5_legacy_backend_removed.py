"""P1-A5 红测试（0824 总纲 §10.1/P1-A）：legacy model backend 清除。

真实事故背景：Pi ModelRuntime 之外并存第二条模型后端——
``models.backend: legacy|modeld`` 配置项、ModeldGateway（批次 D 的
modeld 管理面）、FailoverGateway（无生产实例化）、microcompact
（无消费者）、legacy console 的 /model + service.switch_model、以及
Python/TS 双份 session 查询解析。每一共存面都是"另一套真相"的温床。

本文件断言唯一模型运行时（Pi）：

1. 模块级删除：modeld_gateway / failover / context.compact 不存在；
2. 配置面删除：AgentConfig 无 model_backend；无 `agentd backend`
   子命令；doctor 组件报告不再含 modeld；
3. service 面无 modeld 管理通道（_modeld_mgmt/modeld_providers/
   modeld_models/switch_model）；
4. session 查询解析单份：resume 查询直接传给 Pi 入口（--resume），
   Python 不再重复实现精确 ID→前缀→标题解析。
"""

from __future__ import annotations

import inspect

import pytest

from rosclaw.agentd import cli as agentd_cli
from rosclaw.agentd import onboarding, service


class TestLegacyModulesDeleted:
    @pytest.mark.parametrize(
        "module",
        [
            "rosclaw.agentd.models.modeld_gateway",
            "rosclaw.agentd.models.failover",
            "rosclaw.agentd.context.compact",
        ],
    )
    def test_module_gone(self, module: str) -> None:
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)


class TestBackendConfigSurfaceRemoved:
    def test_agent_config_has_no_model_backend(self) -> None:
        from rosclaw.agentd.config import AgentConfig

        assert not hasattr(AgentConfig(), "model_backend")

    def test_no_backend_subcommand(self, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            agentd_cli.main(["backend"])
        assert excinfo.value.code == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_doctor_components_skip_modeld(self, tmp_path) -> None:
        report = onboarding.doctor(tmp_path)
        assert "modeld" not in report.get("components", {})


class TestServiceModeldPlaneRemoved:
    def test_no_modeld_management_surface(self) -> None:
        source = inspect.getsource(service)
        for marker in (
            "_modeld_mgmt",
            "modeld_providers",
            "modeld_models",
            "ModeldGateway",
            "switch_model",
        ):
            assert marker not in source, f"service 仍含 {marker}"


class TestSessionResolveSingleImplementation:
    def test_python_cli_has_no_duplicate_resolver(self) -> None:
        source = inspect.getsource(agentd_cli)
        assert "resolve_session_query" not in source

    def test_resume_query_passes_through_to_pi(self, tmp_path, monkeypatch) -> None:
        """`rosclaw chat --resume <query>` 原样传查询（TS 单份解析），
        Python 不预先解析成路径。"""
        from rosclaw.agentd import cli as cli_mod
        from rosclaw.agentd.onboarding import configure_model

        seen: list[str] = []

        def fake_chat_pi(home, args):
            seen.append(str(getattr(args, "resume", "")))
            return 0

        monkeypatch.setattr(cli_mod, "_chat_pi", fake_chat_pi)
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        configure_model(tmp_path, "kimi-code")  # chat gate 需要 Pi 配置
        import sys

        from rosclaw.cli import main as rosclaw_main

        monkeypatch.setattr(sys, "argv", ["rosclaw", "chat", "--resume", "abc123"])
        rc = rosclaw_main()
        assert rc == 0
        assert seen == ["abc123"], "查询在 Python 侧被改写/预解析"
