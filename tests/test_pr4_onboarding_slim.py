"""0914 PR-4：上手收敛——setup status 口径 + 交互条款。

实证（0914 审计 §6 + 代码核验 setup_cli.py:195）：
`needs = [k for k,v ... if v.state != "READY"]`——Worker=REMOVED
（H9 有意删除）与可选项（integration/operator/robot_kit）全计入
"未完成"，SIM 聊天用户被永远告知"没配完"。

闭环断言：
- 当前目标（SIM 聊天）的必需项只有 model；REMOVED 永远不进
  "未完成"；可选项移入详细页（--json 仍全量）；
- 交互条款进系统提示：内部 ID（artifact/trace/mission/lease）默认
  不占答案；问候/闲聊只短答不复述上下文状态。
"""

from __future__ import annotations

import inspect
from pathlib import Path

from rosclaw import setup_cli


class TestSetupStatusGoalAware:
    def test_removed_worker_never_in_needs(self, tmp_path, monkeypatch, capsys) -> None:
        """REMOVED（有意删除）不是"未完成"——Worker V2 落地前用户
        不应被要求配置一个不存在的东西。"""
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        args = type("A", (), {"json": False})()
        setup_cli._cmd_status(args)
        out = capsys.readouterr().out
        if "未完成" in out:
            assert "worker" not in out.split("未完成")[1].lower(), out

    def test_sim_chat_goal_needs_only_model(self, tmp_path, monkeypatch, capsys) -> None:
        """SIM 聊天目标：model 就绪即"当前目标已就绪"——integration/
        operator/robot_kit 等可选不计入未完成（高级配置归详细页）。"""
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        # 写好模型配置（内置 kimi-coding 即就绪态；probe 打桩为可用——
        # 状态判定依赖真实探测，单测环境无 engine）。
        from rosclaw.agentd import onboarding
        from rosclaw.agentd.models.gateway import ModelProbeResult
        from rosclaw.agentd.onboarding import configure_model

        async def _probe(home, *, deep=False):
            return ModelProbeResult(
                reachable=True, chat_ok=True, tool_call_ok=True,
                auth_configured=True,
            )

        configure_model(tmp_path, "kimi-code")
        monkeypatch.setattr(onboarding, "probe_home", _probe)
        args = type("A", (), {"json": False})()
        rc = setup_cli._cmd_status(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "未完成" not in out, f"可选/REMOVED 被计入未完成: {out}"

    def test_missing_model_is_the_need(self, tmp_path, monkeypatch, capsys) -> None:
        """model 未配置 = 唯一未完成项（指向 setup model——不是一串
        无关区域）。"""
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        args = type("A", (), {"json": False})()
        setup_cli._cmd_status(args)
        out = capsys.readouterr().out
        assert "未完成" in out
        needs_part = out.split("未完成")[1]
        assert "model" in needs_part
        assert "integration" not in needs_part
        assert "worker" not in needs_part

    def test_json_still_full_report(self, tmp_path, monkeypatch, capsys) -> None:
        """--json 仍输出全量区域（详细页契约不变）。"""
        import json

        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        args = type("A", (), {"json": True})()
        setup_cli._cmd_status(args)
        doc = json.loads(capsys.readouterr().out)
        for area in ("model", "body", "worker", "integration"):
            assert area in doc


class TestInteractionPromptContract:
    def test_prompt_forbids_internal_ids_in_answers(self) -> None:
        """系统提示必须含：内部 ID（artifact/trace/mission/lease 等）
        默认不占答案——0914 实证问候/结果里倒内部标识。"""
        prompt = Path(
            "src/rosclaw/agentd/context/prompts/native_agent_v2.md"
        ).read_text(encoding="utf-8")
        lowered = prompt.lower()
        assert "artifact" in lowered and ("trace" in lowered or "mission" in lowered)
        assert (
            "不占答案" in prompt
            or "do not include internal" in lowered
            or "internal id" in lowered
        ), "提示缺少内部 ID 不占答案条款"

    def test_prompt_greeting_no_context_recital(self) -> None:
        """问候/闲聊只短答——不复述 mission/body/approval 等上下文
        状态（0914 实证问候倒出内部状态）。"""
        prompt = Path(
            "src/rosclaw/agentd/context/prompts/native_agent_v2.md"
        ).read_text(encoding="utf-8")
        assert "问候" in prompt or "greeting" in prompt.lower()
        assert "复述" in prompt or "recite" in prompt.lower(), (
            "问候条款未禁止复述上下文状态"
        )


class TestArtifactSshSemantics:
    def test_open_headless_exit_codes(self, tmp_path, monkeypatch) -> None:
        """U09 契约：无显示环境退出码区分——成功(0 含导出指引)/
        未知(2)/文件缺失(3)。"""
        from rosclaw.agentd import cli as agentd_cli

        src = inspect.getsource(agentd_cli.cmd_artifact_open)
        assert "WAYLAND_DISPLAY" in src or "DISPLAY" in src
        assert "return 2" in src and "return 3" in src
        assert "artifact export" in src, "headless 指引必须给出真实导出命令"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
