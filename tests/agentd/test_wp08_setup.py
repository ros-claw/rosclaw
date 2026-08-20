"""WP-P0-8 红测试（总纲 §11.1/WP-P0-8）：Setup 契约修复。

红测试先行——`rosclaw setup` 声称覆盖模型/机器人/集成，但缺 SIM
策略、语言和"第一个可验证仿真任务"：

1. setup safety：查看/设置 SIM 审批策略（auto|ask-every-time）；
2. setup language：查看/设置 UI 语言（zh-CN|en-US|auto）；
3. setup demo：模型无关的第一个可验证任务（draw_shape 内核直跑，
   VERIFIED + receipt）——不是打印一条建议命令；
4. status 覆盖 safety/language；help 列出全部子命令。
"""

from __future__ import annotations

import json
from pathlib import Path


class TestSetupSafety:
    def test_safety_set_and_show(self, tmp_path: Path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        from rosclaw.setup_cli import dispatch_setup_argv

        assert dispatch_setup_argv(["setup", "safety", "ask-every-time"]) == 0
        data = json.loads(
            (tmp_path / "agent" / "safety.json").read_text(encoding="utf-8")
        )
        assert data["sim_policy"] == "ask"
        assert dispatch_setup_argv(["setup", "safety"]) == 0
        out = capsys.readouterr().out
        assert "ask" in out

    def test_safety_rejects_invalid(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        from rosclaw.setup_cli import dispatch_setup_argv

        assert dispatch_setup_argv(["setup", "safety", "yolo"]) == 2
        assert not (tmp_path / "agent" / "safety.json").exists()


class TestSetupLanguage:
    def test_language_set_and_show(self, tmp_path: Path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        from rosclaw.setup_cli import dispatch_setup_argv

        assert dispatch_setup_argv(["setup", "language", "zh-CN"]) == 0
        data = json.loads(
            (tmp_path / "agent" / "locale.json").read_text(encoding="utf-8")
        )
        assert data.get("ui_locale") == "zh-CN"
        assert dispatch_setup_argv(["setup", "language"]) == 0
        assert "zh-CN" in capsys.readouterr().out

    def test_language_rejects_invalid(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        from rosclaw.setup_cli import dispatch_setup_argv

        assert dispatch_setup_argv(["setup", "language", "klingon"]) == 2


class TestSetupDemo:
    def test_demo_runs_verified_task(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """setup demo：内核直跑 draw_shape（无需模型）——VERIFIED +
        证据等级诚实标注。"""
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        from rosclaw.setup_cli import dispatch_setup_argv

        code = dispatch_setup_argv(["setup", "demo"])
        assert code == 0
        out = capsys.readouterr().out
        assert "VERIFIED" in out, out
        assert "PASS" in out
        # 诚实证据等级（WP-P0-6）：动力学 rollout，非真机证据。
        assert "非真机证据" in out or "不能证明" in out
        # 真实产生了 trace/动画产物（PR-H9：demo 直跑
        # SimTrajectoryService——task_records 已删）。
        assert "trace.json" in out
        assert (tmp_path / "sim" / "traces").exists()


class TestSetupStatusCoversAll:
    def test_status_has_safety_language(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        from rosclaw.setup_cli import _collect_status

        status = _collect_status(tmp_path)
        assert "safety" in status, status.keys()
        assert "language" in status
        assert status["safety"]["state"] in ("READY", "NEEDS_SETUP")

    def test_help_lists_all_subcommands(self) -> None:
        from rosclaw.setup_cli import _HELP

        for sub in ("model", "body", "safety", "language", "worker",
                    "integration", "demo", "status"):
            assert f"setup {sub}" in _HELP, f"help 缺 setup {sub}"
