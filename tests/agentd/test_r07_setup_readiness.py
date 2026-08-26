"""R0-7 红测试（0826 体验审计 §5.R0-7）：模型配置简化与 readiness
状态格。

真实事故（0826 体验旅程）：
- `setup model` 报 MODEL_NOT_READY，随后 chat 正常完成工具调用
  ——tool probe 超时覆盖了 chat 成功的事实；
- setup 默认跑完整 4 步探测（models+chat+严格 tool call，最长
  60s 两次模型请求）+ 一百多行 JSON——首次配置过重；
- configure_model(reasoning_effort="high") 实际忽略该参数——
  high 从未写入 Pi 设置。

断言：
1. effort 真写：configure_model 把 defaultThinkingLevel 写入
   agent/settings.json（保留其他键）；
2. 分级探测：doctor 默认便宜探测（无严格 tool call），
   --deep 才跑完整 tool call；
3. 状态格：UNCONFIGURED/AUTH_READY/CHAT_READY/TOOL_READY/
   DEGRADED——chat_ok=True + tool_call_ok=False → DEGRADED
   （不是 MODEL_NOT_READY）；tool probe 失败不覆盖 chat 事实；
4. 真实工具调用成功 → readiness 升级 TOOL_READY（账本证据
   优先于探测）；
5. setup 默认输出 ≤6 行人类摘要；--json 出完整报告。
"""

from __future__ import annotations

import json
from pathlib import Path


def _stub_pi_entry(monkeypatch) -> None:
    """CI 无内置 dist——stub engine 查找（P1-A1 的 CI 教训）。"""
    from rosclaw.agentd import pi_entry

    monkeypatch.setattr(
        pi_entry, "find_pi_agent_entry", lambda: ("node", "/fake/main.js")
    )


def _configured_home(tmp_path: Path) -> Path:
    from rosclaw.agentd.onboarding import configure_model

    configure_model(tmp_path, "kimi-code", api_key_ref="env:ROSCLAW_KIMI_API_KEY")
    return tmp_path


class TestEffortWritten:
    def test_configure_model_writes_thinking_level(
        self, tmp_path: Path
    ) -> None:
        """reasoning_effort="high" 必须真实写入 Pi settings
        （defaultThinkingLevel）——不再被忽略。"""
        from rosclaw.agentd.onboarding import configure_model

        configure_model(
            tmp_path, "kimi-code",
            api_key_ref="env:ROSCLAW_KIMI_API_KEY", reasoning_effort="high",
        )
        settings = json.loads(
            (tmp_path / "agent" / "settings.json").read_text(
                encoding="utf-8"
            )
        )
        assert settings.get("defaultThinkingLevel") == "high", settings

    def test_effort_preserves_other_settings(self, tmp_path: Path) -> None:
        """写 effort 不得清空其他 settings 键。"""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "settings.json").write_text(
            json.dumps({"custom_key": "keep-me"}), encoding="utf-8"
        )
        from rosclaw.agentd.onboarding import configure_model

        configure_model(
            tmp_path, "kimi-code",
            api_key_ref="env:ROSCLAW_KIMI_API_KEY", reasoning_effort="medium",
        )
        settings = json.loads(
            (agent_dir / "settings.json").read_text(encoding="utf-8")
        )
        assert settings.get("custom_key") == "keep-me"
        assert settings.get("defaultThinkingLevel") == "medium"


class TestProbeLevels:
    def test_doctor_default_cheap_probe(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """doctor 默认便宜探测（不调严格 tool-call 步）；--deep 才
        完整。"""
        _configured_home(tmp_path)
        calls: list[list[str]] = []

        class _Proc:
            def __init__(self, argv):
                self.argv = argv

            async def communicate(self):
                return (b'{"engine": "pi", "reachable": true, '
                        b'"chat_ok": true, "tool_call_ok": true, '
                        b'"models_visible": ["m"], '
                        b'"expected_model_present": true}\n', b"")

            @property
            def returncode(self):
                return 0

        import asyncio

        async def fake_exec(*argv, **kwargs):
            calls.append([str(a) for a in argv])
            return _Proc(argv)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setenv("ROSCLAW_KIMI_API_KEY", "sk-test-fake")
        _stub_pi_entry(monkeypatch)
        from rosclaw.agentd.onboarding import doctor

        report = doctor(tmp_path)
        assert report["status"] in ("CHAT_READY", "TOOL_READY", "DEGRADED")
        probe_argv = calls[-1] if calls else []
        assert not any("--deep" in a for a in probe_argv), probe_argv

        report_deep = doctor(tmp_path, deep=True)
        deep_argv = calls[-1] if calls else []
        assert any("--deep" in a for a in deep_argv), deep_argv
        assert report_deep["probe"]["tool_call_ok"] is True


class TestReadinessTaxonomy:
    def _doctor_with_probe(self, tmp_path: Path, monkeypatch, probe_json: dict):
        _configured_home(tmp_path)

        class _Proc:
            async def communicate(self):
                payload = {"engine": "pi", **probe_json}
                return (json.dumps(payload).encode() + b"\n", b"")

            @property
            def returncode(self):
                return 0

        import asyncio

        async def fake_exec(*argv, **kwargs):
            return _Proc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setenv("ROSCLAW_KIMI_API_KEY", "sk-test-fake")
        _stub_pi_entry(monkeypatch)
        from rosclaw.agentd.onboarding import doctor

        return doctor(tmp_path, deep=True)

    def test_chat_ok_tool_fail_is_degraded(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """chat_ok=True 但 tool probe 失败 → DEGRADED（对话可用、
        工具自检退化）——不是 MODEL_NOT_READY。"""
        report = self._doctor_with_probe(tmp_path, monkeypatch, {
            "reachable": True, "chat_ok": True, "tool_call_ok": False,
            "models_visible": ["m"], "expected_model_present": True,
            "error": "TOOL_PROBE_TIMEOUT",
        })
        assert report["status"] == "DEGRADED", report["status"]
        assert report["probe"]["chat_ok"] is True

    def test_full_pass_is_tool_ready(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        report = self._doctor_with_probe(tmp_path, monkeypatch, {
            "reachable": True, "chat_ok": True, "tool_call_ok": True,
            "models_visible": ["m"], "expected_model_present": True,
        })
        assert report["status"] == "TOOL_READY", report["status"]

    def test_unconfigured(self, tmp_path: Path) -> None:
        from rosclaw.agentd.onboarding import doctor

        report = doctor(tmp_path)
        assert report["status"] == "UNCONFIGURED", report["status"]

    def test_real_tool_success_upgrades_readiness(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """chat 真实工具调用成功（账本证据）→ TOOL_READY——探测
        失败不覆盖真实成功事实。"""
        _configured_home(tmp_path)
        # 账本里有一条真实成功的具身工具完成记录。
        import sqlite3
        from datetime import UTC, datetime

        from rosclaw.storage.migrations import MigrationRunner

        db_path = tmp_path / "agentd" / "missions.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        MigrationRunner().apply(conn, "sqlite")
        conn.execute(
            "INSERT INTO agent_events (event_id, mission_id, sequence, "
            "type, visibility, payload_json, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "evt_1", "mis_1", 1, "tool.completed", "USER",
                json.dumps({"tool_name": "rosclaw_task", "ok": True}),
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

        class _Proc:
            async def communicate(self):
                return (json.dumps({
                    "engine": "pi",
                    "reachable": True, "chat_ok": True,
                    "tool_call_ok": False, "error": "TOOL_PROBE_TIMEOUT",
                    "models_visible": [], "expected_model_present": False,
                }).encode() + b"\n", b"")

            @property
            def returncode(self):
                return 0

        import asyncio

        async def fake_exec(*argv, **kwargs):
            return _Proc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setenv("ROSCLAW_KIMI_API_KEY", "sk-test-fake")
        _stub_pi_entry(monkeypatch)
        from rosclaw.agentd.onboarding import doctor

        report = doctor(tmp_path, deep=True)
        assert report["status"] == "TOOL_READY", report["status"]


class TestSetupOutput:
    def test_cmd_init_default_summary_six_lines(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        """setup 默认输出 ≤6 行人类摘要（不是一百行 JSON）；
        --json 才出完整报告。"""
        from rosclaw.agentd.cli import main as agentd_main

        monkeypatch.setenv("ROSCLAW_KIMI_API_KEY", "sk-test-fake")

        class _Proc:
            async def communicate(self):
                return (json.dumps({
                    "engine": "pi",
                    "reachable": True, "chat_ok": True,
                    "tool_call_ok": True, "models_visible": ["m"],
                    "expected_model_present": True,
                }).encode() + b"\n", b"")

            @property
            def returncode(self):
                return 0

        import asyncio

        async def fake_exec(*argv, **kwargs):
            return _Proc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        _stub_pi_entry(monkeypatch)
        rc = agentd_main([
            "--home", str(tmp_path), "init",
            "--provider", "kimi-code",
        ])
        out = capsys.readouterr().out
        lines = [line for line in out.strip().splitlines() if line.strip()]
        assert len(lines) <= 8, f"默认输出 {len(lines)} 行（>8）:\n{out}"
        assert rc == 0, out
