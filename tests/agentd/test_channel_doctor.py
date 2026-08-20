"""Channel doctor（PR-RC-005）测试。

- ROSClaw 侧检查：home/config/credentials/ACP probe（用 mock-gateway 测试
  server 覆盖 initialize/new/resume/close 全链路）
- OpenClaw 侧缺失时记 SKIP 而不是 FAIL（除非 --require-openclaw）
- doctor 只读：不修改 ROSCLAW_HOME 配置、不写 OpenClaw 配置
"""

from __future__ import annotations

import sys
from pathlib import Path

from rosclaw.integrations.openclaw.doctor import run_doctor_async

_MOCK_CONFIG = """
agent:
  enabled: true
  engine: legacy
  default_profile: mock_default
models:
  profiles:
    mock_default:
      provider: mock
      model: mock-model
      local: true
      capabilities: ["llm.chat"]
"""


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(_MOCK_CONFIG, encoding="utf-8")
    return home


def _names(report) -> dict[str, str]:
    return {c.name: c.status for c in report.checks}


class TestChannelDoctor:
    async def test_missing_home_fails(self, tmp_path: Path) -> None:
        report = await run_doctor_async(tmp_path / "nope", probe_acp=False)
        checks = _names(report)
        assert checks["ROSCLAW_HOME"] == "FAIL"
        assert report.failed

    async def test_model_config_required(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text("agent:\n  enabled: true\n", encoding="utf-8")
        report = await run_doctor_async(home, probe_acp=False)
        assert _names(report)["Native Agent model"] == "FAIL"

    async def test_acp_probe_full_handshake(self, tmp_path: Path) -> None:
        """用 mock-gateway ACP test server 验证 doctor 的 ACP 探测链路。"""
        import os

        home = _home(tmp_path)
        server_script = Path(__file__).parent / "acp_test_server.py"
        report = await run_doctor_async(
            home,
            acp_command=[sys.executable, str(server_script)],
            acp_env=dict(os.environ, ROSCLAW_ACP_TEST_HOME=str(home)),
        )
        checks = _names(report)
        assert checks["ACP initialize"] == "OK"
        assert checks["ACP session/new"] == "OK"
        assert checks["ACP session/resume"] == "OK"
        assert checks["ACP session/close"] == "OK"

    async def test_acp_probe_failure_reports_stderr(self, tmp_path: Path) -> None:
        """ACP server 起不来时，FAIL detail 必须包含 stderr 真因。"""
        home = _home(tmp_path)
        report = await run_doctor_async(
            home,
            acp_command=[sys.executable, "-c", "import sys; sys.exit('boom')"],
        )
        checks = _names(report)
        assert checks["ACP probe"] == "FAIL"
        detail = next(c.detail for c in report.checks if c.name == "ACP probe")
        assert "boom" in detail

    async def test_require_openclaw_promotes_skips(self, tmp_path: Path) -> None:
        """--require-openclaw：SKIP 升级为 FAIL。"""
        home = _home(tmp_path)
        report = await run_doctor_async(home, probe_acp=False, require_openclaw=True)
        # 本机没有 OpenClaw 时 SKIP→FAIL；有 OpenClaw 时 config 探针仍有 SKIP。
        # 两种环境下都不应残留 SKIP。
        assert not report.skipped

    async def test_doctor_is_read_only(self, tmp_path: Path) -> None:
        """doctor 不修改 ROSCLAW_HOME 里的任何配置文件。"""
        home = _home(tmp_path)
        before = (home / "config.yaml").read_bytes()
        await run_doctor_async(home, probe_acp=False)
        assert (home / "config.yaml").read_bytes() == before

    def test_render_summary(self, tmp_path: Path) -> None:
        import asyncio

        report = asyncio.run(run_doctor_async(tmp_path / "nope", probe_acp=False))
        text = report.render()
        assert "ROSClaw Channel Doctor" in text
        assert "NOT READY" in text
