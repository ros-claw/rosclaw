"""0827 体验审计 P0-8 红测试：安装产物 build identity 一致性 Gate。

0827 实证疑点：报告称 verifier 用 ~25mm 地板，真实日志显示 20mm
阈值——"报告测的是新代码，用户跑的是旧 wheel/旧 dist"无法排除。

闭环断言：
1. `rosclaw version --verbose`（=--diagnostic 别名）一次显示：
   CLI/wheel 版本+commit、TS dist commit+digest、schema（migration
   revision）、capability snapshot hash（有运行实例时）、agentd
   version+boot_id（有运行实例时）；
2. 不一致错误码是 INSTALLATION_VERSION_MISMATCH（不是自造词）；
3. pi.status 暴露 agentd_version + boot_id（每次 boot 唯一）；
4. 活的 agentd 在跑时 version --verbose 显示其 boot_id。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestVersionVerboseSurface:
    def test_verbose_alias_and_fields(self, tmp_path: Path, capsys) -> None:
        from rosclaw.version_diag import cmd_version

        rc = cmd_version(diagnostic=True, as_json=True, home=tmp_path)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        # 安装身份五件套（0827 §九.P0-8）。
        assert out.get("rosclaw_version"), out
        assert "wheel_commit" in out, out
        assert (out.get("ts_dist") or {}).get("dist_digest"), out
        assert "migration_revision" in out, out  # schema version
        assert "agentd" in out, "缺 agentd 段（version+boot_id）"

    def test_mismatch_code(self) -> None:
        from rosclaw.version_diag import mixed_build_reason

        reason = mixed_build_reason(
            {"wheel_commit": "aaa", "ts_dist": {"commit": "bbb"}}
        )
        assert reason is not None
        assert reason.startswith("INSTALLATION_VERSION_MISMATCH"), reason


class TestAgentdBootIdentity:
    async def test_status_exposes_version_and_boot_id(
        self, tmp_path: Path
    ) -> None:
        from rosclaw import __version__
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.status",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        assert result.get("ok"), result
        assert result.get("agentd_version") == __version__, result
        boot_id = str(result.get("boot_id") or "")
        assert boot_id, result
        # 每次 boot 唯一（重启即变——会话间状态归因靠它）。
        assert service.boot_id == boot_id
        # capability snapshot hash（有 mission 时必须给出）。
        assert result.get("capability_digest"), result
        await service.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
