"""大道至简 R1-2c：一屏具身事实带安全工作空间窗口。

闭环断言：sim/ur5e 任务的 trusted context envelope 含
safe_radius_m/safe_z_m（规划器硬校验同值——Pi 摆 waypoints 不猜
边界）；非 ur5e 本体不携带。
"""

from __future__ import annotations

import pytest


class TestWorkspaceWindow:
    def test_ur5e_envelope_has_workspace_window(self, tmp_path) -> None:
        import asyncio

        from tests.agentd.test_pi_tool_bridge import _setup

        async def run() -> dict:
            service, mission = await _setup(tmp_path)
            from rosclaw.agentd.pi_bridge.context import build_embodied_context

            envelope = build_embodied_context(service, mission.mission_id)
            await service.close()
            return envelope.body

        body = asyncio.run(run())
        # _setup 的 mission 绑定 sim/ur5e——窗口必须在且与规划器
        # 硬校验同值。
        assert body.get("safe_radius_m") == [0.10, 0.80], body
        assert body.get("safe_z_m") == [0.02, 1.20], body

    def test_window_values_match_planner(self) -> None:
        """窗口数值与规划器硬校验同源（不是抄来的第二份）。"""
        from rosclaw.sim.ur5e_mcp import _SAFE_RADIUS, _SAFE_Z

        assert _SAFE_RADIUS == (0.10, 0.80)
        assert _SAFE_Z == (0.02, 1.20)


class TestSkillRewrite:
    def test_skill_teaches_generic_primitives(self) -> None:
        from pathlib import Path

        skill = (
            Path(__file__).resolve().parents[2]
            / "packages/rosclaw-agent/skills/rosclaw-embodied/SKILL.md"
        ).read_text(encoding="utf-8")
        assert "trajectory_generate_planar_path" in skill
        assert "waypoints" in skill
        # 无五角星步骤/形状特例步骤（shape 只是便捷参数的顺带
        # 提及允许——不得有"画五角星"步骤式指引）。
        assert "画五角星" not in skill
        assert "五角星步骤" not in skill


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
