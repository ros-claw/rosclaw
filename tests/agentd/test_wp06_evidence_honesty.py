"""WP-P0-6 红测试（总纲 §8.3）：仿真证据诚实化。

红测试先行——当前运动学 sandbox 把规划点复制成 trace 再与规划点
比较，却让用户看到"仿真执行 ✓"这类表述。这只能证明路径数据
自洽（COMMAND_REPLAY），不能证明动力学或真实机械臂完成运动。

证据等级：PLANNED / COMMAND_REPLAY / KINEMATIC_SIM / PHYSICS_SIM /
INDEPENDENT_SIM_OBSERVATION / REAL_COMMAND_ACCEPTED / REAL_OBSERVED。
"""

from __future__ import annotations

import json
from pathlib import Path


def _call(name: str, **kwargs):
    from rosclaw.sim import ur5e_mcp

    fn = getattr(ur5e_mcp, name.replace("ur5e.", "").replace(".", "_"), None)
    return json.loads(fn(**kwargs))


class TestEvidenceLevel:
    def test_execute_result_declares_command_replay(self) -> None:
        """执行结果必须声明 evidence_level=COMMAND_REPLAY + 局限性。"""
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        plan = _call(
            "ur5e.plan_cartesian_path",
            shape="star5", center_x=0.35, center_y=0.25, z=0.30, outer_radius=0.10,
        )
        result = _call("ur5e.execute_plan", plan_id=plan["plan_id"])
        assert result.get("evidence_level") == "COMMAND_REPLAY", result
        limitation = str(result.get("limitation", ""))
        assert "自洽" in limitation and "不能证明" in limitation, limitation

    def test_trace_labeled_replay_not_observation(self) -> None:
        """trace 视图必须标 replay——不得自称独立观测。"""
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        plan = _call(
            "ur5e.plan_cartesian_path",
            shape="star5", center_x=0.35, center_y=0.25, z=0.30, outer_radius=0.10,
        )
        _call("ur5e.execute_plan", plan_id=plan["plan_id"])
        trace = _call("ur5e.get_cartesian_trace")
        assert trace["trace"].get("origin") == "command_replay", trace["trace"].keys()
        assert trace.get("evidence_level") == "COMMAND_REPLAY"

    def test_evidence_level_enum_contract(self) -> None:
        """证据等级枚举是合约（顺序=强度，不得改名）。"""
        from rosclaw.contracts.operator.evidence import EVIDENCE_LEVELS

        assert list(EVIDENCE_LEVELS) == [
            "PLANNED",
            "COMMAND_REPLAY",
            "KINEMATIC_SIM",
            "PHYSICS_SIM",
            "INDEPENDENT_SIM_OBSERVATION",
            "REAL_COMMAND_ACCEPTED",
            "REAL_OBSERVED",
        ]


class TestNoOverclaimCopy:
    def test_product_copy_has_no_overclaim(self) -> None:
        """产品文案不得超出证据能力（实际执行轨迹/真实走过/物理仿真
        已完成 等）。大道至简 R0-2b：rosclaw_task（task.ts）已删除，
        扫描面收敛到仍存在的物理工具文案。"""
        banned = ["实际执行轨迹", "真实走过", "实际走过的路径", "物理仿真已完成"]
        for path in (
            Path("src/rosclaw/sim/ur5e_mcp.py"),
            Path("src/rosclaw/agentd/pi_bridge/tool_dispatch.py"),
        ):
            text = path.read_text(encoding="utf-8")
            for phrase in banned:
                assert phrase not in text, f"{path.name} 含过度宣称: {phrase}"

    async def test_simulate_result_declares_evidence_level(
        self, tmp_path: Path
    ) -> None:
        """能力结果必须带证据等级（SIM_DYN_ROLLOUT——动力学仿真
        自洽，不宣称真机）。大道至简 R0-2b：原 rosclaw_task 的
        user_view 文案随工具删除；证据等级纪律由 capability 结果
        投影承接。"""
        import json as _json

        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from tests.agentd.test_pi_tool_bridge import (
            _issue_lease,
            _request,
            _setup,
        )

        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        lease = await _issue_lease(service, mission)
        plan = await dispatcher.execute(
            caller_pid=1, caller_uid=1000,
            request=_request(
                "rosclaw_compute", mission=mission.mission_id,
                idem="ev_plan", lease=lease,
                arguments={
                    "capability_id": "trajectory_generate_planar_path",
                    "arguments": {"shape": "star5",
                                  "center_m": [0.35, 0.25, 0.30],
                                  "scale_m": 0.10, "plane": "xy",
                                  "max_segment_m": 0.02},
                },
            ),
        )
        assert plan.ok, plan.summary
        plan_id = _json.loads(plan.summary)["value"]["plan_id"]
        sim = await dispatcher.execute(
            caller_pid=1, caller_uid=1000,
            request=_request(
                "rosclaw_compute", mission=mission.mission_id,
                idem="ev_sim", lease=lease,
                arguments={
                    "capability_id": "ur5e_simulate_cartesian_trajectory",
                    "arguments": {"plan_id": plan_id},
                },
            ),
        )
        assert sim.ok, sim.summary
        value = _json.loads(sim.summary)["value"]
        assert value.get("evidence_level") == "SIM_DYN_ROLLOUT", value
        await service.close()

