"""PR-SEVEN-2 红测试（七审 §2.1/§2.2）：能力分类与 ActionReadinessV2。

红测试先行——当前缺陷：

1. pi.capabilities 把"非 PHYSICAL_ACTION"全塞进 observation——
   COMPUTE 类的 sim_reach 被显示为"只读观测"，调用时又被
   rosclaw_observe 以"not OBSERVE"拒绝（自相矛盾）；
2. COMPUTE 能力没有免审批调用通道；
3. ActionReadiness 只有 operator/lease/context——缺 capability/
   executor/kit 维度（action count=0 时 Header 仍可能只怪
   Operator OFFLINE）。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _setup


class TestCapabilityBuckets:
    async def test_compute_not_listed_as_observation(self, tmp_path: Path) -> None:
        """sim_reach（COMPUTE）必须只在 compute 桶——不得在
        observation；且带 effect_domain/executor_state 字段。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.capabilities",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        assert result.get("ok"), result
        observation_ids = [
            c["capability_id"] for c in result.get("observation_capabilities", [])
        ]
        compute_ids = [c["capability_id"] for c in result.get("compute_capabilities", [])]
        assert "sim_reach" not in observation_ids, (
            f"COMPUTE 能力被错列为观测: {observation_ids}"
        )
        assert "sim_reach" in compute_ids, f"compute 桶缺 sim_reach: {compute_ids}"
        # 动作条目必须带 effect_domain/executor_state。
        actions = result.get("action_capabilities", [])
        assert actions, "UR5e kit 应自动激活提供动作"
        for entry in actions:
            assert "effect_domain" in entry, "动作条目缺 effect_domain"
            assert "executor_state" in entry, "动作条目缺 executor_state"
        await service.close()

    async def test_compute_callable_without_approval(self, tmp_path: Path) -> None:
        """rosclaw_compute：COMPUTE 能力免审批调用（无副作用）。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        from tests.agentd.test_pi_tool_bridge import _request

        result = await dispatcher.execute(
            _request(
                "rosclaw_compute",
                mission=mission.mission_id,
                idem="idem_seven2_compute",
                arguments={"capability_id": "sim_reach", "arguments": {"x": 0.3, "y": 0.2, "z": 0.4}},
            ),
            caller_pid=1,
            caller_uid=1000,
        )
        assert result.ok, f"compute 调用被拒: {result.error_code} {result.summary[:200]}"
        await service.close()

    async def test_observe_still_rejects_compute(self, tmp_path: Path) -> None:
        """rosclaw_observe 仍只接 OBSERVE——COMPUTE 走 compute 通道。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import (
            PiToolDispatcher,
        )

        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        from tests.agentd.test_pi_tool_bridge import _request

        result = await dispatcher.execute(
            _request(
                "rosclaw_observe",
                mission=mission.mission_id,
                idem="idem_seven2_obs",
                arguments={"capability_id": "sim_reach", "arguments": {}},
            ),
            caller_pid=1,
            caller_uid=1000,
        )
        assert not result.ok
        assert result.error_code in ("NOT_OBSERVABLE", "INVALID_ARGUMENTS")
        await service.close()


class TestReadinessV2:
    async def test_zero_action_capability_blocks_with_kit_reason(
        self, tmp_path: Path
    ) -> None:
        """kit 未激活/executor 缺失时 readiness 必须含
        ROBOT_KIT_INCOMPLETE——不是只怪 OPERATOR_OFFLINE。"""
        from rosclaw.agentd.config import load_agent_config
        from rosclaw.agentd.models.gateway import MockModelGateway
        from rosclaw.agentd.models.profiles import mock_profile
        from rosclaw.agentd.service import AgentService
        from tests.agentd.test_pi_tool_bridge import _turn

        # 禁 kit → 动作能力为 0。
        (tmp_path / "config.yaml").write_text(
            "agent:\n  enabled: true\nkits:\n  disabled: [rosclaw/ur5e-sim]\n",
            encoding="utf-8",
        )
        config = load_agent_config(tmp_path / "config.yaml")
        service = AgentService(
            config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()] * 4)
        )
        blockers = await service.action_blockers()
        codes = [b["code"] for b in blockers]
        assert "ROBOT_KIT_INCOMPLETE" in codes, (
            f"动作能力为 0 时缺 kit blocker: {codes}"
        )
        await service.close()
