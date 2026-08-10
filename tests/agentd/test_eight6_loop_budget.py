"""PR-EIGHT-6 红测试（八审 §4 P0-6）：循环熔断与错误分类。

红测试先行——八审真实会话里模型反复猜 schema/hash 参数，每次
失败都再消耗模型回合（16 次提案）：

1. 同一工具同一参数出错后再次原样调用 → DOOM_LOOP（不再消耗
   模型回合）；成功后重置（合法重复观测不误伤）；
2. 动作参数在 Bridge 本地 JSON Schema 预校验——类型错误
   INVALID_ARGUMENTS，不进授权链（零提案）；
3. 任务级授权卡是人可读的（形状/中心/半径），不是 plan_id。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup


class TestDoomLoop:
    async def test_same_failing_call_twice_is_doom_loop(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        await service._ensure_mcp_discovered()
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        dispatcher = PiToolDispatcher(service)

        async def bad_compute(idem: str):
            return await dispatcher.execute(
                caller_pid=1,
                caller_uid=1000,
                request=_request(
                    "rosclaw_compute",
                    mission=mission.mission_id,
                    idem=idem,
                    lease=await _issue_lease(service, mission),
                    arguments={
                        "capability_id": "ur5e.plan_cartesian_path",
                        "arguments": {"shape": "hexagon"},  # 必然失败
                    },
                ),
            )

        first = await bad_compute("idem_doom_1")
        assert not first.ok
        second = await bad_compute("idem_doom_2")
        assert not second.ok
        assert second.error_code == "DOOM_LOOP", (
            f"同一失败调用原样重复竟不熔断: {second.error_code}"
        )
        await service.close()

    async def test_successful_repeat_not_doomed(self, tmp_path: Path) -> None:
        """成功的重复只读调用不触发熔断（观测两次是合法的）。"""
        service, mission = await _setup(tmp_path)
        await service._ensure_mcp_discovered()
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        dispatcher = PiToolDispatcher(service)
        for i in range(2):
            result = await dispatcher.execute(
                caller_pid=1,
                caller_uid=1000,
                request=_request(
                    "rosclaw_observe",
                    mission=mission.mission_id,
                    idem=f"idem_obs_{i}",
                    lease=await _issue_lease(service, mission),
                    arguments={"capability_id": "ur5e.get_joint_state"},
                ),
            )
            assert result.ok, f"第 {i + 1} 次合法观测被误熔断: {result.error_code}"
        await service.close()


class TestSchemaPrevalidation:
    async def test_wrong_arg_type_rejected_before_card(self, tmp_path: Path) -> None:
        """execute_plan 的 plan_id 传数字 → INVALID_ARGUMENTS，零提案。"""
        service, mission = await _setup(tmp_path)
        await service._ensure_mcp_discovered()
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            caller_pid=1,
            caller_uid=1000,
            request=_request(
                "rosclaw_request_action",
                mission=mission.mission_id,
                idem="idem_schema_1",
                lease=await _issue_lease(service, mission),
                arguments={
                    "capability_id": "ur5e.execute_plan",
                    "arguments": {"plan_id": 123},
                    "expected_effect": "type probe",
                    "risk_tier": "LOW",
                },
            ),
        )
        assert not result.ok
        assert result.error_code == "INVALID_ARGUMENTS", result.error_code
        cards = service._store.connection.execute(
            "SELECT COUNT(*) FROM operator_requests"
        ).fetchone()[0]
        assert cards == 0, f"schema 错误竟进授权链: {cards}"
        await service.close()


class TestTaskCardHumanReadable:
    async def test_task_card_shows_shape_not_plan_id(self, tmp_path: Path) -> None:
        """任务级卡片标题/摘要含形状/中心/半径——不是内部 plan_id。"""
        import json

        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            caller_pid=1,
            caller_uid=1000,
            request=_request(
                "rosclaw_task",
                mission=mission.mission_id,
                idem="idem_card_1",
                lease=await _issue_lease(service, mission),
                arguments={
                    "goal": "draw_shape",
                    "parameters": {
                        "shape": "star5",
                        "center_m": [0.35, 0.25, 0.30],
                        "radius_m": 0.10,
                    },
                },
            ),
        )
        assert result.ok, result.summary
        row = service._store.connection.execute(
            "SELECT request_json FROM operator_requests LIMIT 1"
        ).fetchone()
        assert row, "无授权卡"
        card = json.loads(row[0])
        blob = json.dumps(card, ensure_ascii=False)
        assert "五角星" in blob or "star5" in blob, f"卡片缺人读摘要: {blob[:300]}"
        assert "0.35" in blob and "0.10" in blob, f"卡片缺中心/半径: {blob[:300]}"
        await service.close()
