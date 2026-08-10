"""PR-EIGHT-1 红测试（八审 §4 P0-8）：可量化 token/latency 预算。

红测试先行——当前没有面向用户/CI 的用量面：
pi.usage 必须返回 provider 请求数、token 分项、工具调用计数、
provider 延迟与端到端耗时分离——否则"任务为何昂贵"无法回答。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _turn


async def _service(tmp_path: Path):
    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.models.gateway import MockModelGateway
    from rosclaw.agentd.models.profiles import mock_profile
    from rosclaw.agentd.service import AgentService

    config = load_agent_config(tmp_path / "config.yaml")
    return AgentService(
        config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()] * 4)
    )


class TestUsageReport:
    async def test_usage_report_shape(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service = await _service(tmp_path)
        mission = service.create_mission("usage probe")
        # 记两笔用量（不同延迟，验证分位数）。
        from rosclaw.agentd.usage import UsageRecorder
        from rosclaw.contracts.agent.model_turn import (
            ModelTurnResultV1,
            ModelUsage,
        )

        recorder = UsageRecorder(service._store.connection)
        for i, latency in enumerate((120, 480)):
            recorder.record(
                ModelTurnResultV1(
                    turn_id=f"turn_{i}",
                    mission_id=mission.mission_id,
                    provider="fake",
                    model="fake-k3",
                    profile="p",
                    usage=ModelUsage(
                        prompt_tokens=100 * (i + 1),
                        completion_tokens=50,
                        reasoning_tokens=0,
                        total_tokens=100 * (i + 1) + 50,
                        cost_microunits=0,
                    ),
                    latency_ms=latency,
                    provider_request_id="",
                    context_id="",
                    context_revision=0,
                    finish_reason="stop",
                )
            )
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.usage",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        assert result.get("ok"), result
        report = result.get("usage") or {}
        # provider 请求数与 token 分项。
        assert report.get("model_turns") == 2
        assert report.get("prompt_tokens") == 300
        assert report.get("completion_tokens") == 100
        assert report.get("total_tokens") == 400
        # provider 延迟分布（p50/p95）与端到端跨度分离。
        latency = report.get("provider_latency_ms") or {}
        assert latency.get("p50") is not None and latency.get("p95") == 480
        assert report.get("wall_span_ms") is not None
        # 工具调用计数面（本测试无工具调用——必须为 0 而不是缺字段）。
        tools = report.get("tool_calls") or {}
        assert tools.get("proposed") == 0 and tools.get("completed") == 0
        await service.close()
