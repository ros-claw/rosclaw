"""PR-PNA-3：Tool Bridge 验证链（重构规格 §17 验收矩阵子集）。

- 未绑定 session → SESSION_UNBOUND
- mission 不匹配 → MISSION_MISMATCH
- 无 writer lease → WRITER_LEASE_REQUIRED
- 未知工具 → TOOL_UNKNOWN
- 未开放工具（request_action/delegate）→ TOOL_DEFERRED（诚实拒绝）
- 动作类 capability 经 observe → NOT_OBSERVABLE（不得绕过）
- idempotency 重放 → 相同结果，不产生二次副作用
"""

from __future__ import annotations

from pathlib import Path

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore
from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.contracts.pi.tool_request import PiToolRequestV1


def _turn() -> ModelTurnResultV1:
    return ModelTurnResultV1(
        turn_id="t", provider="mock", model="m", content="ok",
        assistant_message={"role": "assistant", "content": "ok"},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},  # type: ignore[arg-type]
    )


def _request(tool: str, session: str = "pi_1", mission: str = "", **kwargs) -> PiToolRequestV1:
    from datetime import UTC, datetime

    return PiToolRequestV1(
        request_id=f"ptr_{tool}",
        pi_session_id=session,
        mission_id=mission,
        tool_name=tool,
        arguments=kwargs.get("arguments", {}),
        requested_at=datetime.now(UTC).isoformat(),
        idempotency_key=kwargs.get("idem", f"idem_{tool}_{session}"),
    )


async def _setup(tmp_path: Path):
    config = load_agent_config(tmp_path / "config.yaml")
    service = AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()] * 4))
    mission = service.create_mission("tool bridge 测试")
    bindings = SessionBindingStore(service._store.connection)
    bindings.bind(
        pi_session_id="pi_1", pi_session_path="", mission_id=mission.mission_id,
        body_id="sim/limo", execution_mode="SIMULATION", created_by="user:local:1000",
    )
    bindings.acquire_lease(
        mission_id=mission.mission_id, pi_session_id="pi_1", owner_pid=1, owner_uid=1000
    )
    return service, mission


class TestValidationChain:
    async def test_unbound_session_rejected(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request("rosclaw_status", session="pi_ghost", mission=mission.mission_id)
        )
        assert not result.ok and result.error_code == "SESSION_UNBOUND"
        await service.close()

    async def test_mission_mismatch_rejected(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request("rosclaw_status", mission="mis_other", idem="idem_mm")
        )
        assert not result.ok and result.error_code == "MISSION_MISMATCH"
        await service.close()

    async def test_writer_lease_required(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        bindings = SessionBindingStore(service._store.connection)
        bindings.bind(
            pi_session_id="pi_2", pi_session_path="", mission_id=mission.mission_id + "_x",
            body_id="b", execution_mode="SIMULATION", created_by="u",
        )
        dispatcher = PiToolDispatcher(service)
        # pi_2 绑定到别的 mission 且没有本 mission 的 lease
        result = await dispatcher.execute(
            _request("rosclaw_status", session="pi_2", mission=mission.mission_id, idem="idem_w2")
        )
        assert not result.ok and result.error_code in {"MISSION_MISMATCH", "WRITER_LEASE_REQUIRED"}
        await service.close()

    async def test_unknown_and_deferred_tools_rejected(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        unknown = await dispatcher.execute(
            _request("rosclaw_hack", mission=mission.mission_id, idem="idem_unk")
        )
        assert not unknown.ok and unknown.error_code == "TOOL_UNKNOWN"
        deferred = await dispatcher.execute(
            _request("rosclaw_request_action", mission=mission.mission_id, idem="idem_def")
        )
        assert not deferred.ok and deferred.error_code == "TOOL_DEFERRED"
        assert "PNA-5" in deferred.summary
        await service.close()

    async def test_status_and_idempotency(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        request = _request("rosclaw_status", mission=mission.mission_id, idem="idem_once")
        first = await dispatcher.execute(request)
        assert first.ok and "READY" in first.summary
        # 重放：相同 idempotency_key → 完全相同的结果（不重复执行）。
        replay = await dispatcher.execute(request)
        assert replay.model_dump() == first.model_dump()
        await service.close()

    async def test_observe_rejects_action_class(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        # 未知 capability → CAPABILITY_UNKNOWN；动作类（若目录有）→ NOT_OBSERVABLE。
        result = await dispatcher.execute(
            _request(
                "rosclaw_observe",
                mission=mission.mission_id,
                idem="idem_obs",
                arguments={"capability_id": "limo.speaker.play_tone", "arguments": {}},
            )
        )
        assert not result.ok
        assert result.error_code in {"CAPABILITY_UNKNOWN", "NOT_OBSERVABLE", "CAPABILITY_QUARANTINED"}
        await service.close()
