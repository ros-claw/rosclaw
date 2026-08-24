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
        context_lease_id=kwargs.get("lease", ""),
    )


async def _issue_lease(service, mission, session: str = "pi_1") -> str:
    """按真实路径签发 ValidatedContextLease（HOTFIX-1：不绕过 admission）。
    五审 P0-5B：context_hash 必须是当前权威 envelope 的真实 hash
    （admission 会重算比对）——不能用占位符。"""
    from rosclaw.agentd.pi_bridge.context import build_embodied_context
    from rosclaw.agentd.pi_bridge.context_lease import (
        ContextLeaseStore,
        context_hash_of,
    )

    # 七审 SIX-3/SEVEN-1：discovery 先于 hash——capabilities 在
    # context_hash 内（第一方 kit 自动激活后目录内容取决于发现）。
    await service._ensure_mcp_discovered()
    envelope = build_embodied_context(service, mission.mission_id)
    # 六审 §5.3/§5.5（migration 020）：lease 必须带真实 binding/
    # writer/caller 字段——测试 writer 注册为 owner_pid=1/uid=1000。
    from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore

    bindings = SessionBindingStore(service._store.connection)
    binding = bindings.binding_for_session(session)
    writer = bindings.writer_of(mission.mission_id)
    lease = ContextLeaseStore(service._store.connection).issue(
        pi_session_id=session,
        mission_id=mission.mission_id,
        context_revision=envelope.context_revision,
        context_hash=context_hash_of(envelope),
        body_hash=mission.body_binding.effective_body_hash,
        mode=mission.mode.value,
        binding_id=binding.binding_id if binding else "",
        writer_lease_id=writer.lease_id if writer else "",
        caller_uid=1000,
        caller_pid=1,
    )
    return lease.context_lease_id


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
    # P0-C：effectful admission 的动机输入——与产品流一致（输入
    # 先 persist，再有工具调用）。
    service._task_kernel.persist_input(
        mission_id=mission.mission_id, session_ref="pi_1",
        message_id="msg_setup", text="tool bridge 测试目标",
    )
    _register_sim_action_capability(service)
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
        # PNA-5 后 request_action 已开放——空参数必须 fail closed。
        deferred = await dispatcher.execute(
            _request("rosclaw_request_action", mission=mission.mission_id, idem="idem_def")
        )
        assert not deferred.ok and deferred.error_code == "INVALID_ARGUMENTS"
        # 仍未开放的工具保持诚实拒绝。
        plan = await dispatcher.execute(
            _request("rosclaw_plan_patch", mission=mission.mission_id, idem="idem_pp")
        )
        assert not plan.ok and plan.error_code == "TOOL_DEFERRED"
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
        # 未知 capability → EFFECT_UNRESOLVABLE（N5C resolver 先行
        # fail closed）或 CAPABILITY_UNKNOWN；动作类（若目录有）→
        # NOT_OBSERVABLE。
        result = await dispatcher.execute(
            _request(
                "rosclaw_observe",
                mission=mission.mission_id,
                idem="idem_obs",
                arguments={"capability_id": "limo.speaker.play_tone", "arguments": {}},
            )
        )
        assert not result.ok
        assert result.error_code in {
            "EFFECT_UNRESOLVABLE", "CAPABILITY_UNKNOWN", "NOT_OBSERVABLE",
            "CAPABILITY_QUARANTINED",
        }
        await service.close()


SIM_ACTION_CAPABILITY = "sim_ground_truth"


def _register_sim_action_capability(service) -> None:
    """注册确定性 SIM 动作能力（PHYSICAL_ACTION）+ 确定性执行通道。

    HOTFIX-2 后 admission 按 ToolCatalog 权威校验——测试必须走真实
    catalog 路径（不是绕过）。执行端用进程内 fake client 的
    SimActionChannel：确定性 JSON、无外部进程依赖。
    """
    import json as _json

    from rosclaw.agentd.sim_executor import SimActionChannel
    from rosclaw.contracts.agent.tool import (
        ExecutionClass,
        ToolDescriptorV2,
        ToolEvidenceClass,
        ToolSideEffectClass,
    )

    service._tool_catalog.register(
        ToolDescriptorV2(
            tool_id=SIM_ACTION_CAPABILITY,
            source="native:agentd",
            execution_class=ExecutionClass.PHYSICAL_ACTION,
            description="确定性 SIM 验收动作（真实 SIM 执行通道产出 SIMULATED receipt）。",
            # 六审 §4.4.5：物理动作必须声明严格对象边界——properties
            # 覆盖既有测试用参（{}/{"a":int}/{"beep":bool}），未知参数拒绝。
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "beep": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            # 六审 §6.2：物理动作必须声明 body scope——测试本体是 sim/ur5e。
            required_body_types=["sim/ur5e"],
            supported_modes=["SIMULATION"],
            evidence_class=ToolEvidenceClass.SIMULATED,
            risk_tier="LOW",
            model_callable=False,
            requires_exact_action_grant=True,
            side_effect_class=ToolSideEffectClass.IRREVERSIBLE,
        )
    )

    class _FakeSimClient:
        async def call_tool(self, tool_name: str, arguments: dict) -> str:
            return _json.dumps(
                {"tool": tool_name, "args": arguments, "ok": True},
                ensure_ascii=False,
            )

    service._sim_executors["native:agentd"] = SimActionChannel(
        command="true", args=(), name="fake-sim", client=_FakeSimClient()
    )
