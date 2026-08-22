"""PR-N6C 红测试（调整方案 §五.N6C）：结构化恢复梯度 + 截止语义。

红测试先行——恢复注册表/指纹扩展/deadline 语义不存在时必须红。

1. Error Recovery Registry：稳定错误码 → 默认恢复动作（方案表格）；
   envelope 的 recovery 字段从注册表投影（不是各处临时拼）；
2. 相同调用指纹 = capability_id + normalized_args + snapshot_digest——
   同一指纹同错失败禁止原样重试；snapshot 变了不算同一指纹；
3. deadline 语义：executor 未声明 cooperative cancel 不得有
   deadline（默认无墙钟杀死）；声明后超时必须确认停止。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rosclaw.contracts.agent.tool import ExecutionClass, ToolDescriptorV2

#: 方案 §五.N6C 的稳定错误分类（含本仓既有码的对应关系）。
_STABLE_CODES = [
    "INVALID_ARGUMENTS",
    "CAPABILITY_UNKNOWN",          # CAPABILITY_NOT_FOUND
    "EFFECT_UNRESOLVABLE",
    "CAPABILITY_SNAPSHOT_CHANGED",
    "RESOURCE_PROVENANCE_MISSING",  # RESOURCE_NOT_FOUND 族
    "RUNTIME_NOT_READY",
    "TRANSPORT_UNREACHABLE",
    "INVALID_CAPABILITY_OUTPUT",    # OUTPUT_SCHEMA_INVALID
    "ACCEPTANCE_FAILED",
    "SAFETY_DENIED",
    "WAITING_APPROVAL",
]


class TestRecoveryRegistry:
    def test_every_stable_code_has_default_recovery(self) -> None:
        from rosclaw.agentd.tooling.recovery import recovery_for

        for code in _STABLE_CODES:
            action = recovery_for(code)
            assert action, f"{code} 缺默认恢复动作"
            assert isinstance(action, str) and len(action) > 8

    def test_envelope_recovery_comes_from_registry(self) -> None:
        """execute_v2 的 recovery 字段由注册表投影——同一码同一文。"""
        from rosclaw.agentd.tooling.catalog import ToolCatalog
        from rosclaw.agentd.tooling.recovery import recovery_for

        catalog = ToolCatalog()

        async def bad(arguments):
            return {"unexpected": True}

        catalog.register(ToolDescriptorV2(
            tool_id="sim_reach",
            source="native:agentd",
            execution_class=ExecutionClass.COMPUTE,
            input_schema={"type": "object"},
            output_schema={
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
                "additionalProperties": False,
            },
        ), bad)
        env = asyncio.run(catalog.execute_v2("c1", "sim_reach", {}))
        assert env.error is not None
        assert env.error.code == "INVALID_CAPABILITY_OUTPUT"
        assert env.error.recovery == [recovery_for("INVALID_CAPABILITY_OUTPUT")]

    def test_unknown_code_honest_empty(self) -> None:
        from rosclaw.agentd.tooling.recovery import recovery_for

        assert recovery_for("TOTALLY_MADE_UP") == ""


class TestDoomFingerprint:
    async def test_snapshot_digest_part_of_fingerprint(
        self, tmp_path: Path
    ) -> None:
        """同一 capability+args 但不同 snapshot_digest → 不算同一指纹
        （registry 变了之后允许重试一次——重新规划语义）。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from tests.agentd.test_pi_tool_bridge import (
            _issue_lease,
            _request,
            _setup,
        )

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")

        async def bad_executor(arguments):
            return {"broken": True}

        service._tool_catalog._executors["trajectory_generate_planar_path"] = (
            bad_executor
        )
        snap = await bridge._dispatch(
            "user:local:1000", 1, "pi.capability.snapshot",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        digest = snap["snapshot"]["digest"]
        dispatcher = PiToolDispatcher(service)
        args = {
            "capability_id": "trajectory_generate_planar_path",
            "arguments": {"shape": "star5", "center_m": [0.35, 0.25, 0.30],
                          "scale_m": 0.10},
            "snapshot_digest": digest,
        }
        first = await dispatcher.execute(
            caller_pid=1, caller_uid=1000,
            request=_request("rosclaw_compute", mission=mission.mission_id,
                             idem="n6c_f1", lease=await _issue_lease(service, mission),
                             arguments=dict(args)),
        )
        assert first.ok is False
        assert first.error_code == "INVALID_CAPABILITY_OUTPUT"
        # 同一指纹原样重试 → DOOM_LOOP。
        second = await dispatcher.execute(
            caller_pid=1, caller_uid=1000,
            request=_request("rosclaw_compute", mission=mission.mission_id,
                             idem="n6c_f2", lease=await _issue_lease(service, mission),
                             arguments=dict(args)),
        )
        assert second.ok is False
        assert second.error_code == "DOOM_LOOP"
        # registry 变化（隔离某工具）→ 新 snapshot digest → 允许重试
        # 一次（重新规划语义——不再被旧熔断误伤）。
        service._tool_catalog.quarantine_tool("sim_get_state", "probe")
        snap2 = await bridge._dispatch(
            "user:local:1000", 1, "pi.capability.snapshot",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        assert snap2["snapshot"]["digest"] != digest
        third = await dispatcher.execute(
            caller_pid=1, caller_uid=1000,
            request=_request("rosclaw_compute", mission=mission.mission_id,
                             idem="n6c_f3", lease=await _issue_lease(service, mission),
                             arguments={**args,
                                        "snapshot_digest": snap2["snapshot"]["digest"]}),
        )
        assert third.error_code != "DOOM_LOOP", (
            "snapshot 变化后原样重试仍被熔断——指纹没带 snapshot_digest"
        )
        await service.close()


class TestDeadlineSemantics:
    async def test_no_deadline_without_cooperative_cancel(self) -> None:
        """未声明 cooperative cancel 的 executor 不得被墙钟杀死——
        慢执行（超过旧 timeout_ms 默认 2000ms）正常完成。"""
        from rosclaw.agentd.tooling.catalog import ToolCatalog

        catalog = ToolCatalog()

        async def slow(arguments):
            await asyncio.sleep(2.3)  # 超过旧默认 2000ms
            return {"run_id": "run_slow"}

        catalog.register(ToolDescriptorV2(
            tool_id="slow_compute",
            source="native:agentd",
            execution_class=ExecutionClass.COMPUTE,
            timeout_ms=2000,  # 旧语义下会被 wait_for 杀死
            input_schema={"type": "object"},
            output_schema={
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
            },
        ), slow)
        env = await catalog.execute_v2("c1", "slow_compute", {})
        assert env.status.value == "SUCCEEDED", (
            f"未声明 cooperative cancel 的 executor 被墙钟杀死: {env.error}"
        )

    async def test_deadline_requires_cooperative_and_confirms_stop(self) -> None:
        """声明 cooperative cancel 才允许 deadline；超时后必须确认
        停止（返回取消证据，不是只放弃 await）。"""
        from rosclaw.agentd.tooling.catalog import ToolCatalog

        catalog = ToolCatalog()
        stopped = asyncio.Event()

        async def slow(arguments):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                stopped.set()
                raise
            return {"run_id": "never"}

        descriptor = ToolDescriptorV2(
            tool_id="slow_coop",
            source="native:agentd",
            execution_class=ExecutionClass.COMPUTE,
            timeout_ms=200,
            input_schema={"type": "object"},
            output_schema={
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
            },
        )
        # 声明 cooperative cancel 的途径（N6C 新增字段）。
        descriptor2 = descriptor.model_copy(
            update={"cooperative_cancel": True}
        )
        catalog.register(descriptor2, slow)
        env = await catalog.execute_v2("c1", "slow_coop", {})
        assert env.status.value == "FAILED"
        assert env.error is not None
        assert env.error.code == "EXECUTOR_TIMEOUT"
        # 确认停止：取消真实到达 executor（不是只放弃 await）。
        await asyncio.wait_for(stopped.wait(), timeout=2)
