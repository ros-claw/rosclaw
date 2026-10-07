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
        turn_id="t",
        provider="mock",
        model="m",
        content="ok",
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
    service = AgentService(
        config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()] * 4)
    )
    mission = service.create_mission("tool bridge 测试")
    bindings = SessionBindingStore(service._store.connection)
    bindings.bind(
        pi_session_id="pi_1",
        pi_session_path="",
        mission_id=mission.mission_id,
        body_id="sim/limo",
        execution_mode="SIMULATION",
        created_by="user:local:1000",
    )
    bindings.acquire_lease(
        mission_id=mission.mission_id, pi_session_id="pi_1", owner_pid=1, owner_uid=1000
    )
    # P0-C：effectful admission 的动机输入——与产品流一致（输入
    # 先 persist，再有工具调用）。
    service._task_kernel.persist_input(
        mission_id=mission.mission_id,
        session_ref="pi_1",
        message_id="msg_setup",
        text="tool bridge 测试目标",
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
            pi_session_id="pi_2",
            pi_session_path="",
            mission_id=mission.mission_id + "_x",
            body_id="b",
            execution_mode="SIMULATION",
            created_by="u",
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
            "EFFECT_UNRESOLVABLE",
            "CAPABILITY_UNKNOWN",
            "NOT_OBSERVABLE",
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


# ----------------------------------------------------------------------
# 大道至简 R0-2b：r01 生产链文件随 recipe 链删除——其接线 harness
# 助手函数迁移至此（共享测试设施）。
# ----------------------------------------------------------------------


def _kernel(home: Path):
    import sqlite3

    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, home), conn


def _draw_task(kernel, home: Path, text: str = "画一个五角星") -> str:
    kernel.persist_input(
        mission_id="mis_1",
        session_ref="s1",
        message_id="msg_1",
        text=text,
    )
    bound = kernel.ensure_task_for_effect(
        mission_id="mis_1",
        session_ref="s1",
        backend_native_id="s1",
        cwd=str(home),
        body_id="sim/ur5e",
    )
    return str(bound["task_id"])


async def _setup_ur5e(tmp_path: Path):
    """生产级接线 harness：真实 AgentService + ur5e body 绑定。"""
    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.models.gateway import MockModelGateway
    from rosclaw.agentd.models.profiles import mock_profile
    from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore
    from rosclaw.agentd.service import AgentService
    from rosclaw.contracts.agent.model_turn import ModelTurnResultV1

    turn = ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content="ok",
        assistant_message={"role": "assistant", "content": "ok"},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},  # type: ignore[arg-type]
    )
    config = load_agent_config(tmp_path / "config.yaml")
    service = AgentService(
        config,
        tmp_path,
        gateway=MockModelGateway(mock_profile(), [turn]),
    )
    mission = service.create_mission("接线测试")
    bindings = SessionBindingStore(service._store.connection)
    bindings.bind(
        pi_session_id="pi_1",
        pi_session_path="",
        mission_id=mission.mission_id,
        body_id="sim/ur5e",
        execution_mode="SIMULATION",
        created_by="user:local:1000",
    )
    bindings.acquire_lease(
        mission_id=mission.mission_id,
        pi_session_id="pi_1",
        owner_pid=1,
        owner_uid=1000,
    )
    service._task_kernel.persist_input(
        mission_id=mission.mission_id,
        session_ref="pi_1",
        message_id="msg_draw",
        text="画一个五角星",
    )
    return service, mission


# ----------------------------------------------------------------------
# Opt-in declared local artifact schema（有界子集——rosclaw_deliver 登记
# 前校验；拒绝必须零新行、零网络、诊断不回显值/属性名）。
# ----------------------------------------------------------------------

_DECLARED_TABLES = (
    "tasks",
    "artifacts",
    "operations",
    "task_revisions",
    "task_session_bindings",
)

_DECLARED_VALID_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["nonce", "status"],
    "properties": {
        "nonce": {"const": "fixed"},
        "status": {"const": "SOURCE_COMPONENT_PREPARED"},
        "source_artifact_refs": {"type": "object"},
    },
}

_DECLARED_VALID_ARTIFACT = {
    "nonce": "fixed",
    "status": "SOURCE_COMPONENT_PREPARED",
    "source_artifact_refs": {},
}


def _declared_rows(service) -> dict:
    conn = service._store.connection
    return {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in _DECLARED_TABLES
    }


def _declared_files(project: Path, schema, artifact) -> tuple[Path, Path]:
    import json as _json

    project.mkdir(parents=True, exist_ok=True)
    schema_path = project / "schema.json"
    schema_path.write_text(_json.dumps(schema))
    artifact_path = project / "delivery.json"
    artifact_path.write_text(_json.dumps(artifact))
    return artifact_path, schema_path


class TestDeclaredArtifactSchema:
    async def _deliver(self, service, mission, project: Path, idem: str, *, schema: bool = True):
        dispatcher = PiToolDispatcher(service)
        arguments = {
            "path": str(project / "delivery.json"),
            "role": "progress_report",
            "cwd": str(project),
        }
        if schema:
            arguments["schema_path"] = str(project / "schema.json")
        return await dispatcher.execute(
            _request(
                "rosclaw_deliver",
                mission=mission.mission_id,
                idem=idem,
                arguments=arguments,
            )
        )

    async def test_declared_valid_registers(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        _declared_files(project, _DECLARED_VALID_SCHEMA, _DECLARED_VALID_ARTIFACT)
        before = _declared_rows(service)
        result = await self._deliver(service, mission, project, "idem_decl_ok")
        assert result.ok and result.artifact_refs
        after = _declared_rows(service)
        assert after["artifacts"] == before["artifacts"] + 1
        assert after["operations"] == before["operations"]
        await service.close()

    async def test_absent_schema_preserves_legacy(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        _declared_files(project, _DECLARED_VALID_SCHEMA, _DECLARED_VALID_ARTIFACT)
        result = await self._deliver(service, mission, project, "idem_decl_legacy", schema=False)
        assert result.ok and result.artifact_refs
        await service.close()

    async def test_cyclic_local_ref_rejected_zero_rows(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        _declared_files(project, {"$ref": "#"}, _DECLARED_VALID_ARTIFACT)
        before = _declared_rows(service)
        result = await self._deliver(service, mission, project, "idem_decl_cyc")
        assert not result.ok
        assert result.error_code == "DECLARED_SCHEMA_UNSUPPORTED_KEYWORD"
        assert _declared_rows(service) == before
        await service.close()

    async def test_external_https_ref_rejected_zero_rows(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        _declared_files(
            project,
            {"$ref": "https://invalid.example/private"},
            _DECLARED_VALID_ARTIFACT,
        )
        before = _declared_rows(service)
        result = await self._deliver(service, mission, project, "idem_decl_ext")
        assert not result.ok
        assert result.error_code == "DECLARED_SCHEMA_UNSUPPORTED_KEYWORD"
        assert _declared_rows(service) == before
        await service.close()

    async def test_branching_schema_rejected_zero_rows(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        _declared_files(
            project,
            {"anyOf": [{"type": "object"}]},
            _DECLARED_VALID_ARTIFACT,
        )
        before = _declared_rows(service)
        result = await self._deliver(service, mission, project, "idem_decl_anyof")
        assert not result.ok
        assert result.error_code == "DECLARED_SCHEMA_UNSUPPORTED_KEYWORD"
        assert _declared_rows(service) == before
        await service.close()

    async def test_unknown_keyword_rejected_zero_rows(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        _declared_files(
            project,
            {"type": "object", "pattern": "^x"},
            _DECLARED_VALID_ARTIFACT,
        )
        before = _declared_rows(service)
        result = await self._deliver(service, mission, project, "idem_decl_pat")
        assert not result.ok
        assert result.error_code == "DECLARED_SCHEMA_UNSUPPORTED_KEYWORD"
        assert _declared_rows(service) == before
        await service.close()

    async def test_foreign_schema_rejected_zero_rows(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        artifact_path, _ = _declared_files(
            project, _DECLARED_VALID_SCHEMA, _DECLARED_VALID_ARTIFACT
        )
        import json as _json

        outside = tmp_path / "outside-schema.json"
        outside.write_text(_json.dumps(_DECLARED_VALID_SCHEMA))
        dispatcher = PiToolDispatcher(service)
        before = _declared_rows(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_deliver",
                mission=mission.mission_id,
                idem="idem_decl_foreign",
                arguments={
                    "path": str(artifact_path),
                    "role": "progress_report",
                    "cwd": str(project),
                    "schema_path": str(outside),
                },
            )
        )
        assert not result.ok
        assert result.error_code == "DECLARED_SCHEMA_PATH_REJECTED"
        assert _declared_rows(service) == before
        await service.close()

    async def test_secret_values_never_leak_into_diagnostics(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        marker = "SYNTHETIC_SECRET_DO_NOT_EXPORT"
        artifact = dict(_DECLARED_VALID_ARTIFACT, nonce=marker)
        artifact[marker] = marker
        _declared_files(project, _DECLARED_VALID_SCHEMA, artifact)
        before = _declared_rows(service)
        result = await self._deliver(service, mission, project, "idem_decl_sec")
        assert not result.ok
        assert result.error_code == "DECLARED_SCHEMA_VALIDATION_FAILED"
        assert marker not in result.summary
        assert _declared_rows(service) == before
        await service.close()

    async def test_const_mismatch_rejected_zero_rows(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        _declared_files(
            project,
            _DECLARED_VALID_SCHEMA,
            dict(_DECLARED_VALID_ARTIFACT, status="PASS"),
        )
        before = _declared_rows(service)
        result = await self._deliver(service, mission, project, "idem_decl_const")
        assert not result.ok
        assert result.error_code == "DECLARED_SCHEMA_VALIDATION_FAILED"
        assert _declared_rows(service) == before
        await service.close()


class TestDeclaredSchemaHardening:
    """有界子集硬化回归：畸形关键字值/有限读取/typed 深度预算。

    畸形 supported-keyword 值形状必须在登记前 typed 拒绝（零新行，
    既有行不变）——绝不静默忽略；5000 层嵌套在字节上限内也必须
    得到 DECLARED_SCHEMA_BUDGET_EXCEEDED 而不是 RecursionError。"""

    async def _deliver(self, service, mission, project: Path, idem: str):
        dispatcher = PiToolDispatcher(service)
        return await dispatcher.execute(
            _request(
                "rosclaw_deliver",
                mission=mission.mission_id,
                idem=idem,
                arguments={
                    "path": str(project / "delivery.json"),
                    "role": "progress_report",
                    "cwd": str(project),
                    "schema_path": str(project / "schema.json"),
                },
            )
        )

    async def _assert_rejected(self, tmp_path: Path, idem: str, schema, artifact) -> None:
        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        _declared_files(project, schema, artifact)
        before = _declared_rows(service)
        result = await self._deliver(service, mission, project, idem)
        assert not result.ok
        assert result.error_code in {
            "DECLARED_SCHEMA_INVALID",
            "DECLARED_SCHEMA_UNSUPPORTED_KEYWORD",
            "DECLARED_SCHEMA_BUDGET_EXCEEDED",
            "DECLARED_SCHEMA_VALIDATION_FAILED",
        }
        assert _declared_rows(service) == before
        await service.close()

    async def _assert_accepted(self, tmp_path: Path, idem: str, schema, artifact) -> None:
        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        _declared_files(project, schema, artifact)
        result = await self._deliver(service, mission, project, idem)
        assert result.ok and result.artifact_refs
        await service.close()

    async def test_required_wrong_type_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(
            tmp_path, "idem_h_req_type", {"type": "object", "required": "nonce"}, {}
        )

    async def test_required_duplicates_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_req_dup", {"required": ["a", "a"]}, {"a": 1})

    async def test_required_nonstrings_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_req_ns", {"required": [1]}, {})

    async def test_required_empty_valid(self, tmp_path: Path) -> None:
        await self._assert_accepted(
            tmp_path, "idem_h_req_empty", {"type": "object", "required": []}, {}
        )

    async def test_enum_wrong_type_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_enum_type", {"enum": "fixed"}, "different")

    async def test_enum_empty_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_enum_empty", {"enum": []}, 1)

    async def test_enum_numeric_duplicates_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_enum_num", {"enum": [1, 1.0]}, 1)

    async def test_enum_object_duplicates_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(
            tmp_path,
            "idem_h_enum_obj",
            {"enum": [{"a": 1, "b": 2}, {"b": 2, "a": 1}]},
            {"a": 1, "b": 2},
        )

    async def test_enum_bool_number_distinct_valid(self, tmp_path: Path) -> None:
        await self._assert_accepted(tmp_path, "idem_h_enum_bool", {"enum": [True, 1]}, True)

    async def test_enum_ordered_arrays_distinct_valid(self, tmp_path: Path) -> None:
        await self._assert_accepted(tmp_path, "idem_h_enum_arr", {"enum": [[1, 2], [2, 1]]}, [1, 2])

    async def test_enum_nested_object_order_valid(self, tmp_path: Path) -> None:
        await self._assert_accepted(
            tmp_path,
            "idem_h_enum_nest",
            {"enum": [{"a": {"x": 1, "y": 2}, "b": 3}]},
            {"b": 3, "a": {"y": 2, "x": 1}},
        )

    async def test_additional_properties_wrong_type_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(
            tmp_path,
            "idem_h_add_type",
            {"type": "object", "additionalProperties": 7},
            {"foreign": 1},
        )

    async def test_additional_child_valid(self, tmp_path: Path) -> None:
        await self._assert_accepted(
            tmp_path,
            "idem_h_add_ok",
            {"type": "object", "additionalProperties": {"type": "integer"}},
            {"a": 1},
        )

    async def test_additional_child_invalid(self, tmp_path: Path) -> None:
        await self._assert_rejected(
            tmp_path,
            "idem_h_add_bad",
            {"type": "object", "additionalProperties": {"type": "integer"}},
            {"a": "bad"},
        )

    async def test_type_empty_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_type_empty", {"type": []}, 1)

    async def test_type_duplicates_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_type_dup", {"type": ["number", "number"]}, 1)

    async def test_bound_wrong_type_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(
            tmp_path, "idem_h_bound_type", {"type": "string", "minLength": "20"}, "a"
        )

    async def test_bound_boolean_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_bound_bool", {"minLength": True}, "")

    async def test_bound_negative_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_bound_neg", {"minItems": -1}, [])

    async def test_bound_fractional_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_bound_frac", {"maxLength": 1.5}, "")

    async def test_title_wrong_type_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_title", {"title": 1}, {})

    async def test_description_wrong_type_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_desc", {"description": False}, {})

    async def test_items_wrong_type_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(tmp_path, "idem_h_items", {"items": []}, [])

    async def test_properties_child_wrong_type_rejected(self, tmp_path: Path) -> None:
        await self._assert_rejected(
            tmp_path, "idem_h_prop_child", {"properties": {"a": 1}}, {"a": 1}
        )

    async def test_shallow_string_brackets_quotes_valid(self, tmp_path: Path) -> None:
        await self._assert_accepted(
            tmp_path, "idem_h_str", {"type": "string"}, '[{"quoted"}] \\ "escape'
        )

    async def test_shallow_schema_description_valid(self, tmp_path: Path) -> None:
        await self._assert_accepted(
            tmp_path,
            "idem_h_desc_ok",
            {"type": "object", "description": '[{"bracket"}] \\ escaped'},
            {},
        )

    async def test_schema_decoder_depth_typed_budget_error(self, tmp_path: Path) -> None:
        import json as _json

        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        project.mkdir(parents=True, exist_ok=True)
        (project / "schema.json").write_text("[" * 5000 + "0" + "]" * 5000)
        (project / "delivery.json").write_text(_json.dumps(_DECLARED_VALID_ARTIFACT))
        before = _declared_rows(service)
        result = await self._deliver(service, mission, project, "idem_h_schema_depth")
        assert not result.ok
        assert result.error_code == "DECLARED_SCHEMA_BUDGET_EXCEEDED"
        assert _declared_rows(service) == before
        await service.close()

    async def test_artifact_decoder_depth_typed_budget_error(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        project.mkdir(parents=True, exist_ok=True)
        (project / "schema.json").write_text("{}")
        (project / "delivery.json").write_text("[" * 5000 + "0" + "]" * 5000)
        before = _declared_rows(service)
        result = await self._deliver(service, mission, project, "idem_h_art_depth")
        assert not result.ok
        assert result.error_code == "DECLARED_SCHEMA_BUDGET_EXCEEDED"
        assert _declared_rows(service) == before
        await service.close()

    async def test_oversize_schema_bounded_read(self, tmp_path: Path) -> None:
        """超限 schema 在 cap+1 有限读取后即 typed 拒绝（零新行）。"""
        import json as _json

        service, mission = await _setup(tmp_path)
        project = tmp_path / "project"
        project.mkdir(parents=True, exist_ok=True)
        (project / "schema.json").write_text(_json.dumps({"description": "x" * 70000}))
        (project / "delivery.json").write_text(_json.dumps(_DECLARED_VALID_ARTIFACT))
        before = _declared_rows(service)
        result = await self._deliver(service, mission, project, "idem_h_oversize")
        assert not result.ok
        assert result.error_code == "DECLARED_SCHEMA_BUDGET_EXCEEDED"
        assert _declared_rows(service) == before
        await service.close()
