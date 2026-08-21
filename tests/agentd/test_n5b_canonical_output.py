"""PR-N5B 红测试（调整方案 §三.N5B）：canonical 输出——Registry 验证与接线。

红测试先行——execute_v2/输出校验不存在时必须红。

1. Registry 层按 output_schema 验证：多字段/少字段/类型错误 →
   FAILED + INVALID_CAPABILITY_OUTPUT（不抛出、不冒充成功）；
2. executor 返回裸非 JSON 字符串 → INVALID_CAPABILITY_OUTPUT；
3. model-callable 能力缺 output_schema → OUTPUT_SCHEMA_MISSING
   （诚实失败，不猜测）；
4. executor 提交 presentation_meta → 拒绝（只能由可信 projection
   生成）；
5. 守卫映射：quarantine/PHYSICAL → BLOCKED envelope；
6. 产品接线：rosclaw_compute 经 dispatcher 返回 canonical value
   投影（status + value），坏 executor 的错误码模型可见；
7. 结构扫描：全部内置 model-callable 能力 input/output schema 非空。
"""

from __future__ import annotations

import json

from rosclaw.contracts.agent.tool import ExecutionClass, ToolDescriptorV2

_SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string"},
        "physics_steps": {"type": "integer"},
    },
    "required": ["run_id"],
    "additionalProperties": False,
}


def _catalog_with(tool_id="sim_reach", output_schema=None, executor=None):
    from rosclaw.agentd.tooling.catalog import ToolCatalog

    catalog = ToolCatalog()
    descriptor = ToolDescriptorV2(
        tool_id=tool_id,
        source="native:agentd",
        execution_class=ExecutionClass.COMPUTE,
        input_schema={"type": "object", "additionalProperties": False},
        output_schema=_SCHEMA if output_schema is None else output_schema,
    )

    async def _default(arguments):
        return {"run_id": "run_1", "physics_steps": 730}

    catalog.register(descriptor, executor or _default)
    return catalog


class TestRegistryOutputValidation:
    async def test_success_envelope(self) -> None:
        catalog = _catalog_with()
        env = await catalog.execute_v2("call_1", "sim_reach", {})
        assert env.status.value == "SUCCEEDED"
        assert env.call_id == "call_1"
        assert env.capability_id == "sim_reach"
        assert env.value == {"run_id": "run_1", "physics_steps": 730}
        assert env.error is None

    async def test_extra_field_rejected(self) -> None:
        async def bad(arguments):
            return {"run_id": "run_1", "unexpected": True}

        env = await _catalog_with(executor=bad).execute_v2("c", "sim_reach", {})
        assert env.status.value == "FAILED"
        assert env.error is not None
        assert env.error.code == "INVALID_CAPABILITY_OUTPUT"
        assert env.error.retryable is False

    async def test_missing_required_field_rejected(self) -> None:
        async def bad(arguments):
            return {"physics_steps": 730}

        env = await _catalog_with(executor=bad).execute_v2("c", "sim_reach", {})
        assert env.status.value == "FAILED"
        assert env.error.code == "INVALID_CAPABILITY_OUTPUT"

    async def test_wrong_type_rejected(self) -> None:
        async def bad(arguments):
            return {"run_id": 42}

        env = await _catalog_with(executor=bad).execute_v2("c", "sim_reach", {})
        assert env.status.value == "FAILED"
        assert env.error.code == "INVALID_CAPABILITY_OUTPUT"

    async def test_bare_string_masquerade_rejected(self) -> None:
        """executor 随意返回一段字符串冒充结构化结果 → 拒绝。"""
        async def bad(arguments):
            return "完成了，效果很好"

        env = await _catalog_with(executor=bad).execute_v2("c", "sim_reach", {})
        assert env.status.value == "FAILED"
        assert env.error.code == "INVALID_CAPABILITY_OUTPUT"

    async def test_missing_output_schema_honest_failure(self) -> None:
        env = await _catalog_with(output_schema={}).execute_v2(
            "c", "sim_reach", {}
        )
        assert env.status.value == "FAILED"
        assert env.error.code == "OUTPUT_SCHEMA_MISSING"

    async def test_executor_exception_failed_envelope(self) -> None:
        async def boom(arguments):
            raise RuntimeError("physics exploded")

        env = await _catalog_with(executor=boom).execute_v2("c", "sim_reach", {})
        assert env.status.value == "FAILED"
        assert env.error.code == "EXECUTOR_ERROR"
        assert "physics exploded" in env.error.message

    async def test_executor_presentation_meta_rejected(self) -> None:
        """presentationMeta 只能由可信 projection 生成——executor 提交
        即拒绝。"""
        async def bad(arguments):
            return {"run_id": "run_1", "physics_steps": 1,
                    "presentationMeta": {"card": "fake"}}

        # 宽 schema 让其余字段通过——单独测 presentationMeta 规则
        catalog = _catalog_with(output_schema={
            "type": "object",
            "properties": {"run_id": {"type": "string"},
                           "physics_steps": {"type": "integer"}},
            "required": ["run_id"],
            "additionalProperties": True,
        })
        # 替换 executor
        catalog._executors["sim_reach"] = bad
        env = await catalog.execute_v2("c", "sim_reach", {})
        assert env.status.value == "FAILED"
        assert env.error.code == "INVALID_CAPABILITY_OUTPUT"
        assert "presentation" in env.error.message.lower()

    async def test_guards_map_to_blocked(self) -> None:
        from rosclaw.contracts.agent.tool import ToolSideEffectClass

        catalog = _catalog_with()
        catalog.quarantine_tool("sim_reach", "health check failed")
        env = await catalog.execute_v2("c", "sim_reach", {})
        assert env.status.value == "BLOCKED"
        assert env.error.code == "CAPABILITY_QUARANTINED"

        physical = ToolDescriptorV2(
            tool_id="physical_move",
            source="native:agentd",
            execution_class=ExecutionClass.PHYSICAL_ACTION,
            side_effect_class=ToolSideEffectClass.REVERSIBLE,
            model_callable=False,
            requires_exact_action_grant=True,
            input_schema={"type": "object"},
            output_schema=_SCHEMA,
        )
        catalog.register(physical)
        env = await catalog.execute_v2("c", "physical_move", {})
        assert env.status.value == "BLOCKED"
        assert env.error.code == "TOOL_NOT_CALLABLE"


class TestNativeSchemasComplete:
    def test_all_native_model_callable_have_schemas(self) -> None:
        """结构扫描：内置 model-callable 能力 input/output schema 非空
        （CI 硬约束——空 schema 即红）。"""
        from rosclaw.agentd.tooling.catalog import ToolCatalog
        from rosclaw.agentd.tooling.native_tools import register_native_tools
        from rosclaw.agentd.tools import BuiltinToolRegistry

        catalog = ToolCatalog()
        register_native_tools(
            catalog,
            BuiltinToolRegistry(body_id="sim/ur5e", body_summary="UR5e"),
            simulation=True,
        )
        for d in catalog.list(source="native:agentd"):
            if not d.model_callable:
                continue
            assert d.input_schema, f"{d.tool_id} input_schema 为空"
            assert d.output_schema, f"{d.tool_id} output_schema 为空"


class TestDispatcherCanonicalProjection:
    """产品接线：模型看到的是 canonical value 的投影。"""

    async def _service(self, tmp_path):
        from tests.agentd.test_pi_tool_bridge import _setup

        return await _setup(tmp_path)

    async def test_compute_returns_canonical_projection(self, tmp_path) -> None:
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from tests.agentd.test_pi_tool_bridge import _issue_lease, _request

        service, mission = await self._service(tmp_path)
        result = await PiToolDispatcher(service).execute(
            caller_pid=1, caller_uid=1000,
            request=_request(
                "rosclaw_compute",
                mission=mission.mission_id, idem="n5b_1",
                lease=await _issue_lease(service, mission),
                arguments={
                    "capability_id": "trajectory_generate_planar_path",
                    "arguments": {"shape": "star5",
                                  "center_m": [0.35, 0.25, 0.30],
                                  "scale_m": 0.10},
                },
            ),
        )
        assert result.ok, result
        projection = json.loads(result.summary)
        assert projection["status"] == "SUCCEEDED"
        assert projection["capability_id"] == "trajectory_generate_planar_path"
        value = projection["value"]
        assert value["ok"] is True and value["plan_id"]
        # 载荷不带 points（句柄+摘要契约不变）
        assert "points" not in value
        await service.close()

    async def test_broken_executor_error_code_visible(self, tmp_path) -> None:
        """坏 executor → 模型可见 INVALID_CAPABILITY_OUTPUT（诚实失败，
        不再把垃圾文本当结果）。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from tests.agentd.test_pi_tool_bridge import _issue_lease, _request

        service, mission = await self._service(tmp_path)

        async def bad(arguments):
            return {"oops": True}

        service._tool_catalog._executors["trajectory_generate_planar_path"] = bad
        result = await PiToolDispatcher(service).execute(
            caller_pid=1, caller_uid=1000,
            request=_request(
                "rosclaw_compute",
                mission=mission.mission_id, idem="n5b_2",
                lease=await _issue_lease(service, mission),
                arguments={
                    "capability_id": "trajectory_generate_planar_path",
                    "arguments": {"shape": "star5",
                                  "center_m": [0.35, 0.25, 0.30],
                                  "scale_m": 0.10},
                },
            ),
        )
        # 模型可见面：ok=False + 稳定错误码（dispatcher 把
        # ToolBridgeError 转成诚实 REJECTED 结果）。
        assert result.ok is False
        assert result.error_code == "INVALID_CAPABILITY_OUTPUT", result
        await service.close()
