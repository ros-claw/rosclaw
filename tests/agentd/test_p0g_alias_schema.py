"""P0-G 红测试（0824 总纲 §19.P0-G）：Typed Ref 与 Alias 收敛。

红测试先行——canonical alias 模块/Schema 完整性 CI 检查不存在
时必须红。

验收（文档原文）：
- 任一 producer 输出可被下一能力直接消费（WP-2/3 已钉住 plan→
  simulate→render→verify 同 ref）；
- alias 在 setup/header/inspect/verifier 一致（sim/ur5e ↔
  robot:ur5e 唯一权威映射）；
- registry/executor/schema 变异测试全红——output_schema 缺失成为
  CI build error，而非运行时发现。
"""

from __future__ import annotations

from pathlib import Path


class TestCanonicalAlias:
    def test_body_to_resource_canonical(self) -> None:
        """sim/ur5e → robot:ur5e 唯一权威映射（alias 模块存在）。"""
        from rosclaw.cognition.alias import canonical_resource_id

        assert canonical_resource_id("sim/ur5e") == "robot:ur5e"
        assert canonical_resource_id("ur5e") == "robot:ur5e"

    def test_resource_to_body_canonical(self) -> None:
        from rosclaw.cognition.alias import body_id_for_resource

        assert body_id_for_resource("robot:ur5e") == "sim/ur5e"

    def test_unknown_body_honest(self) -> None:
        """未知 body 不猜——原样返回并标记非规范（调用方决定是否
        fail closed）。"""
        from rosclaw.cognition.alias import canonical_resource_id

        assert canonical_resource_id("sim/nonexistent") == "robot:nonexistent"

    def test_verifier_provenance_uses_canonical_alias(
        self, tmp_path: Path
    ) -> None:
        """N4.1 资源证明比对走同一 alias（finish_task 不再手写
        removeprefix）。"""
        import inspect

        from rosclaw.task_kernel import service

        src = inspect.getsource(service.TaskKernel.finish_task)
        assert "canonical_resource_id" in src, (
            "finish_task 资源证明仍手写 removeprefix——alias 不收敛"
        )


class TestOutputSchemaCompleteness:
    def test_all_native_tools_have_output_schema(self) -> None:
        """全部原生工具 output_schema 齐全（CI 期爆炸，不是运行时
        OUTPUT_SCHEMA_MISSING）。"""
        from rosclaw.agentd.tooling.catalog import ToolCatalog
        from rosclaw.agentd.tooling.native_tools import register_native_tools
        from rosclaw.agentd.tools import BuiltinToolRegistry

        catalog = ToolCatalog()
        register_native_tools(
            catalog,
            BuiltinToolRegistry(body_id="sim/ur5e", body_summary="UR5e"),
        )
        missing = [
            d.tool_id
            for d in catalog.list(source="native:agentd")
            if not d.output_schema
        ]
        assert missing == [], f"原生工具缺 output_schema: {missing}"

    def test_all_kit_tools_have_output_schema(self) -> None:
        """第一方 kit 全部工具 output_schema 齐全（N5B 漂移事故的
        CI 化——报告声明的能力必须在构建期可证）。"""
        from rosclaw.sim.robot_kit import kit_for_body, kit_server_spec

        kit = kit_for_body("sim/ur5e")
        assert kit is not None
        spec = kit_server_spec(kit)
        schemas = spec.get("output_schemas") or {}
        all_tools = (
            list(kit.observation_tools)
            + list(kit.compute_tools)
            + list(kit.action_tools)
        )
        # action 工具走 admission（envelope 自带形状）；observation/
        # compute 必须有显式 output_schema。
        strict = list(kit.observation_tools) + list(kit.compute_tools)
        missing = [name for name in strict if name not in schemas]
        assert missing == [], f"kit 工具缺 output_schema: {missing}"
        assert all_tools, "kit 工具集为空——测试无效"


class TestTypedRefConsumption:
    def test_plan_to_simulate_to_render_same_ref(self, tmp_path: Path) -> None:
        """plan_id → simulate → trace_id → render/verify 全链同
        ref（任一 producer 输出可被下一能力直接消费）。"""
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        sim = SimTrajectoryService(tmp_path)
        plan = sim.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
        )
        run = sim.simulate_cartesian_trajectory(plan["plan_id"])
        assert run["trace_id"]
        verify = sim.verify_tracking(
            run["trace_id"], max_tracking_error_m=0.05,
        )
        assert verify["verdict"] == "PASS"
