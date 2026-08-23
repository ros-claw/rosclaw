"""WP-2 红测试（0823 审计 §三.P0-4/§四.WP-2）：Typed Reference 能力组合。

红测试先行——审计实证：kit plan_cartesian_path 产出的 plan_id 被
native simulate 报 unknown plan（两套进程/格式互不兼容）。

1. TypedRefV1 合约（uri/kind/producer/digest/body/task/revision/
   storage_backend）；
2. 共享 PlanStore：kit 与 native 读写同一磁盘 store（kit 子进程必须
   拿到 ROSCLAW_HOME——MCP 默认 env 不含它）；
3. 记录格式互通：kit envelope 记录被 native _load_plan 正确解包，
   native 产出被 kit 读取；
4. 消费方诚实报错：不可解码/来源不兼容 → REF_FORMAT_UNKNOWN /
   REF_PRODUCER_MISMATCH，不再是含糊的 "unknown plan"；
5. Capability 声明 accepts_refs/produces_refs；snapshot 只暴露可连接
   组合（消费者无生产者 → excluded REF_NOT_CONNECTABLE）。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


class TestTypedRefV1:
    def test_contract_and_golden(self) -> None:
        from rosclaw.contracts.agent.typed_ref import TypedRefV1

        ref = TypedRefV1.model_validate_contract({
            "schema_version": "rosclaw.typed_ref.v1",
            "kind": "plan",
            "uri": "rosclaw://plan/plan_abc123",
            "producer_capability": "ur5e.plan_cartesian_path",
            "digest": "sha256:" + "ab" * 32,
            "body_id": "sim/ur5e",
            "world_id": "",
            "task_id": "task_1",
            "revision": 1,
            "storage_backend": "disk:sim/plans",
        })
        assert ref.kind == "plan"
        golden = (
            REPO / "tests" / "contracts" / "golden" / "rosclaw.typed_ref.v1.json"
        )
        current = TypedRefV1.model_json_schema()
        current["$id"] = "rosclaw://schemas/rosclaw.typed_ref.v1"
        current["title"] = "rosclaw.typed_ref.v1"
        assert json.loads(golden.read_text(encoding="utf-8")) == current


class TestSharedPlanStore:
    async def test_kit_plan_consumable_by_native_simulator(
        self, monkeypatch
    ) -> None:
        """审计事故的直接复现：kit plan → native simulate 必须通。"""
        with tempfile.TemporaryDirectory() as td:
            # monkeypatch：测试结束自动还原——裸 os.environ 赋值会
            # 泄漏给同进程后续测试（CI 实证：body 测试集体
            # "Body already linked"）。
            monkeypatch.setenv("ROSCLAW_HOME", td)
            from tests.agentd.test_pi_tool_bridge import _setup

            service, mission = await _setup(Path(td))
            await service._ensure_mcp_discovered()
            plan_env = await service._tool_catalog.execute_v2(
                "c1", "ur5e.plan_cartesian_path",
                {"shape": "star5", "center_x": 0.35, "center_y": 0.25,
                 "z": 0.3, "outer_radius": 0.08},
            )
            assert plan_env.status.value == "SUCCEEDED", plan_env.error
            plan_id = (plan_env.value or {}).get("plan_id")
            assert plan_id
            # 落盘证据：共享 store 必须有这个 plan（kit 子进程看到
            # ROSCLAW_HOME）。
            plans_dir = Path(td) / "sim" / "plans"
            assert (plans_dir / f"{plan_id}.json").exists(), (
                "kit plan 未落盘到共享 store（子进程缺 ROSCLAW_HOME）"
            )
            sim_env = await service._tool_catalog.execute_v2(
                "c2", "ur5e_simulate_cartesian_trajectory",
                {"plan_id": plan_id},
            )
            assert sim_env.status.value == "SUCCEEDED", sim_env.error
            await service.close()

    def test_native_plan_readable_by_kit_store(self) -> None:
        """反向：native SimTrajectoryService 产出的 plan 被 kit 的
        PersistentPlanStore 正确读取（envelope/raw 互通）。"""
        with tempfile.TemporaryDirectory() as td:
            from rosclaw.agentd.sim_trajectory import SimTrajectoryService
            from rosclaw.sim.plan_store import PersistentPlanStore

            home = Path(td)
            sim = SimTrajectoryService(home)
            plan = sim.generate_planar_path(
                shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
            )
            store = PersistentPlanStore(home / "sim" / "plans")
            record = store.get_for_execute(plan["plan_id"])
            assert record["trajectory"]["points"], record.keys()

    def test_undecodable_ref_honest_error(self) -> None:
        """不可解码引用 → REF_FORMAT_UNKNOWN（不是 unknown plan）。"""
        with tempfile.TemporaryDirectory() as td:
            from rosclaw.agentd.sim_trajectory import SimTrajectoryService

            home = Path(td)
            plans = home / "sim" / "plans"
            plans.mkdir(parents=True)
            (plans / "plan_garbage.json").write_text(
                json.dumps({"totally": "foreign"}), encoding="utf-8"
            )
            sim = SimTrajectoryService(home)
            with pytest.raises(ValueError, match="REF_FORMAT_UNKNOWN"):
                sim.simulate_cartesian_trajectory("plan_garbage")

    def test_missing_ref_honest_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            from rosclaw.agentd.sim_trajectory import SimTrajectoryService

            sim = SimTrajectoryService(Path(td))
            with pytest.raises(ValueError, match="REF_NOT_FOUND"):
                sim.simulate_cartesian_trajectory("plan_nonexistent")


class TestRefConnectivity:
    def test_capability_declares_ref_ports(self) -> None:
        """CapabilityDescriptorV2 携带 accepts_refs/produces_refs。"""
        from rosclaw.contracts.agent.capability import CapabilityDescriptorV2

        cap = CapabilityDescriptorV2.model_validate_contract({
            "schema_version": "rosclaw.capability.v2",
            "capability_id": "ur5e_simulate_cartesian_trajectory",
            "source": "rosclaw-builtin",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "accepts_refs": [{"kind": "plan", "from": "ur5e.plan_cartesian_path"}],
            "produces_refs": [{"kind": "trace"}],
        })
        assert cap.accepts_refs[0]["kind"] == "plan"
        assert cap.produces_refs[0]["kind"] == "trace"

    def test_snapshot_excludes_unconnectable_consumer(self) -> None:
        """消费者需要的 ref kind 在 snapshot 中无生产者 → excluded
        REF_NOT_CONNECTABLE（模型看不到天然连不上的工具）。"""
        from rosclaw.agentd.tooling.catalog import ToolCatalog
        from rosclaw.agentd.tooling.snapshot import build_capability_snapshot
        from rosclaw.contracts.agent.tool import (
            ExecutionClass,
            ToolDescriptorV2,
            ToolEvidenceClass,
        )

        catalog = ToolCatalog()
        # 只注册消费者（需要 plan ref），不注册生产者。
        catalog.register(ToolDescriptorV2(
            tool_id="sim_needs_plan",
            source="native:agentd",
            execution_class=ExecutionClass.COMPUTE,
            evidence_class=ToolEvidenceClass.SIMULATED,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        ))
        # 声明它需要 plan ref（N5A 适配或 V2 直写）。
        cap = catalog.capability("sim_needs_plan")
        assert cap is not None
        cap2 = cap.model_copy(update={
            "accepts_refs": [{"kind": "plan", "from": ""}],
        })
        catalog._capabilities["sim_needs_plan"] = cap2
        snap = build_capability_snapshot(
            catalog, body_id="sim/ur5e", mode="SIMULATION"
        )
        excluded = {e.capability_id: e for e in snap.excluded}
        assert "sim_needs_plan" in excluded
        assert excluded["sim_needs_plan"].reason == "REF_NOT_CONNECTABLE"
