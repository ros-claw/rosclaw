"""0827 体验审计 P0-6 红测试：PlanRef 跨进程统一 + 带码透传。

0827 实证：ur5e.plan_cartesian_path 产出 plan_28b6... →
ur5e_simulate_cartesian_trajectory 报 REF_NOT_FOUND——生产者和
消费者没共享同一个 PlanStore（ur5e_mcp 的 _PLAN_STORE 是 import
时解析的模块级单例：进程在 ROSCLAW_HOME 设置前 import 就永远拿
内存 store，或拿到另一个 home）——只是固定 recipe 绕开了问题；
且 REF_NOT_FOUND 被包装成无语义的 EXECUTOR_ERROR（MCP/执行器
包装前缀让锚定正则失配）。

闭环断言：
1. 生产者在调用时解析共享 store（import 后设置 ROSCLAW_HOME 也
   生效）——产出的 plan_id 消费者 SimTrajectoryService 可解析；
2. REF_NOT_FOUND/REF_FORMAT_UNKNOWN 等白名单稳定码即使被包装
   前缀包裹也透传（不得包成 EXECUTOR_ERROR）；
3. 启动 conformance：生产者探针计划消费者不可解析时，工具对
   被排除出快照（不兼容工具不进模型上下文）。
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestSharedPlanStoreResolution:
    def test_producer_store_resolves_at_call_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """import 后设置 ROSCLAW_HOME——生产者写入的 plan 必须落在
        消费者（SimTrajectoryService(home)）可解析的共享目录。"""
        import json as _json

        import rosclaw.sim.ur5e_mcp as ur5e_mcp
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        raw = ur5e_mcp.plan_cartesian_path(
            shape="star5", center_x=0.35, center_y=0.25, z=0.30,
            outer_radius=0.10,
        )
        result = _json.loads(raw)
        plan_id = result["plan_id"]
        # 消费者（生产链用的服务）必须能解析——不共享即 REF_NOT_FOUND。
        consumer = SimTrajectoryService(tmp_path)
        plan = consumer._load_plan(plan_id)
        assert plan["points"], plan
        ur5e_mcp._plan_store().clear()


class TestCodedErrorPassthrough:
    def test_wrapped_ref_not_found_keeps_code(self) -> None:
        """执行器/MCP 包装前缀（'Error executing tool ...: REF_NOT_FOUND:
        ...'）不得让稳定码退化成 EXECUTOR_ERROR。"""
        from rosclaw.agentd.tooling.catalog import stable_error_code

        assert stable_error_code(
            ValueError("REF_NOT_FOUND: plan 'plan_x' 不在共享 PlanStore")
        ) == "REF_NOT_FOUND"
        assert stable_error_code(
            RuntimeError(
                "Error executing tool ur5e.simulate_cartesian_trajectory: "
                "REF_NOT_FOUND: plan_id 'plan_x' 不在共享 PlanStore (fail closed)"
            )
        ) == "REF_NOT_FOUND"
        # 非码散文不冒充稳定码（无 CODE: 前缀 → EXECUTOR_ERROR）。
        assert stable_error_code(ValueError("plain failure")) == "EXECUTOR_ERROR"


class TestPlanRefConformance:
    def test_conformance_excludes_unresolvable_pair(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """conformance 探针：生产者写的 plan 消费者读不到（home
        分裂）→ 工具对进排除清单（带机器原因码）；共享时为空。"""
        from rosclaw.agentd.plan_ref_conformance import (
            plan_ref_conformance,
        )

        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        # 共享 home → 无排除。
        assert plan_ref_conformance(tmp_path) == []
        # 消费者 home 分裂 → 排除生产者+消费者对。
        other = tmp_path / "elsewhere"
        other.mkdir()
        excluded = plan_ref_conformance(other)
        ids = {str(e.get("capability_id")) for e in excluded}
        assert "ur5e.plan_cartesian_path" in ids, excluded
        assert "ur5e.simulate_cartesian_trajectory" in ids, excluded
        assert all(e.get("code") == "REF_CONFORMANCE_FAILED" for e in excluded)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
