"""G-6（0916 三审 B-4）：单一仿真栈——防分叉门禁（ADR-0015）。

三审稿：正在形成新的"双仿真栈"风险。ADR-0015 冻结：
SimulationRuntime（src/rosclaw/sim/）是唯一仿真内核；
agentd/sim_render.py 降级 deprecated 适配层——公开函数集合
冻结（只缩不胀），新仿真能力只能进新栈。

闭环断言：
1. sim_render 公开函数集合 == 冻结 baseline（多一个即红——
   新增能力必须进 src/rosclaw/sim/）；
2. sim_render 模块 docstring 标 deprecated 并指向新栈；
3. 新栈 SimulationRuntime 公开面存在且可用（import 实证——
   "唯一内核"不是纸面宣称）。
"""
from __future__ import annotations

import inspect

from rosclaw.agentd import sim_render

# ADR-0015 冻结清单（2026-09-16，main=cf9aeb3d）——收缩（删函数）
# 是收敛方向，更新本清单必须同 PR 说明归宿；增长永远拒绝。
_FROZEN_PUBLIC = frozenset({
    "apply_camera_framing",
    "os_environ",
    "probe_render_backend",
    "render_from_spec",
    "render_identity_key",
    "render_operation",
    "render_scene_trace",
    "restore_frame_state",
    "select_frame_indices",
    # 例外（ADR-0015 §2 维护性修改——取消传播契约保持，不是新仿
    # 真能力）：G-4b 同步渲染取消注册表（CI 实证本门禁先抓到了
    # 它们——门禁按设计工作，此处带理由登记豁免）。
    "has_active_renders",
    "kill_active_renders",
})


class TestAdapterLayerFrozen:
    def test_public_surface_never_grows(self) -> None:
        current = {
            name
            for name, obj in inspect.getmembers(sim_render, inspect.isfunction)
            if not name.startswith("_") and obj.__module__ == sim_render.__name__
        }
        extra = current - _FROZEN_PUBLIC
        assert not extra, (
            f"sim_render 公开面新增 {sorted(extra)}——ADR-0015：新仿真能力"
            "只能进 src/rosclaw/sim/（单一仿真栈），适配层只缩不胀"
        )
        removed = _FROZEN_PUBLIC - current
        assert not removed, (
            f"sim_render 公开面收缩 {sorted(removed)}——更新冻结清单必须"
            "同 PR 说明归宿（ADR-0015 §后果）"
        )

    def test_module_marked_deprecated(self) -> None:
        doc = inspect.getdoc(sim_render) or ""
        assert "deprecated" in doc.lower(), "适配层必须标注 deprecated"
        assert "rosclaw.sim" in doc, "必须指向唯一内核（新栈）"


class TestSoleKernelReal:
    def test_simulation_runtime_importable_and_capable(self) -> None:
        """SimulationRuntime 是唯一内核——import 实证 + 能力面存在
        （不是纸面宣称）。"""
        from rosclaw.sim.runtime import SimulationRuntime

        runtime = SimulationRuntime(task_root=None)
        caps = runtime.get_capabilities()
        assert caps, "唯一内核无能力面"
        assert caps.get("usable_for_real_execution") is False, (
            "仿真内核永远不得宣称真实执行（ADR-0014 证据语言边界）"
        )
