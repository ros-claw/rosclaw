"""十六审 PR-17.3 红测试：Runtime/Dependency Manager（P0-C）。

红测试先行——修复前必须红：
1. 仿真渲染的 Pillow 依赖由 ROSClaw 托管 runtime 声明（不是让 Worker
   猜哪个 python、更不是改用户 conda/系统 Python）；
2. ensure() 在 ~/.rosclaw/runtimes/<name>/<digest>/ 建 venv + 装包 +
   probe + READY 标记；幂等复用；
3. probe 失败诚实 RuntimeNotReadyError（无 READY 标记，不假成功）；
4. sim render_trace 走托管 runtime（ImportError → 托管激活 → 重试；
   托管也缺 → 诚实 RUNTIME_NOT_READY，不是裸 ModuleNotFoundError）；
5. runtime_requirements.python_packages 任务：控制面 PREFLIGHT 预置
   托管 runtime，Worker PATH 前缀拿到托管 bin；
6. `doctor simulation` 报告托管 runtime 状态（无需模型凭据）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


class TestRuntimeManager:
    def test_simulation_runtime_spec_registered(self) -> None:
        from rosclaw.agentd.runtime_manager import RUNTIME_SPECS

        spec = RUNTIME_SPECS.get("rosclaw-simulation")
        assert spec is not None, "仿真 runtime 未声明"
        assert any("Pillow" in p for p in spec["python_packages"])
        assert spec.get("probe_module"), "缺 readiness probe"

    def test_ensure_builds_and_reuses_managed_venv(self, tmp_path: Path) -> None:
        from rosclaw.agentd.runtime_manager import RuntimeManager

        manager = RuntimeManager(tmp_path)
        handle = manager.ensure(
            "test-rt",
            {"python_packages": [], "probe_module": "json"},
        )
        assert handle.python.exists(), handle.python
        assert handle.site_packages.exists()
        assert (handle.directory / "READY.json").exists()
        # 幂等复用：第二次 ensure 不重建（READY 标记时间不变）。
        marker_mtime = (handle.directory / "READY.json").stat().st_mtime_ns
        again = manager.ensure(
            "test-rt",
            {"python_packages": [], "probe_module": "json"},
        )
        assert again.directory == handle.directory
        assert (handle.directory / "READY.json").stat().st_mtime_ns == marker_mtime
        # 托管目录在 ROSClaw home 下——绝不碰用户 conda/系统 Python。
        assert str(handle.directory).startswith(str(tmp_path))

    def test_ensure_probe_failure_honest(self, tmp_path: Path) -> None:
        from rosclaw.agentd.runtime_manager import (
            RuntimeManager,
            RuntimeNotReadyError,
        )

        manager = RuntimeManager(tmp_path)
        with pytest.raises(RuntimeNotReadyError, match="probe"):
            manager.ensure(
                "test-bad",
                {"python_packages": [],
                 "probe_module": "definitely_missing_module_xyz"},
            )
        # 无 READY 标记——失败诚实落账。
        assert not list((tmp_path / "runtimes").rglob("READY.json"))

    def test_site_packages_activation(self, tmp_path: Path) -> None:
        from rosclaw.agentd.runtime_manager import RuntimeManager

        manager = RuntimeManager(tmp_path)
        handle = manager.ensure(
            "test-rt",
            {"python_packages": [], "probe_module": "json"},
        )
        before = list(sys.path)
        try:
            manager.activate(handle)
            assert str(handle.site_packages) in sys.path
        finally:
            sys.path[:] = before


class TestSimRenderRuntime:
    def test_render_trace_uses_managed_runtime(self, tmp_path: Path,
                                               monkeypatch) -> None:
        """render_trace 必须先 ensure 仿真 runtime（不是裸 import PIL
        碰运气）。托管也缺 → RuntimeNotReadyError（诚实）。"""
        import sys as _sys

        from rosclaw.agentd import sim_trajectory
        from rosclaw.agentd.runtime_manager import RuntimeNotReadyError

        calls: list[str] = []

        class _SpyManager:
            def ensure(self, name, spec=None):
                calls.append(name)
                raise RuntimeNotReadyError("probe failed: PIL")

        # 强制宿主 PIL 缺失（sys.modules[name]=None → ImportError）——
        # 验证的是托管 fallback 路径，不是本机环境巧合。
        monkeypatch.setitem(_sys.modules, "PIL", None)
        service = sim_trajectory.SimTrajectoryService(
            tmp_path, runtime_manager=_SpyManager()
        )
        with pytest.raises(RuntimeNotReadyError):
            service._import_pil()
        assert "rosclaw-simulation" in calls

    def test_render_trace_real_gif_via_manager(self, tmp_path: Path) -> None:
        """真实链路：managed runtime ensure → 渲染 GIF 成功（本机 .venv
        有 Pillow——激活是幂等旁路，不断言绕过）。"""
        pytest.importorskip("PIL")
        from rosclaw.agentd.runtime_manager import RuntimeManager
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        manager = RuntimeManager(tmp_path)
        service = SimTrajectoryService(tmp_path / "sim", runtime_manager=manager)
        plan = service.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.08
        )
        result = service.simulate_cartesian_trajectory(plan["plan_id"])
        render = service.render_trace(result["trace_id"], format="gif")
        artifact = Path(render["artifact"]["path"])
        assert artifact.exists() and artifact.stat().st_size > 1000
        assert render["artifact"]["frames"] >= 30


class TestRuntimePreflightWiring:
    async def test_runtime_requirements_preflighted_to_worker_path(
        self, tmp_path: Path
    ) -> None:
        """runtime_requirements.python_packages → 控制面 PREFLIGHT 预置
        托管 runtime，WorkOrder 拿到 _runtime_bin（Worker PATH 前缀）。"""
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        # 直接驱动（无需 fake worker——preflight 在 hire 前完成）。
        from rosclaw.agentd.runtime_manager import RuntimeManager

        service._runtime_manager = RuntimeManager(tmp_path)
        # CI 无 Node/dist——harness 就绪性不是本测试的对象；直接 ENABLED
        # （只断言 preflight 注入 WorkOrder，不跑真实 worker）。
        if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
            service._registry.set_status(
                "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="ci"
            )
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "需要 Pillow 的任务",
             "effects": "workspace_only",
             "runtime_requirements": {"python_packages": []}},
            idem="17c_preflight",
        )
        # 等 preflight 完成（进入 RUNNING/BLOCKED 任一——关键是 order
        # inputs 里必须有 _runtime_bin）。
        import asyncio

        for _ in range(600):
            row = plane._get(view["execution_id"])
            if row.get("work_order_id") or row["state"] in (
                "SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED",
            ):
                break
            await asyncio.sleep(0.05)
        row = plane._get(view["execution_id"])
        assert row.get("work_order_id"), f"未创建 WorkOrder: {row['state']} {row['summary']}"
        order = service._worker_manager.order(row["work_order_id"])
        runtime_bin = str(order.inputs.get("_runtime_bin") or "")
        assert runtime_bin, "runtime_requirements 未预置进 WorkOrder"
        assert runtime_bin.startswith(str(tmp_path)), (
            f"托管 bin 必须在 ROSClaw home 下: {runtime_bin}"
        )
        assert (Path(runtime_bin) / "python3").exists() or (
            Path(runtime_bin) / "python"
        ).exists()
        await service.close()


class TestDoctorSimulation:
    def test_doctor_simulation_runtime_report(self, tmp_path: Path) -> None:
        """doctor simulation：托管 runtime 状态（无需模型凭据）。"""
        from rosclaw.agentd.runtime_manager import doctor_runtime

        report = doctor_runtime(tmp_path, "simulation")
        assert report["runtime"] == "rosclaw-simulation"
        assert "ready" in report and "packages" in report
