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
    def test_render_trace_pil_missing_honest_no_install(self, tmp_path: Path,
                                                        monkeypatch) -> None:
        """P0-F（0824 总纲 §15.1）：PIL 缺失 → RENDER_DEPS_MISSING
        诚实失败——任务期间绝不安装（不查 runtime manager、不发起
        pip install；Pillow 是主包依赖，安装阶段闭包）。"""
        import sys as _sys

        from rosclaw.agentd import sim_trajectory

        class _SpyManager:
            def ensure(self, name, spec=None):  # noqa: ARG002
                raise AssertionError("任务期仍尝试托管安装——P0-F 违约")

        monkeypatch.setitem(_sys.modules, "PIL", None)
        service = sim_trajectory.SimTrajectoryService(
            tmp_path, runtime_manager=_SpyManager()
        )
        with pytest.raises(ValueError, match="RENDER_DEPS_MISSING"):
            service._import_pil()

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


