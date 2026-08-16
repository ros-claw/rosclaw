"""建议-0816 P0-3/P0-6 红测试：TaskSpec 编译诚实 + 完整指纹。

红测试先行——修复前必须红：
1. profile→capability/effect 映射诚实（developer=code.develop+
   workspace_write，scout=code.repository_analysis+none）——不再
   全部 analysis.text+none；
2. 任务指纹覆盖 inputs/deliverables/acceptance——同 goal 不同参数
   不得 attach（用户改五角星大小不能错挂旧任务）。
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

from rosclaw.agentd.control_plane import TaskControlPlane, _fingerprint
from tests.agentd.test_pi_tool_bridge import _setup


def _enable_fake(service, tmp_path: Path, monkeypatch) -> None:
    from rosclaw.agentd.workers import pi_managed

    fake = tmp_path / "fake-quick"
    fake.write_text(
        "#!/bin/sh\n"
        'echo \'{"kind":"attempt_started"}\'\n'
        'echo \'{"kind":"attempt_finished","report":"done"}\'\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(
        pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake))
    )
    adapter = pi_managed.PiManagedAdapter(
        rosclaw_home=tmp_path, conn=service._store.connection
    )
    service._worker_manager._adapters["pi_managed"] = adapter
    adapter._manager_ref = service._worker_manager
    if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
        service._registry.set_status(
            "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake"
        )


class TestCompilerHonesty:
    async def test_developer_declares_write_effect(self, tmp_path: Path,
                                                   monkeypatch) -> None:
        """code.* 任务 → developer + code.develop + workspace_write
        （账本不得声称 none 副作用而 worker 实际可写）。"""
        service, mission = await _setup(tmp_path)
        _enable_fake(service, tmp_path, monkeypatch)
        plane = service._task_control_plane
        await plane.submit(
            mission.mission_id,
            {"goal": "修复测试", "required_capabilities": ["code.implement"],
             "effects": "workspace_only"},
            idem="p03_dev",
        )
        for _ in range(200):
            orders = service._worker_manager.orders_for_mission(mission.mission_id)
            if orders:
                break
            await asyncio.sleep(0.05)
        order = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        assert order.capability == "code.develop", order.capability
        assert order.side_effect_policy.class_ in (
            "workspace_write", "sandbox_process",
        ), order.side_effect_policy.class_
        for _ in range(200):
            if order.status in ("ACCEPTED", "FAILED", "CANCELLED"):
                break
            await asyncio.sleep(0.05)
        await service.close()

    async def test_read_only_task_honest_none(self, tmp_path: Path,
                                              monkeypatch) -> None:
        """阅读/分析任务 → scout + repository_analysis + none。"""
        service, mission = await _setup(tmp_path)
        _enable_fake(service, tmp_path, monkeypatch)
        plane = service._task_control_plane
        await plane.submit(
            mission.mission_id,
            {"goal": "读一下项目结构", "required_capabilities": ["repo.analyze"],
             "effects": ""},
            idem="p03_scout",
        )
        for _ in range(200):
            orders = service._worker_manager.orders_for_mission(mission.mission_id)
            if orders:
                break
            await asyncio.sleep(0.05)
        order = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        assert order.capability in ("code.repository_analysis", "analysis.text")
        assert order.side_effect_policy.class_ == "none"
        for _ in range(200):
            if order.status in ("ACCEPTED", "FAILED", "CANCELLED"):
                break
            await asyncio.sleep(0.05)
        await service.close()


class TestSimulationRoutingHonesty:
    async def test_unknown_simulation_goes_to_harness(self, tmp_path: Path) -> None:
        """P0-7：非 planar_trajectory 的仿真任务不误进五角星函数——
        无确定性工作流时交 Pi Harness。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.control_plane import ExecutionRouter

        route = ExecutionRouter(service).route(
            {"goal": "G1 双足行走仿真", "required_capabilities": ["simulation.g1_locomotion"],
             "effects": "simulation_only"}
        )
        assert route["domain"] == "agent_harness", route
        route2 = ExecutionRouter(service).route(
            {"goal": "五角星", "required_capabilities": ["simulation.planar_trajectory"],
             "effects": "simulation_only"}
        )
        assert route2["runtime"] == "executor:simulation", route2
        await service.close()


class TestFullFingerprint:
    def test_inputs_change_fingerprint(self) -> None:
        """同 goal 不同 inputs → 不同指纹（用户改五角星半径不得错挂）。"""
        base = {"goal": "画五角星", "required_capabilities": ["simulation.planar_trajectory"],
                "effects": "simulation_only",
                "inputs": {"radius_m": 0.10}}
        changed = {**base, "inputs": {"radius_m": 0.20}}
        assert _fingerprint("mis_1", base) != _fingerprint("mis_1", changed)

    def test_acceptance_change_fingerprint(self) -> None:
        base = {"goal": "x", "acceptance": {"max_tracking_error_m": 0.05}}
        stricter = {"goal": "x", "acceptance": {"max_tracking_error_m": 0.01}}
        assert _fingerprint("mis_1", base) != _fingerprint("mis_1", stricter)

    def test_identical_spec_same_fingerprint(self) -> None:
        spec = {"goal": "x", "inputs": {"a": 1},
                "acceptance": {"max_tracking_error_m": 0.05}}
        assert _fingerprint("mis_1", spec) == _fingerprint("mis_1", dict(spec))

    async def test_changed_inputs_no_attach(self, tmp_path: Path) -> None:
        """同 goal 不同参数 → 新 execution（不 attach 旧任务）。"""
        service, mission = await _setup(tmp_path)
        plane: TaskControlPlane = service._task_control_plane
        first = await plane.submit(
            mission.mission_id,
            {"goal": "画五角星", "required_capabilities": ["simulation.planar_trajectory"],
             "effects": "simulation_only", "inputs": {"radius_m": 0.10}},
            idem="p06_a",
        )
        # 等第一个收尾（SIM executor 会跑完）。
        for _ in range(600):
            row = plane._get(first["execution_id"])
            if row["state"] in ("SUCCEEDED", "FAILED", "BLOCKED"):
                break
            await asyncio.sleep(0.1)
        second = await plane.submit(
            mission.mission_id,
            {"goal": "画五角星", "required_capabilities": ["simulation.planar_trajectory"],
             "effects": "simulation_only", "inputs": {"radius_m": 0.20}},
            idem="p06_b",
        )
        assert second["execution_id"] != first["execution_id"]
        assert not second.get("attached")
        await service.close()
