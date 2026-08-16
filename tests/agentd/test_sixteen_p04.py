"""建议-0816 P0-4 红测试：冻结外部 Harness 发现 + model_snapshot 继承。

红测试先行——修复前必须红：
1. auto_discovery=false（默认）→ codex 二进制+~/.codex 存在也路由
   pi-builtin（绝不偷换执行者）；
2. task_submit 携带的 model_snapshot 传入 WorkOrder inputs（Worker
   继承 Native 当前模型）；
3. execution 卡携带真实模型快照字段（provider/model/thinking）。
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


class TestAutoDiscoveryFrozen:
    async def test_codex_present_but_pi_wins(self, tmp_path: Path,
                                             monkeypatch) -> None:
        """codex + ~/.codex 同时存在也不影响路由（配置冻结）。"""
        monkeypatch.setattr(shutil, "which", lambda b: f"/usr/bin/{b}")
        from rosclaw.agentd.control_plane import ExecutionRouter

        service, mission = await _setup(tmp_path)
        route = ExecutionRouter(service).route(
            {"goal": "写代码", "required_capabilities": ["code.implement"],
             "effects": "workspace_only"}
        )
        assert route["runtime"] == "harness:pi-builtin", route
        await service.close()

    async def test_explicit_enable_unlocks_codex(self, tmp_path: Path,
                                                 monkeypatch) -> None:
        """显式 enabled: [codex-app-server] 才允许 codex 路径。"""
        monkeypatch.setattr(shutil, "which", lambda b: f"/usr/bin/{b}")
        # codex 登录目录也要就位（CI 无 ~/.codex——readiness 是真实检查）。
        (tmp_path / ".codex").mkdir(exist_ok=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        service, mission = await _setup(tmp_path)
        # 写入配置启用 codex。
        service._config.agent_runtime.enabled = ["codex-app-server"]
        service._config.agent_runtime.auto_discovery = True
        from rosclaw.agentd.control_plane import ExecutionRouter

        route = ExecutionRouter(service).route(
            {"goal": "写代码", "required_capabilities": ["code.implement"],
             "effects": "workspace_only"}
        )
        assert route["runtime"] == "harness:codex-app-server", route
        await service.close()


class TestModelSnapshotPropagation:
    async def test_snapshot_flows_to_work_order(self, tmp_path: Path,
                                                monkeypatch) -> None:
        """task_submit 的 model_snapshot 落到 WorkOrder inputs
        （Worker 用同一模型，不是默认推断）。"""
        import stat

        service, mission = await _setup(tmp_path)
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
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_task_submit",
                mission=mission.mission_id,
                idem="idem_p04_snap",
                arguments={
                    "goal": "分析",
                    "model_snapshot": {
                        "provider": "kimi", "model": "k3", "thinking": "high",
                    },
                },
            )
        )
        assert result.ok, result.summary
        for _ in range(200):
            orders = service._worker_manager.orders_for_mission(mission.mission_id)
            if orders:
                break
            await asyncio.sleep(0.05)
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        assert orders, "未创建 WorkOrder"
        snapshot = orders[0].inputs.get("model_snapshot") or {}
        assert snapshot.get("provider") == "kimi"
        assert snapshot.get("model") == "k3"
        assert snapshot.get("thinking") == "high"
        for _ in range(200):
            if orders[0].status in ("ACCEPTED", "FAILED", "CANCELLED"):
                break
            await asyncio.sleep(0.05)
            orders = service._worker_manager.orders_for_mission(mission.mission_id)
        await service.close()
