"""十五审自审回归（修正轮）：自审发现的真实缺陷必须有测试锁定。

1. ACP session/prompt 不得有固定墙钟超时（长任务 30min+）；
2. task_cancel 后 execution 终态是 CANCELLED，绝不被驱动收尾覆盖
   成 FAILED；
3. service.close() 取消 control-plane 驱动任务（不写已关闭的 DB）。
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _setup


class TestAcpPromptNoWallTimeout:
    def test_python_driver_prompt_has_no_timeout(self) -> None:
        """session/prompt 的 _request 调用必须 timeout=None。"""
        from rosclaw.agentd import acp_driver

        source = inspect.getsource(acp_driver.AcpHarnessDriver.run)
        prompt_call = source.split('session/prompt')[1][:200]
        assert "timeout=None" in prompt_call

    def test_ts_client_prompt_has_no_timeout(self) -> None:
        """TS AcpClient.prompt 同样不得带默认 30s 超时。"""
        source = Path(
            "packages/rosclaw-agent/src/agent_runtime/acp-client.ts"
        ).read_text(encoding="utf-8")
        prompt_block = source.split("async prompt(")[1].split("async ")[0]
        assert "null" in prompt_block


class TestCancelStaysCancelled:
    async def test_cancel_not_overwritten_to_failed(self, tmp_path: Path) -> None:
        """用户 task_cancel 后，驱动收尾不得把 CANCELLED 覆盖成 FAILED。"""
        import stat

        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        fake = tmp_path / "fake-forever"
        fake.write_text(
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "while true; do echo '{\"kind\":\"liveness\"}'; sleep 0.5; done\n"
        )
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        monkeypatch_entry = ("/bin/sh", str(fake))
        pi_managed.find_pi_agent_entry = lambda: monkeypatch_entry  # type: ignore
        adapter = pi_managed.PiManagedAdapter(
            rosclaw_home=tmp_path, conn=service._store.connection
        )
        service._worker_manager._adapters["pi_managed"] = adapter
        adapter._manager_ref = service._worker_manager
        if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
            service._registry.set_status(
                "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake"
            )
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "长跑任务", "required_capabilities": [], "effects": ""},
            idem="selfcheck_cancel",
        )
        # 等 worker 起来。
        for _ in range(200):
            row = plane._get(view["execution_id"])
            if row["state"] == "RUNNING" and row.get("work_order_id"):
                break
            await asyncio.sleep(0.05)
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from tests.agentd.test_pi_tool_bridge import _request

        dispatcher = PiToolDispatcher(service)
        cancelled = await dispatcher.execute(
            _request(
                "rosclaw_task_cancel",
                mission=mission.mission_id,
                idem="selfcheck_cancel_2",
                arguments={"execution_id": view["execution_id"]},
            )
        )
        assert cancelled.ok, cancelled.summary
        # 驱动收尾后终态仍是 CANCELLED。
        for _ in range(200):
            row = plane._get(view["execution_id"])
            if row["state"] in ("FAILED", "CANCELLED"):
                break
            await asyncio.sleep(0.05)
        row = plane._get(view["execution_id"])
        assert row["state"] == "CANCELLED", row["state"]
        await service.close()

    async def test_close_cancels_plane_drivers(self, tmp_path: Path) -> None:
        """service.close() 后 control-plane 驱动任务全部收尾（不悬挂）。"""
        service, mission = await _setup(tmp_path)
        plane = service._task_control_plane
        # 提交一个 physical 域任务（驱动立即 BLOCKED 收尾，不等进程）。
        await plane.submit(
            mission.mission_id,
            {"goal": "x", "required_capabilities": [],
             "effects": "physical_real"},
            idem="selfcheck_close",
        )
        await asyncio.sleep(0.2)
        await service.close()
        for driver in plane._drivers.values():
            assert driver.done()
