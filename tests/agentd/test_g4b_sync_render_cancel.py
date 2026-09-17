"""G-4b（0916 三审 B-2b）：同步渲染取消传播。

G09 三轮实证：模型对长渲染任务走**同步** scene_render（不是
process_start）——渲染子进程不在 operation 账本（180s 零注册），
G-4 的 operation 级联对它无感：取消后渲染孤儿跑完全程。

闭环断言：
1. kill_active_renders 灭注册渲染组（不碰无关进程/已退出只清账）；
2. NL-stop（"别做了"）级联报告 renders_killed 且渲染子进程真实死亡；
3. pi.session.interrupt 同样终止渲染子进程；
4. 渲染子进程 start_new_session（killpg 不误伤 agentd 自身组）。
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from rosclaw.agentd import sim_render


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen(
        ["sh", "-c", "sleep 300"], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TestRenderProcRegistry:
    def test_kill_active_renders_kills_tracked_group(self) -> None:
        proc = _sleeper()
        sim_render._track_render_proc(proc)
        killed = sim_render.kill_active_renders()
        assert killed == 1
        proc.wait(timeout=5)  # 回收（不 reap 的僵尸 kill(pid,0) 也算"活"）
        assert proc.returncode is not None, "注册渲染组必须被 killpg 终止"

    def test_untracked_process_untouched(self) -> None:
        tracked = _sleeper()
        bystander = _sleeper()
        sim_render._track_render_proc(tracked)
        try:
            killed = sim_render.kill_active_renders()
            assert killed == 1
            time.sleep(0.2)
            assert _alive(bystander.pid), "未注册进程不得被误伤"
        finally:
            bystander.kill()

    def test_exited_proc_only_cleared(self) -> None:
        proc = subprocess.Popen(["true"], start_new_session=True)
        proc.wait()
        sim_render._track_render_proc(proc)
        assert sim_render.kill_active_renders() == 0

    def test_render_child_owns_new_session(self) -> None:
        """渲染子进程必须独立 pgid——否则 killpg 会灭 agentd 自身组。"""
        import inspect

        src = inspect.getsource(sim_render._render_attempt)
        assert "start_new_session=True" in src, (
            "_render_attempt 必须 start_new_session——共享 pgid 时 "
            "killpg 会杀掉 agentd 自己"
        )


class TestCancelCascadeKillsRenders:
    async def test_nl_stop_kills_render_procs(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        proc = _sleeper()
        sim_render._track_render_proc(proc)
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.input.persist",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "session_ref": "pi_1",
                "message_id": "msg_g4b",
                "text": "别做了",
            },
        )
        disposition = result["turn_disposition"]
        assert disposition["suppress_model_turn"] is True
        report = disposition["cancel_report"]
        assert report["renders_killed"] == 1
        proc.wait(timeout=5)
        assert proc.returncode is not None

    async def test_session_interrupt_kills_render_procs(
        self, tmp_path: Path,
    ) -> None:
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        proc = _sleeper()
        sim_render._track_render_proc(proc)
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.session.interrupt",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "session_ref": "pi_1",
            },
        )
        assert result["renders_killed"] == 1
        proc.wait(timeout=5)
        assert proc.returncode is not None


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    sim_render._ACTIVE_RENDER_PROCS.clear()
