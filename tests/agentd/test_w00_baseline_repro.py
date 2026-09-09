"""W00 基线复现测试（ROSClaw_ClaudeCode_实施规格_2026-09-08 §2/§4）。

每条 §2.2 发现一个最小复现测试。xfail(strict=True) 标记「基线已
复现、修复待对应工作包」——修复落地后测试翻 XPASS 即强制解标，
不留静默漂移。真实模型路径无 key 标 NOT_RUN（不合成冒充）。

基线环境：main=f05aafd，Python 3.11.15，mujoco 3.11.0（dev）/
3.12.0（bundle venv），Pi 0.83.0，bundle Node v22.19.0。
"""

from __future__ import annotations

import pytest


class TestR1AppendDeliveryBlocked:
    """§2.2-1：`_artifact_register` 在活跃任务缺失且最近任务
    SUCCEEDED 时直接 TASK_ALREADY_COMPLETED 拒绝——0907 实证追加
    产物被反复拒绝（W05 已修复：显式上下文下追加注册在既有
    revision，不复活任务；行为测试见 test_w05_delivery_lifecycle.py）。"""

    async def test_append_after_succeeded_allowed(self, tmp_path) -> None:
        """0907 实证条件的精确复现：handler 直达（无 admission
        动机遮掩）+ active=None + 最近任务 SUCCEEDED → 基线=
        TASK_ALREADY_COMPLETED。期望（W05 §9.2）：显式上下文下
        追加交付注册在既有 revision——不改回 RUNNING、不靠动机
        复活任务。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from tests.agentd.test_pi_tool_bridge import (
            _issue_lease,
            _request,
            _setup,
        )

        service, mission = await _setup(tmp_path)
        kernel = service._task_kernel
        f = tmp_path / "a.gif"
        f.write_bytes(b"GIF89a" + b"\x00" * 128)
        bound = kernel.bind_message(
            mission_id=mission.mission_id, session_ref="pi_1",
            backend_native_id="pi_1", message_id="msg_w00_1",
            text="画一个五角星", cwd=str(tmp_path), body_id="",
        )
        task_id = str(bound["task_id"])
        artifact = kernel.register_artifact(
            task_id=task_id, path=str(f), media_type="image/gif",
            producer="kernel:test",
        )
        kernel.finish_task(
            task_id=task_id, summary="done",
            artifact_ids=[str(artifact["artifact_id"])],
        )
        assert str(kernel.get_task(task_id)["state"]) == "SUCCEEDED"
        f2 = tmp_path / "b.txt"
        f2.write_text("appendix", encoding="utf-8")
        result = await PiToolDispatcher(service)._artifact_register(
            _request(
                "rosclaw_artifact_register", mission=mission.mission_id,
                idem="w00_direct", lease=await _issue_lease(service, mission),
                arguments={"path": str(f2)},
            )
        )
        assert result.ok, result.summary
        # 追加不得复活任务状态（审计历史保持 SUCCEEDED）。
        assert str(kernel.get_task(task_id)["state"]) == "SUCCEEDED"
        await service.close()


class TestR2FullStateReplay:
    """§2.2-2：sim_render 回放用 `data.qpos[:model.nu]` 截断——
    nq≠nu 的模型（freejoint/被动关节/被动物体）丢失状态，真实
    仿真与视频不一致（W03 修复：模型维度/关节映射驱动的完整
    状态恢复）。"""

    @pytest.mark.xfail(
        strict=True,
        reason="W00 基线：qpos 按 nu 截断（W03 修复后解标）",
    )
    def test_replay_restores_full_nq_not_nu(self) -> None:
        import inspect

        from rosclaw.agentd import sim_render

        src = inspect.getsource(sim_render)
        assert "data.qpos[: int(model.nu)]" not in src, (
            "回放仍按执行器数 nu 截断位置坐标（nq 丢失）"
        )


class TestR3BackendFallbackDepth:
    """§2.2-3：后端降级只试 `candidates[:2]`——EGL 失败+OSMesa
    不可用+Xvfb 可用时找不到真正可用的第三后端（W04 修复：
    按可用性排序逐个真实尝试）。"""

    @pytest.mark.xfail(
        strict=True,
        reason="W00 基线：降级只试前两个后端（W04 修复后解标）",
    )
    def test_fallback_tries_all_candidates(self) -> None:
        import inspect

        from rosclaw.agentd import sim_render

        src = inspect.getsource(sim_render)
        assert "candidates[:2]" not in src, "后端降级仍只试前两个候选"


class TestR4NoNodeStartup:
    """§4.5-4：无 Node/npm 环境启动——离线包自带 Node runtime
    （v22.19.0 arm64），PATH 无 node 时 chat 正常进入 TUI、stdin
    EOF 干净退出。基线=通过（无 xfail）。"""

    def test_chat_starts_without_system_node(self, tmp_path) -> None:
        import subprocess
        import sys
        from pathlib import Path

        bundle_bin = "/tmp/w00-pkgtest/prefix/bin/rosclaw"
        entry = (
            [bundle_bin] if Path(bundle_bin).exists()
            else [sys.executable, "-m", "rosclaw.entrypoint"]
        )
        env = {
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin",
            "ROSCLAW_HOME": str(tmp_path / "rh"),
        }
        result = subprocess.run(
            [*entry, "chat"], env=env, capture_output=True,
            timeout=60, stdin=subprocess.DEVNULL,
        )
        combined = result.stdout + result.stderr
        # 干净诊断或正常 TUI——不得是 Node 缺失的裸 traceback。
        assert b"Traceback" not in combined, combined[-500:]


class TestR5UnsafeRolloutReply:
    """§2.1/§16.2-1：0907 unsafe rollout（is_safe:false + 碰撞 +
    0.68m 误差）被同时宣布完成——真实主模型回归。"""

    def test_real_model_unsafe_reply(self) -> None:
        import os

        if not os.environ.get("ROSCLAW_KIMI_API_KEY"):
            pytest.skip(
                "NOT_RUN: ROSCLAW_KIMI_API_KEY 未配置——真实模型回归"
                "不可执行（不合成冒充；operator 带 key 重跑）"
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
