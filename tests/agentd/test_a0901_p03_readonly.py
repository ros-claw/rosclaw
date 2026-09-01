"""0901 体验探讨 P0-3 红测试：只读任务/交付物能力。

0901 实证：用户问"你这是啥？"——模型调 task.list_artifacts /
artifact.open 撞 EFFECT_UNRESOLVABLE（猜名字），降级 Shell 被
bwrap 拒，最后把任务重跑一遍制造第二套 artifact。Native Agent
不认识确定性链刚做的事。

闭环断言：
1. pi.artifact.list（task_id 或缺省=最新任务）返回带绝对路径的
   ArtifactRef 视图——解释/交付不用再跑仿真；
2. pi.artifact.resolve（artifact_id）返回 path/kind/size/digest；
3. 模型面注册只读工具：rosclaw_task_inspect / rosclaw_artifact_list
   / rosclaw_artifact_resolve 在 MODEL_TOOL_NAMES；
4. Resolver 漂移修复：EFFECT_UNRESOLVABLE/CAPABILITY_UNKNOWN 撞
   task.*/artifact.* 名字时，恢复提示指向真实只读工具。
"""

from __future__ import annotations

from pathlib import Path

import pytest


async def _setup_with_artifact(tmp_path: Path):
    """真实生产链跑一个任务（有 artifact 可查）。"""
    from rosclaw.agentd.task_execution import TaskExecutionService
    from tests.agentd.test_pi_tool_bridge import _setup

    service, mission = await _setup(tmp_path)
    kernel = service._task_kernel
    kernel.persist_input(
        mission_id=mission.mission_id, session_ref="s1",
        message_id="msg_1", text="画一个五角星",
    )
    bound = kernel.ensure_task_for_effect(
        mission_id=mission.mission_id, session_ref="s1",
        backend_native_id="s1", cwd=str(tmp_path), body_id="sim/ur5e",
        mode="SIMULATION",
    )
    task_id = str(bound["task_id"])
    kernel.note_tool_use(task_id, "rosclaw_task")
    TaskExecutionService(
        kernel=kernel, conn=kernel._conn, home=tmp_path,
    ).execute(
        task_id,
        recipe_inputs={"shape": "star5",
                       "center_m": [0.35, 0.25, 0.30], "scale_m": 0.10},
    )
    return service, mission, task_id


class TestArtifactBridgeReadOnly:
    async def test_artifact_list_latest_task(self, tmp_path: Path) -> None:
        """pi.artifact.list 缺省=最新任务——返回带绝对路径的视图。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service, mission, task_id = await _setup_with_artifact(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "b.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.artifact.list",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        assert result.get("ok"), result
        artifacts = result.get("artifacts") or []
        assert artifacts, "缺 artifact 列表"
        for a in artifacts:
            assert a.get("path"), a
            assert a.get("artifact_id"), a
        await service.close()

    async def test_artifact_resolve_by_id(self, tmp_path: Path) -> None:
        """pi.artifact.resolve(artifact_id) → path/kind/size/digest。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service, mission, task_id = await _setup_with_artifact(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "b.sock")
        listed = await bridge._dispatch(
            "user:local:1000", 1, "pi.artifact.list",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        artifact_id = str(listed["artifacts"][0]["artifact_id"])
        resolved = await bridge._dispatch(
            "user:local:1000", 2, "pi.artifact.resolve",
            {"token": service.control_token,
             "mission_id": mission.mission_id, "artifact_id": artifact_id},
        )
        assert resolved.get("ok"), resolved
        view = resolved.get("artifact") or {}
        assert view.get("path"), view
        assert view.get("kind"), view
        assert int(view.get("size_bytes", 0)) > 0, view
        assert str(view.get("digest", "")).startswith("sha256:"), view
        await service.close()


class TestModelSurfaceReadOnlyTools:
    def test_readonly_tools_in_model_surface(self) -> None:
        """只读工具进模型面——解释追问不再靠猜名字。"""
        surface = (
            Path(__file__).resolve().parents[2]
            / "packages/rosclaw-agent/src/tools/surface.ts"
        ).read_text(encoding="utf-8")
        for name in (
            "rosclaw_task_inspect",
            "rosclaw_artifact_list",
            "rosclaw_artifact_resolve",
        ):
            assert f'"{name}"' in surface, f"{name} 不在模型面"


class TestResolverDriftHint:
    def test_unresolvable_task_artifact_names_hint_readonly(self) -> None:
        """task.list_artifacts / artifact.open 这类漂移名字 →
        恢复提示指向真实只读工具（不是泛泛"查注册表"）。"""
        from rosclaw.agentd.tooling.recovery import recovery_hint

        for name in ("task.list_artifacts", "artifact.open", "artifact.show"):
            hint = recovery_hint("EFFECT_UNRESOLVABLE", context=name)
            assert "rosclaw_task_inspect" in hint or "rosclaw_artifact" in hint, (
                f"{name} 的恢复提示未指向只读工具：{hint}"
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
