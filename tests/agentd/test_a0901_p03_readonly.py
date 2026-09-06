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
    """能力链跑一个任务（有 artifact 可查）。大道至简 R0-2b：
    通用能力链 plan→simulate（capability 产物自动登记），不再走
    recipe。"""
    import json as _json

    from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
    from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup

    service, mission = await _setup(tmp_path)
    lease = await _issue_lease(service, mission)
    dispatcher = PiToolDispatcher(service)
    plan = await dispatcher.execute(
        caller_pid=1, caller_uid=1000,
        request=_request(
            "rosclaw_compute", mission=mission.mission_id,
            idem="p03_plan", lease=lease,
            arguments={
                "capability_id": "trajectory_generate_planar_path",
                "arguments": {"shape": "star5", "center_m": [0.35, 0.25, 0.30],
                              "scale_m": 0.10, "plane": "xy",
                              "max_segment_m": 0.02},
            },
        ),
    )
    assert plan.ok, plan.summary
    plan_id = _json.loads(plan.summary)["value"]["plan_id"]
    sim = await dispatcher.execute(
        caller_pid=1, caller_uid=1000,
        request=_request(
            "rosclaw_compute", mission=mission.mission_id,
            idem="p03_sim", lease=lease,
            arguments={
                "capability_id": "ur5e_simulate_cartesian_trajectory",
                "arguments": {"plan_id": plan_id},
            },
        ),
    )
    assert sim.ok, sim.summary
    row = service._store.connection.execute(
        "SELECT task_id FROM artifacts ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    assert row, "能力链无产物登记"
    return service, mission, str(row["task_id"])



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
