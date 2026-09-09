"""W05 红测试（规格 2026-09-08 §9.2）：追加交付生命周期。

- 首个交付完成（SUCCEEDED）后能补图/补产物——追加注册在既有
  revision，不改回 RUNNING、不靠动机输入复活任务；
- 迟到工具请求不能因为"允许修改"而自动激活新 revision
  （未附着的新输入不被迟到请求绑定）；
- 交付优先路径保留（无任务史时 deliver 即 admission）；
- 模型不能通过填写 task_id 挂到任意任务（请求合约无 task_id
  字段——服务端按 mission+session 解析）。
"""

from __future__ import annotations

import pytest

from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup


async def _succeeded_task(service, mission, tmp_path, text="画一个五角星"):
    kernel = service._task_kernel
    f = tmp_path / "a.gif"
    f.write_bytes(b"GIF89a" + b"\x00" * 128)
    bound = kernel.bind_message(
        mission_id=mission.mission_id, session_ref="pi_1",
        backend_native_id="pi_1", message_id="msg_w05_1",
        text=text, cwd=str(tmp_path), body_id="",
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
    return task_id, bound


class TestAppendAfterTerminal:
    async def test_append_after_succeeded_wire_path(self, tmp_path) -> None:
        """0907 实证场景走完整 wire 路径：SUCCEEDED 后追加交付
        成功登记，任务审计历史保持终态。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        service, mission = await _setup(tmp_path)
        kernel = service._task_kernel
        task_id, bound = await _succeeded_task(service, mission, tmp_path)
        revision_before = int(bound["revision"])
        f2 = tmp_path / "appendix.txt"
        f2.write_text("补充分析", encoding="utf-8")
        result = await PiToolDispatcher(service).execute(
            _request(
                "rosclaw_artifact_register", mission=mission.mission_id,
                idem="w05_append_1",
                lease=await _issue_lease(service, mission),
                arguments={"path": str(f2)},
            )
        )
        assert result.ok, result.summary
        task = kernel.get_task(task_id)
        assert str(task["state"]) == "SUCCEEDED", (
            "追加交付不得复活任务状态（审计历史保持终态）"
        )
        assert int(task["active_revision"]) == revision_before, (
            "追加交付不得 bump revision"
        )
        await service.close()

    async def test_append_marked_post_terminal(self, tmp_path) -> None:
        """追加产物元数据可审计：标记 appended_post_terminal 与
        登记时任务状态——不冒充验收期交付。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        service, mission = await _setup(tmp_path)
        kernel = service._task_kernel
        task_id, _ = await _succeeded_task(service, mission, tmp_path)
        f2 = tmp_path / "b.txt"
        f2.write_text("appendix", encoding="utf-8")
        result = await PiToolDispatcher(service).execute(
            _request(
                "rosclaw_artifact_register", mission=mission.mission_id,
                idem="w05_append_2",
                lease=await _issue_lease(service, mission),
                arguments={"path": str(f2)},
            )
        )
        assert result.ok, result.summary
        rows = kernel._conn.execute(
            "SELECT metadata_json FROM artifacts WHERE task_id = ? "
            "ORDER BY rowid DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert rows is not None
        import json

        meta = json.loads(rows["metadata_json"] or "{}")
        assert meta.get("appended_post_terminal") is True, meta
        assert meta.get("task_state_at_registration") == "SUCCEEDED"
        await service.close()

    async def test_late_request_does_not_auto_activate_revision(
        self, tmp_path
    ) -> None:
        """迟到请求不得自动激活新 revision：SUCCEEDED 后到达的
        新用户输入（未附着）保持未附着——新 revision 由该输入
        自己的回合创建，不由迟到的工具请求绑定。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        service, mission = await _setup(tmp_path)
        kernel = service._task_kernel
        task_id, _ = await _succeeded_task(service, mission, tmp_path)
        # 新用户输入已持久化但尚未附着（下一回合才绑定）。
        kernel.persist_input(
            mission_id=mission.mission_id, session_ref="pi_1",
            message_id="msg_w05_new", text="再补一张俯视图",
        )
        f2 = tmp_path / "late.txt"
        f2.write_text("late appendix", encoding="utf-8")
        result = await PiToolDispatcher(service).execute(
            _request(
                "rosclaw_artifact_register", mission=mission.mission_id,
                idem="w05_late",
                lease=await _issue_lease(service, mission),
                arguments={"path": str(f2)},
            )
        )
        assert result.ok, result.summary
        row = kernel._conn.execute(
            "SELECT task_id FROM user_inputs WHERE message_id = ?",
            ("msg_w05_new",),
        ).fetchone()
        assert row is not None and row["task_id"] is None, (
            "迟到工具请求把未附着新输入绑定成了新 revision"
        )
        assert str(kernel.get_task(task_id)["state"]) == "SUCCEEDED"
        await service.close()

    async def test_deliver_first_still_admits(self, tmp_path) -> None:
        """P0-C 金丝雀保留：无任务史 + 未附着输入时 deliver 即
        admission（首个 effectful call 建任务）——不被 W05 改没。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        service, mission = await _setup(tmp_path)
        kernel = service._task_kernel
        f = tmp_path / "first.txt"
        f.write_text("first delivery", encoding="utf-8")
        result = await PiToolDispatcher(service).execute(
            _request(
                "rosclaw_artifact_register", mission=mission.mission_id,
                idem="w05_first",
                lease=await _issue_lease(service, mission),
                arguments={"path": str(f)},
            )
        )
        assert result.ok, result.summary
        # mock gateway 下 coordinator 会随即验收终态——断言任务已
        # 创建（admission 生效），不要求仍活跃。
        task = kernel.latest_task_for(mission.mission_id, "pi_1")
        assert task is not None, "交付优先 admission 未建任务"
        await service.close()

    async def test_request_contract_has_no_task_id(self) -> None:
        """§9.2：模型不能通过填写 task_id 挂到任意任务——请求
        合约不接受 task_id 字段（任务解析只在服务端）。"""
        from rosclaw.contracts.pi.tool_request import PiToolRequestV1

        assert "task_id" not in PiToolRequestV1.model_fields


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
