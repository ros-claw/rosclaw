"""R0-1 红测试（0826 体验审计 §5.R0-1）：删除生产双链，接通
TaskSpecV2 → TaskRouter → PlanGraph → PlanExecutor。

真实事故（0826 体验旅程）：
- 新 PlanGraph（plan_templates.run_draw_path_plan）只被测试直调；
  生产 ``rosclaw_task`` 仍走 tool_dispatch._task 的旧 SIM 管线——
  P1-C2 是"组件完成、生产未接线"；
- 旧管线读到了 mp4_artifact，但返回给模型的 payload.artifacts 没
  有 MP4——内核成功、产品失败（用户只看到 GIF）。

断言：
1. 结构扫描：tool_dispatch.py 不再含旧 SIM 管线实现
   （SimTrajectoryService/generate_planar_path/simulate_/render_
   trace 直调）——draw path 在生产代码只有一条执行链；
2. TaskRouter：draw_path intent → recipe；未知 intent → None；
3. TaskExecutionService 端到端：frozen spec → recipe → PlanGraph
   node 事件落账 + typed refs + GIF/MP4 登记 + 任务终态；
4. 生产接线（dispatcher 级）：rosclaw_task 一次调用 → 调用栈经
   TaskExecutionService（plan.node 事件存在）+ 模型可见 payload
   含 GIF 和 MP4（MP4 漏出修复）；
5. 诚实拒绝：无 recipe 的 intent / 缺 frozen spec / 未知任务——
   稳定错误码，不猜不编；
6. 幂等：同一 idempotency key 重放不重复执行（plan node 事件
   不翻倍）。
"""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _issue_lease, _request

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestSingleChainStructure:
    """Gate R0-1 结构面：生产代码对 draw path 只存在一条执行链。"""

    def test_dispatcher_has_no_inline_sim_pipeline(self) -> None:
        source = (
            REPO_ROOT / "src/rosclaw/agentd/pi_bridge/tool_dispatch.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "SimTrajectoryService",
            "generate_planar_path",
            "simulate_cartesian_trajectory",
            "render_trace",
            "verify_tracking",
        ):
            assert forbidden not in source, (
                f"tool_dispatch.py 仍内联旧 SIM 管线（{forbidden}）——"
                "生产双链未删除"
            )

    def test_plan_template_not_called_outside_service(self) -> None:
        """draw_path recipe 只能经 TaskExecutionService 到达——禁止
        测试/其他生产代码绕过生产 service 直调模板。"""
        import subprocess

        hits = subprocess.run(
            ["grep", "-rn", "draw_path_recipe", "src/", "tests/"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        ).stdout
        allowed = (
            "agentd/plan_templates.py",
            "agentd/task_execution.py",
        )
        for line in hits.splitlines():
            if "test_r01_production_chain.py" in line:
                continue  # 扫描器自身的模式串不算引用
            assert any(a in line for a in allowed), (
                f"recipe handler 被生产 service 之外引用：{line}"
            )


class TestTaskRouter:
    def test_draw_path_intent_routes_to_recipe(self) -> None:
        from rosclaw.task_kernel.task_router import route_recipe

        recipe = route_recipe({"goal": {"intent": "manipulation.draw_path"}})
        assert recipe == "recipe:sim.draw_path"

    def test_unknown_intent_no_recipe(self) -> None:
        from rosclaw.task_kernel.task_router import route_recipe

        assert route_recipe({"goal": {"intent": "task.unknown"}}) is None
        assert route_recipe({"goal": {"intent": "conversation.chat"}}) is None


def _kernel(home: Path):
    import sqlite3

    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, home), conn


def _draw_task(kernel, home: Path, text: str = "画一个五角星") -> str:
    kernel.persist_input(
        mission_id="mis_1", session_ref="s1",
        message_id="msg_1", text=text,
    )
    bound = kernel.ensure_task_for_effect(
        mission_id="mis_1", session_ref="s1", backend_native_id="s1",
        cwd=str(home), body_id="sim/ur5e",
    )
    return str(bound["task_id"])


def _plan_events(kernel, task_id: str) -> list[str]:
    rows = kernel._conn.execute(
        "SELECT event_type FROM task_events WHERE task_id = ? "
        "AND event_type LIKE 'plan.node_%' ORDER BY seq",
        (task_id,),
    ).fetchall()
    return [str(r["event_type"]) for r in rows]


class TestTaskExecutionService:
    def test_execute_draw_path_end_to_end(self, tmp_path: Path) -> None:
        """唯一生产入口：frozen TaskSpecV2 → router → recipe →
        PlanGraph node 事件 + typed refs + GIF/MP4 登记 + 终态。"""
        from rosclaw.agentd.task_execution import TaskExecutionService

        kernel, conn = _kernel(tmp_path)
        task_id = _draw_task(kernel, tmp_path)
        service = TaskExecutionService(
            kernel=kernel, conn=conn, home=tmp_path,
        )
        outcome = service.execute(
            task_id,
            recipe_inputs={
                "shape": "star5", "center_m": [0.35, 0.25, 0.30],
                "scale_m": 0.10,
            },
        )
        assert outcome.ok, outcome.failure
        assert outcome.recipe_id == "recipe:sim.draw_path"
        # typed refs 真实持久化（不是"目录里有文件"）。
        for ref in ("ResourceRef", "PlanRef", "TraceRef",
                    "RenderRef", "VerificationRef"):
            assert ref in outcome.refs, f"缺 {ref}"
        # plan node 事件链完整（5 节点 started+completed）。
        events = _plan_events(kernel, task_id)
        assert events.count("plan.node_started") == 5, events
        assert events.count("plan.node_completed") == 5, events
        # GIF 和 MP4 都进产物账本。
        media = {a["media_type"] for a in outcome.artifacts}
        assert "image/gif" in media and "video/mp4" in media, media
        task = kernel.get_task(task_id)
        assert task["state"] == "SUCCEEDED", task["state"]

    def test_unknown_intent_honest_refusal(self, tmp_path: Path) -> None:
        from rosclaw.agentd.task_execution import TaskExecutionService

        kernel, conn = _kernel(tmp_path)
        task_id = _draw_task(kernel, tmp_path, text="帮我查一下天气")
        service = TaskExecutionService(
            kernel=kernel, conn=conn, home=tmp_path,
        )
        outcome = service.execute(task_id, recipe_inputs={})
        assert not outcome.ok
        assert outcome.error_code == "TASK_NO_RECIPE"

    def test_unknown_task_honest_error(self, tmp_path: Path) -> None:
        from rosclaw.agentd.task_execution import TaskExecutionService

        kernel, conn = _kernel(tmp_path)
        service = TaskExecutionService(
            kernel=kernel, conn=conn, home=tmp_path,
        )
        outcome = service.execute("task_nonexistent", recipe_inputs={})
        assert not outcome.ok
        assert outcome.error_code == "TASK_NOT_FOUND"


async def _setup_ur5e(tmp_path: Path):
    """生产级接线 harness：真实 AgentService + ur5e body 绑定。"""
    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.models.gateway import MockModelGateway
    from rosclaw.agentd.models.profiles import mock_profile
    from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore
    from rosclaw.agentd.service import AgentService
    from rosclaw.contracts.agent.model_turn import ModelTurnResultV1

    turn = ModelTurnResultV1(
        turn_id="t", provider="mock", model="m", content="ok",
        assistant_message={"role": "assistant", "content": "ok"},
        usage={"prompt_tokens": 1, "completion_tokens": 1,
               "total_tokens": 2},  # type: ignore[arg-type]
    )
    config = load_agent_config(tmp_path / "config.yaml")
    service = AgentService(
        config, tmp_path,
        gateway=MockModelGateway(mock_profile(), [turn]),
    )
    mission = service.create_mission("r01 接线测试")
    bindings = SessionBindingStore(service._store.connection)
    bindings.bind(
        pi_session_id="pi_1", pi_session_path="",
        mission_id=mission.mission_id, body_id="sim/ur5e",
        execution_mode="SIMULATION", created_by="user:local:1000",
    )
    bindings.acquire_lease(
        mission_id=mission.mission_id, pi_session_id="pi_1",
        owner_pid=1, owner_uid=1000,
    )
    service._task_kernel.persist_input(
        mission_id=mission.mission_id, session_ref="pi_1",
        message_id="msg_draw", text="画一个五角星",
    )
    return service, mission


class TestProductionWiring:
    async def test_rosclaw_task_reaches_execution_service(
        self, tmp_path: Path
    ) -> None:
        """生产接线：rosclaw_task（模型入口）→ TaskExecutionService
        → PlanGraph node 事件——不是旧 _task 内联管线。"""
        service, mission = await _setup_ur5e(tmp_path)
        lease = await _issue_lease(service, mission)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_task", mission=mission.mission_id,
                idem="r01_wiring", lease=lease,
                arguments={
                    "goal": "draw_shape",
                    "parameters": {
                        "shape": "star5",
                        "center_m": [0.35, 0.25, 0.30],
                        "scale_m": 0.10,
                    },
                },
            )
        )
        assert result.ok, result.summary
        task = service._task_kernel.latest_task_for(
            mission.mission_id, "pi_1"
        )
        assert task is not None
        events = _plan_events(service._task_kernel, str(task["task_id"]))
        assert "plan.node_started" in events, (
            f"生产路径未产生 PlanGraph node 事件（走的不是 "
            f"TaskExecutionService）：{events}"
        )
        await service.close()

    async def test_model_payload_includes_mp4(self, tmp_path: Path) -> None:
        """MP4 漏出修复：模型可见 payload.artifacts 同时含 GIF 和
        MP4（不是只在数据库里）。"""
        service, mission = await _setup_ur5e(tmp_path)
        lease = await _issue_lease(service, mission)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_task", mission=mission.mission_id,
                idem="r01_mp4", lease=lease,
                arguments={
                    "goal": "draw_shape",
                    "parameters": {"shape": "star5"},
                },
            )
        )
        assert result.ok, result.summary
        payload = json.loads(result.summary)
        artifacts = payload.get("artifacts", {})
        assert artifacts.get("mp4"), (
            f"模型可见 payload 缺 MP4——用户只能看到 GIF："
            f"{sorted(artifacts.keys())}"
        )
        assert artifacts.get("gif")
        await service.close()

    async def test_idempotent_replay_no_double_execution(
        self, tmp_path: Path
    ) -> None:
        """同一 idempotency key 重放：返回首个结果，plan node 事件
        不翻倍。"""
        service, mission = await _setup_ur5e(tmp_path)
        lease = await _issue_lease(service, mission)
        dispatcher = PiToolDispatcher(service)
        request = _request(
            "rosclaw_task", mission=mission.mission_id,
            idem="r01_idem", lease=lease,
            arguments={
                "goal": "draw_shape", "parameters": {"shape": "star5"},
            },
        )
        first = await dispatcher.execute(request)
        second = await dispatcher.execute(request)
        assert first.ok and second.ok
        task = service._task_kernel.latest_task_for(
            mission.mission_id, "pi_1"
        )
        events = _plan_events(service._task_kernel, str(task["task_id"]))
        assert events.count("plan.node_started") == 5, events
        await service.close()

    async def test_non_sim_mode_denied(self, tmp_path: Path) -> None:
        """SIM recipe 在非 SIMULATION mode 下诚实拒绝（REAL/SHADOW
        门禁不变——不静默降级执行）。"""
        from rosclaw.agentd.task_execution import TaskExecutionService

        kernel, conn = _kernel(tmp_path)
        kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_shadow", text="画一个五角星",
        )
        bound = kernel.ensure_task_for_effect(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            cwd=str(tmp_path), mode="SHADOW", body_id="sim/ur5e",
        )
        service = TaskExecutionService(
            kernel=kernel, conn=conn, home=tmp_path,
        )
        outcome = service.execute(
            str(bound["task_id"]),
            recipe_inputs={"shape": "star5"},
        )
        assert not outcome.ok
        assert outcome.error_code == "MODE_DENIED", outcome.failure
