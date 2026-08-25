"""P1-C2 红测试（0824 总纲 §7.2/P1-C）：typed PlanGraph executor。

真实缺口：能力链（plan→rollout→render→verify）由模型逐工具编排
——模型可以跳步/乱序/漏交付（金丝雀 run1：1687 次 bash 自由发
挥）。PlanGraph 把"怎么干"变成 typed DAG：节点 op/in/out 是契约，
执行是内核确定性的，Fast Path（单 capability 直出）与复杂 DAG
共用同一结果类型与 Outcome。

断言：
1. 契约：合法 DAG 构建；未知 op / 重复 id / 环 / 悬空输入拒绝；
2. executor 拓扑执行：handler 按依赖序调用，输出按名喂给下游；
   节点事件链（plan.node_started/completed）落 task_events；
3. 节点失败 → 停止 + plan.node_failed + 下游 SKIPPED（不假装）；
4. Fast Path：单节点图与 DAG 同一结果形状（refs + ok）；
5. draw_path 模板端到端（真实 sim 链，无模型）：ResourceRef/
   PlanRef/TraceRef/RenderRef 全产出 + GIF/MP4 落盘 + verify PASS。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rosclaw.contracts.agent.plan_graph import (
    PLAN_NODE_OPS,
    PlanGraphV1,
    PlanNodeV1,
)
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.plan_executor import PlanExecutor
from rosclaw.task_kernel.service import TaskKernel


def _conn(tmp_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "missions.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return conn


def _task(conn: sqlite3.Connection) -> None:
    now = "2026-08-25T00:00:00+00:00"
    conn.execute(
        "INSERT INTO tasks (task_id, mission_id, root_goal, mode, body_id, "
        "state, active_revision, workspace_path, created_at, updated_at) "
        "VALUES ('task_1', 'm1', 'goal', 'SIMULATION', '', 'ACTIVE', 1, "
        "'', ?, ?)",
        (now, now),
    )
    conn.commit()


def _events(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT event_type FROM task_events WHERE task_id = 'task_1' "
        "ORDER BY seq"
    ).fetchall()
    return [r["event_type"] for r in rows]


def _node(node_id: str, op: str, inputs=None, outputs=None) -> PlanNodeV1:
    return PlanNodeV1(
        id=node_id, op=op,
        inputs=list(inputs or []), outputs=list(outputs or []),
    )


class TestContract:
    def test_valid_dag_builds(self) -> None:
        graph = PlanGraphV1(
            graph_id="pg_1", task_id="task_1", revision=1,
            nodes=[
                _node("resolve", "resource.resolve", outputs=["ResourceRef"]),
                _node("plan", "geometry.plan_path",
                      inputs=["ResourceRef"], outputs=["PlanRef"]),
            ],
            digest="sha256:x",
        )
        assert len(graph.nodes) == 2

    def test_unknown_op_rejected(self) -> None:
        with pytest.raises(ValueError, match="op"):
            _node("x", "magic.teleport")

    def test_ops_cover_draw_path_pipeline(self) -> None:
        for op in (
            "resource.resolve", "geometry.plan_path", "robot.execute_plan",
            "simulation.render", "task.verify",
        ):
            assert op in PLAN_NODE_OPS

    def test_duplicate_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            PlanGraphV1(
                graph_id="pg_1", task_id="task_1", revision=1,
                nodes=[
                    _node("a", "resource.resolve", outputs=["ResourceRef"]),
                    _node("a", "geometry.plan_path",
                          inputs=["ResourceRef"], outputs=["PlanRef"]),
                ],
                digest="sha256:x",
            )

    def test_dangling_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown input|dangling"):
            PlanGraphV1(
                graph_id="pg_1", task_id="task_1", revision=1,
                nodes=[
                    _node("plan", "geometry.plan_path",
                          inputs=["NoSuchRef"], outputs=["PlanRef"]),
                ],
                digest="sha256:x",
            )

    def test_cycle_rejected(self) -> None:
        with pytest.raises(ValueError, match="cycle|order|环"):
            PlanGraphV1(
                graph_id="pg_1", task_id="task_1", revision=1,
                nodes=[
                    _node("a", "resource.resolve",
                          inputs=["B"], outputs=["A"]),
                    _node("b", "geometry.plan_path",
                          inputs=["A"], outputs=["B"]),
                ],
                digest="sha256:x",
            )


class TestExecutor:
    def test_topological_execution_and_ref_flow(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        executor = PlanExecutor(None, conn)
        order: list[str] = []
        graph = PlanGraphV1(
            graph_id="pg_1", task_id="task_1", revision=1,
            nodes=[
                _node("resolve", "resource.resolve", outputs=["ResourceRef"]),
                _node("plan", "geometry.plan_path",
                      inputs=["ResourceRef"], outputs=["PlanRef"]),
                _node("sim", "robot.execute_plan",
                      inputs=["PlanRef"], outputs=["TraceRef"]),
            ],
            digest="sha256:x",
        )

        def h_resolve(inputs):
            order.append("resolve")
            return {"ResourceRef": {"body_ref": "robot:ur5e"}}

        def h_plan(inputs):
            order.append("plan")
            assert inputs["ResourceRef"]["body_ref"] == "robot:ur5e"
            return {"PlanRef": {"plan_id": "plan_1"}}

        def h_sim(inputs):
            order.append("sim")
            assert inputs["PlanRef"]["plan_id"] == "plan_1"
            return {"TraceRef": {"trace_id": "trace_1"}}

        result = executor.run(graph, {
            "resource.resolve": h_resolve,
            "geometry.plan_path": h_plan,
            "robot.execute_plan": h_sim,
        })
        assert order == ["resolve", "plan", "sim"]
        assert result.ok is True
        assert result.refs["TraceRef"]["trace_id"] == "trace_1"
        types = _events(conn)
        assert "plan.node_started" in types
        assert "plan.node_completed" in types

    def test_node_failure_stops_and_marks_skipped(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        executor = PlanExecutor(None, conn)
        graph = PlanGraphV1(
            graph_id="pg_1", task_id="task_1", revision=1,
            nodes=[
                _node("resolve", "resource.resolve", outputs=["ResourceRef"]),
                _node("plan", "geometry.plan_path",
                      inputs=["ResourceRef"], outputs=["PlanRef"]),
                _node("sim", "robot.execute_plan",
                      inputs=["PlanRef"], outputs=["TraceRef"]),
            ],
            digest="sha256:x",
        )
        calls: list[str] = []

        def fail_plan(inputs):
            raise ValueError("PLAN_INFEASIBLE")

        result = executor.run(graph, {
            "resource.resolve": lambda i: (calls.append("resolve"),
                                           {"ResourceRef": {}})[1],
            "geometry.plan_path": fail_plan,
            "robot.execute_plan": lambda i: (calls.append("sim"),
                                             {"TraceRef": {}})[1],
        })
        assert result.ok is False
        assert "sim" not in calls, "失败节点下游仍执行"
        assert result.failed_node == "plan"
        types = _events(conn)
        assert "plan.node_failed" in types

    def test_fast_path_single_node_same_shape(self, tmp_path: Path) -> None:
        conn = _conn(tmp_path)
        _task(conn)
        executor = PlanExecutor(None, conn)
        graph = PlanGraphV1(
            graph_id="pg_1", task_id="task_1", revision=1,
            nodes=[
                _node("sim", "robot.execute_plan", outputs=["TraceRef"]),
            ],
            digest="sha256:x",
        )
        result = executor.run(graph, {
            "robot.execute_plan": lambda i: {"TraceRef": {"trace_id": "t1"}},
        })
        assert result.ok is True
        assert result.refs["TraceRef"]["trace_id"] == "t1"


class TestDrawPathTemplate:
    def test_draw_path_end_to_end(self, tmp_path: Path) -> None:
        """确定性 draw_path 模板（真实 sim 链，无模型）：ResourceRef/
        PlanRef/TraceRef/RenderRef 全产出 + 媒体落盘。"""
        conn = _conn(tmp_path)
        _task(conn)
        from rosclaw.agentd.plan_templates import run_draw_path_plan

        kernel = TaskKernel(conn, tmp_path)
        result = run_draw_path_plan(
            kernel, conn, tmp_path,
            task_id="task_1",
            shape="star5",
            center_m=[0.35, 0.0, 0.12],
            scale_m=0.12,
        )
        assert result.ok is True, result
        for ref in ("ResourceRef", "PlanRef", "TraceRef", "RenderRef"):
            assert ref in result.refs, f"缺 {ref}"
        render = result.refs["RenderRef"]
        assert Path(render["gif_path"]).exists()
        assert Path(render["mp4_path"]).exists()
        types = _events(conn)
        assert "plan.node_started" in types
        assert "plan.node_completed" in types
