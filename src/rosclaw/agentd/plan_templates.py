"""draw_path PlanGraph 模板（P1-C2，0824 总纲 §7.2/§7.3）。

确定性任务级入口：画路径任务（五角星/圆/任意形状）不再由模型逐
工具编排——内核按模板执行同一 typed DAG：

    resolve_robot → make_path → simulate → render → verify

Fast Path（模型/调用方单 capability 直出 TraceRef/RenderRef）与
本模板共用 PlanExecutionResult 形状与 TaskOutcomeV2——两条路径的
终态语义一致。模板是**通用的**：形状/中心/缩放/平面都是参数，
无五角星特例。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from rosclaw.contracts.agent.plan_graph import PlanGraphV1, PlanNodeV1
from rosclaw.contracts.common import new_id
from rosclaw.task_kernel.plan_executor import (
    PlanExecutionResult,
    PlanExecutor,
    PlanNodeHandler,
)
from rosclaw.task_kernel.service import TaskKernel


def build_draw_path_graph(task_id: str, revision: int) -> PlanGraphV1:
    """draw_path 标准 DAG（参数经 handler 闭包传递，不在图里写死）。"""
    nodes = [
        PlanNodeV1(id="resolve_robot", op="resource.resolve", outputs=["ResourceRef"]),
        PlanNodeV1(
            id="make_path", op="geometry.plan_path", inputs=["ResourceRef"], outputs=["PlanRef"]
        ),
        PlanNodeV1(
            id="simulate", op="robot.execute_plan", inputs=["PlanRef"], outputs=["TraceRef"]
        ),
        PlanNodeV1(id="render", op="simulation.render", inputs=["TraceRef"], outputs=["RenderRef"]),
        PlanNodeV1(
            id="verify",
            op="task.verify",
            inputs=["PlanRef", "TraceRef", "RenderRef"],
            outputs=["VerificationRef"],
        ),
    ]
    digest = hashlib.sha256(
        json.dumps(
            [[n.id, n.op, n.inputs, n.outputs] for n in nodes],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return PlanGraphV1(
        graph_id=new_id("pg"),
        task_id=task_id,
        revision=revision,
        nodes=nodes,
        digest=f"sha256:{digest}",
    )


def run_draw_path_plan(
    kernel: TaskKernel,
    conn: sqlite3.Connection,
    home: Path,
    *,
    task_id: str,
    shape: str,
    center_m: list[float],
    scale_m: float,
    plane: str = "xy",
    max_segment_m: float = 0.02,
    body_id: str = "sim/ur5e",
) -> PlanExecutionResult:
    """执行 draw_path 模板（真实 sim 链——无模型、确定性、可审计）。"""
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService
    from rosclaw.cognition.alias import canonical_resource_id

    service = SimTrajectoryService(home)
    task = kernel.get_task(task_id) or {}
    revision = int(task.get("active_revision") or 1)
    artifact_ids: list[str] = []

    def h_resolve(inputs: dict) -> dict:
        return {"ResourceRef": {"body_ref": canonical_resource_id(body_id)}}

    def h_make_path(inputs: dict) -> dict:
        plan = service.generate_planar_path(
            shape=shape,
            center_m=center_m,
            scale_m=scale_m,
            plane=plane,
            max_segment_m=max_segment_m,
        )
        return {
            "PlanRef": {
                "plan_id": plan["plan_id"],
                "digest": plan["hash"],
                "point_count": plan["point_count"],
            }
        }

    def h_simulate(inputs: dict) -> dict:
        rollout = service.simulate_cartesian_trajectory(inputs["PlanRef"]["plan_id"])
        return {
            "TraceRef": {
                "trace_id": rollout["trace_id"],
                "evidence_level": rollout["evidence_level"],
                "tracking": rollout["tracking"],
            }
        }

    def h_render(inputs: dict) -> dict:
        rendered = service.render_trace(inputs["TraceRef"]["trace_id"])
        gif = rendered["artifact"]
        mp4 = rendered["mp4_artifact"]
        for artifact, media_type in ((gif, "image/gif"), (mp4, "video/mp4")):
            record = kernel.register_artifact(
                task_id=task_id,
                path=artifact["path"],
                media_type=media_type,
                producer="kernel:plan_template:draw_path",
            )
            artifact_ids.append(str(record["artifact_id"]))
        return {
            "RenderRef": {
                "gif_path": gif["path"],
                "mp4_path": mp4["path"],
                "frames": gif["frames"],
            }
        }

    def h_verify(inputs: dict) -> dict:
        verdict = kernel.finish_task(
            task_id=task_id,
            summary=f"draw_path 模板执行（{len(artifact_ids)} 项产物）",
            artifact_ids=artifact_ids,
        )
        return {
            "VerificationRef": {
                "status": verdict.get("status", ""),
                "failures": verdict.get("failures", []),
            }
        }

    handlers: dict[str, PlanNodeHandler] = {
        "resource.resolve": h_resolve,
        "geometry.plan_path": h_make_path,
        "robot.execute_plan": h_simulate,
        "simulation.render": h_render,
        "task.verify": h_verify,
    }
    graph = build_draw_path_graph(task_id, revision)
    return PlanExecutor(kernel, conn).run(graph, handlers)


__all__ = ["build_draw_path_graph", "run_draw_path_plan"]
