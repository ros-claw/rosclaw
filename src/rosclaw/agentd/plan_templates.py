"""draw_path recipe（P1-C2 模板 → R0-1 生产 recipe handler）。

确定性任务级入口：画路径任务（五角星/圆/任意形状）不再由模型逐
工具编排——内核按模板执行同一 typed DAG：

    resolve_robot → make_path → simulate → render → verify

**生产接线纪律（0826 体验审计 §5.R0-1）**：本模块的 handler 只经
TaskExecutionService（唯一生产入口）到达——测试也不得绕过
service 直调（绕过 = 证明了一条用户永远到不了的路径）。

Fast Path（模型/调用方单 capability 直出 TraceRef/RenderRef）与
本 recipe 共用 PlanExecutionResult 形状与 TaskOutcomeV2——两条
路径的终态语义一致。模板是**通用的**：形状/中心/缩放/平面都是
参数，无五角星特例。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

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


def draw_path_recipe(
    *,
    kernel: TaskKernel,
    conn: sqlite3.Connection,
    home: Path,
    task_id: str,
    inputs: dict[str, Any],
    runtime_manager=None,
) -> PlanExecutionResult:
    """draw_path recipe handler（真实 sim 链——无模型、确定性、
    可审计）。只经 TaskExecutionService 调用。

    inputs（recipe 参数，全部可选、有确定性缺省）：
    shape / center_m / scale_m（radius_m 兼容回落）/ plane /
    max_segment_m / acceptance{max_tracking_error_m,
    animation_min_frames}。
    """
    from rosclaw.agentd.runtime_manager import RuntimeNotReadyError
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService
    from rosclaw.cognition.alias import canonical_resource_id

    if runtime_manager is not None:
        # SIM 动力学由托管 runtime 预置——不可用时诚实失败（不是
        # 渲染时裸 ModuleNotFoundError）。
        try:
            runtime_manager.ensure("rosclaw-simulation")
        except RuntimeNotReadyError as exc:
            raise RecipeInputError("RUNTIME_NOT_READY", str(exc)) from exc

    shape = str(inputs.get("shape", "star5"))
    center_m = list(inputs.get("center_m") or [0.35, 0.25, 0.30])
    scale_m = float(inputs.get("scale_m", inputs.get("radius_m", 0.10)))
    plane = str(inputs.get("plane", "xy"))
    max_segment_m = float(inputs.get("max_segment_m", 0.03))
    acceptance = inputs.get("acceptance") or {}
    threshold = float(acceptance.get("max_tracking_error_m", 0.05))
    min_frames = int(acceptance.get("animation_min_frames", 30))

    service = SimTrajectoryService(home, runtime_manager=runtime_manager)
    task = kernel.get_task(task_id) or {}
    revision = int(task.get("active_revision") or 1)
    body_id = str(task.get("body_id") or "sim/ur5e")
    artifact_ids: list[str] = []

    def h_resolve(inputs_: dict) -> dict:
        return {"ResourceRef": {"body_ref": canonical_resource_id(body_id)}}

    def h_make_path(inputs_: dict) -> dict:
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

    def h_simulate(inputs_: dict) -> dict:
        rollout = service.simulate_cartesian_trajectory(
            inputs_["PlanRef"]["plan_id"]
        )
        return {
            "TraceRef": {
                "trace_id": rollout["trace_id"],
                "evidence_level": rollout["evidence_level"],
                "tracking": rollout["tracking"],
                "is_safe": bool(rollout.get("is_safe")),
                "violations": rollout.get("violations") or [],
                "resource": rollout.get("resource") or {},
                "artifacts": {
                    k: str(v)
                    for k, v in (rollout.get("artifacts") or {}).items()
                },
            }
        }

    def h_render(inputs_: dict) -> dict:
        trace = inputs_["TraceRef"]
        rendered = service.render_trace(trace["trace_id"])
        gif = rendered["artifact"]
        mp4 = rendered["mp4_artifact"]
        # 产物登记进 kernel 账本（受信管道 producer=kernel:*——
        # 资源证明 N4.1 + preview 血缘 WP-4：2D 预演是 COMMAND_REPLAY
        # 可视化，血缘到 trace，不当场景渲染证据）。
        for artifact, media_type in (
            (gif, "image/gif"),
            (mp4, "video/mp4"),
            ({"path": trace["artifacts"].get("trace_json", "")},
             "application/json"),
        ):
            path = str(artifact.get("path") or "")
            if not path:
                continue
            meta: dict[str, Any] = {"resource": trace["resource"]}
            if media_type.startswith(("image/", "video/")):
                # gif 与 mp4 都是 2D 轨迹预览（COMMAND_REPLAY 可视
                # 化）——preview_2d kind；场景视频（scene_3d）是
                # 另一条渲染链（R0-3），kind 分野是硬边界。
                meta["lineage"] = {
                    "trace_id": trace["trace_id"],
                    "kind": "preview_2d",
                }
            try:
                record = kernel.register_artifact(
                    task_id=task_id,
                    path=path,
                    media_type=media_type,
                    producer="kernel:plan_template:draw_path",
                    metadata=meta,
                )
            except ValueError:
                continue  # 文件缺失由验收 failures 表达——不阻断执行
            artifact_ids.append(str(record["artifact_id"]))
        return {
            "RenderRef": {
                "gif_path": gif["path"],
                "mp4_path": mp4["path"],
                "frames": gif["frames"],
                "artifact_ids": list(artifact_ids),
            }
        }

    def h_verify(inputs_: dict) -> dict:
        trace = inputs_["TraceRef"]
        render = inputs_["RenderRef"]
        verify = service.verify_tracking(
            trace["trace_id"], max_tracking_error_m=threshold
        )
        metrics = verify["metrics"]
        failures: list[str] = []
        if verify["verdict"] != "PASS":
            failures.append(
                f"跟踪误差 {metrics['max_error_m']}m > {threshold}m"
            )
        if int(render["frames"]) < min_frames:
            failures.append(f"动画帧数 {render['frames']} < {min_frames}")
        if not trace["is_safe"]:
            failures.append(f"rollout 不安全: {trace['violations']}")
        if failures:
            # recipe 级验收失败 → 不调 finish_task（不制造
            # "账本 SUCCEEDED、交付 FAILED"的双真相）；任务保持
            # ACTIVE，failure 即 RepairDirective 原料。
            return {
                "VerificationRef": {
                    "status": "FAIL",
                    "verification_id": "",
                    "failures": failures,
                    "metrics": metrics,
                    "threshold_m": threshold,
                    "min_frames": min_frames,
                }
            }
        # 验收真跑决定终态（frozen acceptance——模型自述不算数）。
        verdict = kernel.finish_task(
            task_id=task_id,
            summary=f"draw_path recipe 执行（{len(artifact_ids)} 项产物）",
            artifact_ids=artifact_ids,
        )
        kernel_failures = [str(f) for f in verdict.get("failures", [])]
        status = "PASS" if verdict.get("status") == "SUCCEEDED" else "FAIL"
        return {
            "VerificationRef": {
                "status": status,
                "verification_id": str(verdict.get("verification_id", "")),
                "failures": kernel_failures,
                "metrics": metrics,
                "threshold_m": threshold,
                "min_frames": min_frames,
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


class RecipeInputError(RuntimeError):
    """recipe 前置条件失败（runtime 未就绪等）——带稳定错误码，
    由 TaskExecutionService 原样透传（不包成裸异常）。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


__all__ = ["RecipeInputError", "build_draw_path_graph", "draw_path_recipe"]
