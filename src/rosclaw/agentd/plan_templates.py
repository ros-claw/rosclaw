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


def build_draw_path_graph(
    task_id: str, revision: int, *, include_scene: bool = False
) -> PlanGraphV1:
    """draw_path 标准 DAG（参数经 handler 闭包传递，不在图里写死）。

    include_scene（R0-3）：spec 要求 scene_video 时在 2D 渲染后
    插入场景渲染节点（scene_3d kind——与 preview_2d 硬边界）。
    """
    nodes = [
        PlanNodeV1(id="resolve_robot", op="resource.resolve", outputs=["ResourceRef"]),
        PlanNodeV1(
            id="make_path", op="geometry.plan_path", inputs=["ResourceRef"], outputs=["PlanRef"]
        ),
        PlanNodeV1(
            id="simulate", op="robot.execute_plan", inputs=["PlanRef"], outputs=["TraceRef"]
        ),
        PlanNodeV1(id="render", op="simulation.render", inputs=["TraceRef"], outputs=["RenderRef"]),
    ]
    verify_inputs = ["PlanRef", "TraceRef", "RenderRef"]
    if include_scene:
        nodes.append(
            PlanNodeV1(
                id="render_scene",
                op="simulation.render",
                inputs=["TraceRef"],
                outputs=["SceneRef"],
            )
        )
        verify_inputs.append("SceneRef")
    nodes.append(
        PlanNodeV1(
            id="verify",
            op="task.verify",
            inputs=verify_inputs,
            outputs=["VerificationRef"],
        )
    )
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
    # R0-5：尺度/平台自适应阈值（不是 50mm 平阈值——实测下限为
    # 地板，scale*5% 为目标，3mm 绝对地板）。
    from rosclaw.task_kernel.embodied_verifier import (
        contact_failures,
        scene_media_failures,
        tool_axis_failures,
        tracking_acceptance,
    )

    threshold = float(
        acceptance.get("max_tracking_error_m")
        or tracking_acceptance(scale_m)
    )
    min_frames = int(acceptance.get("animation_min_frames", 30))

    service = SimTrajectoryService(home, runtime_manager=runtime_manager)
    task = kernel.get_task(task_id) or {}
    revision = int(task.get("active_revision") or 1)
    body_id = str(task.get("body_id") or "sim/ur5e")
    artifact_ids: list[str] = []
    # R0-3：场景语义来自 frozen spec（world/tool/deliverables）——
    # spec 要求 scene_video 才跑场景渲染。
    spec = kernel.get_task_spec(task_id) or {}
    subjects = spec.get("subjects") or {}
    scene_world = str(subjects.get("world_ref", "")).removeprefix("world:")
    scene_tool = str(subjects.get("tool_ref", ""))
    scene_required = any(
        d.get("kind") == "scene_video" and d.get("required")
        for d in (spec.get("deliverables") or [])
    )
    # R2-3（0902 §4.3）：需求驱动 overlay——任务要求"显示实际轨迹"
    # 时场景渲染必须带 actual_eef_trace overlay（RenderSpec 链），
    # 且场景渲染因此成为必需（overlay 画在场景视频上）。
    requirements = spec.get("requirements") or []
    overlay_trace_required = any(
        str(r.get("verifier") or "") == "receipt.overlays.actual_eef_trace"
        and str(r.get("level") or "") == "must"
        for r in requirements
        if isinstance(r, dict)
    )
    if overlay_trace_required:
        scene_required = True

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
            if media_type == "application/json":
                # R0-5：证据等级元数据（拆分——不用单个
                # SIM_DYN_ROLLOUT 覆盖全部）。轨迹数据产物承载
                # 几何规划/运动学跟踪/动力学推演证据；接触仿真
                # 在有接触段样本时成立。
                levels = [
                    "GEOMETRY_PLAN",
                    "KINEMATIC_TRACKING",
                    "DYNAMIC_ROLLOUT",
                ]
                if int((trace.get("tracking") or {}).get(
                    "contact_samples", 0
                ) or 0) > 0:
                    levels.append("CONTACT_SIMULATION")
                meta["evidence"] = {"levels": levels}
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

    def h_scene(inputs_: dict) -> dict:
        """R0-3：场景渲染节点（spec 要求 scene_video 时入图）。

        world/tool 来自 frozen spec；失败不炸图——SceneRef 记
        FAILED + 稳定错误码，verify 节点产出 PARTIAL（2D 交付不
        被场景故障拖死）。
        """
        from rosclaw.agentd.sim_render import render_scene_trace

        trace = inputs_["TraceRef"]
        trace_id = trace["trace_id"]
        try:
            if overlay_trace_required:
                # R2-3：RenderSpec 驱动——overlay 证据绑定到本次
                # trace（render_from_spec 父进程校验，旧证据冒充
                # 会被 RENDER_EVIDENCE_MISMATCH 拒）。
                from rosclaw.agentd.sim_render import render_from_spec
                from rosclaw.contracts.agent.render_spec import RenderSpecV1

                result = render_from_spec(
                    home,
                    RenderSpecV1(
                        body_ref=f"robot:{body_id}",
                        world_ref=f"world:{scene_world or 'empty'}",
                        overlays=[{
                            "kind": "actual_eef_trace",
                            "source_ref": f"trace:{trace_id}",
                        }],
                        outputs=["gif", "mp4"],
                    ),
                    trace_id,
                )
            else:
                result = render_scene_trace(
                    home, trace_id,
                    world_id=scene_world or "empty",
                    tool_ref=scene_tool,
                )
        except ValueError as exc:
            return {
                "SceneRef": {
                    "status": "FAILED",
                    "failure": str(exc)[:300],
                    "trace_id": trace_id,
                }
            }
        receipt_path = (
            home / "sim" / "traces" / trace_id / "render_receipt.json"
        )
        for key, media_type in (("gif", "image/gif"), ("mp4", "video/mp4")):
            item = (result.get("artifacts") or {}).get(key) or {}
            path = str(item.get("path") or "")
            if not path:
                continue
            try:
                record = kernel.register_artifact(
                    task_id=task_id,
                    path=path,
                    media_type=media_type,
                    producer="kernel:plan_template:draw_path",
                    metadata={
                        "resource": trace["resource"],
                        "lineage": {
                            "trace_id": trace_id,
                            "kind": "scene_3d",
                            "render_receipt_path": str(receipt_path),
                        },
                    },
                )
            except ValueError as exc:
                return {
                    "SceneRef": {
                        "status": "FAILED",
                        "failure": f"SCENE_ARTIFACT_REGISTER: {exc}"[:300],
                        "trace_id": trace_id,
                    }
                }
            artifact_ids.append(str(record["artifact_id"]))
        return {
            "SceneRef": {
                "status": "OK",
                "trace_id": trace_id,
                "receipt": result.get("receipt") or {},
                "frames": int((result.get("artifact") or {}).get("frames", 0)),
                "gif_path": str(
                    ((result.get("artifacts") or {}).get("gif") or {}).get("path", "")
                ),
                "mp4_path": str(
                    ((result.get("artifacts") or {}).get("mp4") or {}).get("path", "")
                ),
            }
        }

    def h_verify(inputs_: dict) -> dict:
        trace = inputs_["TraceRef"]
        render = inputs_["RenderRef"]
        scene = inputs_.get("SceneRef") or {}
        scene_failure = (
            str(scene.get("failure", ""))
            if scene.get("status") == "FAILED"
            else ""
        )
        verify = service.verify_tracking(
            trace["trace_id"], max_tracking_error_m=threshold
        )
        metrics = verify["metrics"]
        failures: list[str] = []
        # P0-5（0827 审计）：误差分级——≥90% 阈值占用是
        # PASS_NEAR_LIMIT（诚实"接近阈值"），不是普通 PASS。
        from rosclaw.task_kernel.embodied_verifier import tracking_grade

        grade = tracking_grade(float(metrics["max_error_m"]), threshold)
        if grade == "FAIL":
            failures.append(
                f"跟踪误差 {metrics['max_error_m']}m > {threshold}m"
            )
        if int(render["frames"]) < min_frames:
            failures.append(f"动画帧数 {render['frames']} < {min_frames}")
        if not trace["is_safe"]:
            failures.append(f"rollout 不安全: {trace['violations']}")
        # R0-5：接触证据 + 工具轴对齐（spec 声明驱动——不查没有
        # 声明的，但也不放过声明了没证据的）。
        constraints = spec.get("constraints") or {}
        failures += contact_failures(constraints, metrics)
        axis_limit = constraints.get(
            "tool_axis_aligned_with_plane_normal_deg"
        )
        failures += tool_axis_failures(
            metrics,
            limit_deg=float(axis_limit) if axis_limit is not None else 3.0,
        )
        if failures:
            # recipe 级验收失败 → 不调 finish_task（不制造
            # "账本 SUCCEEDED、交付 FAILED"的双真相）；任务保持
            # ACTIVE，failure 即 RepairDirective 原料。
            return {
                "VerificationRef": {
                    "status": "FAIL",
                    "verification_id": "",
                    "failures": failures + ([scene_failure] if scene_failure else []),
                    "metrics": metrics,
                    "threshold_m": threshold,
                    "min_frames": min_frames,
                }
            }
        # R0-5：场景媒体可解码/帧数/分辨率核验（文件存在≠可交付
        # ——MP4 损坏注入不得完整 PASS）。
        media_failures: list[str] = []
        scene_mp4 = str(scene.get("mp4_path", ""))
        if scene_mp4:
            scene_deliverable = next(
                (d for d in (spec.get("deliverables") or [])
                 if d.get("kind") == "scene_video"),
                {},
            )
            media_failures = scene_media_failures(
                scene_mp4,
                min_frames=int(scene_deliverable.get("min_frames", 0) or 0),
                min_resolution=list(
                    scene_deliverable.get("min_resolution") or []
                ),
            )
        # 验收真跑决定终态（frozen acceptance + R0-2 required
        # deliverables——场景缺失即 DELIVERABLE_MISSING，不整体 PASS）。
        verdict = kernel.finish_task(
            task_id=task_id,
            summary=f"draw_path recipe 执行（{len(artifact_ids)} 项产物）",
            artifact_ids=artifact_ids,
            grade=grade,
            tracking_max_error_m=float(metrics["max_error_m"]),
        )
        kernel_failures = [str(f) for f in verdict.get("failures", [])]
        status = (
            "PASS"
            if verdict.get("status") == "SUCCEEDED"
            and not scene_failure
            and not media_failures
            else "FAIL"
        )
        return {
            "VerificationRef": {
                "status": status,
                "verification_id": str(verdict.get("verification_id", "")),
                "failures": kernel_failures + (
                    [scene_failure] if scene_failure else []
                ) + media_failures,
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
    graph = build_draw_path_graph(task_id, revision, include_scene=scene_required)
    if scene_required:
        # render_scene 与 render 同 op——executor 按节点 id 优先
        # 分派（同 op 多实例的合法机制）。
        handlers["render_scene"] = h_scene
    return PlanExecutor(kernel, conn).run(graph, handlers)


class RecipeInputError(RuntimeError):
    """recipe 前置条件失败（runtime 未就绪等）——带稳定错误码，
    由 TaskExecutionService 原样透传（不包成裸异常）。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


__all__ = ["RecipeInputError", "build_draw_path_graph", "draw_path_recipe"]
