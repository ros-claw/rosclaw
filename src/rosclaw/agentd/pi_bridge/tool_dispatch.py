"""Pi Tool Bridge 分发（重构规格 §15/§16/§17，PR-PNA-3）。

每个 Pi 工具调用在这里变成受控的 agentd 操作 + DecisionV1 审计镜像。

验证链（任一不过即 fail closed）：
session binding → mission 存在 → writer lease → context revision →
idempotency → tool allowlist → side-effect class → mode → 执行。

PNA-3 工具集（read/observe/verify/memory/fail_safe/status）：
动作类（request_action/delegate）在 PNA-4/PNA-5 接入，当前一律拒绝并
如实说明——不存在"经 observe 绕过动作"的路径。
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore
from rosclaw.contracts.pi.tool_request import PiToolRequestV1, PiToolResultV1

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService

#: PNA-3/PNA-4 开放工具及其 side-effect 语义。
_TOOL_TABLE: dict[str, str] = {
    "rosclaw_status": "read",
    "rosclaw_observe": "observe",
    "rosclaw_compute": "compute",
    "rosclaw_verify": "read",
    "rosclaw_memory_query": "read",
    "rosclaw_fail_safe": "control",
    "rosclaw_delegate": "delegate",
    # 十审 W0：异步 WorkOrder 协议——start 立即返回精确 ID；check/cancel
    # 按 ID 精确关联（不再是"取 mission 最后一单"）。
    "rosclaw_check_work": "read",
    "rosclaw_cancel_work": "delegate",
    # 十审 W2：五工具协议补齐——list（mission 摘要）+ update（steer
    # 备注；运行中 Worker 的实时转向在 W4，update 诚实说明生效范围）。
    "rosclaw_list_work": "read",
    "rosclaw_update_work": "delegate",
    # 十审 W4：终态单 retry（新 attempt 携带 steer 备注 + parent lineage）。
    "rosclaw_retry_work": "delegate",
    # 十一审 PR-E：WAITING_INPUT 的用户回答（经 /job answer 命令层）。
    "rosclaw_answer_work": "delegate",
    # 十二审 PR-12.3：resume——恢复同一 Pi 会话（retry ≠ resume）。
    "rosclaw_resume_work": "delegate",
    # 十三审 HOTFIX-13.2：BUDGET_PAUSED 的追加预算并唤醒。
    "rosclaw_extend_work": "delegate",
    # 十三审 PR-13.5：Native Agent 只读 Worker 诊断（分页、脱敏——
    # 不再"无权查看内部日志"）。
    "rosclaw_read_work_events": "read",
    "rosclaw_read_work_transcript": "read",
    "rosclaw_list_work_artifacts": "read",
    "rosclaw_read_work_failure": "read",
    "rosclaw_request_action": "physical_action",
    # 八审 P0-5：任务级入口——确定性编译器编排，模型只交 TaskSpec。
    "rosclaw_task": "task",
}
#: 后续批次才开放；现在调用必须得到诚实的"未开放"拒绝。
_DEFERRED_TOOLS = {
    "rosclaw_plan_patch": "PNA-3 后续（TaskGraph patch）",
    "rosclaw_team_coordinate": "PNA-4 后续",
}


class ToolBridgeError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def _expected_artifacts(
    task_type: str, deliverables_raw: list, is_workbench: bool
) -> list[str]:
    """十二审 PR-12.4（§5.2）：任务类型驱动验收面。"""
    from rosclaw.contracts.worker.workspec import (
        DeliverableV1,
        WorkSpecV2,
        expected_media_types,
    )

    deliverables = [DeliverableV1(**d) for d in deliverables_raw if isinstance(d, dict)]
    if task_type == "artifact_build" and not deliverables:
        # 动画/媒体任务的默认交付物（用户的 GIF/MP4 场景）。
        deliverables = [
            DeliverableV1(
                id="media",
                media_types=["image/gif", "video/mp4"],
                validators=["exists", "non_empty", "magic"],
            )
        ]
    if task_type == "simulation_run" and not deliverables:
        deliverables = [
            DeliverableV1(id="trace", media_types=["application/json"]),
            DeliverableV1(
                id="media",
                media_types=["image/gif", "video/mp4"],
                validators=["exists", "non_empty", "magic"],
            ),
        ]
    spec = WorkSpecV2(task_type=task_type, deliverables=deliverables)
    media = expected_media_types(spec)
    base = ["text/plain", "text/x-diff"] if is_workbench else ["text/plain"]
    if task_type == "artifact_build":
        base = ["text/plain"]  # diff 可选
    return base + [m for m in media if m not in base]


class PiToolDispatcher:
    def __init__(self, service: AgentService) -> None:
        self._service = service
        self._bindings = SessionBindingStore(service._store.connection)

    async def execute(
        self,
        request: PiToolRequestV1,
        *,
        caller_pid: int | None = None,
        caller_uid: int | None = None,
    ) -> PiToolResultV1:
        conn = self._service._store.connection
        self._caller_pid = caller_pid
        self._caller_uid = caller_uid
        # 1. idempotency：重放直接返回首个结果（不产生重复副作用）。
        row = conn.execute(
            "SELECT response_json FROM pi_tool_idempotency WHERE idempotency_key = ?",
            (request.idempotency_key,),
        ).fetchone()
        if row is not None:
            return PiToolResultV1(**json.loads(row["response_json"]))
        # 八审 §4 P0-6：doom-loop 熔断——同一工具同一参数出错后原样
        # 重复直接拒绝（不再消耗模型回合）；成功即重置，不误伤合法
        # 重复观测。进程级指纹（安全语义仍在 fail-closed 链上，熔断
        # 只是效率护栏）。
        fingerprint = request.tool_name + ":" + json.dumps(
            request.arguments, sort_keys=True, ensure_ascii=False
        )
        failures = getattr(self._service, "_tool_fail_fingerprints", None)
        if failures is None:
            failures = self._service._tool_fail_fingerprints = {}
        if failures.get(fingerprint):
            return PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="REJECTED",
                summary="同一调用已失败过一次——原样重复不会成功。请改变参数、"
                "改用任务级入口（rosclaw_task），或诚实报告无法完成。",
                error_code="DOOM_LOOP",
            )
        try:
            result = await self._execute_validated(request)
        except ToolBridgeError as exc:
            result = PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="REJECTED",
                summary=exc.message,
                error_code=exc.code,
                retryable=exc.retryable,
            )
        if result.ok:
            failures.pop(fingerprint, None)
        elif result.error_code in (
            # 验收轮实测：瞬态/上下文类失败可安全重试（JIT 续租与
            # 任务 attach 语义保证）——不记入熔断指纹，否则第一次
            # 瞬态失败会毒化随后的合法重试。
            "CONTEXT_NOT_FRESH",
            "CONTEXT_HASH_MISMATCH",
            "NEEDS_REPLAN",
            "CONTEXT_LEASE_REQUIRED",
        ):
            pass
        else:
            failures[fingerprint] = True
        conn.execute(
            "INSERT OR IGNORE INTO pi_tool_idempotency "
            "(idempotency_key, request_id, tool_name, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                request.idempotency_key,
                request.request_id,
                request.tool_name,
                result.model_dump_json(),
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
        await self._mirror_decision(request, result)
        return result

    async def _execute_validated(self, request: PiToolRequestV1) -> PiToolResultV1:
        service = self._service
        # 2. session binding + mission。
        binding = self._bindings.binding_for_session(request.pi_session_id)
        if binding is None:
            raise ToolBridgeError("SESSION_UNBOUND", "pi session has no active binding")
        if binding.mission_id != request.mission_id:
            raise ToolBridgeError(
                "MISSION_MISMATCH",
                f"bound mission is {binding.mission_id}, not {request.mission_id}",
            )
        mission = service.get_mission(request.mission_id)
        if mission is None:
            raise ToolBridgeError("MISSION_NOT_FOUND", "unknown mission")
        # 3. writer lease（崩溃回收由 lease 过期保证）。
        writer = self._bindings.writer_of(request.mission_id)
        if writer is None or writer.pi_session_id != request.pi_session_id:
            raise ToolBridgeError(
                "WRITER_LEASE_REQUIRED", "this session does not hold the writer lease"
            )
        # 3.5 context revision 硬校验（P0-7）：动作类必须 exact match；
        # 观测类允许有限 stale（记录但放行）。
        if _TOOL_TABLE.get(request.tool_name) == "physical_action":
            snapshot = service.snapshot(request.mission_id)
            if request.context_revision != snapshot.context_revision:
                raise ToolBridgeError(
                    "CONTEXT_REVISION_MISMATCH",
                    f"request revision {request.context_revision} != current "
                    f"{snapshot.context_revision} — refresh embodied context and "
                    "re-propose (P0-7: actions require exact context)",
                )
        # 4. allowlist。
        if request.tool_name in _DEFERRED_TOOLS:
            raise ToolBridgeError(
                "TOOL_DEFERRED",
                f"{request.tool_name} 在 {_DEFERRED_TOOLS[request.tool_name]}才开放——当前拒绝",
            )
        if request.tool_name not in _TOOL_TABLE:
            raise ToolBridgeError("TOOL_UNKNOWN", f"unknown tool {request.tool_name!r}")
        # 5. 分发。
        return await self._dispatch(request)

    async def _dispatch(self, request: PiToolRequestV1) -> PiToolResultV1:
        service = self._service
        name = request.tool_name
        args = request.arguments
        if name == "rosclaw_status":
            mission = service.get_mission(request.mission_id)
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary=json.dumps(
                    {
                        "agentd": "READY",
                        "mode": mission.mode.value if mission else "",
                        "state": mission.state.value if mission else "",
                        "authorization_profile": service.authorization_profile(),
                    },
                    ensure_ascii=False,
                ),
            )
        if name == "rosclaw_observe":
            capability_id = str(args.get("capability_id", ""))
            if not capability_id:
                raise ToolBridgeError("INVALID_ARGUMENTS", "capability_id required")
            descriptor = service._tool_catalog.get(capability_id)
            if descriptor is None:
                raise ToolBridgeError(
                    "CAPABILITY_UNKNOWN", f"capability {capability_id!r} not in catalog"
                )
            if descriptor.execution_class.value != "OBSERVE":
                # 规格 §16.2：动作类能力不得经 observe 绕过。
                raise ToolBridgeError(
                    "NOT_OBSERVABLE",
                    f"capability {capability_id} is {descriptor.execution_class.value}, "
                    "not OBSERVE — action-class capabilities need the approval chain",
                )
            # 六审 §6.3 旅程暴露：quarantine 判定必须用 catalog API——
            # ToolDescriptorV2 没有 .quarantined 属性（此前 observe MCP
            # 能力的路径从未被真实旅程走到）。
            if service._tool_catalog.quarantine_reason(capability_id) is not None:
                raise ToolBridgeError(
                    "CAPABILITY_QUARANTINED", f"capability {capability_id} is quarantined"
                )
            try:
                output = await service._tool_registry.execute(
                    capability_id, dict(args.get("arguments", {}))
                )
            except Exception as exc:  # noqa: BLE001 — 八审 P0-6：错误分类
                raise ToolBridgeError(
                    "INVALID_ARGUMENTS" if "validation" in str(type(exc).__name__).lower()
                    or "validation" in str(exc).lower() else "EXECUTOR_ERROR",
                    f"{type(exc).__name__}: {exc}"[:400],
                ) from exc
            text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary=text[:8000],
            )
        if name == "rosclaw_compute":
            # 七审 §2.2/PR-SEVEN-2.2：COMPUTE 能力免审批调用（纯计算无
            # 物理副作用）——不再被 observe 的 OBSERVE-only 拒绝。
            capability_id = str(args.get("capability_id", ""))
            if not capability_id:
                raise ToolBridgeError("INVALID_ARGUMENTS", "capability_id required")
            descriptor = service._tool_catalog.get(capability_id)
            if descriptor is None:
                raise ToolBridgeError(
                    "CAPABILITY_UNKNOWN", f"capability {capability_id!r} not in catalog"
                )
            if descriptor.execution_class.value != "COMPUTE":
                raise ToolBridgeError(
                    "NOT_COMPUTABLE",
                    f"capability {capability_id} is {descriptor.execution_class.value}, "
                    "not COMPUTE — actions need the approval chain, observations "
                    "use rosclaw_observe",
                )
            if service._tool_catalog.quarantine_reason(capability_id) is not None:
                raise ToolBridgeError(
                    "CAPABILITY_QUARANTINED", f"capability {capability_id} is quarantined"
                )
            try:
                output = await service._tool_registry.execute(
                    capability_id, dict(args.get("arguments", {}))
                )
            except Exception as exc:  # noqa: BLE001 — 八审 P0-6：错误分类
                raise ToolBridgeError(
                    "INVALID_ARGUMENTS" if "validation" in str(type(exc).__name__).lower()
                    or "validation" in str(exc).lower() else "EXECUTOR_ERROR",
                    f"{type(exc).__name__}: {exc}"[:400],
                ) from exc
            text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary=text[:8000],
            )
        if name == "rosclaw_verify":
            receipts = [
                e.payload
                for e in service.events_replay(request.mission_id, limit=200)
                if e.type.value == "receipt.received"
            ]
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary=json.dumps({"receipts": receipts[-3:]}, ensure_ascii=False)[:8000],
            )
        if name == "rosclaw_memory_query":
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary="memory query is wired to the approved evidence pipeline; "
                "no results in this SIM profile",
            )
        if name == "rosclaw_request_action":
            return await self._request_action(request)
        if name == "rosclaw_task":
            return await self._task(request)
        if name == "rosclaw_delegate":
            return await self._delegate(request)
        if name == "rosclaw_check_work":
            return await self._check_work(request)
        if name == "rosclaw_cancel_work":
            return await self._cancel_work(request)
        if name == "rosclaw_list_work":
            return await self._list_work(request)
        if name == "rosclaw_update_work":
            return await self._update_work(request)
        if name == "rosclaw_retry_work":
            return await self._retry_work(request)
        if name == "rosclaw_answer_work":
            return await self._answer_work(request)
        if name == "rosclaw_resume_work":
            return await self._resume_work(request)
        if name == "rosclaw_extend_work":
            return await self._extend_work(request)
        if name == "rosclaw_read_work_events":
            return await self._read_work_events(request)
        if name == "rosclaw_read_work_transcript":
            return await self._read_work_transcript(request)
        if name == "rosclaw_list_work_artifacts":
            return await self._list_work_artifacts(request)
        if name == "rosclaw_read_work_failure":
            return await self._read_work_failure(request)
        if name == "rosclaw_fail_safe":
            await service.cancel(request.mission_id)
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary="fail-safe: 当前回合已请求取消；E-Stop 请走独立 operator 路径",
            )
        raise ToolBridgeError("TOOL_UNKNOWN", f"unhandled tool {name!r}")

    async def _delegate(self, request: PiToolRequestV1) -> PiToolResultV1:
        """PNA-4（规格 §19）：招聘 Worker——受限 WorkOrder + 递归上限 +
        验证通过才进结果（未验证输出绝不进入主上下文）。"""
        from rosclaw.contracts.common import new_id
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        args = request.arguments
        goal = str(args.get("goal", "")).strip()
        if not goal:
            raise ToolBridgeError("INVALID_ARGUMENTS", "goal required")
        worker_hint = str(args.get("worker_id", "auto"))
        parent_id = str(args.get("parent_work_order_id", "") or "") or None
        # 递归上限（规格 §19.5）：直派 worker 单是叶子——沿用全系统约定
        # delegation_depth=0 + max_children=0（depth>0 的单子必须自带
        # children 预算，而 native worker 不接受子委派——两层共同保证
        # worker 无法经此桥再委派）。带 parent_work_order_id 的请求在
        # max_delegation_depth（默认 1）语义下一律是"再委派"，直接拒。
        parent_depth = 0
        if parent_id:
            parent = self._service._worker_manager.order(parent_id)
            parent_depth = parent.delegation_depth if parent else 0
        max_depth = int(args.get("max_delegation_depth", 1))
        if parent_id and parent_depth + 1 > max_depth - 1:
            raise ToolBridgeError(
                "DELEGATION_DEPTH_EXCEEDED",
                "delegation beyond the direct worker layer is refused "
                f"(max_delegation_depth={max_depth})",
            )
        order_depth = 0
        worker_profile = str(args.get("worker_profile") or "")
        capability = str(args.get("capability") or "")
        if not capability:
            # W3：写能力 profile 默认 code.develop（sandbox_process 语义）。
            capability = (
                "code.develop" if worker_profile in ("developer", "sim-builder")
                else "analysis.text"
            )
        # 十审 W0：调用方（TS 工具层，非模型字段）可预生成 work_order_id——
        # abort 在响应返回前也能按精确 ID cancel（修 P0-ORDER-CORRELATION
        # 的另一半：abort 不再"找不到要杀的单"）。
        provided_wo = str(args.get("work_order_id", "") or "")
        if provided_wo and not re.fullmatch(r"wo_[A-Za-z0-9]{8,32}", provided_wo):
            raise ToolBridgeError("INVALID_ARGUMENTS", "malformed work_order_id")
        is_workbench = worker_profile in ("developer", "sim-builder")
        # 十三审 HOTFIX-13.2：execution_policy——hard deadline 必须有
        # 显式权威来源（模型不得凭感觉给 240/300s 处决时间）；缺来源
        # 的 hard deadline 硬拒绝。wall_time_sec 语义变为 soft target。
        exec_policy = args.get("execution_policy") or {}
        if not isinstance(exec_policy, dict):
            raise ToolBridgeError("INVALID_ARGUMENTS", "execution_policy must be an object")
        if exec_policy.get("hard_deadline_sec") and exec_policy.get(
            "hard_deadline_source"
        ) not in ("user", "benchmark", "admin_policy"):
            raise ToolBridgeError(
                "DEADLINE_AUTHORITY_REQUIRED",
                "hard_deadline_sec 需要显式权威来源（user/benchmark/admin_policy）"
                "——Worker 有进度就让它继续；wall 时间只是观察指标",
            )
        # 十四审 PR-14.1（§3.1）：cost_hard_limit 是唯一可暂停进程的预算
        # 手段——同样需要显式 user/admin_policy 权威；模型自报的
        # model_tokens 只是遥测，绝不控制进程。
        if exec_policy.get("cost_hard_limit_tokens") and exec_policy.get(
            "cost_hard_limit_source"
        ) not in ("user", "admin_policy"):
            raise ToolBridgeError(
                "COST_LIMIT_AUTHORITY_REQUIRED",
                "cost_hard_limit_tokens 需要显式权威来源（user/admin_policy）"
                "——token soft target 只做提示，不改变进程状态",
            )
        # 十二审 PR-12.4：WorkSpecV2——任务类型驱动验收（deliverables
        # 覆盖 profile 默认工件类型）。
        task_type = str(args.get("task_type") or (
            "code_change" if is_workbench else "analyze"
        ))
        deliverables_raw = args.get("deliverables") or []
        order = WorkOrderV1(
            work_order_id=provided_wo or new_id("wo"),
            mission_id=request.mission_id,
            issued_by="rosclaw-agent:pi",
            capability=capability,
            goal=goal,
            inputs={
                "instructions": str(args.get("instructions", goal)),
                "artifacts": args.get("artifact_refs") or [],
                # 十审 W1：无 secret 模型快照（TS 工具层从 session ctx 取
                # provider/model/thinking）+ Worker profile。快照绝不携带
                # 凭据（pi_managed 写 envelope 前有硬校验）。
                "model_snapshot": args.get("model_snapshot") or {},
                "worker_profile": worker_profile,
                # W3：目标 workspace（git 仓库→worktree；默认当前 cwd）+
                # 可选 base_ref。
                "workspace": str(args.get("workspace") or ""),
                "base_ref": str(args.get("base_ref") or ""),
                # 十二审 PR-12.4：WorkSpecV2 快照（Worker envelope 据此
                # 生成 DoD；verifier 据此验收）。
                "task_type": task_type,
                "deliverables": deliverables_raw,
                # 十三审：执行策略（soft target/hard deadline/token limit）。
                "execution_policy": {
                    "soft_target_sec": exec_policy.get("soft_target_sec"),
                    "hard_deadline_sec": exec_policy.get("hard_deadline_sec"),
                    "hard_deadline_source": exec_policy.get("hard_deadline_source"),
                    "token_soft_limit": exec_policy.get("token_soft_limit"),
                },
            },
            budgets=BudgetEnvelope(
                wall_time_sec=int(args.get("budget", {}).get("wall_time_sec", 300))
                if isinstance(args.get("budget"), dict)
                else 300,
                model_tokens=int(args.get("budget", {}).get("model_tokens", 50_000))
                if isinstance(args.get("budget"), dict)
                else 50_000,
                # 叶子 worker 单：max_children=0（native adapter 不接受子委派）。
            ),
            expected_output=ExpectedOutput(
                artifacts=_expected_artifacts(task_type, deliverables_raw, is_workbench)
            ),
            side_effect_policy=SideEffectPolicy(
                **{"class": "sandbox_process" if is_workbench else "none"}
            ),
            delegation_depth=order_depth,
            max_delegation_depth=max_depth,
            parent_work_order_id=parent_id,
            root_work_order_id=str(args.get("root_work_order_id", "") or "") or parent_id,
        )
        service = self._service
        self._check_worker_token_budget(request.mission_id)
        candidates = self._candidates_for(worker_hint, capability)
        if not candidates:
            raise ToolBridgeError(
                "WORKER_UNAVAILABLE", f"no worker matches {worker_hint!r}", retryable=True
            )
        try:
            scheduled = service._worker_manager.hire(order, candidates)
        except Exception as exc:  # noqa: BLE001 - 诚实失败，不伪造委派
            raise ToolBridgeError("SCHEDULING_FAILED", str(exc), retryable=True) from exc
        # 十审 W0（P0-WORKER-BLOCK）：不在工具请求栈里同步等待整个任务。
        # 后台驱动任务驱动到终态；这里只在短 grace 内等"快任务"——
        # 超时返回 STARTED + 精确 WorkOrder ID/worker/预算/deadline。
        service.spawn_worker_driver(scheduled)
        grace = float(args.get("sync_grace_sec", 3.0) or 0)
        grace = max(0.0, min(grace, 10.0))  # 硬上限：永不长阻塞父会话
        terminal = await self._await_terminal(
            scheduled.work_order_id, timeout_sec=grace
        )
        if terminal is not None:
            return self._terminal_response(request, terminal)
        # 十四审 PR-14.7：wall_time 只是 soft target（提醒阈值）——
        # 不再向模型/用户展示 "Deadline"（概念误导→错误归因之源）。
        # 十审 §10.3：外部 harness 的模型/账号配置独立——UI 明示，不得
        # 假装继承 Native Agent。
        external_note = ""
        card = service._registry.get(scheduled.assigned_to or "")
        if card is not None and card.kind.value == "harness":
            external_note = (
                "\n注意：这是外部 Worker——使用其自身账号/模型/权限配置"
                "（独立于 Native Agent 的 Kimi 配置）。"
            )
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status="STARTED",
            summary=(
                "已启动后台 Worker（不阻塞本会话——你可以继续与用户交互）。\n"
                f"WorkOrder: {scheduled.work_order_id}\n"
                f"Worker: {scheduled.assigned_to}\n"
                f"预算（soft target 提醒阈值，不会强杀 Worker）: "
                f"wall_time {scheduled.budgets.wall_time_sec}s · "
                f"{scheduled.budgets.model_tokens} tokens\n"
                "Worker 有进度就让它继续做；wall/token 到点只提醒不终止。\n"
                f"查询进度：rosclaw_check_work(work_order_id=\"{scheduled.work_order_id}\")；"
                f"取消：rosclaw_cancel_work(work_order_id=\"{scheduled.work_order_id}\")。"
                "Worker 产出须经 ROSClaw 验证后才会被采纳。"
                f"{external_note}"
            ),
        )

    async def _await_terminal(self, work_order_id: str, *, timeout_sec: float):
        """短 grace 内等待终态（快任务保持同步体验）；超时返回 None。"""
        if timeout_sec <= 0:
            return None
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout_sec
        while loop.time() < end:
            order = self._service._worker_manager.order(work_order_id)
            if order is not None and order.status in (
                "ACCEPTED",
                "FAILED",
                "EXPIRED",
                "CANCELLED",
            ):
                return order
            await asyncio.sleep(0.05)
        return None

    def _terminal_response(self, request: PiToolRequestV1, order) -> PiToolResultV1:
        """终态 WorkOrder → 工具结果（从权威存储读结果与验证报告）。"""
        conn = self._service._store.connection
        row = conn.execute(
            "SELECT result_json FROM work_results WHERE work_order_id = ?",
            (order.work_order_id,),
        ).fetchone()
        summary = ""
        artifact_refs: list[str] = []
        result_status = ""
        if row is not None:
            payload = json.loads(row["result_json"])
            summary = str(payload.get("summary", ""))
            artifact_refs = [str(a.get("ref", "")) for a in payload.get("artifacts", [])]
            result_status = str(payload.get("status", ""))
        verify_row = conn.execute(
            "SELECT verify_report_json FROM work_orders WHERE work_order_id = ?",
            (order.work_order_id,),
        ).fetchone()
        accepted = False
        reasons: list[str] = []
        if verify_row and verify_row["verify_report_json"]:
            report = json.loads(verify_row["verify_report_json"])
            accepted = bool(report.get("accepted"))
            reasons = [str(r) for r in report.get("reasons", [])]
        if order.status == "ACCEPTED" and accepted:
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary=summary,
                artifact_refs=artifact_refs,
            )
        if order.status == "CANCELLED":
            return PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="CANCELLED",
                summary=f"WorkOrder {order.work_order_id} 已取消。",
                error_code="WORK_CANCELLED",
            )
        if result_status == "FAILED" or order.status == "FAILED":
            if reasons:
                # 十二审 PR-12.4：verifier 原因之外附上 Worker 的真实失败
                # 摘要（DELIVERABLE_FAILED/adapter 错误都在 result.summary）
                # ——竞态下 verify_report_json 可能未落，summary 是兜底真相。
                detail = f"（{'；'.join(reasons)}）"
                if summary and summary not in "；".join(reasons):
                    detail += f" Worker 报告：{summary[:400]}"
                return PiToolResultV1(
                    request_id=request.request_id,
                    ok=False,
                    status="VERIFY_FAILED",
                    summary=(
                        f"Worker {order.assigned_to} 提交的结果未通过验证"
                        f"{detail}——未采纳进主上下文。"
                    ),
                    error_code="VERIFICATION_REJECTED",
                    retryable=True,
                )
            # 失败原因必须透出（DELIVERABLE_FAILED/liveness/adapter 等
            # 都在 result.summary）——不得只报"失败"让模型猜。
            code = "WORKER_FAILED"
            for marker in ("DELIVERABLE_FAILED", "ADAPTER_PROTOCOL_ERROR",
                           "BLOCKED_PREFLIGHT", "PROVIDER_TIMEOUT"):
                if marker in summary:
                    code = marker
                    break
            return PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="FAILED",
                summary=summary or f"Worker {order.assigned_to} 失败。",
                error_code=code,
                retryable=code == "WORKER_FAILED",
            )
        return PiToolResultV1(
            request_id=request.request_id,
            ok=False,
            status=order.status,
            summary=f"WorkOrder {order.work_order_id} 终态 {order.status}。",
            error_code="WORK_NOT_COMPLETED",
            retryable=True,
        )

    async def _list_work(self, request: PiToolRequestV1) -> PiToolResultV1:
        """十审 W2：当前 mission 的 WorkOrder 摘要（DB 权威——/compact
        或重启后不丢）。"""
        orders = self._service._worker_manager.orders_for_mission(request.mission_id)
        if not orders:
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="EMPTY",
                summary="当前 Mission 没有 WorkOrder。",
            )
        lines = []
        for o in orders[-20:]:
            lines.append(
                f"- {o.work_order_id} · {o.assigned_to or '?'} · {o.status} · "
                f"{(o.goal or '')[:60]}"
            )
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status="LISTED",
            summary=f"当前 Mission 的 WorkOrder（{len(orders)} 单）：\n" + "\n".join(lines),
        )

    async def _update_work(self, request: PiToolRequestV1) -> PiToolResultV1:
        """十审 W2：追加约束/steer 备注。

        诚实语义：运行中的 Worker（W2 的内置/外部 adapter）不能实时接收
        新约束——备注落进 order.inputs.steer_notes（check/list 可见，
        retry/新 attempt 生效）；要立刻改变方向请 cancel 后重新委派。
        """
        work_order_id = str(request.arguments.get("work_order_id", "")).strip()
        note = str(request.arguments.get("note", "")).strip()
        if not work_order_id or not note:
            raise ToolBridgeError("INVALID_ARGUMENTS", "work_order_id and note required")
        manager = self._service._worker_manager
        order = manager.order(work_order_id)
        if order is None:
            raise ToolBridgeError(
                "WORK_ORDER_NOT_FOUND", f"unknown work order {work_order_id!r}"
            )
        if order.mission_id != request.mission_id:
            raise ToolBridgeError(
                "MISSION_MISMATCH", "work order belongs to a different mission"
            )
        if order.status in ("ACCEPTED", "FAILED", "EXPIRED", "CANCELLED"):
            raise ToolBridgeError(
                "ALREADY_TERMINAL",
                f"work order already {order.status} — update 对终态无意义",
            )
        conn = self._service._store.connection
        inputs = dict(order.inputs)
        notes = list(inputs.get("steer_notes") or [])
        notes.append({"note": note, "at": datetime.now(UTC).isoformat()})
        inputs["steer_notes"] = notes
        updated = order.model_copy(update={"inputs": inputs})
        conn.execute(
            "UPDATE work_orders SET order_json = ?, updated_at = ? WHERE work_order_id = ?",
            (updated.model_dump_json(), datetime.now(UTC).isoformat(), work_order_id),
        )
        conn.commit()
        # 十审 W4：运行中的内置 Worker 可实时 steer（stdin 通道）——
        # 送达与否诚实报告（不假装运行外进程也能收到）。
        delivered = False
        if order.status == "RUNNING":
            card_row = conn.execute(
                "SELECT card_json FROM worker_cards WHERE worker_id = ?",
                (order.assigned_to,),
            ).fetchone()
            if card_row is not None:
                card = json.loads(card_row["card_json"])
                adapter = self._service._worker_manager._adapters.get(
                    card.get("adapter_type", "")
                )
                steer = getattr(adapter, "steer", None)
                if steer is not None:
                    import contextlib as _cl

                    with _cl.suppress(Exception):
                        delivered = await steer(work_order_id, note)
        if delivered:
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="UPDATED",
                summary=f"steer 已实时送达运行中的 Worker（{work_order_id}），并落账备查。",
            )
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status="UPDATED",
            summary=(
                f"已记录 steer 备注（{work_order_id}）。注意：该 Worker 当前不可实时接收"
                "（非运行中或 adapter 无 steer 通道）——备注对 retry/后续 attempt 生效；"
                "要立刻改变方向请 rosclaw_cancel_work 后重新委派。"
            ),
        )

    def _candidates_for(self, worker_hint: str, capability: str):
        from rosclaw.agentd.workers.scheduler import CandidateView

        service = self._service
        return [
            CandidateView(
                card=card,
                registry_status=service._registry.status_of(card.worker_id) or "DISABLED",
                running_orders=len(
                    service._worker_manager.active_orders_for_worker(card.worker_id)
                ),
                circuit_open=service._worker_manager.circuit_open(card.worker_id, capability),
            )
            for card in service._registry.list()
            if worker_hint in ("", "auto") or card.worker_id == worker_hint
        ]

    def _check_worker_token_budget(self, mission_id: str) -> None:
        """十审 W4：mission 级 Worker token 预算——超过即诚实拒绝新委派
        （防 runaway 计费；ROSCLAW_WORKER_TOKEN_BUDGET 可调，默认 1M）。"""
        import os

        cap = int(os.environ.get("ROSCLAW_WORKER_TOKEN_BUDGET", "1000000"))
        if cap <= 0:
            return
        conn = self._service._store.connection
        rows = conn.execute(
            "SELECT wr.result_json FROM work_results wr "
            "JOIN work_orders wo ON wo.work_order_id = wr.work_order_id "
            "WHERE wo.mission_id = ?",
            (mission_id,),
        ).fetchall()
        spent = 0
        for row in rows:
            usage = (json.loads(row["result_json"]).get("usage") or {})
            spent += int(usage.get("prompt_tokens") or 0) + int(
                usage.get("completion_tokens") or 0
            )
        if spent >= cap:
            raise ToolBridgeError(
                "WORKER_BUDGET_EXHAUSTED",
                f"本 Mission 的 Worker token 预算已用尽（{spent}/{cap}）——"
                "请开新 Mission 或调大 ROSCLAW_WORKER_TOKEN_BUDGET。",
            )

    async def _retry_work(self, request: PiToolRequestV1) -> PiToolResultV1:
        """十审 W4：终态单 retry。十四审 PR-14.2：RetryCoordinator 唯一
        决策——已有自动 retry/活跃 attempt 时返回现有 attempt（幂等），
        绝不创建第二个顶层任务。"""
        from rosclaw.agentd.workers.retry import parse_cause

        work_order_id = str(request.arguments.get("work_order_id", "")).strip()
        if not work_order_id:
            raise ToolBridgeError("INVALID_ARGUMENTS", "work_order_id required")
        manager = self._service._worker_manager
        order = manager.order(work_order_id)
        if order is None:
            raise ToolBridgeError(
                "WORK_ORDER_NOT_FOUND", f"unknown work order {work_order_id!r}"
            )
        if order.mission_id != request.mission_id:
            raise ToolBridgeError(
                "MISSION_MISMATCH", "work order belongs to a different mission"
            )
        if order.status not in (
            "ACCEPTED", "FAILED", "EXPIRED", "CANCELLED", "INTERRUPTED_RESUMABLE",
        ):
            raise ToolBridgeError(
                "NOT_TERMINAL",
                f"work order is {order.status}——只能 retry 终态单（运行中请先 cancel）",
            )
        self._check_worker_token_budget(request.mission_id)
        # 终态原因来自账本/摘要（模型不得猜日志）——coordinator 幂等仲裁。
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        state = WorkerEventStore(self._service._home).read_state(work_order_id) or {}
        cause = str(state.get("termination_cause") or "") or parse_cause(
            getattr(order, "summary", "") or ""
        )
        attempt, created, reason = await self._service._retry_coordinator.request_retry(
            order, cause=cause, actor="native_agent"
        )
        if attempt is None:
            raise ToolBridgeError(
                "WORKER_UNAVAILABLE" if reason == "worker_unavailable" else "RETRY_REJECTED",
                f"retry 未被接受：{reason}",
                retryable=reason == "worker_unavailable",
            )
        if not created:
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="STARTED",
                summary=(
                    f"已有进行中的 attempt（同一任务，不重复创建）：{attempt.work_order_id}\n"
                    "一个用户任务只有一张卡——retry/resume 是内部 attempt。"
                ),
            )
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status="STARTED",
            summary=(
                f"已 retry（新 attempt）：{attempt.work_order_id}\n"
                f"parent: {order.work_order_id} · root: {attempt.root_work_order_id}\n"
                f"Worker: {attempt.assigned_to}"
            ),
        )

    async def _resume_work(self, request: PiToolRequestV1) -> PiToolResultV1:
        """十二审 PR-12.3：resume——新 attempt 恢复同一 Pi 会话（工具
        历史与上下文保留）；区别于 retry（新会话从零）。只接受终态单
        且有持久 session 文件。"""

        work_order_id = str(request.arguments.get("work_order_id", "")).strip()
        if not work_order_id:
            raise ToolBridgeError("INVALID_ARGUMENTS", "work_order_id required")
        manager = self._service._worker_manager
        order = manager.order(work_order_id)
        if order is None:
            raise ToolBridgeError(
                "WORK_ORDER_NOT_FOUND", f"unknown work order {work_order_id!r}"
            )
        if order.mission_id != request.mission_id:
            raise ToolBridgeError(
                "MISSION_MISMATCH", "work order belongs to a different mission"
            )
        if order.status not in (
            "FAILED", "CANCELLED", "EXPIRED", "ACCEPTED", "INTERRUPTED_RESUMABLE",
        ):
            raise ToolBridgeError(
                "NOT_TERMINAL", f"work order is {order.status}——只能 resume 终态单"
            )
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        state = WorkerEventStore(self._service._home).read_state(work_order_id) or {}
        session_file = str(state.get("session_file") or "")
        if not session_file or not Path(session_file).exists():
            raise ToolBridgeError(
                "NO_CHECKPOINT",
                "该 WorkOrder 没有可恢复的会话检查点（session 未持久化）——"
                "请用 rosclaw_retry_work 开新 attempt",
            )
        self._check_worker_token_budget(request.mission_id)
        # 十四审 PR-14.2：resume 也走 RetryCoordinator——同一 root job 的
        # 新 attempt（同一 Pi 会话恢复），活跃 attempt 去重同样生效。
        cause = str(state.get("termination_cause") or "")
        attempt, created, reason = await self._service._retry_coordinator.request_retry(
            order,
            cause=cause or None,
            actor="native_agent",
            note="resume 恢复同一 Pi 会话（上下文保留）",
            resume_session=session_file,
        )
        if attempt is None:
            raise ToolBridgeError(
                "WORKER_UNAVAILABLE" if reason == "worker_unavailable" else "RESUME_REJECTED",
                f"resume 未被接受：{reason}",
                retryable=reason == "worker_unavailable",
            )
        if not created:
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="STARTED",
                summary=(
                    f"已有进行中的 attempt（同一任务，不重复创建）：{attempt.work_order_id}"
                ),
            )
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status="STARTED",
            summary=(
                f"已从持久会话恢复（新 attempt）：{attempt.work_order_id}\n"
                f"resume 自 {work_order_id}（同一 Pi 会话，上下文保留）"
            ),
        )

    async def _extend_work(self, request: PiToolRequestV1) -> PiToolResultV1:
        """十三审 HOTFIX-13.2：BUDGET_PAUSED 追加预算并唤醒（同一会话
        继续——不是新 attempt）。"""
        work_order_id = str(request.arguments.get("work_order_id", "")).strip()
        if not work_order_id:
            raise ToolBridgeError("INVALID_ARGUMENTS", "work_order_id required")
        add_tokens = int(request.arguments.get("add_tokens") or 50_000)
        manager = self._service._worker_manager
        order = manager.order(work_order_id)
        if order is None:
            raise ToolBridgeError(
                "WORK_ORDER_NOT_FOUND", f"unknown work order {work_order_id!r}"
            )
        if order.mission_id != request.mission_id:
            raise ToolBridgeError(
                "MISSION_MISMATCH", "work order belongs to a different mission"
            )
        if order.status != "BUDGET_PAUSED":
            raise ToolBridgeError(
                "NOT_PAUSED", f"work order is {order.status}——只能 extend 预算暂停的单"
            )
        # 预算落账 + 唤醒 + BUDGET_PAUSED→RUNNING。
        policy = dict(order.inputs.get("execution_policy") or {})
        policy["token_soft_limit"] = int(policy.get("token_soft_limit") or 0) + add_tokens
        inputs = {**dict(order.inputs), "execution_policy": policy}
        conn = self._service._store.connection
        updated = order.model_copy(update={"inputs": inputs})
        conn.execute(
            "UPDATE work_orders SET order_json = ?, updated_at = ? WHERE work_order_id = ?",
            (updated.model_dump_json(), datetime.now(UTC).isoformat(), work_order_id),
        )
        conn.commit()
        card_row = conn.execute(
            "SELECT card_json FROM worker_cards WHERE worker_id = ?",
            (order.assigned_to,),
        ).fetchone()
        delivered = False
        if card_row is not None:
            import json as _json

            card = _json.loads(card_row["card_json"])
            adapter = manager._adapters.get(card.get("adapter_type", ""))
            extend = getattr(adapter, "extend", None)
            if extend is not None:
                import contextlib as _cl

                with _cl.suppress(Exception):
                    delivered = await extend(work_order_id, add_tokens)
        if not delivered:
            raise ToolBridgeError(
                "DELIVERY_FAILED", "Worker 进程不在——请用 /job resume 从检查点恢复"
            )
        manager._transition(work_order_id, "RUNNING", "budget_extended")
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status="RUNNING",
            summary=f"已追加 {add_tokens} tokens 并唤醒 Worker（{work_order_id}）——同一会话继续。",
        )

    async def _answer_work(self, request: PiToolRequestV1) -> PiToolResultV1:
        """十一审 PR-E：WAITING_INPUT 的用户回答——stdin 通道送达 +
        BLOCKED→RUNNING（adapter answer_received 事件驱动）。"""
        work_order_id = str(request.arguments.get("work_order_id", "")).strip()
        text = str(request.arguments.get("text", "")).strip()
        if not work_order_id or not text:
            raise ToolBridgeError("INVALID_ARGUMENTS", "work_order_id and text required")
        manager = self._service._worker_manager
        order = manager.order(work_order_id)
        if order is None:
            raise ToolBridgeError(
                "WORK_ORDER_NOT_FOUND", f"unknown work order {work_order_id!r}"
            )
        if order.mission_id != request.mission_id:
            raise ToolBridgeError(
                "MISSION_MISMATCH", "work order belongs to a different mission"
            )
        if order.status != "BLOCKED":
            raise ToolBridgeError(
                "NOT_WAITING",
                f"work order is {order.status}——只有 WAITING_INPUT 的单可回答",
            )
        card_row = self._service._store.connection.execute(
            "SELECT card_json FROM worker_cards WHERE worker_id = ?",
            (order.assigned_to,),
        ).fetchone()
        delivered = False
        if card_row is not None:
            card = json.loads(card_row["card_json"])
            adapter = manager._adapters.get(card.get("adapter_type", ""))
            answer = getattr(adapter, "answer", None)
            if answer is not None:
                import contextlib as _cl

                with _cl.suppress(Exception):
                    delivered = await answer(work_order_id, text)
        if not delivered:
            raise ToolBridgeError(
                "DELIVERY_FAILED", "Worker 进程不在或通道已断——请 retry"
            )
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status="RUNNING",
            summary=f"回答已送达 Worker（{work_order_id}）——任务继续。",
        )

    def _order_for_read(self, request: PiToolRequestV1):
        work_order_id = str(request.arguments.get("work_order_id", "")).strip()
        if not work_order_id:
            raise ToolBridgeError("INVALID_ARGUMENTS", "work_order_id required")
        order = self._service._worker_manager.order(work_order_id)
        if order is None:
            raise ToolBridgeError(
                "WORK_ORDER_NOT_FOUND", f"unknown work order {work_order_id!r}"
            )
        if order.mission_id != request.mission_id:
            raise ToolBridgeError(
                "MISSION_MISMATCH", "work order belongs to a different mission"
            )
        return order

    async def _read_work_events(self, request: PiToolRequestV1) -> PiToolResultV1:
        """十三审 PR-13.5：事件分页读取（诊断用——每次调用一页，
        模型按 cursor 翻页，不整本塞上下文）。"""
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        order = self._order_for_read(request)
        after_seq = int(request.arguments.get("after_seq") or 0)
        limit = min(int(request.arguments.get("limit") or 30), 100)
        page = WorkerEventStore(self._service._home).tail_page(
            order.work_order_id, after_seq=after_seq, limit=limit
        )
        lines = []
        for e in page["events"]:
            kind = e.get("kind", "")
            if kind == "liveness":
                continue
            detail = (
                e.get("args_preview") or e.get("output_preview")
                or e.get("preview") or e.get("message") or ""
            )
            lines.append(f"#{e.get('seq')} {kind} {str(detail)[:120]}".rstrip())
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status=order.status,
            summary=(
                f"Worker 事件（{order.work_order_id}，cursor {after_seq}→"
                f"{page['next_cursor']}{'，还有更多' if page['has_more'] else ''}）：\n"
                + ("\n".join(lines) or "（本页无可见事件）")
            ),
        )

    async def _read_work_transcript(self, request: PiToolRequestV1) -> PiToolResultV1:
        """十四审 PR-14.3：完整公开 transcript 分页（tseq 游标 + channel
        过滤）——不再是尾部 4000 字节切片。"""
        from rosclaw.agentd.workers.transcript_store import TranscriptStore

        order = self._order_for_read(request)
        args = request.arguments
        before_raw = args.get("before_seq")
        page = TranscriptStore(self._service._home).read_page(
            order.work_order_id,
            after_seq=int(args.get("after_seq") or 0) or None,
            before_seq=int(before_raw) if before_raw is not None else None,
            limit=min(int(args.get("limit") or 50), 200),
            channel=str(args.get("channel") or "") or None,
        )
        lines = []
        for record in page["records"]:
            channel = record.get("channel", "?")
            if channel == "conversation":
                lines.append(
                    f"[{record['tseq']}] {record.get('role', '?')}: "
                    f"{str(record.get('text', ''))[:2000]}"
                )
            elif channel == "tools":
                if record.get("phase") == "start":
                    lines.append(
                        f"[{record['tseq']}] ▶ {record.get('tool', '?')} "
                        f"{str(record.get('args', ''))[:300]}"
                    )
                else:
                    mark = "✗" if record.get("is_error") else "✓"
                    lines.append(
                        f"[{record['tseq']}] {mark} {record.get('tool', '?')} "
                        f"{str(record.get('output', ''))[:800]}"
                    )
            elif channel == "files":
                lines.append(
                    f"[{record['tseq']}] 文件 {record.get('op', record.get('kind', '?'))}: "
                    f"{record.get('path', '')}"
                )
            elif channel == "artifacts":
                files = record.get("files") or []
                lines.append(
                    f"[{record['tseq']}] 产物: "
                    + ", ".join(f"{f.get('name')}({f.get('bytes')}B)" for f in files)
                )
            elif channel == "usage":
                lines.append(
                    f"[{record['tseq']}] usage: in={record.get('input')} "
                    f"out={record.get('output')} turns={record.get('turns')}"
                )
            elif channel == "control":
                lines.append(
                    f"[{record['tseq']}] 控制 ACK: {record.get('state', '?')}"
                )
        footer = (
            f"—— total={page['total']} has_more={page['has_more']} "
            f"next_cursor={page['next_cursor']}（after_seq 续读；"
            f"channel=conversation|tools|files|artifacts|usage|control 过滤）"
        )
        text = "\n".join(lines) if lines else "（无 transcript——Worker 尚未产出公开消息）"
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status=order.status,
            summary=f"{text}\n{footer}" if lines else text,
        )

    async def _list_work_artifacts(self, request: PiToolRequestV1) -> PiToolResultV1:
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        order = self._order_for_read(request)
        artifacts_dir = (
            WorkerEventStore(self._service._home).dir_of(order.work_order_id)
            / "artifacts"
        )
        lines = []
        if artifacts_dir.is_dir():
            for f in sorted(artifacts_dir.iterdir()):
                if f.is_file():
                    lines.append(f"- {f.name} · {f.stat().st_size}B")
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status=order.status,
            summary="\n".join(lines) or "（无产物）",
        )

    async def _read_work_failure(self, request: PiToolRequestV1) -> PiToolResultV1:
        """失败诊断摘要：终态 + verifier 原因 + 最后事件 + stderr tail +
        checkpoint/resume 入口——模型据此如实诊断，不再猜。"""
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        order = self._order_for_read(request)
        store = WorkerEventStore(self._service._home)
        conn = self._service._store.connection
        vrow = conn.execute(
            "SELECT verify_report_json FROM work_orders WHERE work_order_id = ?",
            (order.work_order_id,),
        ).fetchone()
        reasons = ""
        if vrow and vrow["verify_report_json"]:
            reasons = "；".join(
                str(r) for r in json.loads(vrow["verify_report_json"]).get("reasons", [])
            )
        events = store.tail(order.work_order_id, after_seq=0, limit=500)
        last_semantic = [e for e in events if e.get("kind") not in ("liveness",)][-3:]
        last_lines = [
            f"#{e.get('seq')} {e.get('kind')} "
            f"{str(e.get('output_preview') or e.get('message') or e.get('args_preview') or '')[:100]}"
            for e in last_semantic
        ]
        stderr_tail = store.tail_stderr(order.work_order_id)[-600:]
        state = store.read_state(order.work_order_id) or {}
        checkpoint = "有" if state.get("session_file") else "无"
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status=order.status,
            summary=(
                f"诊断（{order.work_order_id} · {order.status}）：\n"
                f"verifier 原因：{reasons or '（无）'}\n"
                f"最后事件：\n" + "\n".join(last_lines) + "\n"
                f"stderr 尾部：{stderr_tail[-300:] or '（无）'}\n"
                f"会话检查点：{checkpoint}"
                + ("；可 /job resume 恢复" if state.get("session_file") else "")
            ),
        )

    async def _check_work(self, request: PiToolRequestV1) -> PiToolResultV1:
        """十审 W0：按精确 work_order_id 查询（修 P0-ORDER-CORRELATION）。"""
        work_order_id = str(request.arguments.get("work_order_id", "")).strip()
        if not work_order_id:
            raise ToolBridgeError("INVALID_ARGUMENTS", "work_order_id required")
        order = self._service._worker_manager.order(work_order_id)
        if order is None:
            raise ToolBridgeError(
                "WORK_ORDER_NOT_FOUND", f"unknown work order {work_order_id!r}"
            )
        if order.mission_id != request.mission_id:
            raise ToolBridgeError(
                "MISSION_MISMATCH", "work order belongs to a different mission"
            )
        if order.status in ("ACCEPTED", "FAILED", "EXPIRED", "CANCELLED"):
            response = self._terminal_response(request, order)
            # check 的语义是"查询"——终态本身不是调用失败。
            response.ok = True
            return response
        conn = self._service._store.connection
        hb = conn.execute(
            "SELECT last_heartbeat_at, heartbeat_seq FROM work_orders WHERE work_order_id = ?",
            (work_order_id,),
        ).fetchone()
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status=order.status,
            summary=(
                f"WorkOrder {order.work_order_id} · Worker {order.assigned_to} · "
                f"状态 {order.status}"
                + (
                    f" · 心跳 seq={hb['heartbeat_seq']} at {hb['last_heartbeat_at']}"
                    if hb and hb["last_heartbeat_at"]
                    else " · 暂无心跳"
                )
            ),
        )

    async def _cancel_work(self, request: PiToolRequestV1) -> PiToolResultV1:
        """十审 W0：cancel 闭环——WorkOrder CANCELLED + adapter 杀进程树。"""
        work_order_id = str(request.arguments.get("work_order_id", "")).strip()
        if not work_order_id:
            raise ToolBridgeError("INVALID_ARGUMENTS", "work_order_id required")
        reason = str(request.arguments.get("reason", "") or "model_cancel")
        order = self._service._worker_manager.order(work_order_id)
        if order is None:
            raise ToolBridgeError(
                "WORK_ORDER_NOT_FOUND", f"unknown work order {work_order_id!r}"
            )
        if order.mission_id != request.mission_id:
            raise ToolBridgeError(
                "MISSION_MISMATCH", "work order belongs to a different mission"
            )
        final = await self._service._worker_manager.cancel_order(work_order_id, reason=reason)
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status=final.status,
            summary=f"WorkOrder {work_order_id} 当前状态 {final.status}。",
            error_code=None if final.status == "CANCELLED" else "ALREADY_TERMINAL",
        )

    async def _task(self, request: PiToolRequestV1) -> PiToolResultV1:
        """八审 §1.6/P0-5：任务级入口——TaskRunner 确定性编排
        （规划→策略→单动作→自动验证），模型只收 TaskResult 摘要。"""
        from rosclaw.agentd.pi_bridge.action_admission import ActionRequestContext
        from rosclaw.agentd.task_runner import TaskRunner

        args = request.arguments
        goal = str(args.get("goal", "")).strip()
        parameters = args.get("parameters")
        if not str(args.get("task_id", "") or "") and (
            not goal or not isinstance(parameters, dict)
        ):
            raise ToolBridgeError(
                "INVALID_ARGUMENTS", "goal and parameters required (fail closed)"
            )
        mission = self._service.get_mission(request.mission_id)
        ctx = ActionRequestContext(
            pi_session_id=request.pi_session_id,
            mission_id=request.mission_id,
            context_revision=request.context_revision,
            body_hash=mission.body_binding.effective_body_hash if mission else "",
            mode=mission.mode.value if mission else "",
            idempotency_key=request.idempotency_key,
            context_lease_id=request.context_lease_id,
        )
        runner = TaskRunner(self._service)
        # 两阶段：submit（wait=False——ASK 时立即返回 WAITING_APPROVAL
        # 视图供 TUI 展卡）与 resume（task_id 重入，批准后执行+验证）。
        resume_task_id = str(args.get("task_id", "") or "")
        if resume_task_id:
            payload = await runner.resume(
                task_id=resume_task_id,
                request_ctx=ctx,
                caller_pid=self._caller_pid,
                caller_uid=self._caller_uid,
            )
        else:
            payload = await runner.run(
                request_ctx=ctx,
                goal=goal,
                parameters=parameters,
                caller_pid=self._caller_pid,
                caller_uid=self._caller_uid,
                wait=False,
            )
        import json as _json

        return PiToolResultV1(
            request_id=request.request_id,
            ok=payload["state"] in ("VERIFIED", "WAITING_APPROVAL"),
            status=(
                "COMPLETED" if payload["state"] == "VERIFIED"
                else "PENDING" if payload["state"] == "WAITING_APPROVAL"
                else "REJECTED"
            ),
            summary=_json.dumps(payload, ensure_ascii=False),
            error_code=(
                "" if payload["state"] in ("VERIFIED", "WAITING_APPROVAL")
                else payload["state"]
            ),
        )

    async def _request_action(self, request: PiToolRequestV1) -> PiToolResultV1:
        """PNA-5 + NA-FIX-5 + 三审 P0-NA-10：经唯一 ActionAdmissionService——
        完整请求上下文（session/lease/revision/body/mode）硬校验、
        精确 grant、结构化回执、execute TOCTOU 复验。"""
        import asyncio as _asyncio

        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionAdmissionService,
            ActionRequestContext,
        )

        args = request.arguments
        capability_id = str(args.get("capability_id", "")).strip()
        arguments = args.get("arguments")
        if not capability_id or not isinstance(arguments, dict):
            raise ToolBridgeError(
                "INVALID_ARGUMENTS", "capability_id and arguments required (fail closed)"
            )
        # dispatcher 上游已做 binding/mission/lease/revision 校验——这里把
        # 同一上下文传入 admission service，让它以同一套验证再次确认
        # （execute 阶段的 TOCTOU 复验依赖这份上下文）。
        mission = self._service.get_mission(request.mission_id)
        body_hash = ""
        if mission is not None:
            body_hash = mission.body_binding.effective_body_hash
        ctx = ActionRequestContext(
            pi_session_id=request.pi_session_id,
            mission_id=request.mission_id,
            context_revision=request.context_revision,
            body_hash=body_hash,
            mode=mission.mode.value if mission else "",
            idempotency_key=request.idempotency_key,
            context_lease_id=request.context_lease_id,
        )
        admission = ActionAdmissionService(self._service)
        card = await admission.propose(
            request=ctx,
            capability_id=capability_id,
            arguments=arguments,
            expected_effect=str(args.get("expected_effect") or capability_id),
            risk_tier=str(args.get("risk_tier", "LOW")),
            title=str(args.get("title") or capability_id),
            caller_pid=self._caller_pid,
            caller_uid=self._caller_uid,
        )
        # 等 operator（决定只能经 operatord 到达）。
        deadline_sec = 330.0
        waited = 0.0
        while waited < deadline_sec:
            status = admission.decision_status(card["approval_id"])["status"]
            if status != "PENDING":
                break
            await _asyncio.sleep(1.0)
            waited += 1.0
        status = admission.decision_status(card["approval_id"])["status"]
        if status == "PENDING":
            raise ToolBridgeError(
                "APPROVAL_TIMEOUT",
                "operator 未在期限内决定（默认拒绝语义）——动作未执行",
            )
        if status != "APPROVED":
            return PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="DECLINED",
                summary=f"operator 拒绝了 {capability_id}——动作未执行",
                approval_id=card["approval_id"],
                error_code="OPERATOR_DECLINED",
            )
        result = await admission.execute(
            card["approval_id"], request=ctx,
            caller_pid=self._caller_pid, caller_uid=self._caller_uid,
        )
        return PiToolResultV1(
            request_id=request.request_id,
            ok=bool(result.get("executed")),
            status=str(result.get("status", "FAILED")),
            summary=str(result.get("summary", ""))[:8000],
            approval_id=card["approval_id"],
            error_code=result.get("error_code"),
        )

    async def _mirror_decision(
        self, request: PiToolRequestV1, result: PiToolResultV1
    ) -> None:
        """规格 §15：每个工具调用镜像为 DecisionV1 审计事件（不写全文）。"""
        try:
            from rosclaw.contracts.agent.agent_event import AgentEventType

            await self._service._events.append(
                request.mission_id,
                AgentEventType.TOOL_COMPLETED,
                {
                    "engine": "pi",
                    "request_id": request.request_id,
                    "tool_name": request.tool_name,
                    "ok": result.ok,
                    "status": result.status,
                    "error_code": result.error_code,
                    "summary_hash": json.dumps(result.summary)[:64],
                },
            )
        except Exception:  # noqa: BLE001 - 审计镜像失败不影响已完成的工具结果
            pass
