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

import json
from datetime import UTC, datetime
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
    "rosclaw_inspect": "read",
    "rosclaw_fail_safe": "control",
    "rosclaw_request_action": "physical_action",
    # 八审 P0-5：任务级入口——确定性编译器编排，模型只交 TaskSpec。
    "rosclaw_task": "task",
    # 十五审 PR-RF-1/RF-2：治理工具（无为而治）——模型只交目标合同，
    # 观察/steer/回答/暂停/恢复/取消都作用于同一 owning execution。
    # PR-H3：process 工具（长进程 Operation——立即返回/事件流/终态
    # followUp）。
    "rosclaw_process_start": "delegate",
    "rosclaw_process_status": "read",
    "rosclaw_process_output": "read",
    "rosclaw_process_stop": "delegate",
    # PR-H4：Product Pack——交付登记/收尾/阻塞（验收决定终态）。
    "rosclaw_artifact_register": "delegate",
    "rosclaw_task_finish": "task",
    "rosclaw_task_blocked": "delegate",
    # PR-H5：统一执行入口 + operation 等待/停止。
    "rosclaw_execute": "task",
    "rosclaw_wait_operation": "read",
    "rosclaw_stop_operation": "delegate",
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


def _envelope_result(request: PiToolRequestV1, envelope) -> PiToolResultV1:
    """N5B：canonical envelope → 模型可见投影（status + capability_id +
    value）；FAILED/BLOCKED 以稳定错误码诚实抛出。"""
    if envelope.status.value == "SUCCEEDED":
        projection = {
            "status": envelope.status.value,
            "capability_id": envelope.capability_id,
            "value": envelope.value,
        }
        if envelope.artifact_refs:
            projection["artifact_refs"] = list(envelope.artifact_refs)
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status="COMPLETED",
            summary=json.dumps(projection, ensure_ascii=False)[:8000],
        )
    error = envelope.error
    code = error.code if error else "EXECUTOR_ERROR"
    message = error.message if error else f"capability {envelope.status.value}"
    retryable = error.retryable if error else False
    raise ToolBridgeError(code, message[:400], retryable=retryable)


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

    def _note_embodiment_use(self, request: PiToolRequestV1) -> None:
        """N4.1：具身执行工具落账——行为任务的判定依据是实际调用，
        不是 body 在场。"""
        task = self._service._task_kernel.active_task_for(
            request.mission_id, request.pi_session_id
        )
        if task is not None:
            self._service._task_kernel.note_tool_use(
                str(task["task_id"]), request.tool_name
            )

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
            envelope = await service._tool_catalog.execute_v2(
                request.request_id, capability_id, dict(args.get("arguments", {}))
            )
            return _envelope_result(request, envelope)
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
            envelope = await service._tool_catalog.execute_v2(
                request.request_id, capability_id, dict(args.get("arguments", {}))
            )
            return _envelope_result(request, envelope)
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
            self._note_embodiment_use(request)
            return await self._request_action(request)
        if name == "rosclaw_task":
            self._note_embodiment_use(request)
            return await self._task(request)
        # PR-H3：process 工具——长进程 = Operation（立即返回，事件流
        # 可查，终态 followUp 一次）。
        # PR-H5：统一执行入口 + operation 控制。
        if name == "rosclaw_execute":
            self._note_embodiment_use(request)
            return await self._execute(request)
        if name == "rosclaw_wait_operation":
            return await self._wait_operation(request)
        if name == "rosclaw_stop_operation":
            return await self._process_stop(request)
        # PR-H4：Product Pack。
        if name == "rosclaw_artifact_register":
            return await self._artifact_register(request)
        if name == "rosclaw_task_finish":
            return await self._task_finish(request)
        if name == "rosclaw_task_blocked":
            return await self._task_blocked(request)
        if name == "rosclaw_process_start":
            return await self._process_start(request)
        if name == "rosclaw_process_status":
            return await self._process_status(request)
        if name == "rosclaw_process_output":
            return await self._process_output(request)
        if name == "rosclaw_process_stop":
            return await self._process_stop(request)
        # 十五审 PR-RF-1/RF-2：治理工具——同一 owning execution 的
        # 提交/观察/steer/回答/暂停/恢复/取消。
        if name == "rosclaw_inspect":
            # PR-N3：生态索引自检——程序探测（read 类，免任务绑定）。
            from rosclaw.agentd.pi_bridge.server import PiBridgeServer  # noqa: F401
            from rosclaw.cognition.index.query import robot_chain, search
            from rosclaw.cognition.inspect_cli import ensure_index, inspect_self

            kind = str(request.arguments.get("kind", "self"))
            query = str(request.arguments.get("query", ""))
            if kind == "self":
                info = inspect_self(self._service._home)
                return PiToolResultV1(
                    request_id=request.request_id, ok=True, status="COMPLETED",
                    summary=json.dumps(info, ensure_ascii=False),
                )
            idx = ensure_index(self._service._home)
            if kind == "robot":
                chain = robot_chain(idx, query)
                if chain is None:
                    return PiToolResultV1(
                        request_id=request.request_id, ok=False, status="FAILED",
                        summary=f"未知机器人 {query!r}（索引无权威链）",
                        error_code="UNKNOWN_ROBOT",
                    )
                return PiToolResultV1(
                    request_id=request.request_id, ok=True, status="COMPLETED",
                    summary=json.dumps(chain, ensure_ascii=False),
                )
            hits = search(idx, query or kind, limit=20)
            return PiToolResultV1(
                request_id=request.request_id, ok=True, status="COMPLETED",
                summary=json.dumps({"hits": hits}, ensure_ascii=False),
            )
        if name == "rosclaw_fail_safe":
            await service.cancel(request.mission_id)
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary="fail-safe: 当前回合已请求取消；E-Stop 请走独立 operator 路径",
            )
        raise ToolBridgeError("TOOL_UNKNOWN", f"unhandled tool {name!r}")

    async def _execute(self, request: PiToolRequestV1) -> PiToolResultV1:
        """PR-H5（§10.2）：统一能力执行入口——按 execution_class 路由：
        OBSERVE→观测 / COMPUTE→内联免审批 / PHYSICAL_ACTION→同一
        admission 链（SIM 安全自动、REAL 永远 rosclawd+审批）。未知
        ID 诚实拒绝（不猜不编）。"""
        capability_id = str(request.arguments.get("capability_id", "")).strip()
        if not capability_id:
            raise ToolBridgeError("INVALID_ARGUMENTS", "capability_id required")
        await self._service._ensure_mcp_discovered()
        descriptor = None
        for d in self._service._tool_catalog.list():
            if d.tool_id == capability_id:
                descriptor = d
                break
        if descriptor is None:
            raise ToolBridgeError(
                "UNKNOWN_CAPABILITY",
                f"未知能力 {capability_id!r}——用 rosclaw_capabilities 查"
                "当前 body 的精确 ID（不要编造）",
            )
        cls = descriptor.execution_class.value
        if cls == "OBSERVE":
            # 复用观测路径（同一分发器，克隆请求换工具名——幂等键加
            # 后缀避免与外层 execute 互吞）。
            return await self._execute_validated(
                request.model_copy(update={
                    "tool_name": "rosclaw_observe",
                    "idempotency_key": request.idempotency_key + ":observe",
                })
            )
        if cls == "COMPUTE":
            return await self._execute_validated(
                request.model_copy(update={
                    "tool_name": "rosclaw_compute",
                    "idempotency_key": request.idempotency_key + ":compute",
                })
            )
        # PHYSICAL_ACTION：同一 admission 链（policy AUTO/ASK/DENY——
        # REAL 永远 rosclawd+operator；execute 不是绕过的旁路）。
        return await self._request_action(request)

    async def _wait_operation(self, request: PiToolRequestV1) -> PiToolResultV1:
        """等有界（模型主动等——默认上限 120s；终态返回末段输出）。"""
        import asyncio as _aio

        operation_id = str(request.arguments.get("operation_id", ""))
        op = self._service._operation_manager.get(operation_id)
        if not op:
            raise ToolBridgeError("NOT_FOUND", f"unknown operation {operation_id!r}")
        timeout = min(float(request.arguments.get("timeout_sec", 120) or 120), 120)
        deadline = _aio.get_running_loop().time() + timeout
        while _aio.get_running_loop().time() < deadline:
            op = self._service._operation_manager.get(operation_id)
            if op["state"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                events = self._service._operation_manager.events_since(
                    op["task_id"], 0
                )
                tail = "".join(
                    str(e["payload"].get("text", ""))
                    for e in events
                    if e["event_type"] == "operation.output"
                    and e.get("operation_id") == operation_id
                )[-1500:]
                return PiToolResultV1(
                    request_id=request.request_id,
                    ok=op["state"] == "SUCCEEDED",
                    status=str(op["state"]),
                    summary=f"operation 终态 {op['state']}\n{tail}",
                )
            await _aio.sleep(0.5)
        return PiToolResultV1(
            request_id=request.request_id, ok=True, status="RUNNING",
            summary=f"operation 仍在运行（等待 {timeout}s 未见终态——"
            "可以稍后 process_status 再查，或等完成通知）",
        )

    async def _artifact_register(
        self, request: PiToolRequestV1
    ) -> PiToolResultV1:
        """PR-H4：交付物登记（实读文件算 hash——口头提到不算）。"""
        kernel = self._service._task_kernel
        task = kernel.active_task_for(request.mission_id, request.pi_session_id)
        if task is None:
            raise ToolBridgeError("NO_ACTIVE_TASK", "无活跃任务")
        path = str(request.arguments.get("path", ""))
        if not path:
            raise ToolBridgeError("INVALID_ARGUMENTS", "path required")
        # PR-N0：确定性解析——相对路径依次按会话 cwd（模型实际工作
        # 目录）与任务 workspace 解析成绝对路径后交 kernel；两处都不
        # 存在时报错列出两个实际根（禁止反复猜路径）。真正的单一事实
        # 源（ActiveTaskContext）在 PR-N1。
        from pathlib import Path as _Path

        session_cwd = str(request.arguments.get("cwd", "") or "")
        task_ws = str(task["workspace_path"])
        if _Path(path).is_absolute():
            resolved = path
        elif session_cwd and (_Path(session_cwd) / path).exists():
            resolved = str(_Path(session_cwd) / path)
        elif (_Path(task_ws) / path).exists():
            resolved = str(_Path(task_ws) / path)
        else:
            raise ToolBridgeError(
                "ARTIFACT_MISSING",
                f"artifact 不存在: {path}（已查会话目录 {session_cwd or '—'} "
                f"与任务工作区 {task_ws}）",
            )
        try:
            artifact = kernel.register_artifact(
                task_id=task["task_id"], path=resolved,
                media_type=str(request.arguments.get("media_type", "application/octet-stream")),
                producer="model:rosclaw_artifact_register",
            )
        except ValueError as exc:
            raise ToolBridgeError("ARTIFACT_MISSING", str(exc)) from exc
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True, status="REGISTERED",
            summary=f"交付物已登记：{_Path(artifact['path']).name}（{artifact['size_bytes']}B）artifact_id={artifact['artifact_id']}",
        )

    async def _task_finish(self, request: PiToolRequestV1) -> PiToolResultV1:
        """PR-H4：FinishRequest——验收真跑决定终态（模型自述不算数）。
        REPAIR_REQUIRED 回同一 session（task 保持活跃）。"""
        kernel = self._service._task_kernel
        task = kernel.active_task_for(request.mission_id, request.pi_session_id)
        if task is None:
            raise ToolBridgeError("NO_ACTIVE_TASK", "无活跃任务")
        result = kernel.finish_task(
            task_id=task["task_id"],
            summary=str(request.arguments.get("summary", "")),
            artifact_ids=[
                str(a) for a in (request.arguments.get("artifact_ids") or [])
            ],
        )

        if result["status"] == "SUCCEEDED":
            return PiToolResultV1(
                request_id=request.request_id, ok=True, status="SUCCEEDED",
                summary=f"验收通过——任务完成（{result['verification_id']}）",
            )
        failures = "；".join(result.get("failures", []))[:300]
        return PiToolResultV1(
            request_id=request.request_id, ok=False, status="REPAIR_REQUIRED",
            summary=f"验收未过，同一任务内修复后重试：{failures}",
            error_code="VERIFICATION_FAILED", retryable=True,
        )

    async def _task_blocked(self, request: PiToolRequestV1) -> PiToolResultV1:
        """PR-H4：诚实阻塞（稳定原因码 + 恢复动作）。"""
        kernel = self._service._task_kernel
        task = kernel.active_task_for(request.mission_id, request.pi_session_id)
        if task is None:
            raise ToolBridgeError("NO_ACTIVE_TASK", "无活跃任务")
        reason_code = str(request.arguments.get("reason_code", "")).strip()
        if not reason_code:
            raise ToolBridgeError("INVALID_ARGUMENTS", "reason_code required")
        kernel.block_task(
            task_id=task["task_id"], reason_code=reason_code,
            detail=str(request.arguments.get("detail", "")),
            recovery=[
                str(r) for r in (request.arguments.get("recovery") or [])
            ],
        )
        return PiToolResultV1(
            request_id=request.request_id, ok=True, status="BLOCKED",
            summary=f"任务已标记阻塞：{reason_code}",
        )

    async def _process_start(self, request: PiToolRequestV1) -> PiToolResultV1:
        """PR-H3：长进程 → Operation（立即返回 operation_id）。"""
        command = str(request.arguments.get("command", "")).strip()
        if not command:
            raise ToolBridgeError("INVALID_ARGUMENTS", "command required")
        kernel = self._service._task_kernel
        task = kernel.active_task_for(request.mission_id, request.pi_session_id)
        if task is None:
            raise ToolBridgeError(
                "NO_ACTIVE_TASK", "无活跃任务——先发送任务消息（输入事务绑定）"
            )
        op = await self._service._operation_manager.start(
            task_id=task["task_id"],
            attempt_id="main",
            kind="process",
            argv=["sh", "-c", command],
            cwd=task["workspace_path"],
        )
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status="STARTED",
            summary=(
                f"Operation 已启动：{op['operation_id']}（后台运行——"
                "progress/输出经 process_output 查看；完成时会收到一次"
                "通知，不要在回合里死等）"
            ),
        )

    async def _process_status(self, request: PiToolRequestV1) -> PiToolResultV1:
        op = self._service._operation_manager.get(
            str(request.arguments.get("operation_id", ""))
        )
        if not op:
            raise ToolBridgeError("NOT_FOUND", "unknown operation")
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status=str(op["state"]),
            summary=(
                f"operation {op['operation_id']}: {op['state']}"
                + (f"（{op['failure_code']}）" if op.get("failure_code") else "")
            ),
        )

    async def _process_output(self, request: PiToolRequestV1) -> PiToolResultV1:
        operation_id = str(request.arguments.get("operation_id", ""))
        tail = min(int(request.arguments.get("tail", 50) or 50), 200)
        op = self._service._operation_manager.get(operation_id)
        if not op:
            raise ToolBridgeError("NOT_FOUND", "unknown operation")
        events = self._service._operation_manager.events_since(
            op["task_id"], 0
        )
        lines = [
            str(e["payload"].get("text", ""))
            for e in events
            if e["event_type"] == "operation.output"
            and e.get("operation_id") == operation_id
        ]
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status=str(op["state"]),
            summary="".join(lines[-tail:])[-3000:] or "（暂无输出）",
        )

    async def _process_stop(self, request: PiToolRequestV1) -> PiToolResultV1:
        operation_id = str(request.arguments.get("operation_id", ""))
        op = self._service._operation_manager.get(operation_id)
        if not op:
            raise ToolBridgeError("NOT_FOUND", "unknown operation")
        await self._service._operation_manager.cancel(
            operation_id, reason="model_request"
        )
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True, status="CANCELLED",
            summary=f"operation {operation_id} 已取消（账本先行）",
        )

    async def _task(self, request: PiToolRequestV1) -> PiToolResultV1:
        """rosclaw_task（PR-H9 重接）：确定性 SIM 任务闭环——
        SimTrajectoryService 直跑（规划→ rollout → 渲染 → 跟踪验证，
        总纲 §5.1），产物登记进 TaskKernel 产物账本。

        TaskRunner/task_records 旧链已删除；SIM 动力学由托管
        rosclaw-simulation runtime 预置（不可用时诚实 BLOCKED，
        不是渲染时裸 ModuleNotFoundError）。
        """
        import asyncio as _asyncio
        import json as _json

        args = request.arguments
        goal = str(args.get("goal", "")).strip()
        parameters = args.get("parameters")
        if goal not in ("simulate_trajectory", "draw_shape") or not isinstance(
            parameters, dict
        ):
            raise ToolBridgeError(
                "UNKNOWN_CAPABILITY",
                f"未知确定性任务 {goal!r}——当前支持 simulate_trajectory"
                "（其余长任务走 rosclaw_execute/process_start）",
            )
        service = self._service
        mission = service.get_mission(request.mission_id)
        if mission is not None and mission.mode.value != "SIMULATION":
            raise ToolBridgeError(
                "MODE_DENIED",
                f"rosclaw_task 仅 SIMULATION——当前 {mission.mode.value}",
            )
        from rosclaw.agentd.runtime_manager import RuntimeNotReadyError
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        try:
            service._runtime_manager.ensure("rosclaw-simulation")
        except RuntimeNotReadyError as exc:
            return PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="REJECTED",
                summary=f"RUNTIME_NOT_READY：{exc}"[:400],
                error_code="RUNTIME_NOT_READY",
            )
        sim = SimTrajectoryService(
            service._home, runtime_manager=service._runtime_manager
        )
        acceptance = args.get("acceptance") or {}
        plan = await _asyncio.to_thread(
            sim.generate_planar_path,
            shape=str(parameters.get("shape", "star5")),
            center_m=parameters.get("center_m") or [0.35, 0.25, 0.30],
            scale_m=float(parameters.get("scale_m", parameters.get("radius_m", 0.10))),
            plane=str(parameters.get("plane", "xy")),
            max_segment_m=float(parameters.get("max_segment_m", 0.03)),
        )
        result = await _asyncio.to_thread(
            sim.simulate_cartesian_trajectory, plan["plan_id"]
        )
        render = await _asyncio.to_thread(
            sim.render_trace, result["trace_id"], format="gif"
        )
        threshold = float(acceptance.get("max_tracking_error_m", 0.05))
        verify = await _asyncio.to_thread(
            sim.verify_tracking, result["trace_id"],
            max_tracking_error_m=threshold,
        )
        min_frames = int(acceptance.get("animation_min_frames", 30))
        failures: list[str] = []
        if verify["verdict"] != "PASS":
            failures.append(
                f"跟踪误差 {verify['metrics']['max_error_m']}m > {threshold}m"
            )
        if render["artifact"]["frames"] < min_frames:
            failures.append(
                f"动画帧数 {render['artifact']['frames']} < {min_frames}"
            )
        if not result.get("is_safe"):
            failures.append(f"rollout 不安全: {result.get('violations')}")
        metrics = verify["metrics"]
        gif_path = str(render["artifact"]["path"])
        # 产物登记进 kernel 账本（当前会话有活跃任务时——登记才算交付）。
        task = service._task_kernel.active_task_for(
            request.mission_id, request.pi_session_id
        )
        if task is not None:
            for path, media in (
                (gif_path, "image/gif"),
                (str(result["artifacts"]["trace_json"]), "application/json"),
            ):
                import contextlib as _cl

                with _cl.suppress(ValueError):
                    # 产物文件缺失由验收 failures 表达；受信管道登记
                    # （producer=kernel——PR-N0）+ 资源证明元数据
                    # （N4.1——producer 只是来源身份，资源证明才算数）。
                    service._task_kernel.register_artifact(
                        task_id=task["task_id"], path=path, media_type=media,
                        producer="kernel:sim_pipeline",
                        metadata={"resource": result.get("resource") or {}},
                    )
        state = "VERIFIED" if not failures else "FAILED"
        payload = {
            "state": state,
            "goal": goal,
            "artifacts": {
                "gif": gif_path,
                **{k: str(v) for k, v in result["artifacts"].items()},
                "metrics": metrics,
                "evidence_level": "SIM_DYN_ROLLOUT",
            },
            "failures": failures,
            "verification": {
                "verdict": "PASS" if not failures else "FAIL",
                "max_error_m": metrics["max_error_m"],
                "threshold_m": threshold,
                "frames": render["artifact"]["frames"],
                "min_frames": min_frames,
            },
            # WP06：证据等级与局限必须显式——动力学 rollout 自洽，
            # 不能证明真机执行。
            "user_view": (
                f"动力学仿真 rollout 完成：最大跟踪误差 "
                f"{metrics['max_error_m'] * 1000:.0f}mm，动画 {gif_path}。"
                "证据等级 SIM_DYN_ROLLOUT——仿真动力学自洽，"
                "不能证明真机执行效果。"
                if not failures else
                f"验收未过：{'；'.join(failures)}（证据等级 SIM_DYN_ROLLOUT）"
            ),
            "evidence_level": "SIM_DYN_ROLLOUT",
        }
        return PiToolResultV1(
            request_id=request.request_id,
            ok=not failures,
            status="COMPLETED" if not failures else "REJECTED",
            summary=_json.dumps(payload, ensure_ascii=False),
            error_code="" if not failures else "ACCEPTANCE_FAILED",
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
