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
    # 十五审 PR-RF-1/RF-2：治理工具（无为而治）——模型只交目标合同，
    # 观察/steer/回答/暂停/恢复/取消都作用于同一 owning execution。
    # PR-H3：process 工具（长进程 Operation——立即返回/事件流/终态
    # followUp）。
    "rosclaw_process_start": "delegate",
    "rosclaw_process_status": "read",
    "rosclaw_process_output": "read",
    "rosclaw_process_stop": "delegate",
    # PR-H4：Product Pack——交付登记/收尾/阻塞（验收决定终态）。
    # P0-D：rosclaw_deliver 是模型面唯一幂等交付入口。
    "rosclaw_deliver": "delegate",
    "rosclaw_artifact_register": "delegate",
    "rosclaw_task_finish": "task",
    "rosclaw_task_blocked": "delegate",
    # PR-H5：统一执行入口 + operation 等待/停止。
    "rosclaw_execute": "task",
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


#: 自动登记扫描的 artifact 键（0824 总纲 §8.1：可信 capability
#: 产生的 artifact 必须自动登记——模型不需要也不应该手动 deliver
#: capability 产物）。
_ARTIFACT_SCALAR_KEYS = ("artifact", "mp4_artifact")
_FORMAT_MEDIA = {
    "gif": "image/gif", "mp4": "video/mp4",
    "json": "application/json", "csv": "text/csv",
}


def _auto_register_artifacts(
    service, request: PiToolRequestV1, value: object
) -> list[dict]:
    """capability 产物自动登记（producer=kernel:capability:<id>，
    幂等——同内容重复登记返回同一 ArtifactRef）。

    产物生成即 effectful（0824 总纲 §8.1：可信 capability 产生的
    artifact 必须自动登记）——首次产出时原子 admission（模型不
    需要也不应该手动 deliver capability 产物）。

    R0-4：返回登记的 ArtifactRef 列表（id/kind/media/digest/
    open_command）——ToolResult 投影必须带回模型（登记了但模型
    看不到 = 交付失败）。"""
    if not isinstance(value, dict):
        return []
    kernel = service._task_kernel
    mission = service.get_mission(request.mission_id)
    bound = kernel.ensure_task_for_effect(
        mission_id=request.mission_id,
        session_ref=request.pi_session_id,
        backend_native_id=request.pi_session_id,
        cwd="",
        mode=mission.mode.value if mission else "SIMULATION",
        body_id=(mission.body_binding.body_id if mission else ""),
    )
    task = kernel.get_task(str(bound["task_id"]))
    if task is None:
        return []
    candidates: list[dict] = []
    for key in _ARTIFACT_SCALAR_KEYS:
        item = value.get(key)
        if isinstance(item, dict) and item.get("path"):
            candidates.append(item)
    nested = value.get("artifacts")
    if isinstance(nested, dict):
        for item in nested.values():
            if isinstance(item, dict) and item.get("path"):
                candidates.append(item)
            elif isinstance(item, str) and item.endswith((".json", ".csv")):
                candidates.append({
                    "path": item,
                    "format": item.rsplit(".", 1)[-1],
                })
    # 归因用真实 capability_id（不是 wire 入口名）。
    capability_id = str(
        (request.arguments or {}).get("capability_id") or request.tool_name
    )
    registered: list[dict] = []
    for item in candidates:
        path = str(item["path"])
        fmt = str(item.get("format") or path.rsplit(".", 1)[-1])
        try:
            record = kernel.register_artifact(
                task_id=str(task["task_id"]), path=path,
                media_type=_FORMAT_MEDIA.get(fmt, "application/octet-stream"),
                producer=f"kernel:capability:{capability_id}",
            )
        except ValueError:
            continue  # 文件缺失等由验收表达——登记不阻断工具结果
        registered.append({
            "artifact_id": str(record["artifact_id"]),
            "media_type": str(record["media_type"]),
            "path": str(record["path"]),
            "size_bytes": int(record["size_bytes"]),
            "digest": str(record["sha256"]),
            "open_command": f"rosclaw artifact open {record['artifact_id']}",
        })
    return registered


def _envelope_result(
    request: PiToolRequestV1, envelope, *, auto_refs: list[dict] | None = None
) -> PiToolResultV1:
    """N5B：canonical envelope → 模型可见投影（status + capability_id +
    value）；FAILED/BLOCKED 以稳定错误码诚实抛出。

    R0-4：artifact_refs = 能力声明 refs + 内核自动登记 refs（按
    artifact_id 去重）——登记了但模型看不到 = 交付失败。"""
    if envelope.status.value == "SUCCEEDED":
        projection = {
            "status": envelope.status.value,
            "capability_id": envelope.capability_id,
            "value": envelope.value,
        }
        refs: list[dict] = []
        seen: set[str] = set()
        for ref in [*(auto_refs or []), *(envelope.artifact_refs or [])]:
            if not isinstance(ref, dict):
                continue
            key = str(ref.get("artifact_id") or ref.get("path") or "")
            if key and key in seen:
                continue
            seen.add(key)
            refs.append(ref)
        if refs:
            projection["artifact_refs"] = refs
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


#: 熔断指纹的易变参数（R0-3，0826 体验审计 §5.R0-3）：基础设施
#: 故障的重试熔断按 capability+参数实质判定——换 camera 不绕过
#: renderer 熔断（事故实证：同一渲染故障换 camera 重试三次）。
_VOLATILE_FINGERPRINT_KEYS = frozenset({"camera"})


def _failure_fingerprint(request: PiToolRequestV1) -> str:
    def _strip(value: object) -> object:
        if isinstance(value, dict):
            return {
                k: _strip(v)
                for k, v in value.items()
                if k not in _VOLATILE_FINGERPRINT_KEYS
            }
        if isinstance(value, list):
            return [_strip(v) for v in value]
        return value

    return request.tool_name + ":" + json.dumps(
        _strip(request.arguments), sort_keys=True, ensure_ascii=False
    )


#: R0-9（0826 体验审计 §5.R0-9）：transient 错误码（可安全重试——
#: 状态已变化/等待外部事件，不记熔断也不计预算）。
_TRANSIENT_CODES = frozenset({
    "CONTEXT_NOT_FRESH",
    "CONTEXT_HASH_MISMATCH",
    "NEEDS_REPLAN",
    "CONTEXT_LEASE_REQUIRED",
    "CAPABILITY_SNAPSHOT_CHANGED",
    "WAITING_APPROVAL",
})

#: 基础设施/配置错误前缀（模型重试预算为 0——重试不会成功，
#: 修复路径在 recovery_action）。
_INFRA_PREFIXES = (
    "RENDER_", "RUNTIME_", "TRANSPORT_", "PI_ENGINE",
    "WORLD_ASSET_", "TOOL_ASSET_", "RESOURCE_",
)


def _error_envelope_details(code: str, *, tool_name: str = "") -> dict:
    """ErrorEnvelope 语义（R0-9）：scope / attempt_budget /
    recovery_action——基础设施/配置/确定性错误模型重试预算为 0；
    只有明确 transient 且状态已变化时可重试。

    0901 P0-3：tool_name 进恢复提示——漂移的 task.*/artifact.*
    名字指向真实只读工具（不再泛泛"查注册表"）。"""
    from rosclaw.agentd.tooling.recovery import recovery_hint

    if code in _TRANSIENT_CODES:
        return {
            "scope": "transient",
            "attempt_budget": 1,
            "retry_after_condition": "状态已变化（context/审批/快照刷新后）",
            "recovery_action": recovery_hint(code, context=tool_name),
        }
    scope = (
        "infrastructure"
        if code.startswith(_INFRA_PREFIXES)
        else "deterministic"
    )
    return {
        "scope": scope,
        "attempt_budget": 0,
        "retry_after_condition": "",
        "recovery_action": recovery_hint(code, context=tool_name),
    }


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
        # 只是效率护栏）。R0-3：指纹剔除易变参数（camera 等）——
        # 基础设施故障换参数不能绕过。
        fingerprint = _failure_fingerprint(request)
        failures = getattr(self._service, "_tool_fail_fingerprints", None)
        if failures is None:
            failures = self._service._tool_fail_fingerprints = {}
        if failures.get(fingerprint):
            return PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="REJECTED",
                summary="同一调用已失败过一次——原样重复不会成功。请改变参数、"
                "换用其他通用原语工具组合，或诚实报告无法完成。",
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
                details=_error_envelope_details(
                    exc.code,
                    tool_name=str(request.tool_name or ""),
                ),
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
        # 4.5 PR-N5C 单一 Effect Contract：rosclaw_* 工具在执行前解析
        # 并冻结 effect（tool.effect_resolved 事件，含 capability/
        # arguments digest）；不可解析 fail closed——审批/并发/Verifier
        # 读冻结结果，不再有第二份手写分类。
        if request.tool_name.startswith("rosclaw_"):
            import contextlib

            from rosclaw.agentd.tooling.effect_resolver import (
                GENERIC_ENTRY_TOOLS,
                EffectResolver,
                EffectUnresolvableError,
            )
            from rosclaw.contracts.agent.agent_event import AgentEventType

            # 通用入口的解析依赖完整注册表——先确保 MCP 发现完成
            # （与 _execute 分支的发现顺序一致；幂等）。
            if request.tool_name in GENERIC_ENTRY_TOOLS:
                await service._ensure_mcp_discovered()
            # PR-N5D：snapshot digest 校验——调用方钉住的工具面与当前
            # registry 不一致时不静默换工具（下一步重新规划一次）。
            claimed_digest = str(
                (request.arguments or {}).get("snapshot_digest", "")
            )
            if claimed_digest:
                current = service.capability_snapshot(mission)
                if claimed_digest != current.digest:
                    raise ToolBridgeError(
                        "CAPABILITY_SNAPSHOT_CHANGED",
                        "capability snapshot changed（registry 在本回合内变化）"
                        f"——current digest {current.digest[:23]}…；请重新获取 "
                        "pi.capability.snapshot 并按新工具面重新规划一次",
                        retryable=True,
                    )
            try:
                frozen = EffectResolver(service._tool_catalog).resolve(
                    request.tool_name, dict(request.arguments or {})
                )
            except EffectUnresolvableError as exc:
                raise ToolBridgeError(
                    "EFFECT_UNRESOLVABLE", str(exc)[:400]
                ) from exc
            with contextlib.suppress(Exception):
                await service._events.append(
                    request.mission_id,
                    AgentEventType.TOOL_EFFECT_RESOLVED,
                    frozen.to_event_payload(),
                )
        # 5. 分发。
        return await self._dispatch(request)

    def _coordinator_consider(
        self, request: PiToolRequestV1, result: PiToolResultV1
    ) -> None:
        """P0-D：effectful 完成后的自动收尾评估——outcome 摘要附进
        工具结果 details（模型看到结果，无需新回合）。"""
        try:
            kernel = self._service._task_kernel
            task = kernel.latest_task_for(
                request.mission_id, request.pi_session_id
            )
            if task is None:
                return
            from rosclaw.task_kernel.coordinator import TaskCoordinator

            outcome = TaskCoordinator(kernel).consider(str(task["task_id"]))
            if outcome is not None:
                details = dict(result.details or {})
                details["task_outcome"] = {
                    "lifecycle": outcome["lifecycle"],
                    "verification": outcome["verification"],
                    "delivery": outcome["delivery"],
                }
                result.details = details
        except Exception:
            # 收尾评估失败不影响工具结果本身（下轮再评估）。
            return

    def _ensure_task_for_effect(self, request: PiToolRequestV1) -> None:
        """P0-C（0824 总纲 §6.2）：effectful wire 工具执行前的原子
        admission——缺动机输入诚实拒绝（INPUT_MOTIVATION_MISSING）。
        mode 取 mission 权威值（request 不携带 mode 字段）。"""
        kernel = self._service._task_kernel
        mission = self._service.get_mission(request.mission_id)
        kernel.ensure_task_for_effect(
            mission_id=request.mission_id,
            session_ref=request.pi_session_id,
            backend_native_id=request.pi_session_id,
            cwd="",
            mode=mission.mode.value if mission else "SIMULATION",
            # N0 熔断：body 缺省回落 mission 绑定（执行面首条即武装）。
            body_id=(mission.body_binding.body_id if mission else ""),
        )

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
            auto_refs = _auto_register_artifacts(
                service, request, envelope.value
            )
            return _envelope_result(request, envelope, auto_refs=auto_refs)
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
            auto_refs = _auto_register_artifacts(
                service, request, envelope.value
            )
            return _envelope_result(request, envelope, auto_refs=auto_refs)
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
            self._ensure_task_for_effect(request)
            return await self._request_action(request)
        # PR-H3：process 工具——长进程 = Operation（立即返回，事件流
        # 可查，终态 followUp 一次）。
        # PR-H5：统一执行入口 + operation 控制。
        if name == "rosclaw_execute":
            self._note_embodiment_use(request)
            self._ensure_task_for_effect(request)
            result = await self._execute(request)
            self._coordinator_consider(request, result)
            return result
        if name == "rosclaw_stop_operation":
            return await self._process_stop(request)
        # PR-H4：Product Pack。P0-D：rosclaw_deliver 是模型面唯一
        # 幂等交付入口（普通文件工具创建的交付物）；capability 产物
        # 自动登记不走模型。
        if name == "rosclaw_deliver":
            # P0-C：deliver 也是 effectful——首个 effectful call 就是
            # 交付时先原子 admission（否则裸 NO_ACTIVE_TASK——
            # 金丝雀实证模型先 deliver 后干活的路径）。
            self._ensure_task_for_effect(request)
            result = await self._artifact_register(request)
            self._coordinator_consider(request, result)
            return result
        if name == "rosclaw_artifact_register":
            self._ensure_task_for_effect(request)
            result = await self._artifact_register(request)
            self._coordinator_consider(request, result)
            return result
        if name == "rosclaw_task_finish":
            return await self._task_finish(request)
        if name == "rosclaw_task_blocked":
            return await self._task_blocked(request)
        if name == "rosclaw_process_start":
            self._ensure_task_for_effect(request)
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

    async def _artifact_register(
        self, request: PiToolRequestV1
    ) -> PiToolResultV1:
        """PR-H4：交付物登记（实读文件算 hash——口头提到不算）。"""
        kernel = self._service._task_kernel
        task = kernel.active_task_for(request.mission_id, request.pi_session_id)
        if task is None:
            # 终态后语义（0824 金丝雀实证）：Coordinator 已验收完成的任务
            # 不需要再交付——给可行动的引导，不是裸错误。
            latest = kernel.latest_task_for(
                request.mission_id, request.pi_session_id
            )
            if latest is not None and str(latest.get("state")) == "SUCCEEDED":
                raise ToolBridgeError(
                    "TASK_ALREADY_COMPLETED",
                    "任务已验收完成（SUCCEEDED）——无需再交付；"
                    "用户有新目标时会开始新任务，请直接回答即可",
                )
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

    async def _request_action(self, request: PiToolRequestV1) -> PiToolResultV1:
        """PNA-5 + NA-FIX-5 + 三审 P0-NA-10：经唯一 ActionAdmissionService——
        完整请求上下文（session/lease/revision/body/mode）硬校验、
        精确 grant、结构化回执、execute TOCTOU 复验。"""

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
        # PR-N7：审批续接——携带既有 approval_id 的调用直接进入
        # 执行路径（operator 已决定），不重建卡、不轮询。
        resume_approval_id = str(args.get("approval_id", "") or "")
        if resume_approval_id:
            status = admission.decision_status(resume_approval_id)["status"]
            if status == "APPROVED":
                result = await admission.execute(
                    resume_approval_id, request=ctx,
                    caller_pid=self._caller_pid, caller_uid=self._caller_uid,
                )
                return PiToolResultV1(
                    request_id=request.request_id,
                    ok=bool(result.get("executed")),
                    status=str(result.get("status", "FAILED")),
                    summary=str(result.get("summary", ""))[:8000],
                    approval_id=resume_approval_id,
                    error_code=result.get("error_code"),
                )
            if status == "PENDING":
                return PiToolResultV1(
                    request_id=request.request_id,
                    ok=False,
                    status="WAITING_APPROVAL",
                    summary=f"审批 {resume_approval_id} 仍待决定——让出回合等待事件",
                    approval_id=resume_approval_id,
                    error_code="WAITING_APPROVAL",
                    retryable=True,
                )
            raise ToolBridgeError(
                "APPROVAL_NOT_FOUND",
                f"approval {resume_approval_id} 状态 {status}——不可续接执行",
            )
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
        # PR-N7（调整方案 §六）：不再在模型工具里轮询 330 秒——
        # 创建审批后立即返回 WAITING_APPROVAL + approval_id；operator
        # 决定事件恢复原 session，模型携带 approval_id 续接执行。
        # 审批可安全过期，但绝不卡住 Harness 回合。
        status = admission.decision_status(card["approval_id"])["status"]
        if status == "PENDING":
            from rosclaw.agentd.tooling.recovery import recovery_for

            return PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="WAITING_APPROVAL",
                summary=(
                    f"已创建审批 {card['approval_id']}——等待 operator 决定；"
                    f"{recovery_for('WAITING_APPROVAL') or '让出回合，等待事件恢复'}"
                ),
                approval_id=card["approval_id"],
                error_code="WAITING_APPROVAL",
                retryable=True,
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
