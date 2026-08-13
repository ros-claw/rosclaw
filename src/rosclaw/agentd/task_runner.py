"""Task Compiler + Task Runner（八审 §1.6/P0-5）。

把"已知任务"从模型的即兴工具链收回到内核确定性编排：

```text
TaskSpec(goal=draw_shape, parameters)
→ 编译（默认参数/工作区校验——非法参数零提案即败）
→ PLANNING（plan_cartesian_path → PlanStore 句柄）
→ POLICY_CHECK（ admission 链：SIM AUTO / ask ASK / REAL 门禁）
→ EXECUTING（恰好一个 ExactAction：ur5e.execute_plan）
→ VERIFYING（verify_drawing 自动后验）
→ VERIFIED | FAILED | DENIED | CANCELLED | INCONCLUSIVE
```

红线：
- 模型只见到 TaskResult 摘要——不搬轨迹/hash/lease/grant；
- 恰好一次 action proposal；verifier PASS 才能 VERIFIED；
- 幂等：同一 idempotency_key 重放返回既有任务，不产生第二动作；
- verifier 不可用/失败 → INCONCLUSIVE/FAILED，绝不报喜；
- 无安全 batch executor 时诚实失败——禁止模型逐点 fallback 替代。
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from rosclaw.contracts.common import new_id

if TYPE_CHECKING:
    from rosclaw.agentd.pi_bridge.action_admission import ActionRequestContext
    from rosclaw.agentd.service import AgentService

#: TaskStateV1（八审 §2.2）——终态：VERIFIED/FAILED/DENIED/CANCELLED/
#: INCONCLUSIVE。
TASK_STATES = (
    "CREATED",
    "PLANNING",
    "PLANNED",
    "POLICY_CHECK",
    "WAITING_APPROVAL",
    "EXECUTING",
    "VERIFYING",
    "VERIFIED",
    "FAILED",
    "DENIED",
    "CANCELLED",
    "INCONCLUSIVE",
)
TERMINAL_STATES = frozenset({"VERIFIED", "FAILED", "DENIED", "CANCELLED", "INCONCLUSIVE"})


class TaskStore:
    """task_records 表（migration 021）——任务权威状态。"""

    def __init__(self, conn) -> None:
        self._conn = conn

    def create(
        self,
        *,
        task_id: str,
        idempotency_key: str,
        mission_id: str,
        goal: str,
        params: dict,
        caused_by_turn_id: str = "",
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT INTO task_records (task_id, idempotency_key, mission_id, "
            "goal, params_json, state, caused_by_turn_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'CREATED', ?, ?, ?)",
            (task_id, idempotency_key, mission_id, goal, json.dumps(params),
             caused_by_turn_id, now, now),
        )

    def get_by_id(self, task_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM task_records WHERE task_id = ?", (task_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_by_idempotency(self, idempotency_key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM task_records WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return dict(row) if row else None

    def transition(self, task_id: str, state: str, **fields) -> None:
        assert state in TASK_STATES, f"unknown task state {state}"
        assignments = ", ".join(f"{k} = ?" for k in ("state", *fields, "updated_at"))
        self._conn.execute(
            f"UPDATE task_records SET {assignments} WHERE task_id = ?",  # noqa: S608
            (state, *fields.values(), datetime.now(UTC).isoformat(), task_id),
        )


def _unwrap_tool_output(output: Any) -> dict:
    """tool_registry 的返回是包装 JSON：{"tool":..., "content":
    ["<内层 JSON>"]}（内层转义）——解析到内层对象。"""
    data = output
    # 双重编码/包装层数不定——循环解到 dict 为止（上限防死循环）。
    for _ in range(6):
        if isinstance(data, str):
            data = json.loads(data)
            continue
        if isinstance(data, dict) and isinstance(data.get("content"), list):
            data = data["content"][0]
            continue
        break
    if not isinstance(data, dict):
        raise ValueError(f"unexpected tool output shape: {type(data)}")
    return data


def evidence_user_label(level: str) -> str:
    """十一审 PR-E（§P0-7/Gate H）：三层证据语言——低层证据绝不得
    用高层完成文案。"""
    return {
        "COMMAND_REPLAY": "路径预演/几何验证完成（命令回放）",
        "SIM_DYN_ROLLOUT": "MuJoCo 动力学仿真完成",
        "REAL_RECEIPT": "真机动作完成",
    }.get(level, level or "未知证据等级")


class TaskRunner:
    """确定性任务编排器——模型只交 TaskSpec，只收 TaskResult。"""

    def __init__(self, service: AgentService) -> None:
        self._service = service
        self._store = TaskStore(service._store.connection)

    def _compile_draw_shape(self, params: dict) -> dict:
        """draw_shape 编译：默认参数 + 工作区边界校验（编译期失败，
        零提案）。返回 plan_cartesian_path 的精确参数。"""
        shape = str(params.get("shape", "star5"))
        if shape != "star5":
            raise ValueError(f"unsupported shape {shape!r} (supported: star5)")
        center = params.get("center_m") or [0.35, 0.25, 0.30]
        if not isinstance(center, list) or len(center) != 3:
            raise ValueError("center_m must be [x, y, z]")
        radius = float(params.get("radius_m", 0.10))
        # 与 ur5e 安全工作空间一致的编译期边界（越界零提案即败）。
        if not (0.02 <= radius <= 0.35):
            raise ValueError(f"radius_m {radius} outside [0.02, 0.35]")
        import math

        cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
        reach = math.hypot(cx, cy) + radius
        if reach > 0.80 or not (0.02 <= cz <= 1.20):
            raise ValueError(
                f"shape at center {center} radius {radius} exceeds safe workspace"
            )
        return {
            "shape": shape,
            "center_x": cx,
            "center_y": cy,
            "z": cz,
            "outer_radius": radius,
        }

    async def run(
        self,
        *,
        request_ctx: ActionRequestContext,
        goal: str,
        parameters: dict,
        caller_pid: int | None,
        caller_uid: int | None,
        wait: bool = True,
    ) -> dict[str, Any]:
        """同步跑到终态（kinematic sandbox 亚秒级）；WAITING_APPROVAL
        由调用方等待后重入（同一 idempotency_key）。"""
        from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        idem = request_ctx.idempotency_key
        existing = self._store.get_by_idempotency(idem)
        if existing is not None:
            return self._result(existing)
        # WP-P0-7（总纲 §7.4）：crash 恢复只能 attach——同 mission/
        # goal/参数的非终态任务重提（新幂等键）返回既有任务，绝不
        # 二次提交。
        fingerprint = json.dumps(
            {"mission": request_ctx.mission_id, "goal": goal,
             "parameters": parameters},
            sort_keys=True, ensure_ascii=False,
        )
        row = self._store._conn.execute(
            "SELECT task_id FROM task_records WHERE mission_id = ? AND goal = ? "
            "AND state NOT IN ('VERIFIED','FAILED','DENIED','CANCELLED','INCONCLUSIVE') "
            "ORDER BY rowid DESC LIMIT 1",
            (request_ctx.mission_id, goal),
        ).fetchone()
        if row is not None:
            candidate = self._store.get_by_id(row[0])
            if candidate is not None and candidate.get("params_json"):
                try:
                    same = json.loads(candidate["params_json"])
                    same_fp = json.dumps(
                        {"mission": request_ctx.mission_id, "goal": goal,
                         "parameters": same},
                        sort_keys=True, ensure_ascii=False,
                    )
                except Exception:  # noqa: BLE001
                    same_fp = ""
                if same_fp == fingerprint:
                    result = self._result(candidate)
                    result["attached"] = True
                    return result
        if goal != "draw_shape":
            raise ToolBridgeError(
                "TASK_UNKNOWN", f"unknown task goal {goal!r} (supported: draw_shape)"
            )
        task_id = new_id("task")
        # 九审 §7：caused_by_turn_id——任务必须可追溯到用户 turn
        # （该 session 最近的 interactive 输入；无则空=历史/命令路径）。
        caused_by = ""
        with contextlib.suppress(Exception):
            from rosclaw.agentd.turn_store import TurnStore

            latest = TurnStore(self._service._store.connection).latest_for_session(
                request_ctx.pi_session_id
            )
            caused_by = str(latest["turn_id"]) if latest else ""
        self._store.create(
            task_id=task_id,
            idempotency_key=idem,
            mission_id=request_ctx.mission_id,
            goal=goal,
            params=parameters,
            caused_by_turn_id=caused_by,
        )
        try:
            compiled = self._compile_draw_shape(parameters)
        except ValueError as exc:
            self._store.transition(task_id, "FAILED", error=str(exc))
            return self._result(self._store.get_by_idempotency(idem))

        service = self._service
        await service._ensure_mcp_discovered()
        # PLANNING：内核调 COMPUTE 规划——载荷进 PlanStore，句柄留内核。
        self._store.transition(task_id, "PLANNING")
        try:
            plan_raw = await service._tool_registry.execute(
                "ur5e.plan_cartesian_path", compiled
            )
            plan = _unwrap_tool_output(plan_raw)
            if not plan.get("ok"):
                raise ValueError(str(plan)[:200])
        except Exception as exc:  # noqa: BLE001
            self._store.transition(task_id, "FAILED", error=f"planning: {exc}")
            return self._result(self._store.get_by_idempotency(idem))
        plan_id = str(plan["plan_id"])
        self._store.transition(
            task_id, "PLANNED", plan_id=plan_id, plan_digest=str(plan.get("digest", ""))
        )

        # POLICY_CHECK → EXECUTING：恰好一个 ExactAction。
        self._store.transition(task_id, "POLICY_CHECK")
        admission = ActionAdmissionService(service)
        # 八审 §3.3：任务级卡片人读——形状/中心/半径，不是 plan_id。
        card_title = (
            f"绘制五角星（中心 ({compiled['center_x']:.2f}, "
            f"{compiled['center_y']:.2f}, {compiled['z']:.2f})m，"
            f"半径 {compiled['outer_radius']:.2f}m，运动学仿真）"
        )
        card = await admission.propose(
            request=request_ctx,
            capability_id="ur5e.execute_plan",
            arguments={"plan_id": plan_id},
            expected_effect=card_title,
            risk_tier="LOW",
            title=card_title,
            caller_pid=caller_pid,
            caller_uid=caller_uid,
        )
        approval_id = card["approval_id"]
        self._store.transition(task_id, "POLICY_CHECK", approval_id=approval_id)
        # POLICY_AUTO 在 propose 内已完成政策决定；ask 策略下
        # wait=False 立即返回 WAITING_APPROVAL 视图（TS 展卡 → 人工
        # 决定 → resume 重入），wait=True 则与 rosclaw_request_action
        # 同语义原地等待。
        import asyncio

        waited = 0.0
        while True:
            status = admission.decision_status(approval_id)["status"]
            if status != "PENDING":
                break
            self._store.transition(task_id, "WAITING_APPROVAL")
            if not wait:
                result = self._result(self._store.get_by_idempotency(idem))
                result["approval_id"] = approval_id
                result["display_hash"] = str(card.get("display_hash") or "")
                return result
            await asyncio.sleep(1.0)
            waited += 1.0
            if waited >= 330.0:
                break
        if status != "APPROVED":
            self._store.transition(
                task_id,
                "DENIED" if status == "DENIED" else "FAILED",
                error=f"decision={status}",
            )
            return self._result(self._store.get_by_idempotency(idem))
        return await self._execute_and_verify(
            task_id, idem, admission, approval_id, request_ctx,
            caller_pid=caller_pid, caller_uid=caller_uid,
        )

    async def resume(
        self,
        *,
        task_id: str,
        request_ctx: ActionRequestContext,
        caller_pid: int | None,
        caller_uid: int | None,
    ) -> dict[str, Any]:
        """人工批准后的重入：execute + verify。task 状态机是权威——
        重复 resume 幂等（已终态直接返回）。"""
        from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        record = self._store.get_by_id(task_id)
        if record is None:
            raise ToolBridgeError("TASK_NOT_FOUND", f"unknown task {task_id!r}")
        if record["mission_id"] != request_ctx.mission_id:
            raise ToolBridgeError(
                "TASK_MISSION_MISMATCH", "task belongs to another mission (fail closed)"
            )
        if record["state"] in TERMINAL_STATES:
            return self._result(record)
        if record["state"] != "WAITING_APPROVAL":
            raise ToolBridgeError(
                "TASK_STATE_FORBIDDEN",
                f"task is {record['state']}, not WAITING_APPROVAL (fail closed)",
            )
        admission = ActionAdmissionService(self._service)
        status = admission.decision_status(record["approval_id"])["status"]
        if status != "APPROVED":
            raise ToolBridgeError(
                "APPROVAL_PENDING", f"decision is {status}, not APPROVED"
            )
        return await self._execute_and_verify(
            task_id, record["idempotency_key"], admission, record["approval_id"],
            request_ctx, caller_pid=caller_pid, caller_uid=caller_uid,
        )

    def cancel(self, task_id: str) -> dict[str, Any]:
        """八审 §4 P0-9：/cancel 取消真实任务（不只是当前 LLM
        回合）。非终态 → CANCELLED（待批准卡随之失效——resume 拒绝）；
        终态 → 诚实 no-op。"""
        record = self._store.get_by_id(task_id)
        if record is None:
            return {"ok": False, "error": f"unknown task {task_id!r}", "code": "TASK_NOT_FOUND"}
        if record["state"] in TERMINAL_STATES:
            return {"ok": True, "state": record["state"], "changed": False}
        self._store.transition(task_id, "CANCELLED")
        # WP-P0-7（总纲 §7.4）：取消传播——撤销未消费的审批卡
        # （此后 decide 即拒，绝不产 grant/执行）。
        approval_id = record.get("approval_id") or ""
        if approval_id:
            with contextlib.suppress(Exception):
                self._service._broker.cancel_request(
                    approval_id, principal="task.cancel"
                )
        return {"ok": True, "state": "CANCELLED", "changed": True}

    async def _execute_and_verify(
        self,
        task_id: str,
        idem: str,
        admission,
        approval_id: str,
        request_ctx: ActionRequestContext,
        *,
        caller_pid: int | None,
        caller_uid: int | None,
    ) -> dict[str, Any]:
        service = self._service
        self._store.transition(task_id, "EXECUTING")
        executed = await admission.execute(
            approval_id,
            request=request_ctx,
            caller_pid=caller_pid,
            caller_uid=caller_uid,
        )
        txn_id = str(executed.get("txn_id") or "")
        if not executed.get("executed"):
            self._store.transition(
                task_id, "FAILED", txn_id=txn_id,
                error=str(executed.get("error") or executed.get("status"))[:300],
            )
            return self._result(self._store.get_by_idempotency(idem))
        self._store.transition(task_id, "EXECUTING", txn_id=txn_id)

        # VERIFYING：自动后验——PASS 才 VERIFIED。
        self._store.transition(task_id, "VERIFYING")
        try:
            verify_raw = await service._tool_registry.execute("ur5e.verify_drawing", {})
            verify = _unwrap_tool_output(verify_raw)
            verification = verify.get("verification") or {}
        except Exception as exc:  # noqa: BLE001
            self._store.transition(
                task_id, "INCONCLUSIVE", error=f"verifier: {exc}"[:300]
            )
            return self._result(self._store.get_by_idempotency(idem))
        verdict = verification.get("verdict")
        self._store.transition(
            task_id,
            "VERIFIED" if verdict == "PASS" else "INCONCLUSIVE",
            verification_json=json.dumps(verification, ensure_ascii=False),
        )
        return self._result(self._store.get_by_idempotency(idem))

    def _result(self, record: dict) -> dict[str, Any]:
        """模型可见 TaskResult——摘要级，无内部载荷。"""
        verification = {}
        if record.get("verification_json"):
            with contextlib.suppress(Exception):
                verification = json.loads(record["verification_json"])
        # 决策权威如实呈现（AUTO_SIM=政策自动 / OPERATOR=人工卡）。
        policy = "OPERATOR"
        approval_id = record.get("approval_id") or ""
        if approval_id:
            row = self._store._conn.execute(
                "SELECT decided_by FROM operator_requests WHERE request_id = ?",
                (approval_id,),
            ).fetchone()
            if row and row[0] and "POLICY_AUTO" in str(row[0]):
                policy = "AUTO_SIM"
        # 八审 §4 P0-7 三通道：user_view 一行进度（model_view 是本
        # 结构；audit_view 全量在 task_records/receipt，/trace 展开）。
        if record["state"] == "VERIFIED":
            # WP-P0-6（总纲 §8.3）+ 十一审 PR-E（§P0-7 三层证据语言）：
            # COMMAND_REPLAY 只显示"路径预演"——不宣称动力学仿真或真机。
            user_view = evidence_user_label("COMMAND_REPLAY") + (
                "：路径数据自洽、几何闭合验证 PASS——不能证明动力学或"
                "真实机械臂完成运动"
            )
        elif record["state"] == "WAITING_APPROVAL":
            user_view = "规划 ✓  安全校验 ✓  等待人工确认…"
        else:
            user_view = f"任务 {record['state']}" + (
                f"：{record['error'][:80]}" if record.get("error") else ""
            )
        return {
            "task_id": record["task_id"],
            "state": record["state"],
            "goal": record["goal"],
            "policy": policy,
            "user_view": user_view,
            "evidence_level": "COMMAND_REPLAY",
            "plan_id": record.get("plan_id") or "",
            "summary": (
                "UR5e 命令回放沙盒绘制闭合五角星（路径自洽）"
                if record["state"] == "VERIFIED"
                else f"任务{record['state']}"
            ),
            "verification": verification,
            "error": record.get("error") or "",
        }
