"""Task Control Plane + ExecutionRouter（十五审 PR-RF-2，ADR-0011）。

核心不变量（Gate 3 不裂变）：
- 一个用户任务 = 一个 owning execution（同一 mission+目标指纹的重复
  提交 attach 到既有执行，绝不创建第二个 Worker）；
- 验收失败的证据反馈给同一 execution（REPAIRING）修复——不自动
  新建 Worker；
- 运行中禁止横跳 Runtime（router 只在启动前选择）；
- Native Agent 不指定 worker_id——ExecutionRouter 按注册表+策略
  确定性选择执行域。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from rosclaw.contracts.common import new_id

#: 执行域（ADR-0011）：simulation/executor/agent_harness/physical。
EXECUTION_DOMAINS = ("executor", "agent_harness", "physical")

#: execution 状态机（单一权威——不映射 WorkOrder 内部态给用户）。
EXECUTION_STATES = (
    "PREFLIGHT",
    "RUNNING",
    "INPUT_REQUIRED",
    "PERMISSION_REQUIRED",
    "REPAIRING",
    "PAUSED",
    "VERIFYING",
    "SUCCEEDED",
    "FAILED",
    "BLOCKED",
    "CANCELLED",
    "INTERRUPTED",
)
EXECUTION_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"})

#: 路由策略（ADR-0011 §7.2）：先 effect 排除 → capability 匹配 →
#: readiness preflight → 策略优先。prefer 只用于启动前选择。
_ROUTING_RULES: list[tuple[str, str, str]] = [
    # (capability 前缀, domain, runtime)
    ("simulation.", "executor", "executor:simulation"),
    ("trajectory", "executor", "executor:simulation"),
    ("render.", "executor", "executor:simulation"),
    ("robot.observe.", "executor", "executor:body-observer"),
    ("code.", "agent_harness", "harness:pi-builtin"),
    ("repo.", "agent_harness", "harness:pi-builtin"),
    ("research.", "agent_harness", "harness:pi-builtin"),
]


def _fingerprint(mission_id: str, spec: dict) -> str:
    payload = json.dumps(
        {
            "mission": mission_id,
            "goal": str(spec.get("goal", "")),
            "capabilities": sorted(spec.get("required_capabilities") or []),
            "effects": str(spec.get("effects", "")),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class ExecutionRouter:
    """确定性执行域选择——Native Agent 永不直接挑 Worker。"""

    def __init__(self, service) -> None:
        self._service = service

    def route(self, spec: dict) -> dict:
        """返回 {domain, runtime, reason}；不可路由抛 ValueError。"""
        effects = str(spec.get("effects", "") or "")
        capabilities = [str(c) for c in (spec.get("required_capabilities") or [])]
        # 1. effect/risk 排除：physical 效果不走 executor/harness。
        if effects in ("physical_real", "physical_shadow"):
            return {
                "domain": "physical",
                "runtime": "rosclawd",
                "reason": "物理效果只能经 rosclawd 准入链（Worker 只能产出提案）",
            }
        # 2. capability 精确匹配路由表。
        for capability in capabilities:
            for prefix, domain, runtime in _ROUTING_RULES:
                if capability.startswith(prefix) or capability == prefix.rstrip("."):
                    return {
                        "domain": domain,
                        "runtime": runtime,
                        "reason": f"capability {capability} → {runtime}",
                    }
        # 3. 无匹配 capability 的开放任务 → 默认 Harness（编码/调研）。
        return {
            "domain": "agent_harness",
            "runtime": "harness:pi-builtin",
            "reason": "开放式任务默认内置 Pi Harness（共享 Native provider）",
        }


class TaskControlPlane:
    """任务唯一权威：提交/路由/状态/恢复——一个任务一个 owning execution。"""

    def __init__(self, service) -> None:
        self._service = service
        self._router = ExecutionRouter(service)
        self._drivers: dict[str, asyncio.Task] = {}

    @property
    def _conn(self):
        return self._service._store.connection

    def _insert_execution(
        self, *, execution_id: str, mission_id: str, spec: dict,
        route: dict, fingerprint: str, idem: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT INTO task_executions (execution_id, mission_id, spec_json, "
            "fingerprint, idempotency_key, domain, runtime, state, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'PREFLIGHT', ?, ?)",
            (
                execution_id, mission_id, json.dumps(spec, ensure_ascii=False),
                fingerprint, idem, route["domain"], route["runtime"], now, now,
            ),
        )

    def _update_state(self, execution_id: str, state: str, **fields) -> None:
        assert state in EXECUTION_STATES, f"unknown execution state {state}"
        assignments = ", ".join(f"{k} = ?" for k in ("state", *fields, "updated_at"))
        self._conn.execute(
            f"UPDATE task_executions SET {assignments} WHERE execution_id = ?",  # noqa: S608
            (state, *fields.values(), datetime.now(UTC).isoformat(), execution_id),
        )

    def _get(self, execution_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM task_executions WHERE execution_id = ?", (execution_id,)
        ).fetchone()
        return dict(row) if row else None

    def executions_for(self, mission_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM task_executions WHERE mission_id = ? ORDER BY created_at",
            (mission_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    async def submit(
        self, mission_id: str, spec: dict, *, idem: str
    ) -> dict[str, Any]:
        """提交任务合同。幂等：同 idem 重放返回既有执行；同 mission+
        目标指纹的活跃执行存在 → attach（不裂变）。"""
        existing = self._conn.execute(
            "SELECT * FROM task_executions WHERE idempotency_key = ?", (idem,)
        ).fetchone()
        if existing is not None:
            return self._view(dict(existing), attached=True)
        fingerprint = _fingerprint(mission_id, spec)
        active = self._conn.execute(
            "SELECT * FROM task_executions WHERE mission_id = ? AND fingerprint = ? "
            "AND state NOT IN ('SUCCEEDED','FAILED','BLOCKED','CANCELLED') "
            "ORDER BY created_at DESC LIMIT 1",
            (mission_id, fingerprint),
        ).fetchone()
        if active is not None:
            # Gate 3：一个任务一个 owning execution——重复提交只 attach。
            return self._view(dict(active), attached=True)
        if not str(spec.get("goal", "")).strip():
            raise ValueError("TaskSpec.goal required")
        route = self._router.route(spec)
        execution_id = new_id("exec")
        self._insert_execution(
            execution_id=execution_id, mission_id=mission_id, spec=spec,
            route=route, fingerprint=fingerprint, idem=idem,
        )
        self._drivers[execution_id] = asyncio.create_task(
            self._drive(execution_id, mission_id, spec, route)
        )
        return self._view(self._get(execution_id))

    def _view(self, row: dict | None, *, attached: bool = False) -> dict[str, Any]:
        assert row is not None
        view = {
            "execution_id": row["execution_id"],
            "state": row["state"],
            "domain": row["domain"],
            "runtime": row["runtime"],
            "summary": row.get("summary") or "",
        }
        if attached:
            view["attached"] = True
        return view

    async def report_verifier_failure(
        self, execution_id: str, *, evidence: str
    ) -> None:
        """验收失败 → REPAIRING：证据反馈同一 execution（不新建 Worker）。"""
        row = self._get(execution_id)
        if row is None:
            raise ValueError(f"unknown execution {execution_id!r}")
        if row["state"] in EXECUTION_TERMINAL:
            raise ValueError(f"execution is {row['state']}（终态不可修复）")
        self._update_state(execution_id, "REPAIRING", verifier_feedback=evidence)

    async def _drive(
        self, execution_id: str, mission_id: str, spec: dict, route: dict
    ) -> None:
        """执行驱动：executor 域走确定性 TaskRunner 组合；harness 域走
        内置 Pi Worker（RF-3 后走 ACP）。任何异常 → FAILED（诚实）。"""
        try:
            if route["domain"] == "executor" and route["runtime"] == "executor:simulation":
                await self._drive_simulation(execution_id, mission_id, spec)
            elif route["domain"] == "physical":
                self._update_state(
                    execution_id, "BLOCKED",
                    summary="物理效果任务必须经 rosclawd 准入链——"
                    "execution 只做提案，不直接执行",
                )
            else:
                await self._drive_harness(execution_id, mission_id, spec)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 执行失败是数据
            self._update_state(execution_id, "FAILED", summary=str(exc)[:500])

    async def _drive_simulation(
        self, execution_id: str, mission_id: str, spec: dict
    ) -> None:
        """SIM 闭环：直接调 SimTrajectoryService 组合（确定性，零
        Agent Worker——总纲 §5.1）。"""
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        self._update_state(execution_id, "RUNNING")
        sim = SimTrajectoryService(self._service._home)
        inputs = spec.get("inputs") or {}
        acceptance = spec.get("acceptance") or {}
        plan = await asyncio.to_thread(
            sim.generate_planar_path,
            shape=str(inputs.get("shape", "star5")),
            center_m=inputs.get("center_m") or [0.35, 0.25, 0.30],
            scale_m=float(inputs.get("scale_m", inputs.get("radius_m", 0.10))),
            plane=str(inputs.get("plane", "xy")),
            max_segment_m=float(inputs.get("max_segment_m", 0.03)),
        )
        result = await asyncio.to_thread(
            sim.simulate_cartesian_trajectory, plan["plan_id"]
        )
        render = await asyncio.to_thread(sim.render_trace, result["trace_id"], format="gif")
        self._update_state(execution_id, "VERIFYING")
        threshold = float(acceptance.get("max_tracking_error_m", 0.05))
        verify = await asyncio.to_thread(
            sim.verify_tracking, result["trace_id"], max_tracking_error_m=threshold
        )
        min_frames = int(acceptance.get("animation_min_frames", 30))
        failures = []
        if verify["verdict"] != "PASS":
            failures.append(
                f"跟踪误差 {verify['metrics']['max_error_m']}m > {threshold}m"
            )
        if render["artifact"]["frames"] < min_frames:
            failures.append(f"动画帧数 {render['artifact']['frames']} < {min_frames}")
        if not result.get("is_safe"):
            failures.append(f"rollout 不安全: {result.get('violations')}")
        metrics = verify["metrics"]
        summary = (
            f"动力学仿真完成：动画 {render['artifact']['path']} · "
            f"trace {result['artifacts']['trace_json']} · 最大误差 "
            f"{metrics['max_error_m'] * 1000:.0f}mm · 验证 {verify['verdict']}"
            if not failures
            else f"验收未过：{'；'.join(failures)}"
        )
        self._update_state(
            execution_id,
            "SUCCEEDED" if not failures else "FAILED",
            summary=summary,
            artifacts_json=json.dumps(
                {
                    "gif": render["artifact"]["path"],
                    **result["artifacts"],
                    "metrics": metrics,
                    "evidence_level": "SIM_DYN_ROLLOUT",
                },
                ensure_ascii=False,
            ),
        )

    async def _drive_harness(
        self, execution_id: str, mission_id: str, spec: dict
    ) -> None:
        """Harness 域：经现有 worker manager 起一个内置 Pi 执行会话
        （RF-3 后改走 ACP——execution 语义不变）。"""
        from rosclaw.agentd.workers.scheduler import CandidateView
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        manager = self._service._worker_manager
        registry = self._service._registry
        if registry.status_of("worker:rosclaw:pi") != "ENABLED":
            self._update_state(
                execution_id, "BLOCKED",
                summary="内置 Pi Harness 未就绪（Node/dist 不可用）——"
                "preflight 失败，未创建执行",
            )
            return
        card = registry.get("worker:rosclaw:pi")
        deliverables = spec.get("deliverables") or []
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id=mission_id,
            issued_by="task-control-plane",
            capability="analysis.text",
            goal=str(spec.get("goal", "")),
            inputs={
                "instructions": str(spec.get("goal", "")),
                "worker_profile": str((spec.get("inputs") or {}).get("profile", "developer")),
                "_execution_id": execution_id,
            },
            budgets=BudgetEnvelope(),
            expected_output=ExpectedOutput(
                artifacts=[str(d.get("type", "text/plain")) for d in deliverables]
                or ["text/plain"]
            ),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
        )
        scheduled = manager.hire(
            order,
            [CandidateView(card=card, registry_status="ENABLED",
                           running_orders=0, circuit_open=False)],
        )
        self._update_state(
            execution_id, "RUNNING", work_order_id=scheduled.work_order_id
        )
        result, _report = await manager.run_to_completion(scheduled)
        if result.status == "COMPLETED":
            self._update_state(
                execution_id, "SUCCEEDED", summary=result.summary[:500]
            )
        elif result.status == "INTERRUPTED":
            self._update_state(
                execution_id, "INTERRUPTED",
                summary=result.summary[:500],
            )
        else:
            self._update_state(
                execution_id, "FAILED", summary=result.summary[:500]
            )
