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
from pathlib import Path
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
#: 建议-0816 P0-7：只有明确匹配的确定性工作流（planar trajectory）
#: 才进 executor:simulation——其余 simulation.* 交 Pi Harness 开发/
#: 调用合适能力，绝不误进五角星函数。
_SIMULATION_WORKFLOWS = frozenset({
    "simulation.planar_trajectory",
    "trajectory",
    "render.trace",
})
_ROUTING_RULES: list[tuple[str, str, str]] = [
    # (capability 前缀, domain, runtime)
    ("robot.observe.", "executor", "executor:body-observer"),
    ("code.", "agent_harness", "harness:pi-builtin"),
    ("repo.", "agent_harness", "harness:pi-builtin"),
    ("research.", "agent_harness", "harness:pi-builtin"),
]


def _fingerprint(mission_id: str, spec: dict) -> str:
    """任务指纹（建议-0816 P0-6）：覆盖完整规范化 TaskSpec——goal/
    capabilities/effects/inputs/deliverables/acceptance/model_snapshot。
    只排除时间戳等非语义字段；用户改参数绝不错挂旧任务。"""
    payload = json.dumps(
        {
            "mission": mission_id,
            "goal": str(spec.get("goal", "")),
            "kind": str(spec.get("kind", "")),
            "capabilities": sorted(spec.get("required_capabilities") or []),
            "effects": spec.get("effects") or "",
            "inputs": spec.get("inputs") or {},
            "deliverables": spec.get("deliverables") or [],
            "acceptance": spec.get("acceptance") or {},
            "runtime_requirements": spec.get("runtime_requirements") or {},
            "model": spec.get("model_snapshot") or {},
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _goal_similarity(a: str, b: str) -> float:
    """目标相似度（十六审 P0-E 防裂变）：归一化（去空白/标点/大小写）
    后的 SequenceMatcher 比率。只用于"同一 mission 内活跃执行"的
    attach 判定——终态任务永不被 attach（新目标是新任务）。"""
    import difflib
    import re

    def _norm(text: str) -> str:
        return re.sub(r"[\s，。、,.;；:：'\"\"''（）()【】《》<>_-]+", "", text).lower()

    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


class ExecutionRouter:
    """确定性执行域选择——Native Agent 永不直接挑 Worker。"""

    def __init__(self, service) -> None:
        self._service = service

    def route(self, spec: dict) -> dict:
        """返回 {domain, runtime, reason}；不可路由抛 ValueError。"""
        from rosclaw.agentd.task_compiler import normalize_effects

        effects = normalize_effects(spec.get("effects"))
        capabilities = [str(c) for c in (spec.get("required_capabilities") or [])]
        # 1. effect/risk 排除：physical 效果不走 executor/harness。
        if effects & {"physical.shadow", "physical.real"}:
            return {
                "domain": "physical",
                "runtime": "rosclawd",
                "reason": "物理效果只能经 rosclawd 准入链（Worker 只能产出提案）",
            }
        # 2. 确定性仿真工作流精确匹配（P0-7：只有 planar trajectory
        # 进五角星能力链；其余 simulation.* 落 Harness）。
        for capability in capabilities:
            if capability in _SIMULATION_WORKFLOWS:
                return {
                    "domain": "executor",
                    "runtime": "executor:simulation",
                    "reason": f"capability {capability} → executor:simulation",
                }
        # 3. capability 前缀匹配路由表（Harness 类规则走启动前健康
        # 选择——不硬编码某个具体 runtime）。
        for capability in capabilities:
            for prefix, domain, runtime in _ROUTING_RULES:
                if capability.startswith(prefix) or capability == prefix.rstrip("."):
                    if domain == "agent_harness":
                        runtime = self._preferred_harness()
                    return {
                        "domain": domain,
                        "runtime": runtime,
                        "reason": f"capability {capability} → {runtime}",
                    }
        # 4. 无匹配 capability 的开放任务 → Harness（readiness preflight：
        # ACP Harness 就绪优先，否则内置 Pi——总纲 §7.2 prefer 只用于
        # 启动前选择）。
        return {
            "domain": "agent_harness",
            "runtime": self._preferred_harness(),
            "reason": "开放式任务 Harness 路由（readiness preflight 后选定）",
        }

    def _preferred_harness(self) -> str:
        """启动前健康选择（readiness preflight）。建议-0816 P0-4：
        默认冻结——auto_discovery=false 时永远 pi-builtin，不因机器
        装了 Codex/Claude 偷换执行者；显式 enabled 才解锁。"""
        import shutil

        runtime_cfg = getattr(self._service._config, "agent_runtime", None)
        enabled = list(getattr(runtime_cfg, "enabled", ["pi-sdk"]))
        auto_discovery = bool(getattr(runtime_cfg, "auto_discovery", False))
        if not auto_discovery:
            return "harness:pi-builtin"
        if "codex-app-server" in enabled and shutil.which("codex") and (
            Path.home() / ".codex"
        ).exists():
            return "harness:codex-app-server"
        if "claude-acp" in enabled and shutil.which("claude-code-acp"):
            return "harness:acp:claude-local"
        if "pi-acp" in enabled and shutil.which("pi-acp"):
            return "harness:acp:pi-acp"
        return "harness:pi-builtin"


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
        # 十六审 A1/A3：终态权威——已终态的 execution 不得被后续驱动
        # 收尾覆盖（取消在飞的任务完成后曾把 CANCELLED 涂改成 FAILED）。
        current = self._conn.execute(
            "SELECT state FROM task_executions WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        if current is not None and current["state"] in EXECUTION_TERMINAL:
            if current["state"] != state:
                import logging

                logging.getLogger(__name__).warning(
                    "execution %s 已终态 %s——拒绝覆盖为 %s",
                    execution_id, current["state"], state,
                )
            return
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
        # 十六审 P0-E：模糊防裂变——模型换措辞重提同一目标（指纹不同）
        # 也 attach 到活跃执行（归一化 + 相似度阈值，同 mission 内）。
        goal = str(spec.get("goal", ""))
        for cand in self._conn.execute(
            "SELECT * FROM task_executions WHERE mission_id = ? "
            "AND state NOT IN ('SUCCEEDED','FAILED','BLOCKED','CANCELLED') "
            "ORDER BY created_at DESC",
            (mission_id,),
        ).fetchall():
            try:
                cand_goal = str(json.loads(cand["spec_json"]).get("goal", ""))
            except (ValueError, TypeError):
                continue
            if _goal_similarity(goal, cand_goal) >= 0.80:
                return self._view(dict(cand), attached=True)
        if not str(spec.get("goal", "")).strip():
            raise ValueError("TaskSpec.goal required")
        self._validate_acceptance(spec.get("acceptance") or {})
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

    @staticmethod
    def _validate_acceptance(acceptance: dict) -> None:
        """验收契约 fail-fast（十六审 A2/安全）：模型提交的 shell 字符串
        tests_command 是命令注入面——submit 即拒绝；只允许结构化形式
        （required_files 列表 / run.argv 数组 / 仿真数值阈值）。"""
        if not isinstance(acceptance, dict):
            raise ValueError("TaskSpec.acceptance 必须是对象")
        if "tests_command" in acceptance:
            raise ValueError(
                "acceptance.tests_command（shell 字符串）已禁止——命令注入面。"
                "改用结构化形式：acceptance.run = {\"argv\": [...], "
                "\"timeout_sec\": N}（argv 不走 shell，解释器白名单）"
            )
        run = acceptance.get("run")
        if run is not None:
            if not isinstance(run, dict) or not isinstance(run.get("argv"), list):
                raise ValueError("acceptance.run 必须是 {\"argv\": [...]} 形式")
            if not run["argv"] or not all(
                isinstance(a, str) for a in run["argv"]
            ):
                raise ValueError("acceptance.run.argv 必须是非空字符串数组")

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
        内置 Pi Worker（RF-3 后走 ACP）。任何异常 → FAILED（诚实）。

        十六审 P0-B：编译期集合检查——授权不满足在启动前 BLOCKED
        （零 Worker 预算燃烧），不抱侥幸启动只读 profile。"""
        from rosclaw.agentd.task_compiler import compile_task

        try:
            plan = compile_task(spec)
        except ValueError as exc:
            self._update_state(
                execution_id, "BLOCKED", summary=f"编译期拦截：{exc}"[:400]
            )
            return
        if plan.blocked_reason:
            self._update_state(
                execution_id, "BLOCKED",
                summary=f"编译期拦截：{plan.blocked_reason}"[:400],
            )
            return
        # 十六审 P0-C：runtime_requirements 是 ROSClaw 的责任——
        # PREFLIGHT 在托管 runtime 里确定性预置（Worker 不装环境）。
        runtime_bin = ""
        packages = plan.runtime_requirements.get("python_packages")
        if packages is not None and route["domain"] == "agent_harness":
            from rosclaw.agentd.runtime_manager import RuntimeNotReadyError

            try:
                handle = self._service._runtime_manager.ensure(
                    "rosclaw-task", {"python_packages": list(packages)}
                )
                runtime_bin = str(handle.bin_dir)
            except RuntimeNotReadyError as exc:
                self._update_state(
                    execution_id, "BLOCKED",
                    summary=f"RUNTIME_NOT_READY：{exc}"[:400],
                )
                return
        try:
            if route["domain"] == "executor" and route["runtime"] == "executor:simulation":
                await self._drive_simulation(execution_id, mission_id, spec)
            elif route["domain"] == "executor":
                # RF-6：其余 executor runtime（body-observer 等）走确定性
                # capability 调用——不落 harness（不开 Agent Worker）。
                await self._drive_capability_executor(execution_id, spec, route)
            elif route["domain"] == "physical":
                self._update_state(
                    execution_id, "BLOCKED",
                    summary="物理效果任务必须经 rosclawd 准入链——"
                    "execution 只做提案，不直接执行",
                )
            else:
                await self._drive_harness(
                    execution_id, mission_id, spec, route, plan,
                    runtime_bin=runtime_bin,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 执行失败是数据
            self._update_state(execution_id, "FAILED", summary=str(exc)[:500])

    async def _drive_capability_executor(
        self, execution_id: str, spec: dict, route: dict
    ) -> None:
        """RF-6：确定性 capability 执行域（robot.observe.* 等）——经
        tool registry 直接调用，零 Agent Worker，结果即证据。"""
        service = self._service
        inputs = spec.get("inputs") or {}
        capability_id = str(
            inputs.get("capability_id")
            or (spec.get("required_capabilities") or [""])[0]
        )
        if not capability_id:
            self._update_state(
                execution_id, "BLOCKED",
                summary="executor 任务缺 capability_id——编译期失败，未执行",
            )
            return
        await service._ensure_mcp_discovered()
        self._update_state(execution_id, "RUNNING")
        try:
            raw = await service._tool_registry.execute(
                capability_id, dict(inputs.get("arguments") or {})
            )
        except Exception as exc:  # noqa: BLE001
            self._update_state(
                execution_id, "FAILED",
                summary=f"capability {capability_id} 执行失败: {exc}"[:400],
            )
            return
        text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        self._update_state(
            execution_id, "SUCCEEDED",
            summary=text[:500],
            artifacts_json=json.dumps(
                {"capability": capability_id, "result": text[:2000]},
                ensure_ascii=False,
            ),
        )

    async def _drive_simulation(
        self, execution_id: str, mission_id: str, spec: dict
    ) -> None:
        """SIM 闭环：直接调 SimTrajectoryService 组合（确定性，零
        Agent Worker——总纲 §5.1）。十六审 P0-C：渲染依赖（Pillow）
        由托管 rosclaw-simulation runtime 预置——preflight 失败诚实
        BLOCKED，不是渲染时裸 ModuleNotFoundError。"""
        from rosclaw.agentd.runtime_manager import RuntimeNotReadyError
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        try:
            self._service._runtime_manager.ensure("rosclaw-simulation")
        except RuntimeNotReadyError as exc:
            self._update_state(
                execution_id, "BLOCKED",
                summary=f"RUNTIME_NOT_READY：{exc}"[:400],
            )
            return
        self._update_state(execution_id, "RUNNING")
        sim = SimTrajectoryService(
            self._service._home,
            runtime_manager=self._service._runtime_manager,
        )
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

    @property
    def _pi_runner(self):
        """十六审 P0-D：PiTaskRunner 是内置 Pi 的唯一执行层。"""
        from rosclaw.agentd.pi_task_runner import PiTaskRunner

        runner = getattr(self, "_pi_runner_cache", None)
        if runner is None:
            runner = PiTaskRunner(self)
            self._pi_runner_cache = runner
        return runner

    async def _drive_pi_builtin(
        self, execution_id: str, mission_id: str, spec: dict, plan,
        *, runtime_bin: str = "",
    ) -> None:
        await self._pi_runner._drive_pi_builtin(
            execution_id, mission_id, spec, plan, runtime_bin=runtime_bin
        )

    async def _verify_acceptance(self, spec: dict, workspace, *, report: str = "") -> dict:
        return await self._pi_runner._verify_acceptance(spec, workspace, report=report)

    async def _drive_harness(
        self, execution_id: str, mission_id: str, spec: dict, route: dict,
        plan, *, runtime_bin: str = "",
    ) -> None:
        """Harness 域：codex app-server 原生路径（RF-5）/ACP（RF-3）/
        pi-builtin（RF-9 前保留）。"""
        if route["runtime"] == "harness:codex-app-server":
            await self._drive_codex(execution_id, spec)
            return
        if route["runtime"].startswith("harness:acp:"):
            await self._drive_acp(execution_id, spec, route["runtime"])
            return
        await self._drive_pi_builtin(
            execution_id, mission_id, spec, plan, runtime_bin=runtime_bin
        )

    async def _drive_codex(self, execution_id: str, spec: dict) -> None:
        """Codex app-server：单 thread 单 turn；sandbox（RF-4）+ 原生
        thread/resume/compaction。事件落 EventStore。"""
        from rosclaw.agentd.codex_driver import CodexAppServerDriver, codex_binary

        if codex_binary() is None:
            self._update_state(
                execution_id, "BLOCKED",
                summary="codex CLI 未安装——preflight 失败，未创建执行",
            )
            return
        self._update_state(execution_id, "RUNNING")
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        events = WorkerEventStore(self._service._home)

        async def sink(kind: str, payload: dict) -> None:
            events.append_event(execution_id, "", kind, payload)

        driver = CodexAppServerDriver(
            cwd=str(self._service._home),
            event_sink=sink,
            sandbox_home=self._service._home / "work" / execution_id,
        )
        result = await driver.run(str(spec.get("goal", "")))
        self._update_state(
            execution_id,
            "SUCCEEDED" if result["ok"] else "FAILED",
            summary=result["detail"][:500],
        )

    async def _drive_acp(
        self, execution_id: str, spec: dict, runtime: str
    ) -> None:
        """ACP Harness：单 session 跑到底；事件落 EventStore。"""
        from rosclaw.agentd.acp_driver import AcpHarnessDriver, acp_binary_for

        if acp_binary_for(runtime) is None:
            self._update_state(
                execution_id, "BLOCKED",
                summary=f"ACP Harness {runtime} 未安装——preflight 失败，"
                "未创建执行（可安装后重试同一任务）",
            )
            return
        self._update_state(execution_id, "RUNNING")
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        events = WorkerEventStore(self._service._home)

        async def sink(kind: str, payload: dict) -> None:
            events.append_event(execution_id, "", kind, payload)

        driver = AcpHarnessDriver(
            runtime, cwd=str(self._service._home), event_sink=sink
        )
        result = await driver.run(str(spec.get("goal", "")))
        self._update_state(
            execution_id,
            "SUCCEEDED" if result["ok"] else "FAILED",
            summary=result["detail"][:500],
        )

