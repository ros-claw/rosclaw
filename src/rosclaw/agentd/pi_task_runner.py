"""PiTaskRunner（十六审 P0-D）：内置 Pi 是 ROSClaw 的原生任务执行层，
不是外部 Worker。

边界：
- Task Control Plane 拥有任务身份与状态机；本类拥有"一次 Pi 执行"的
  完整生命周期（编译授权信封 → WorkOrder 机制（内部进程监管）→
  驱动 → 验收/修复/能力升级 → 终态）；
- WorkOrder/WorkerManager 只是内部进程监管机制——用户永不见"雇佣了
  Worker"；attempts（repair/escalate）折叠在同一个 execution 下；
- 验收只认结构化证据（文件/argv 检查/纯报告），零检查绝不成功；
- 能力升级恢复同一 session（≤1 次）；验收修复 ≤2 轮。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rosclaw.contracts.common import new_id


class PiTaskRunner:
    """内置 Pi 任务执行器（Task Control Plane 的内部执行层）。"""

    def __init__(self, plane) -> None:
        self._plane = plane

    async def _drive_pi_builtin(
        self, execution_id: str, mission_id: str, spec: dict, plan,
        *, runtime_bin: str = "",
    ) -> None:
        """内置 Pi Harness（RF-9 前的默认路径）。十六审 P0-B：profile/
        capability/effect 来自 Task Compiler 的集合编译（不再是
        capability 名称前缀猜测）。"""
        from rosclaw.agentd.workers.scheduler import CandidateView
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        manager = self._plane._service._worker_manager
        registry = self._plane._service._registry
        if registry.status_of("worker:rosclaw:pi") != "ENABLED":
            self._plane._update_state(
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
            capability=plan.capability,
            goal=str(spec.get("goal", "")),
            inputs={
                "instructions": str(spec.get("goal", "")),
                # profile 由 Task Compiler 集合编译（授权信封——账本与
                # 实际工具面逐字一致）。
                "worker_profile": plan.profile,
                "_execution_id": execution_id,
                # 建议-0816 P0-4：Native 当前模型快照继承（Worker 用同一
                # provider/model/thinking——不是默认推断）。
                "model_snapshot": dict(spec.get("model_snapshot") or {}),
                # 十六审 P0-C：托管 runtime 的 bin 前缀（Worker PATH
                # 前置——装依赖是 ROSClaw 的活，Worker 直接用）。
                **({"_runtime_bin": runtime_bin} if runtime_bin else {}),
            },
            budgets=BudgetEnvelope(),
            expected_output=ExpectedOutput(
                artifacts=[str(d.get("type", "text/plain")) for d in deliverables]
                or ["text/plain"]
            ),
            side_effect_policy=SideEffectPolicy(**{"class": plan.effect_class}),
        )
        scheduled = manager.hire(
            order,
            [CandidateView(card=card, registry_status="ENABLED",
                           running_orders=0, circuit_open=False)],
        )
        self._plane._update_state(
            execution_id, "RUNNING", work_order_id=scheduled.work_order_id
        )
        result, _report = await manager.run_to_completion(scheduled)
        # 十六审 A3：Worker 终态权威映射——状态只来自结构化字段
        # （result.status / [CAUSE] 括号 cause），禁止摘要子串推断。
        if result.status == "COMPLETED":
            # 建议-0816 P0-2：COMPLETED ≠ SUCCEEDED——先 VERIFYING 跑
            # acceptance；WorkOrder 层 deliverable 拒绝同样进 REPAIRING
            # （验收失败反馈同一 session 修复，不新建 Worker）。
            await self._verify_and_repair(
                execution_id, spec, scheduled, result.summary,
                report=self._pure_report(result),
            )
        elif result.status == "FAILED":
            from rosclaw.agentd.workers.retry import parse_cause

            cause = parse_cause(result.summary)
            if cause == "DELIVERABLE_FAILED":
                # 交付物被 WorkOrder 层拒绝 = 验收失败 → 同 session 修复。
                await self._verify_and_repair(
                    execution_id, spec, scheduled, result.summary,
                    report=self._pure_report(result),
                )
            else:
                # 终态守卫在 _update_state 内（CANCELLED 在飞不被覆盖）。
                self._plane._update_state(
                    execution_id, "FAILED", summary=result.summary[:500]
                )
        elif result.status == "BLOCKED":
            # 十六审 P0-B：Worker 诚实 BLOCKED → 能力协商——若缺的
            # 能力有可升级 profile（process.exec/workspace.write），
            # 恢复同一 session 升级授权继续（同一 execution，attempt
            # 折叠）；无授权路径（network/physical）才如实 BLOCKED。
            escalated = await self._maybe_escalate(
                execution_id, mission_id, spec, plan, scheduled, result.summary
            )
            if not escalated:
                self._plane._update_state(
                    execution_id, "BLOCKED",
                    summary=result.summary[:500] or "worker BLOCKED",
                )
        elif result.status == "INTERRUPTED":
            self._plane._update_state(
                execution_id, "INTERRUPTED",
                summary=result.summary[:500],
            )
        elif result.status == "CANCELLED":
            # 用户取消是 CANCELLED——绝不能落成 FAILED（自审修复：
            # task_cancel 后 driver 收尾曾把 CANCELLED 覆盖成 FAILED）。
            self._plane._update_state(
                execution_id, "CANCELLED",
                summary=result.summary[:500] or "已取消",
            )
        else:  # pragma: no cover - 契约外的未来状态：诚实 FAILED
            self._plane._update_state(
                execution_id, "FAILED",
                summary=f"未知 worker 终态 {result.status}: "
                f"{result.summary[:300]}",
            )

    async def _maybe_escalate(
        self, execution_id: str, mission_id: str, spec: dict, plan,
        blocked_order, blocked_summary: str,
    ) -> bool:
        """能力升级协商（十六审 P0-B）：Worker BLOCKED + 结构化缺能力
        标记 → 升级到覆盖该副作用的 profile，恢复同一 session 继续。

        返回 True = 已接管后续驱动（升级 attempt 跑完并进验收）；
        False = 无授权路径（调用方落 BLOCKED）。最多升级一次——升级后
        仍 BLOCKED 即诚实终态，不循环。"""
        from rosclaw.agentd.task_compiler import (
            PROFILE_CAPABILITY,
            PROFILE_EFFECT_CLASS,
            escalation_profile_for,
            missing_capability_of,
        )
        from rosclaw.agentd.workers.event_store import WorkerEventStore
        from rosclaw.agentd.workers.scheduler import CandidateView
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        missing = missing_capability_of(blocked_summary)
        if not missing:
            return False
        upgraded = escalation_profile_for(missing, plan.profile)
        if not upgraded:
            return False
        state = (
            WorkerEventStore(self._plane._service._home).read_state(
                blocked_order.work_order_id
            )
            or {}
        )
        session_file = str(state.get("session_file") or "")
        if not session_file:
            return False  # 无恢复点——不盲重试
        manager = self._plane._service._worker_manager
        card = self._plane._service._registry.get("worker:rosclaw:pi")
        deliverables = spec.get("deliverables") or []
        self._plane._update_state(
            execution_id, "REPAIRING",
            verifier_feedback=(
                f"worker 报告缺能力 {missing}——升级授权信封 "
                f"{plan.profile}→{upgraded}，恢复同一 session 继续"
            ),
        )
        fresh = manager.order(blocked_order.work_order_id) or blocked_order
        escalated_order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id=mission_id,
            issued_by="task-control-plane",
            capability=PROFILE_CAPABILITY[upgraded],
            goal=str(spec.get("goal", "")),
            inputs={
                "instructions": (
                    str(spec.get("goal", ""))
                    + f"\n\n（授权已升级：你现在拥有 {upgraded} profile 的"
                    "完整工具面——继续完成目标，不要从零开始。）"
                ),
                "worker_profile": upgraded,
                "_execution_id": execution_id,
                "model_snapshot": dict(spec.get("model_snapshot") or {}),
                "_resume_session": session_file,
                **(
                    {"_reuse_workspace": str(fresh.inputs["workspace"])}
                    if fresh.inputs.get("workspace")
                    else {}
                ),
            },
            budgets=BudgetEnvelope(),
            expected_output=ExpectedOutput(
                artifacts=[str(d.get("type", "text/plain")) for d in deliverables]
                or ["text/plain"]
            ),
            side_effect_policy=SideEffectPolicy(
                **{"class": PROFILE_EFFECT_CLASS[upgraded]}
            ),
            parent_work_order_id=blocked_order.work_order_id,
            root_work_order_id=(
                blocked_order.root_work_order_id or blocked_order.work_order_id
            ),
        )
        scheduled = manager.hire(
            escalated_order,
            [CandidateView(card=card, registry_status="ENABLED",
                           running_orders=0, circuit_open=False)],
            attempt_actor="escalate",
            attempt_fingerprint=f"escalate:{missing}",
        )
        self._plane._update_state(
            execution_id, "RUNNING", work_order_id=scheduled.work_order_id
        )
        result, _report = await manager.run_to_completion(scheduled)
        if result.status == "COMPLETED":
            await self._verify_and_repair(
                execution_id, spec, scheduled, result.summary,
                report=self._pure_report(result),
            )
        elif result.status == "BLOCKED":
            self._plane._update_state(
                execution_id, "BLOCKED",
                summary=result.summary[:500] or "升级后仍 BLOCKED",
            )
        elif result.status == "CANCELLED":
            self._plane._update_state(
                execution_id, "CANCELLED",
                summary=result.summary[:500] or "已取消",
            )
        elif result.status == "INTERRUPTED":
            self._plane._update_state(
                execution_id, "INTERRUPTED", summary=result.summary[:500]
            )
        else:
            self._plane._update_state(
                execution_id, "FAILED", summary=result.summary[:500]
            )
        return True

    async def _verify_acceptance(
        self, spec: dict, workspace: Path | None, *, report: str = ""
    ) -> dict:
        """acceptance 真验收（十六审 A2 重写）：

        - checks==0 绝不 pass（零检查成功 = 假成功）；
        - required_files ∪ deliverables 声明的文件路径——存在性检查；
        - run.argv 结构化执行（无 shell、解释器白名单、无凭据 env、
          cwd 限定 workspace）——模型提交的 shell 字符串在 submit 已被拒；
        - 纯文本任务（无文件/无 run）：报告非空是最后一条真实检查；
        - 以上全无 → ACCEPTANCE_MISSING（pass=False）。
        """
        acceptance = spec.get("acceptance") or {}
        failures: list[str] = []
        checks = 0
        required = [str(r) for r in (acceptance.get("required_files") or [])]
        # 交付物即验收：deliverables 声明的 path/name 必须真实存在。
        for d in spec.get("deliverables") or []:
            if isinstance(d, dict):
                rel = str(d.get("path") or d.get("name") or "")
                if rel and rel not in required:
                    required.append(rel)
        for rel in required:
            checks += 1
            candidate = (workspace / rel).resolve() if workspace else None
            if (
                candidate is None
                or workspace is None
                or not str(candidate).startswith(str(workspace.resolve()))
                or not candidate.exists()
            ):
                failures.append(f"缺交付文件 {rel}")
        run = acceptance.get("run")
        if run and workspace is not None:
            checks += 1
            failures.extend(await self._run_structured_check(run, workspace))
        elif run:
            failures.append("验收 run 需要 workspace（缺失）")
            checks += 1
        if checks == 0:
            # 纯文本/问答任务：Worker 必须真的产出了报告（非空）。
            checks = 1
            if not report.strip():
                failures.append(
                    "ACCEPTANCE_MISSING: 无验收定义且 worker 未产出任何"
                    "报告——零证据不得成功"
                )
        return {"pass": not failures, "failures": failures, "checks": checks}

    #: 验收解释器白名单（结构化 argv 的 argv[0]——拒绝 sh/bash/任意二进制）。
    _VERIFIER_ARGV0_ALLOWLIST = frozenset({"python3", "python", "pytest"})

    async def _run_structured_check(
        self, run: dict, workspace: Path
    ) -> list[str]:
        """结构化验收命令：create_subprocess_exec（无 shell），env 从零
        构建（无 API key/无宿主 HOME——凭据不进验收进程），cwd 限定
        workspace。诚实命名：GUARDED_VERIFIER（argv 过滤+凭据隔离，
        不是完整沙箱）。"""
        import os as _os
        import shutil

        argv = [str(a) for a in (run.get("argv") or [])]
        if not argv:
            return ["验收 run.argv 为空"]
        if argv[0] not in self._VERIFIER_ARGV0_ALLOWLIST:
            return [
                f"验收解释器 {argv[0]!r} 不在白名单 "
                f"{sorted(self._VERIFIER_ARGV0_ALLOWLIST)}——拒绝执行"
            ]
        executable = shutil.which(argv[0])
        if executable is None:
            return [f"验收解释器 {argv[0]!r} 未安装"]
        env = {
            k: _os.environ[k]
            for k in ("PATH", "LANG", "LC_ALL", "TZ")
            if k in _os.environ
        }
        env["HOME"] = str(workspace)  # 凭据隔离：验收进程拿不到用户 HOME
        timeout = min(float(run.get("timeout_sec", 600)), 600)
        proc = await asyncio.create_subprocess_exec(
            executable, *argv[1:],
            cwd=str(workspace),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return [f"验收测试超时({timeout}s): {' '.join(argv)}"]
        if proc.returncode != 0:
            tail = (out or b"").decode(errors="replace")[-300:]
            return [f"验收测试失败(rc={proc.returncode}): {tail}"]
        return []

    @staticmethod
    def _pure_report(result) -> str | None:
        """模型原始报告（evidence.final_report）——验收的报告检查
        用纯报告，不用含 [workbench] 注解的摘要。返回 None = 该结果
        没有结构化报告证据（调用方回退摘要）。"""
        for item in getattr(result, "evidence", None) or []:
            if isinstance(item, dict) and item.get("kind") == "final_report":
                return str(item.get("text") or "")
        return None

    async def _verify_and_repair(
        self,
        execution_id: str,
        spec: dict,
        first_order,
        first_summary: str,
        *,
        report: str | None = None,
    ) -> None:
        """VERIFYING →（失败）REPAIRING 同一 session（≤2 轮）→
        SUCCEEDED/FAILED。artifacts/usage 回灌 task_executions。"""
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        manager = self._plane._service._worker_manager
        current_order = first_order
        summary = first_summary
        # None = 无结构化报告证据（旧 adapter）回退摘要；"" = 结构化
        # 证据确认报告为空——不得被 [workbench] 注解稀释成"有内容"。
        pure_report = first_summary if report is None else report
        verdict = {"pass": False, "failures": ["未验收"], "checks": 0}
        max_rounds = 2
        for round_no in range(max_rounds + 1):
            self._plane._update_state(execution_id, "VERIFYING")
            # hire() 返回的对象没有 workspace 注解（pi_managed 运行时才
            # 回写）——必须重读 DB 行，否则验收读不到工作区。
            fresh = manager.order(current_order.work_order_id) or current_order
            workspace_raw = str(fresh.inputs.get("workspace") or "")
            verdict = await self._verify_acceptance(
                spec, Path(workspace_raw) if workspace_raw else None,
                report=pure_report,
            )
            if verdict["pass"]:
                break
            if round_no >= max_rounds:
                break
            # REPAIRING：验收证据反馈同一 session（resume——不是新 Worker）。
            self._plane._update_state(
                execution_id, "REPAIRING",
                verifier_feedback="；".join(verdict["failures"])[:400],
            )
            state = (
                WorkerEventStore(self._plane._service._home).read_state(
                    current_order.work_order_id
                )
                or {}
            )
            session_file = str(state.get("session_file") or "")
            if not session_file:
                break  # 无恢复点——诚实失败，不盲重试
            retry_order, created, _reason = (
                await self._plane._service._retry_coordinator.request_retry(
                    current_order,
                    cause="DELIVERABLE_REJECTED",
                    actor="repair",
                    note="验收反馈（同一 session 修复，不要从零开始）："
                    + "；".join(verdict["failures"])[:300],
                    resume_session=session_file,
                )
            )
            if not created or retry_order is None:
                break
            self._plane._update_state(
                execution_id, "RUNNING",
                work_order_id=retry_order.work_order_id,
            )
            # coordinator 的 spawn_fn 驱动新 attempt——等终态（不双驱动）。
            for _ in range(24000):
                current = manager.order(retry_order.work_order_id)
                if current and current.status in (
                    "ACCEPTED", "FAILED", "CANCELLED", "EXPIRED",
                ):
                    break
                await asyncio.sleep(0.05)
            current = manager.order(retry_order.work_order_id)
            if current is None or current.status != "ACCEPTED":
                row = self._plane._conn.execute(
                    "SELECT result_json FROM work_results WHERE work_order_id = ?",
                    (retry_order.work_order_id,),
                ).fetchone()
                if row is not None:
                    summary = str(
                        json.loads(row["result_json"]).get("summary", "")
                    )[:500]
                verdict = {
                    "pass": False,
                    "failures": [f"修复 attempt 终态 {current.status if current else '?'}"],
                    "checks": verdict["checks"],
                }
                current_order = retry_order
                continue
            row = self._plane._conn.execute(
                "SELECT result_json FROM work_results WHERE work_order_id = ?",
                (retry_order.work_order_id,),
            ).fetchone()
            if row is not None:
                payload = json.loads(row["result_json"])
                summary = str(payload.get("summary", ""))[:500]
                pure_report = next(
                    (str(e.get("text") or "")[:2000]
                     for e in payload.get("evidence", [])
                     if isinstance(e, dict) and e.get("kind") == "final_report"),
                    summary,
                )
            current_order = retry_order
        if verdict["pass"]:
            # 回灌 artifacts/usage（不再是 500 字摘要了事）。
            row = self._plane._conn.execute(
                "SELECT result_json FROM work_results WHERE work_order_id = ?",
                (current_order.work_order_id,),
            ).fetchone()
            artifacts_json = ""
            if row is not None:
                payload = json.loads(row["result_json"])
                artifacts_json = json.dumps(
                    {
                        "artifacts": payload.get("artifacts", []),
                        "usage": payload.get("usage", {}),
                        "verifier": {
                            "checks": verdict["checks"],
                            "verdict": "PASS",
                        },
                    },
                    ensure_ascii=False,
                )
            # 十六审：SUCCEEDED 摘要必须是干净结论——失败 attempt 的
            # AdapterError 文本不得出现在成功摘要里（验收在文件落盘后
            # 才过，摘要却挂着 429 错误 = 用户无法理解的假矛盾）。
            if "AdapterError" in summary or "worker attempt failed" in summary:
                summary = (
                    pure_report.strip()[:400]
                    if pure_report.strip()
                    else "验收通过——交付物已确认"
                )
            self._plane._update_state(
                execution_id, "SUCCEEDED",
                summary=f"{summary[:400]}（验收 PASS·{verdict['checks']} 项）",
                artifacts_json=artifacts_json,
            )
        else:
            # 十六审 A2：零证据（无验收定义且无产出）是 BLOCKED 不是
            # FAILED——任务根本无法被诚实验收，重试无意义。
            if any(f.startswith("ACCEPTANCE_MISSING") for f in verdict["failures"]):
                self._plane._update_state(
                    execution_id, "BLOCKED",
                    summary=(
                        "ACCEPTANCE_MISSING：任务无验收定义且执行未产出"
                        "任何证据——不能宣称成功，也未真实失败"
                    ),
                )
            else:
                self._plane._update_state(
                    execution_id, "FAILED",
                    summary=(
                        f"验收未过：{'；'.join(verdict['failures'])[:300]}"
                        f"（修复 {max_rounds} 轮预算内未达标）"
                    ),
                )
