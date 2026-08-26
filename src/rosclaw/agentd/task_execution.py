"""TaskExecutionService（R0-1，0826 体验审计 §5.R0-1）——已知任务
的唯一生产入口。

    frozen TaskSpecV2（task_kernel 权威账本）
      → TaskRouter（intent → recipe）
      → recipe handler（typed PlanGraph）
      → PlanExecutor（plan.node_* 事件落 task_events）

纪律：
- 生产 draw path 只有这一条执行链（旧 tool_dispatch._task 内联
  SIM 管线已物理删除）；
- 无 recipe 的 intent / 缺 frozen spec / 未知任务 → 稳定错误码
  诚实拒绝（不猜不编）；
- recipe 失败一次成形（PlanExecutionResult.failed_node/failure
  即 RepairDirective 原料）——模型不得重新编排同一 recipe。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.agentd.plan_templates import RecipeInputError, draw_path_recipe
from rosclaw.task_kernel.plan_executor import PlanExecutionResult
from rosclaw.task_kernel.service import TaskKernel
from rosclaw.task_kernel.task_router import route_recipe

#: recipe_id → handler。recipe 声明的执行域（mode 门禁在 service
#: 层统一执行——recipe 不自判）。
RecipeHandler = Callable[..., PlanExecutionResult]

_RECIPE_REGISTRY: dict[str, RecipeHandler] = {
    "recipe:sim.draw_path": draw_path_recipe,
}

#: recipe 的执行域要求（SIM recipe 只在 SIMULATION mode 下运行；
#: SHADOW/REAL 永远走 rosclawd 权威链——recipe 不是旁路）。
_RECIPE_MODE: dict[str, str] = {
    "recipe:sim.draw_path": "SIMULATION",
}


@dataclass
class TaskExecutionOutcome:
    """任务级执行结果（模型可见 payload 与 TUI 任务卡同一原料）。"""

    ok: bool
    task_id: str
    recipe_id: str = ""
    refs: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    verification_id: str = ""
    failures: list[str] = field(default_factory=list)
    failed_node: str = ""
    failure: str = ""
    error_code: str = ""
    task_state: str = ""


class TaskExecutionService:
    """execute(task_id) 是已知任务的唯一生产入口。"""

    def __init__(
        self,
        *,
        kernel: TaskKernel,
        conn: sqlite3.Connection,
        home: Path,
        runtime_manager=None,
    ) -> None:
        self._kernel = kernel
        self._conn = conn
        self._home = home
        self._runtime_manager = runtime_manager

    def execute(
        self,
        task_id: str,
        *,
        recipe_inputs: dict[str, Any] | None = None,
        goal_hint: str = "",
    ) -> TaskExecutionOutcome:
        task = self._kernel.get_task(task_id)
        if task is None:
            return TaskExecutionOutcome(
                ok=False, task_id=task_id,
                error_code="TASK_NOT_FOUND",
                failure=f"未知任务 {task_id!r}",
            )
        spec = self._kernel.get_task_spec(task_id)
        if spec is None:
            return TaskExecutionOutcome(
                ok=False, task_id=task_id,
                error_code="TASK_SPEC_MISSING",
                failure=f"任务 {task_id} 缺 frozen TaskSpecV2——无法路由",
            )
        recipe_id = route_recipe(spec, goal_hint=goal_hint)
        if recipe_id is None:
            intent = str((spec.get("goal") or {}).get("intent", ""))
            return TaskExecutionOutcome(
                ok=False, task_id=task_id,
                error_code="TASK_NO_RECIPE",
                failure=(
                    f"intent {intent!r} 无确定性 recipe——改用类型化能力"
                    "规划（rosclaw_execute/observe），或诚实报告无法完成"
                ),
            )
        required_mode = _RECIPE_MODE.get(recipe_id, "")
        task_mode = str(task.get("mode") or "SIMULATION")
        if required_mode and task_mode != required_mode:
            return TaskExecutionOutcome(
                ok=False, task_id=task_id, recipe_id=recipe_id,
                error_code="MODE_DENIED",
                failure=(
                    f"{recipe_id} 仅 {required_mode}——当前 {task_mode}"
                    "（REAL/SHADOW 走 rosclawd 权威链）"
                ),
            )
        handler = _RECIPE_REGISTRY[recipe_id]
        try:
            result = handler(
                kernel=self._kernel,
                conn=self._conn,
                home=self._home,
                task_id=task_id,
                inputs=dict(recipe_inputs or {}),
                runtime_manager=self._runtime_manager,
            )
        except RecipeInputError as exc:
            return TaskExecutionOutcome(
                ok=False, task_id=task_id, recipe_id=recipe_id,
                error_code=exc.code, failure=exc.message[:400],
            )
        artifacts = self._ledger_artifacts(task_id)
        verification = result.refs.get("VerificationRef") or {}
        failures = [str(f) for f in verification.get("failures", [])]
        verify_ok = verification.get("status") == "PASS"
        ok = result.ok and verify_ok
        return TaskExecutionOutcome(
            ok=ok,
            task_id=task_id,
            recipe_id=recipe_id,
            refs=result.refs,
            artifacts=artifacts,
            verification_id=str(verification.get("verification_id", "")),
            failures=failures or ([result.failure] if result.failure else []),
            failed_node=result.failed_node,
            failure=(
                "" if ok else (
                    result.failure or "；".join(failures)[:300]
                )
            ),
            error_code="" if ok else (
                "ACCEPTANCE_FAILED" if result.ok else "RECIPE_NODE_FAILED"
            ),
            task_state=str(
                (self._kernel.get_task(task_id) or {}).get("state", "")
            ),
        )

    def _ledger_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        """产物账本的用户可见视图（R0-4：id/kind/media/size/
        digest/open_command——不是"目录里有文件"）。"""
        return self._kernel.artifact_refs_for(task_id)


__all__ = ["TaskExecutionOutcome", "TaskExecutionService"]
