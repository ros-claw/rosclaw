"""输入路由自动执行（R0-1.5，金丝雀 4/10 实证 + 0826 审计收口）。

已知 recipe 的画路径指令在用户输入持久化时由**内核**自动执行
（TaskExecutionService 唯一生产链）——零模型工具调用。金丝雀
实证：任务级入口在场不等于模型会选它（绕链手拼/竖直平面被拒
后降级调试都是产品侧失败，不是模型侧借口）。

纪律：
- 仅 SIM 模式（REAL/SHADOW 永远走 rosclawd 权威链——自动路由
  不是审批旁路）；
- 疑问句/讨论形式不自动执行（is_task_directive 护栏）；
- 幂等：同一 message_id 只路由一次（进程内去重 + input.task_id
  回写检查）；
- 执行失败=数据（plan.node_failed/PARTIAL outcome）——不崩输入
  通道。
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService

#: 进程内已路由去重（message_id）——与 input.task_id 回写双保险。
_routed: set[str] = set()


async def maybe_auto_route(
    service: AgentService,
    *,
    mission_id: str,
    session_ref: str,
    message_id: str,
    text: str,
) -> dict[str, Any] | None:
    """指令性画路径输入 → 自动路由执行。返回 auto_task 描述或
    None（不可路由/疑问句/重放）。"""
    from rosclaw.task_kernel.task_router import (
        compile_recipe_inputs,
        is_task_directive,
        route_recipe,
    )
    from rosclaw.task_kernel.task_spec import _classify_intent

    text = text.strip()
    if not text or not is_task_directive(text):
        return None
    kernel = service._task_kernel
    # P0-1/P0-2（0827 审计·双控制者根治）：重放优先——同一
    # message_id 已认领（进程内去重）或已附着任务（持久层去重，
    # 覆盖 daemon 重启后的重投递）时，返回同一任务的 auto_task
    # 描述。若重放返回 None，输入会掉进模型路径=双执行。
    if message_id in _routed:
        row = kernel._conn.execute(
            "SELECT task_id FROM user_inputs WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        if row is not None and row["task_id"]:
            return {
                "task_id": str(row["task_id"]),
                "recipe_id": "",
                "state": "RUNNING",
                "inputs": {},
                "replayed": True,
            }
        return None
    intent = _classify_intent(text)
    recipe_id = route_recipe({"goal": {"intent": intent}})
    if recipe_id is None:
        return None
    # 0902 R0-2（§3.1 硬规则）：语义覆盖率门禁——recipe 只在材料性
    # 要求 100% 覆盖时执行。0902 实证：关键词命中吞掉"红色圆柱笔/
    # 3D 轨迹/不要 2D"→ 旧视频复用 → PASS/DELIVERED 假成功。任一
    # must/forbidden 条款未被 recipe 声明覆盖 → 不自动路由（交
    # Native Agent 编译 TaskSpec——不猜）。门禁在任务创建之前——
    # 未覆盖不产生幽灵任务。
    from rosclaw.task_kernel.requirements import compile_requirements
    from rosclaw.task_kernel.task_router import RECIPE_COVERAGE

    coverage = RECIPE_COVERAGE.get(recipe_id, frozenset())
    uncovered = [r for r in compile_requirements(text)
                 if r.verifier not in coverage]
    if uncovered:
        import logging

        logging.getLogger("rosclaw.auto_route").info(
            "coverage gate: recipe %s 未覆盖 %s——转 Native Agent 路径",
            recipe_id,
            [r.verifier for r in uncovered],
        )
        return None
    mission = service.get_mission(mission_id)
    if mission is not None and mission.mode.value != "SIMULATION":
        return None  # REAL/SHADOW 不自动路由（rosclawd 权威链）
    # 重放/已附着输入不重路由（持久层双保险：返回同一任务而不是
    # None——None 意味着"交给模型"，对重放即双控制者）。
    row = kernel._conn.execute(
        "SELECT task_id FROM user_inputs WHERE message_id = ?", (message_id,),
    ).fetchone()
    if row is not None and row["task_id"]:
        return {
            "task_id": str(row["task_id"]),
            "recipe_id": "",
            "state": "RUNNING",
            "inputs": {},
            "replayed": True,
        }
    try:
        bound = kernel.ensure_task_for_effect(
            mission_id=mission_id,
            session_ref=session_ref,
            backend_native_id=session_ref,
            cwd="",
            mode="SIMULATION",
            body_id=(
                mission.body_binding.body_id if mission else ""
            ),
            explicit_goal=text,
        )
    except ValueError:
        return None  # 动机缺失等——诚实不路由
    task_id = str(bound["task_id"])
    recipe_id = route_recipe(
        kernel.get_task_spec(task_id) or {"goal": {"intent": intent}}
    )
    if recipe_id is None:
        return None
    _routed.add(message_id)
    inputs = compile_recipe_inputs(text)

    async def _run() -> None:
        try:
            outcome = await asyncio.to_thread(
                service._task_execution.execute,
                task_id,
                recipe_inputs=inputs,
            )
            # 0827 复核实证：断链不能沉默——execute 失败（node_failed/
            # 验收失败）时任务停在 RUNNING、无 task.terminal，用户
            # 永远等不到回复。无模型可修复的路径（suppress 了模型
            # 回合）必须诚实终态：FAILED + task.terminal——watcher
            # 由此呈现失败回复（用户可追问开新 revision）。
            if not outcome.ok:
                kernel.transition(
                    task_id, "FAILED",
                    reason=(
                        f"{outcome.error_code or 'EXECUTION_FAILED'}: "
                        f"{(outcome.failure or ';'.join(outcome.failures))[:200]}"
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - 执行失败是任务数据
            # 不沉默——失败写诊断日志（金丝雀实证：静默吞错导致
            # "任务 RUNNING 无事件"无迹可查）。
            import logging
            import traceback

            logging.getLogger("rosclaw.auto_route").error(
                "auto-route execution failed for %s: %s\n%s",
                task_id, exc, traceback.format_exc()[-2000:],
            )
            # 同上：异常断链也到 FAILED 终态（不沉默）。
            try:
                kernel.transition(
                    task_id, "FAILED",
                    reason=f"EXECUTION_EXCEPTION: {exc}"[:220],
                )
            except Exception:  # noqa: BLE001 - 终态迁移失败只记日志
                logging.getLogger("rosclaw.auto_route").exception(
                    "FAILED transition failed for %s", task_id,
                )

    asyncio.get_event_loop().create_task(_run())
    return {
        "task_id": task_id,
        "recipe_id": recipe_id,
        "state": "RUNNING",
        "inputs": inputs,
    }


def reset_routed_for_tests() -> None:
    _routed.clear()


__all__ = ["maybe_auto_route", "reset_routed_for_tests"]
