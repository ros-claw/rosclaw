"""解释性追问的只读确定性回答（0901 体验探讨 P0-4，硬 Gate A）。

0901 实证：任务 FAIL+PARTIAL 后用户问"你这是啥？"——系统没有
只读解释路径，模型重跑了整个任务（新 trace + 第二套 artifact）。

解释/查看类追问（这是啥/结果呢/文件在哪/为什么失败…）且会话里
已有任务时：由 EXPLAIN_HANDLER 认领——TurnOutcome + 交付物列表
直接回答（零模型回合、零新 task/trace/artifact、零仿真渲染）。
"""

from __future__ import annotations

from typing import Any

#: 解释/查看追问标记（通用词类——问"刚才的结果"不是新指令）。
_EXPLAIN_MARKERS = (
    "这是啥", "这是什么", "什么意思", "刚才", "结果呢", "结果是什么",
    "怎么样", "怎么样了", "文件在哪", "在哪呢", "在哪里", "为什么失败",
    "怎么失败", "成功了吗", "成功了么", "给我看", "给我看看", "看看结果",
    "什么情况", "发生什么了", "咋样了", "how did", "what happened",
    "where is", "show me", "what is this",
)


def is_explain_followup(text: str) -> bool:
    """解释/查看追问判定。"""
    lowered = text.lower()
    return any(m in lowered or m in text for m in _EXPLAIN_MARKERS)


def maybe_explain_last_task(
    service: Any, *, mission_id: str, session_ref: str,
) -> dict[str, Any] | None:
    """会话最近任务 + outcome + 交付物 → explain 负载（无任务/无
    outcome 则不劫持——正常聊天走模型）。"""
    kernel = service._task_kernel
    latest = kernel.latest_task_for(mission_id, session_ref)
    if latest is None:
        # session 无绑定回落 mission 最近任务（与 pi.artifact.list 同
        # 语义——session 轮换不丢解释面）。
        row = kernel._conn.execute(
            "SELECT task_id FROM tasks WHERE mission_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (mission_id,),
        ).fetchone()
        if row is None:
            return None
        latest = kernel.get_task(str(row["task_id"]))
    if latest is None:
        return None
    task_id = str(latest["task_id"])
    # outcome 幂等读取（coordinator.consider 是只读评估——已存
    # outcome 原样返回；没有 outcome 的任务给诚实"还在进行"）。
    outcome = None
    try:
        from rosclaw.task_kernel.coordinator import TaskCoordinator

        outcome = TaskCoordinator(kernel).consider(task_id)
    except Exception:  # noqa: BLE001 - outcome 不可得≠不能解释
        outcome = None
    return {
        "task_id": task_id,
        "state": str(latest.get("state", "")),
        "goal": str(latest.get("root_goal", "") or "")[:200],
        "revision": int(latest.get("active_revision") or 1),
        "outcome": outcome,
        "artifacts": kernel.artifact_refs_for(task_id),
    }


__all__ = ["is_explain_followup", "maybe_explain_last_task"]
