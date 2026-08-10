"""Behavior Gate 判定器（八审 §4 P0-1）。

把"任务效率"从人眼审计变成机器判定：一次会话的结构化指标
（模型请求数、动作提案数、重试分类、verifier 结果、用户交互）
对照 SLO 给出 PASS/FAIL + 逐项违规。

判定器是唯一权威——真实模型 journey、失败回归 fixture、CI artifact
分析全部走这里，不允许各测试自造口径。
"""

from __future__ import annotations

from typing import Any

#: 五角星默认 SIM 任务的 SLO（八审 §4 P0-8 表 + §7.3 发布阻断条件）。
STAR_TASK_SLO: dict[str, Any] = {
    "user_messages": 1,  # 用户只发一句话（0 次"继续"）
    "user_confirmations": 0,  # 默认 SIM 零确认
    "model_requests_max": 3,  # ≤3，目标 2
    "action_proposals_exact": 1,  # 恰好一次任务级提案
    "retry_budget": 0,  # schema/hash/lease/CONTEXT_NOT_FRESH 重试全零
    "verifier_pass_required": True,  # verifier PASS 才能声称完成
}


def judge_session(metrics: dict[str, Any]) -> dict[str, Any]:
    """对照 SLO 判定一次会话。metrics 键见 STAR_TASK_SLO 与测试。

    返回 {"verdict": "PASS"|"FAIL", "violations": [str, ...]}——
    violations 逐条列出（不是一句笼统 fail），供 CI artifact 与报告
    直接引用。
    """
    violations: list[str] = []

    user_messages = int(metrics.get("user_messages", 1))
    if user_messages > STAR_TASK_SLO["user_messages"]:
        violations.append(
            f"user_messages={user_messages} > {STAR_TASK_SLO['user_messages']}"
            "（要求用户回复'继续'——发布阻断）"
        )
    confirmations = int(metrics.get("user_confirmations", 0))
    if confirmations > STAR_TASK_SLO["user_confirmations"]:
        violations.append(
            f"user_confirmations={confirmations} > 0（默认 SIM 不应打断）"
        )
    model_requests = int(metrics.get("model_requests", 0))
    if model_requests > STAR_TASK_SLO["model_requests_max"]:
        violations.append(
            f"model_requests={model_requests} > {STAR_TASK_SLO['model_requests_max']}"
        )
    proposals = int(metrics.get("action_proposals", 0))
    if proposals != STAR_TASK_SLO["action_proposals_exact"]:
        violations.append(
            f"action_proposals={proposals} != {STAR_TASK_SLO['action_proposals_exact']}"
            "（必须恰好一次任务级提案）"
        )
    retries = {
        name: int(metrics.get(key, 0))
        for name, key in (
            ("context_not_fresh", "context_not_fresh_retries"),
            ("schema", "schema_retries"),
            ("hash", "hash_retries"),
            ("lease", "lease_retries"),
        )
    }
    for name, count in retries.items():
        if count > STAR_TASK_SLO["retry_budget"]:
            violations.append(f"{name}_retries={count} > 0（确定性协议工作交给了 LLM）")
    # verifier 未 PASS 却声称完成——单独即 FAIL。
    if metrics.get("task_completed") and not metrics.get("verifier_pass"):
        violations.append("verifier 未 PASS 却声称完成（完成真实性违规）")
    if metrics.get("conflict_with_kernel"):
        violations.append("模型叙述与内核权威结果冲突")
    return {"verdict": "FAIL" if violations else "PASS", "violations": violations}
