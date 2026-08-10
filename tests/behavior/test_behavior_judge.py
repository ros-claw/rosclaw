"""PR-EIGHT-1 红测试（八审 §4 P0-1）：Behavior Gate 判定器。

红测试先行——当前没有行为判定器：真实会话的 25 次模型阶段/16 次
动作提案/12 次 CONTEXT_NOT_FRESH/3-边五角星只能靠人眼审计。

判定器必须把八审 §0 的失败特征固化为回归：同签名会话必须 FAIL；
满足 SLO 的会话 PASS。禁止用"模型偶尔发挥不好"豁免。
"""

from __future__ import annotations


def _judge(metrics: dict):
    from rosclaw.agentd.behavior_judge import judge_session

    return judge_session(metrics)


def _failed_user_session() -> dict:
    """八审 §0 记录的真实失败会话签名（2026-08-10 用户实测）。"""
    return {
        "goal": "draw star",
        "user_messages": 4,  # 多次"继续"
        "model_requests": 25,
        "action_proposals": 16,
        "observation_calls": 5,
        "capability_queries": 2,
        "context_not_fresh_retries": 12,
        "schema_retries": 3,  # trajectory/points/hash 猜测
        "hash_retries": 2,
        "lease_retries": 2,
        "task_completed": True,  # 模型自称完成
        "verifier_pass": False,  # 只画完 3/5 边；无 verifier PASS
        "user_confirmations": 0,
        "conflict_with_kernel": True,  # 模型叙述与内核结果冲突
    }


def _good_session() -> dict:
    """目标体验：一句话 → 1 次 task → verifier PASS。"""
    return {
        "goal": "draw star",
        "user_messages": 1,
        "model_requests": 2,
        "action_proposals": 1,
        "observation_calls": 0,
        "capability_queries": 1,
        "context_not_fresh_retries": 0,
        "schema_retries": 0,
        "hash_retries": 0,
        "lease_retries": 0,
        "task_completed": True,
        "verifier_pass": True,
        "user_confirmations": 0,
        "conflict_with_kernel": False,
    }


class TestBehaviorJudge:
    def test_documented_failure_session_fails(self) -> None:
        verdict = _judge(_failed_user_session())
        assert verdict["verdict"] == "FAIL", (
            f"八审记录的真实失败会话竟被判过: {verdict}"
        )
        reasons = verdict["violations"]
        # 关键违规逐项命中（不是一句笼统 fail）。
        assert any("model_requests" in r for r in reasons)
        assert any("action_proposals" in r for r in reasons)
        assert any("context_not_fresh" in r for r in reasons)
        assert any("verifier" in r for r in reasons)  # 未 PASS 却称完成
        assert any("user_messages" in r for r in reasons)  # 要求"继续"

    def test_good_session_passes(self) -> None:
        verdict = _judge(_good_session())
        assert verdict["verdict"] == "PASS", f"达标会话被误判: {verdict}"

    def test_unverified_completion_is_fail(self) -> None:
        """verifier 未 PASS 而自称完成——单独即 FAIL（发布阻断条件）。"""
        metrics = _good_session()
        metrics["verifier_pass"] = False
        verdict = _judge(metrics)
        assert verdict["verdict"] == "FAIL"
        assert any("verifier" in r for r in verdict["violations"])

    def test_slo_thresholds_visible(self) -> None:
        """判定器暴露 SLO 阈值（CI/报告引用同一定义）。"""
        from rosclaw.agentd.behavior_judge import STAR_TASK_SLO

        assert STAR_TASK_SLO["model_requests_max"] == 3
        assert STAR_TASK_SLO["action_proposals_exact"] == 1
        assert STAR_TASK_SLO["user_messages"] == 1
        assert STAR_TASK_SLO["retry_budget"] == 0
