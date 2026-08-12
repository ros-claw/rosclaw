"""NINE-4 红测试（九审 §6/§21.10/§25.9/11/12）：通用合约 + Judge
硬门 + replay 命名降级。

红测试先行：

1. 通用合约存在且不含 star 字段：GoalEnvelopeV1/OutcomeContractV1/
   TaskSpecV3/TaskGraphV1/ArtifactRefV1/EvidenceClaimV1/WorkOrderV2；
2. Behavior Judge 硬门：input_persisted=False 或 causal_integrity=
   False 直接 FAIL（哪怕其余全绿）；
3. SIM_KIND 降级命名 command-replay（不再叫 kinematic-sandbox）。
"""

from __future__ import annotations


class TestGeneralContracts:
    def test_contracts_exist_and_star_free(self) -> None:
        from rosclaw.contracts.agent import task_contracts as tc

        for name in (
            "GoalEnvelopeV1", "OutcomeContractV1", "TaskSpecV3",
            "TaskGraphV1", "ArtifactRefV1", "EvidenceClaimV1", "WorkOrderV2",
        ):
            cls = getattr(tc, name, None)
            assert cls is not None, f"缺合约 {name}"
            schema = getattr(cls, "SCHEMA", "")
            assert schema.startswith("rosclaw."), f"{name} 缺 SCHEMA"
            fields = " ".join(getattr(cls, "model_fields", {}).keys())
            assert "star" not in fields.lower(), f"{name} 写死了 star 字段: {fields}"


class TestJudgeHardGates:
    def test_input_not_persisted_is_fail(self) -> None:
        from rosclaw.agentd.behavior_judge import judge_session

        metrics = {
            "user_messages": 1, "model_requests": 2, "action_proposals": 1,
            "user_confirmations": 0, "task_completed": True, "verifier_pass": True,
            "input_persisted": False,
        }
        verdict = judge_session(metrics)
        assert verdict["verdict"] == "FAIL"
        assert any("input_persisted" in v for v in verdict["violations"])

    def test_causal_split_is_fail(self) -> None:
        from rosclaw.agentd.behavior_judge import judge_session

        metrics = {
            "user_messages": 1, "model_requests": 2, "action_proposals": 1,
            "user_confirmations": 0, "task_completed": True, "verifier_pass": True,
            "causal_integrity": False,
        }
        verdict = judge_session(metrics)
        assert verdict["verdict"] == "FAIL"
        assert any("causal_integrity" in v for v in verdict["violations"])

    def test_clean_session_still_passes(self) -> None:
        from rosclaw.agentd.behavior_judge import judge_session

        metrics = {
            "user_messages": 1, "model_requests": 2, "action_proposals": 1,
            "user_confirmations": 0, "task_completed": True, "verifier_pass": True,
            "input_persisted": True, "causal_integrity": True,
        }
        assert judge_session(metrics)["verdict"] == "PASS"


class TestReplayNaming:
    def test_sim_kind_is_command_replay(self) -> None:
        from rosclaw.sim.ur5e_mcp import SIM_KIND

        assert SIM_KIND == "command-replay", SIM_KIND
