"""Benchmark harness + learning pipeline tests (PR-EV-080/081 exits)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.bench.harness import (
    GROUP_A_NATIVE_ONLY,
    GROUP_B_NATIVE_WORKERS,
    BenchmarkRunner,
    aggregate,
    default_scenarios,
)
from rosclaw.agentd.context.sources import EvidenceClass
from rosclaw.agentd.learning.pipeline import (
    EvidenceRejectedError,
    LearningPipeline,
    PromotionGateError,
)
from rosclaw.agentd.mission import MissionStore
from tests.agentd.conftest import LOCAL_PRINCIPAL


def _home_factory(base: Path):
    def make(seed: int) -> Path:
        home = base / f"run_{seed}"
        home.mkdir(parents=True, exist_ok=True)
        return home

    return make


class TestBenchmark:
    async def test_matrix_ab_groups(self, tmp_path: Path) -> None:
        runner = BenchmarkRunner(
            _home_factory(tmp_path / "homes"), reporter_dir=tmp_path / "reports"
        )
        scenarios = default_scenarios()
        results = await runner.run_matrix(
            scenarios, seeds=[1, 2, 3], groups=[GROUP_A_NATIVE_ONLY, GROUP_B_NATIVE_WORKERS]
        )
        assert len(results) == 2 * 2 * 3
        report = aggregate(results)
        chat_a = report["families"]["chat_answer/A"]
        assert chat_a["success_rate"] == 1.0
        assert chat_a["unsupported_claim_rate"] == 0.0
        # Delegation succeeds only with the worker channel (group B).
        assert report["families"]["delegation_analysis/B"]["success_rate"] == 1.0
        assert report["families"]["delegation_analysis/B"]["delegation_accept_rate"] == 1.0
        assert report["families"]["delegation_analysis/A"]["success_rate"] == 0.0
        # Reports are persisted artifacts.
        artifacts = list((tmp_path / "reports").glob("*.json"))
        assert len(artifacts) == 12

    async def test_deterministic_given_seed(self, tmp_path: Path) -> None:
        runner = BenchmarkRunner(_home_factory(tmp_path / "h1"))
        scenario = default_scenarios()[0]
        r1 = await runner.run_once(scenario, seed=7, group_id=GROUP_B_NATIVE_WORKERS)
        runner2 = BenchmarkRunner(_home_factory(tmp_path / "h2"))
        r2 = await runner2.run_once(scenario, seed=7, group_id=GROUP_B_NATIVE_WORKERS)
        assert r1.success == r2.success
        assert r1.tokens_used == r2.tokens_used


@pytest.fixture
def store(tmp_path: Path) -> MissionStore:
    return MissionStore(tmp_path / "m.db")


class TestLearningPipeline:
    def test_evidence_gate_rejects_unverified(self, store: MissionStore) -> None:
        pipe = LearningPipeline(store.connection, actor_id="agent:test")
        with pytest.raises(EvidenceRejectedError):
            pipe.propose(
                kind="MEMORY",
                title="rumor",
                content={},
                evidence_class=EvidenceClass.UNVERIFIED,
                evidence_refs=["mem://x"],
            )
        with pytest.raises(EvidenceRejectedError):
            pipe.propose(
                kind="HOW",
                title="guess",
                content={},
                evidence_class=EvidenceClass.INFERRED,
                evidence_refs=[],
            )

    def test_promotion_gate(self, store: MissionStore) -> None:
        pipe = LearningPipeline(store.connection, actor_id="agent:test")
        cid = pipe.propose(
            kind="HOW",
            title="retry-with-backoff pattern",
            content={"pattern": "exponential backoff"},
            evidence_class=EvidenceClass.CURATED,
            evidence_refs=["dec://1"],
        )
        # No auto-promotion: agent principal rejected.
        with pytest.raises(PromotionGateError, match="human principal"):
            pipe.promote(cid, principal="agent:rosclaw-native:b1", evaluation_ref="eval://1")
        # No evaluation reference rejected.
        with pytest.raises(PromotionGateError, match="evaluation reference"):
            pipe.promote(cid, principal=LOCAL_PRINCIPAL, evaluation_ref="")
        # Human + evaluation → promoted.
        pipe.promote(cid, principal=LOCAL_PRINCIPAL, evaluation_ref="eval://darwin/1")
        rows = pipe.list(status="PROMOTED")
        assert len(rows) == 1
        assert rows[0]["promoted_by"] == LOCAL_PRINCIPAL

    def test_extract_from_accepted_work(self, tmp_path: Path) -> None:
        from rosclaw.agentd.config import load_agent_config
        from rosclaw.agentd.models.gateway import MockModelGateway
        from rosclaw.agentd.models.profiles import mock_profile
        from rosclaw.agentd.service import AgentService
        from rosclaw.contracts.agent.model_turn import ModelTurnResultV1

        def hire(request):
            decision = {
                "schema_version": "rosclaw.decision.v1",
                "decision_id": "d1",
                "mission_id": request.mission_id,
                "context_id": request.context_id,
                "context_revision": request.context_revision,
                "next_intent": "HIRE_WORKER",
                "summary": "委派",
                "evidence_refs": ["a://1"],
                "proposed_operation": {"type": "create_work_order", "payload": {"goal": "分析"}},
                "verification": {
                    "schema_version": "rosclaw.decision_verification.v1",
                    "verifiers": ["deterministic:x"],
                },
            }
            return ModelTurnResultV1(
                turn_id="t",
                provider="mock",
                model="m",
                content=f"```json\n{json.dumps(decision)}\n```",
                assistant_message={"role": "assistant", "content": None},
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},  # type: ignore[arg-type]
            )

        def worker(request):
            return ModelTurnResultV1(
                turn_id="t2",
                provider="mock",
                model="m",
                content="分析结果",
                assistant_message={"role": "assistant", "content": "x"},
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},  # type: ignore[arg-type]
            )

        config = load_agent_config(tmp_path / "config.yaml")
        service = AgentService(
            config, tmp_path, gateway=MockModelGateway(mock_profile(), [hire, worker])
        )
        import asyncio

        async def flow():
            mission = service.create_mission("学习测试")
            await service.send_turn(mission.mission_id, "委派分析")
            return mission

        mission = asyncio.run(flow())
        pipe = LearningPipeline(service.store.connection, actor_id="agent:test")
        created = pipe.extract_from_mission(mission.mission_id)
        assert created  # verified receipt formed a candidate
        candidates = pipe.list(status="CANDIDATE")
        classes = {c["evidence_class"] for c in candidates}
        assert classes <= {"verified_receipt", "curated"}
        asyncio.run(service.close())
