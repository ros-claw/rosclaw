"""Audit-remediation regression tests (对照总纲审计的 5 项违规 + 归因链).

V1: EXACT_ACTION 必须声明动作意图（省略/mismatch 都拒绝）
V2: broker 签名密钥不可由公开 policy_hash 推导
V3: 成员失联时副作用任务不盲目重排队
V4: world merge 是 latest-valid（后到旧观测不得覆盖）
V5: 预算超限进入 WAIT_INPUT 且不执行决策
G4: decisions/context_manifests/work_results/trace_id 归因链落库
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.mission import MissionStore
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.mission import Budgets, MissionState
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.contracts.operator.approval import ActionDisplayV1, ApprovalRequestV2
from rosclaw.contracts.team.member import MemberBody, TeamMemberCardV1
from rosclaw.contracts.team.world import ObjectState, SharedWorldDeltaV1
from rosclaw.operator import GrantDeniedError, OperatorBroker
from rosclaw.team import TeamCoordinator
from rosclaw.team.allocator import Bid, TaskAnnouncement
from tests.agentd.conftest import LOCAL_PRINCIPAL

NOW = datetime.now(UTC)


def _approval_request() -> ApprovalRequestV2:
    return ApprovalRequestV2(
        request_id="appr_audit",
        mission_id="mis_x",
        principal=LOCAL_PRINCIPAL,
        body_id="sim/ur5e",
        effective_body_hash="body_abc",
        mode="SIMULATION",
        action_display=ActionDisplayV1(title="move", summary="joints → home", risk_tier="LOW"),
        context_id="ctx_1",
        context_revision=1,
        created_at=NOW.isoformat(),
        expires_at=(NOW + timedelta(minutes=10)).isoformat(),
    )


@pytest.fixture
def broker(tmp_path: Path) -> OperatorBroker:
    store = MissionStore(tmp_path / "m.db")
    return OperatorBroker(store.connection, policy_hash="pol_audit")


class TestV1ExactActionBinding:
    def test_missing_action_intent_denied(self, broker: OperatorBroker) -> None:
        broker.create_request(_approval_request())
        grant = broker.decide("appr_audit", principal=LOCAL_PRINCIPAL, approve=True)
        with pytest.raises(GrantDeniedError, match="missing_action_intent"):
            broker.verify(
                grant.grant_id,
                principal=LOCAL_PRINCIPAL,
                body_hash="body_abc",
                mode="SIMULATION",
                risk_tier="LOW",
                action_intent=None,
            )

    def test_broker_computed_intent_accepted_then_consumed(self, broker: OperatorBroker) -> None:
        broker.create_request(_approval_request())
        grant = broker.decide("appr_audit", principal=LOCAL_PRINCIPAL, approve=True)
        intent = broker.action_intent_for_grant(grant.grant_id)
        assert intent is not None
        broker.verify(
            grant.grant_id,
            principal=LOCAL_PRINCIPAL,
            body_hash="body_abc",
            mode="SIMULATION",
            risk_tier="LOW",
            action_intent=intent,
        )

    def test_wrong_intent_denied(self, broker: OperatorBroker) -> None:
        broker.create_request(_approval_request())
        grant = broker.decide("appr_audit", principal=LOCAL_PRINCIPAL, approve=True)
        with pytest.raises(GrantDeniedError, match="action_intent_mismatch"):
            broker.verify(
                grant.grant_id,
                principal=LOCAL_PRINCIPAL,
                body_hash="body_abc",
                mode="SIMULATION",
                risk_tier="LOW",
                action_intent="deadbeef" * 8,
            )


class TestV2BrokerKey:
    def test_key_not_derivable_from_policy_hash(self, broker: OperatorBroker) -> None:
        row = broker._conn.execute(
            "SELECT value FROM broker_state WHERE key = 'signing_key'"
        ).fetchone()
        assert row is not None
        derivable = hashlib.sha256(b"operator:pol_audit").digest()
        assert bytes(row["value"]) != derivable

    def test_signature_forgery_via_public_input_fails(self, broker: OperatorBroker) -> None:
        broker.create_request(_approval_request())
        grant = broker.decide("appr_audit", principal=LOCAL_PRINCIPAL, approve=True)
        # Attacker recomputes HMAC with the previously-derivable key.
        attacker_key = hashlib.sha256(b"operator:pol_audit").digest()
        forged = hmac.new(attacker_key, grant.public_hash.encode(), hashlib.sha256).hexdigest()
        broker._conn.execute(
            "UPDATE mission_grants SET private_signature = ? WHERE grant_id = ?",
            (forged, grant.grant_id),
        )
        with pytest.raises(GrantDeniedError, match="forged_grant"):
            broker.verify(
                grant.grant_id,
                principal=LOCAL_PRINCIPAL,
                body_hash="body_abc",
                mode="SIMULATION",
                risk_tier="LOW",
                action_intent=broker.action_intent_for_grant(grant.grant_id),
            )

    def test_key_stable_across_broker_instances(self, tmp_path: Path) -> None:
        store = MissionStore(tmp_path / "m.db")
        b1 = OperatorBroker(store.connection, policy_hash="pol_audit")
        b2 = OperatorBroker(store.connection, policy_hash="pol_other")
        row1 = b1._conn.execute("SELECT value FROM broker_state").fetchone()
        assert b1._secret == b2._secret == bytes(row1["value"])


class TestV3SideEffectRequeue:
    def test_side_effect_task_not_requeued(self, tmp_path: Path) -> None:
        store = MissionStore(tmp_path / "m.db")
        coord = TeamCoordinator(store.connection, team_id="t", actor_id="c", policy_hash="p")
        coord.join_member(
            TeamMemberCardV1(
                team_id="t",
                member_id="r1",
                body=MemberBody(**{"body_id": "r1", "effective_body_hash": "h", "class": "mb"}),
                capabilities=["x.y"],
            )
        )
        for i, side_effect in enumerate(("none", "workspace_write")):
            ann = TaskAnnouncement(
                task_id=f"task_{i}",
                team_id="t",
                team_epoch=coord.epoch(),
                required_capabilities=("x.y",),
                side_effect_class=side_effect,
            )
            bids = [
                Bid(
                    member_id="r1",
                    eta_ms=1,
                    energy_cost=1,
                    capability_fit=1.0,
                    reliability=0.9,
                    current_load=0.0,
                    comms_quality=1.0,
                )
            ]
            coord.announce_and_award(ann, bids)
            coord.accept_task(f"task_{i}", "r1")
        store.connection.execute(
            "UPDATE team_members SET last_seen_at = '2000-01-01' WHERE member_id = 'r1'"
        )
        coord.membership.sweep_ttl(suspect_after_ms=1, lost_after_ms=2)
        coord.member_lost("r1")
        rows = {
            r["task_id"]: r["status"]
            for r in store.connection.execute("SELECT task_id, status FROM team_tasks")
        }
        assert rows["task_0"] == "ANNOUNCED"  # 无副作用：可重公告
        assert rows["task_1"] == "ACCEPTED"  # 副作用：保持，等 reconcile


class TestV4LatestValid:
    def test_older_observation_losing_race_ignored(self, tmp_path: Path) -> None:
        store = MissionStore(tmp_path / "m.db")
        coord = TeamCoordinator(store.connection, team_id="t", actor_id="c", policy_hash="p")
        coord.join_member(
            TeamMemberCardV1(
                team_id="t",
                member_id="r1",
                body=MemberBody(**{"body_id": "r1", "effective_body_hash": "h", "class": "mb"}),
                capabilities=[],
            )
        )

        def delta(x: float, observed: datetime) -> SharedWorldDeltaV1:
            return SharedWorldDeltaV1(
                team_id="t",
                team_epoch=coord.epoch(),
                world_revision=1,
                base_revision=0,
                source_member="r1",
                observed_at=observed.isoformat(),
                published_at=observed.isoformat(),
                objects=[ObjectState(object_id="ball", pose={"x": x}, confidence=0.9)],
            )

        newer = datetime.now(UTC)
        older = newer - timedelta(milliseconds=300)
        coord.merge_world_delta(delta(x=2.0, observed=newer), now=newer)
        warnings = coord.merge_world_delta(delta(x=1.0, observed=older), now=newer)
        assert warnings and "stale observation ignored" in warnings[0]
        fresh = coord.world.fresh_objects(now=newer, max_age_ms=1000)
        assert fresh["ball"].pose["x"] == 2.0


class _BudgetBuster:
    """Scripted gateway that always answers with big token usage."""

    def __call__(self, request) -> ModelTurnResultV1:
        decision = {
            "schema_version": "rosclaw.decision.v1",
            "decision_id": "d",
            "mission_id": request.mission_id,
            "context_id": request.context_id,
            "context_revision": request.context_revision,
            "next_intent": "ANSWER",
            "summary": "ok",
            "evidence_refs": [],
        }
        return ModelTurnResultV1(
            turn_id="t",
            provider="mock",
            model="m",
            content=f"```json\n{json.dumps(decision)}\n```",
            assistant_message={"role": "assistant", "content": "x"},
            usage={"prompt_tokens": 10_000, "completion_tokens": 5_000, "total_tokens": 15_000},  # type: ignore[arg-type]
        )


class TestV5BudgetGate:
    async def test_budget_exceeded_parks_mission(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        gateway = MockModelGateway(mock_profile(), [_BudgetBuster()] * 5)
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            mission = service.create_mission("预算测试")
            # Tighten the durable budget below one turn.
            service.store.connection.execute(
                "UPDATE missions SET budgets_json = ? WHERE mission_id = ?",
                (
                    Budgets(model_tokens=1000).model_dump_json(),
                    mission.mission_id,
                ),
            )
            result = await service.send_turn(mission.mission_id, "开始")
            assert result.degraded == "budget_exceeded"
            assert result.state is MissionState.WAIT_INPUT
            assert "暂停" in result.reply
        finally:
            await service.close()


class TestAttributionChain:
    async def test_decisions_manifests_trace(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        gateway = MockModelGateway(mock_profile(), [_BudgetBuster()])
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            mission = service.create_mission("归因测试")
            await service.send_turn(mission.mission_id, "你好")
            conn = service.store.connection
            decision = conn.execute(
                "SELECT context_id, validated FROM decisions WHERE mission_id = ?",
                (mission.mission_id,),
            ).fetchone()
            assert decision is not None and decision["validated"] == 1
            manifest = conn.execute(
                "SELECT bundle_hash, prompt_hash FROM context_manifests WHERE mission_id = ?",
                (mission.mission_id,),
            ).fetchone()
            assert manifest is not None
            assert manifest["prompt_hash"].startswith("prompt_")
            assert manifest["bundle_hash"].startswith("ctxb_")
            traces = [
                e["trace_id"]
                for e in service.store.events(mission.mission_id)
                if e["event_type"] == "rosclaw.agent.mission.transition.v1"
            ]
            assert traces and all(t and t.startswith("tr_") for t in traces)
        finally:
            await service.close()
