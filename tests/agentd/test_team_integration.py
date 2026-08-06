"""Agent-level Team Fabric integration tests (PR-TF + agentd).

- team disabled by default → TEAM_COORDINATE honest refusal
- team enabled → team_task_claim allocates via contract net with journal
- team state surfaces in the L6 org layer (epoch/membership changes trigger
  recompile reasons)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.contracts.team.member import MemberBody, TeamMemberCardV1


def _team_decision(request) -> ModelTurnResultV1:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "dec_team",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": "TEAM_COORDINATE",
        "summary": "请求团队分配巡检任务",
        "evidence_refs": [],
        "proposed_operation": {
            "type": "team_task_claim",
            "payload": {
                "required_capabilities": ["navigation.local"],
                "success_criteria": "区域巡检完成",
                "idempotency_key": "team-test-1",
            },
        },
        "verification": {
            "schema_version": "rosclaw.decision_verification.v1",
            "verifiers": ["deterministic:schema"],
        },
    }
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="mock-model",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


def _service(tmp_path: Path, team_enabled: bool) -> AgentService:
    config = load_agent_config(tmp_path / "config.yaml")
    config.raw["team"] = {"enabled": team_enabled, "team_id": "blue_team"}
    gateway = MockModelGateway(mock_profile(), [_team_decision] * 10)
    return AgentService(config, tmp_path, gateway=gateway)


def _member(member_id: str) -> TeamMemberCardV1:
    return TeamMemberCardV1(
        team_id="blue_team",
        member_id=member_id,
        body=MemberBody(
            **{"body_id": member_id, "effective_body_hash": "h", "class": "mobile_base"}
        ),
        capabilities=["navigation.local"],
    )


class TestTeamIntegration:
    async def test_disabled_honest_refusal(self, tmp_path: Path) -> None:
        service = _service(tmp_path, team_enabled=False)
        mission = service.create_mission("团队测试")
        result = await service.send_turn(mission.mission_id, "协调团队")
        assert "尚未启用" in result.reply
        await service.close()

    async def test_task_claim_allocated_with_journal(self, tmp_path: Path) -> None:
        service = _service(tmp_path, team_enabled=True)
        coord = service._team_coordinator
        coord.join_member(_member("robot:limo:blue_01"))
        coord.join_member(_member("robot:limo:blue_02"))
        mission = service.create_mission("团队巡检")
        result = await service.send_turn(mission.mission_id, "请团队分配巡检任务")
        assert "已按 contract_net.v1 分配" in result.reply
        assert "本地 Native Agent 与" in result.reply  # local-safety disclaimer
        # Award is journaled with bids feature vectors.
        row = service.store.connection.execute(
            "SELECT bids_json, status, awardee FROM team_tasks WHERE task_id LIKE 'ttask_%' "
            "OR idempotency_key = 'team-test-1'"
        ).fetchone()
        assert row is not None
        bids = json.loads(row["bids_json"])
        assert len(bids) == 2
        assert all("features" in b for b in bids)
        # Idempotent: same claim returns the same award.
        task_row = service.store.connection.execute(
            "SELECT COUNT(*) AS n FROM team_tasks WHERE idempotency_key = 'team-test-1'"
        ).fetchone()
        assert task_row["n"] == 1
        await service.close()

    async def test_epoch_change_invalidates_old_claims(self, tmp_path: Path) -> None:
        service = _service(tmp_path, team_enabled=True)
        coord = service._team_coordinator
        coord.join_member(_member("robot:limo:blue_01"))
        mission = service.create_mission("epoch 测试")
        await service.send_turn(mission.mission_id, "分配任务")
        first_epoch = coord.epoch()
        coord.membership.leave("robot:limo:blue_01")  # bumps epoch
        assert coord.epoch() > first_epoch
        # Old-epoch announcement is rejected by the coordinator directly.
        from rosclaw.team import TeamError
        from rosclaw.team.allocator import TaskAnnouncement

        with pytest.raises(TeamError, match="epoch"):
            coord.announce_and_award(
                TaskAnnouncement(
                    task_id="old",
                    team_id="blue_team",
                    team_epoch=first_epoch,
                    required_capabilities=("navigation.local",),
                ),
                [],
            )
        await service.close()
