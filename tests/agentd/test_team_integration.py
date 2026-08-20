"""Agent-level Team Fabric integration tests (PR-TF + agentd).

- team disabled by default → TEAM_COORDINATE honest refusal
- team enabled → team_task_claim allocates via contract net with journal
- team state surfaces in the L6 org layer (epoch/membership changes trigger
  recompile reasons)
"""

from __future__ import annotations

import json
from pathlib import Path

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
