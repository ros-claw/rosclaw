"""AgentEventV2 + /v2 API tests (PR-02 exits).

- journal: per-mission monotonic sequence, transactional
- replay via after_sequence; visibility filtering
- submit decoupled from stream: turn completes even when SSE never connects
- reconnect: replay + live continuation without duplicates
- busy mission: second concurrent submit rejected
- model/tool/verification events emitted during a turn
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.events import AgentEventStore
from rosclaw.agentd.mission import MissionStore
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService, create_app
from rosclaw.contracts.agent.agent_event import AgentEventType, Visibility
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1, ToolCall


def _answer(request) -> ModelTurnResultV1:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "d1",
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
        content=f"回答。\n```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": "回答。"},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


def _tool_then_answer(request) -> ModelTurnResultV1:
    if len(getattr(_tool_then_answer, "calls", [])) == 0:
        _tool_then_answer.calls = [1]
        return ModelTurnResultV1(
            turn_id="t0",
            provider="mock",
            model="m",
            content="",
            tool_calls=[
                ToolCall(call_id="c1", name="sim_get_state", arguments_json='{"verbose": true}')
            ],
            assistant_message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "sim_get_state", "arguments": '{"verbose": true}'},
                    }
                ],
            },
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
        )
    return _answer(request)


@pytest.fixture
def service(tmp_path: Path) -> AgentService:
    config = load_agent_config(tmp_path / "config.yaml")
    gateway = MockModelGateway(mock_profile(), [_answer] * 30)
    return AgentService(config, tmp_path, gateway=gateway)


class TestJournal:
    async def test_monotonic_sequence_and_replay(self, tmp_path: Path) -> None:
        store = MissionStore(tmp_path / "m.db")
        events = AgentEventStore(store.connection)
        e1 = await events.append("mis_a", AgentEventType.TURN_ACCEPTED, {"n": 1})
        e2 = await events.append("mis_a", AgentEventType.TOOL_STARTED, {"n": 2})
        e3 = await events.append("mis_b", AgentEventType.TURN_ACCEPTED, {"n": 3})
        assert (e1.sequence, e2.sequence, e3.sequence) == (1, 2, 1)
        replayed = events.replay("mis_a", after_sequence=1)
        assert [e.sequence for e in replayed] == [2]
        assert events.latest_sequence("mis_a") == 2

    async def test_visibility_filter(self, tmp_path: Path) -> None:
        store = MissionStore(tmp_path / "m.db")
        events = AgentEventStore(store.connection)
        await events.append("mis_a", AgentEventType.TOOL_STARTED, {}, visibility=Visibility.USER)
        await events.append(
            "mis_a", AgentEventType.MODEL_REQUEST_STARTED, {}, visibility=Visibility.DEBUG
        )
        user_only = events.replay("mis_a", visibility=Visibility.USER)
        assert len(user_only) == 1
        assert user_only[0].type is AgentEventType.TOOL_STARTED


class TestV2Api:
    async def test_turn_completes_without_any_sse_client(self, service: AgentService) -> None:
        mission = service.create_mission("解耦测试")
        turn_id = await service.submit_turn_v2(mission.mission_id, "你好")
        task = service._turn_tasks[mission.mission_id]
        await task
        events = service.events_replay(mission.mission_id)
        types = [e.type for e in events]
        assert AgentEventType.TURN_ACCEPTED in types
        assert AgentEventType.MISSION_COMPLETED in types
        accepted = next(e for e in events if e.type is AgentEventType.TURN_ACCEPTED)
        assert accepted.turn_id == turn_id

    async def test_busy_mission_rejects_second_submit(self, service: AgentService) -> None:
        import asyncio

        mission = service.create_mission("并发测试")

        async def slow_answer(request):
            await asyncio.sleep(0.3)
            return _answer(request)

        # Replace gateway with a slow one for this test.
        service._gateway = MockModelGateway(mock_profile(), [slow_answer, slow_answer])
        service._loops = {}
        first = await service.submit_turn_v2(mission.mission_id, "第一句")
        with pytest.raises(Exception, match="already running"):
            await service.submit_turn_v2(mission.mission_id, "第二句")
        await service._turn_tasks[mission.mission_id]
        assert first

    async def test_tool_and_verification_events(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        gateway = MockModelGateway(mock_profile(), [_tool_then_answer, _answer])
        service = AgentService(config, tmp_path, gateway=gateway)
        mission = service.create_mission("事件测试")
        await service.submit_turn_v2(mission.mission_id, "读状态")
        await service._turn_tasks[mission.mission_id]
        types = [e.type for e in service.events_replay(mission.mission_id)]
        assert AgentEventType.MODEL_REQUEST_STARTED in types
        assert AgentEventType.MODEL_TOOL_CALL_PROPOSED in types
        assert AgentEventType.TOOL_STARTED in types
        assert AgentEventType.TOOL_COMPLETED in types
        assert AgentEventType.VERIFICATION_COMPLETED in types
        debug_events = [
            e for e in service.events_replay(mission.mission_id) if e.visibility is Visibility.DEBUG
        ]
        assert any(e.type is AgentEventType.MODEL_REQUEST_STARTED for e in debug_events)

    async def test_http_v2_endpoints(self, service: AgentService) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(create_app(service), headers={'x-rosclaw-token': service.control_token})
        mission = service.create_mission("HTTP 测试")
        r = client.post(f"/v2/missions/{mission.mission_id}/turns", json={"text": "你好"})
        assert r.status_code == 202
        task = service._turn_tasks[mission.mission_id]
        await task
        # Replay endpoint returns the journaled events immediately.
        with client.stream(
            "GET", f"/v2/missions/{mission.mission_id}/events?after_sequence=0&follow=false"
        ) as response:
            assert response.status_code == 200
            lines = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    lines.append(line)
                if len(lines) >= 2:
                    break
        assert lines
        first = json.loads(lines[0][6:])
        assert first["schema_version"] == "rosclaw.agent_event.v2"
        assert first["sequence"] == 1

    async def test_reconnect_replays_without_duplicates(self, service: AgentService) -> None:
        mission = service.create_mission("重连测试")
        await service.submit_turn_v2(mission.mission_id, "你好")
        await service._turn_tasks[mission.mission_id]
        all_events = service.events_replay(mission.mission_id)
        midpoint = all_events[len(all_events) // 2].sequence
        tail = service.events_replay(mission.mission_id, after_sequence=midpoint)
        assert all(e.sequence > midpoint for e in tail)
        assert [e.sequence for e in tail] == [
            e.sequence for e in all_events[len(all_events) // 2 + 1 :]
        ]
