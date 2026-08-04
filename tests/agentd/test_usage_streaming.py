"""Usage metering + streaming UX + conversation persistence tests (PR-NA-030b).

- model_usage rows recorded per turn, per-mission totals
- cost computed from profile pricing and charged to monetary budget
- streaming deltas reach on_text_delta (mock) and aggregate correctly
- conversation journaled and restored after service restart
- SSE endpoint streams delta + final events
"""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService, create_app
from rosclaw.agentd.usage import estimate_cost_microunits
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1


def _answer(request, text: str = "回答完毕") -> ModelTurnResultV1:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "dec_1",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": "ANSWER",
        "summary": "ok",
        "evidence_refs": [],
    }
    return ModelTurnResultV1(
        turn_id="t1",
        provider="mock",
        model="mock-model",
        content=f"{text}\n```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": text},
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "reasoning_tokens": 10,
            "total_tokens": 150,
        },  # type: ignore[arg-type]
    )


def _service(tmp_path: Path, price_in: int = 2_000_000, price_out: int = 6_000_000):
    config = load_agent_config(tmp_path / "config.yaml")
    profile = mock_profile()
    profile = type(profile)(
        **{
            **profile.__dict__,
            "price_input_per_mtok_microunits": price_in,
            "price_output_per_mtok_microunits": price_out,
        }
    )
    gateway = MockModelGateway(profile, [_answer] * 50)
    return AgentService(config, tmp_path, gateway=gateway)


class TestUsageMetering:
    async def test_rows_and_totals(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        mission = service.create_mission("计量测试")
        await service.send_turn(mission.mission_id, "你好")
        totals = service.mission_usage(mission.mission_id)
        assert totals["total_tokens"] == 150
        assert totals["model_turns"] == 1
        # 100*2e6 + 50*6e6 µ / 1e6 = 200+300 = 500 microunits
        assert totals["cost_microunits"] == 500
        rows = service._usage.rows(mission.mission_id)
        assert len(rows) == 1
        assert rows[0]["reasoning_tokens"] == 10
        await service.close()

    async def test_monetary_budget_charged(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        mission = service.create_mission("预算测试")
        await service.send_turn(mission.mission_id, "你好")
        usage = service.store.budget_usage(mission.mission_id)
        assert usage["monetary_microunits"] == 500
        assert usage["model_tokens"] == 150
        await service.close()

    async def test_cost_estimator(self) -> None:
        assert (
            estimate_cost_microunits(
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
                price_input_per_mtok=2_000_000,
                price_output_per_mtok=6_000_000,
            )
            == 8_000_000
        )
        assert (
            estimate_cost_microunits(
                prompt_tokens=0,
                completion_tokens=0,
                price_input_per_mtok=1,
                price_output_per_mtok=1,
            )
            == 0
        )


class TestStreamingUx:
    async def test_deltas_delivered(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        mission = service.create_mission("流式测试")
        pieces: list[str] = []
        result = await service.send_turn(mission.mission_id, "你好", pieces.append)
        assert "".join(pieces) == "回答完毕" or "回答完毕" in "".join(pieces)
        assert "回答完毕" in result.reply
        await service.close()

    async def test_sse_endpoint(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        service = _service(tmp_path)
        client = TestClient(create_app(service), headers={'x-rosclaw-token': service.control_token})
        mission = service.create_mission("SSE 测试")
        events: list[dict] = []
        with client.stream(
            "POST", f"/missions/{mission.mission_id}/turns/stream", json={"text": "hi"}
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        types = [e["type"] for e in events]
        assert "delta" in types
        assert "final" in types
        final = events[types.index("final")]
        assert final["state"] == "IDLE"
        await service.close()


class TestConversationPersistence:
    async def test_resume_after_restart(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        mission = service.create_mission("连续性测试")
        await service.send_turn(mission.mission_id, "第一句话")
        mid = mission.mission_id
        await service.close()

        # "Restart": fresh service, new loop — history comes from the journal.
        service2 = _service(tmp_path)
        history = service2.conversation(mid)
        roles = [m.get("role") for m in history]
        assert "user" in roles and "assistant" in roles
        assert any("第一句话" in str(m.get("content")) for m in history)
        # New turn on the resumed mission keeps full history in the request.
        await service2.send_turn(mid, "第二句话")
        request = service2._gateway.requests[-1]
        contents = [str(m.get("content")) for m in request.messages]
        assert any("第一句话" in c for c in contents)
        assert any("第二句话" in c for c in contents)
        await service2.close()
