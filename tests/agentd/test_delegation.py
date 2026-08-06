"""K4 delegation closed-loop tests (PR-WF-053).

Mock: main agent emits HIRE_WORKER → ServiceIntentHandlers builds a bounded
WorkOrder → native worker executes with an isolated conversation → result
verified → reply reports acceptance. Faults: worker crash, scheduling
failure, secret never enters work orders.
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


def _hire_decision(request) -> ModelTurnResultV1:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "dec_h1",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": "HIRE_WORKER",
        "summary": "委派日志分析",
        "evidence_refs": ["artifact://logs/failure-1"],
        "proposed_operation": {
            "type": "create_work_order",
            "payload": {
                "goal": "分析失败日志并提出修复建议",
                "capability": "analysis.text",
                "instructions": "只基于给定日志内容",
                "artifacts": ["log://failure-1"],
            },
        },
        "verification": {
            "schema_version": "rosclaw.decision_verification.v1",
            "verifiers": ["deterministic:schema"],
        },
    }
    return ModelTurnResultV1(
        turn_id="t1",
        provider="mock",
        model="mock-model",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},  # type: ignore[arg-type]
    )


def _worker_answer(request) -> ModelTurnResultV1:
    return ModelTurnResultV1(
        turn_id="t2",
        provider="mock",
        model="mock-model",
        content="根因：连接超时配置 3s 过短（日志第 42 行）。建议：提高到 30s 并加退避。[事实+推断]",
        assistant_message={"role": "assistant", "content": "..."},
        usage={"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},  # type: ignore[arg-type]
    )


@pytest.fixture
def service(tmp_path: Path) -> AgentService:
    config = load_agent_config(tmp_path / "config.yaml")
    gateway = MockModelGateway(mock_profile(), [_hire_decision, _worker_answer])
    return AgentService(config, tmp_path, gateway=gateway)


class TestDelegationLoop:
    async def test_hire_worker_accepted_and_attributed(self, service: AgentService) -> None:
        mission = service.create_mission("分析失败原因")
        result = await service.send_turn(mission.mission_id, "分析一下失败日志，交给 worker 做。")
        assert "已完成并通过验证" in result.reply
        assert "根因" in result.reply
        assert result.state.value == "IDLE"
        # Attribution: work order persisted with full lifecycle.
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        assert len(orders) == 1
        assert orders[0].status == "ACCEPTED"
        assert orders[0].issued_by == service.actor_id
        assert orders[0].assigned_to == "worker:native:basic"
        # Scheduler decision was journaled with the feature vector.
        events = service.store.connection.execute(
            "SELECT payload_json FROM worker_events WHERE event_type = "
            "'rosclaw.worker.work_order.offered.v1'"
        ).fetchall()
        payload = json.loads(events[0]["payload_json"])
        assert payload["policy"] == "scheduler.v1"
        assert "reliability" in payload["features"]
        await service.close()

    async def test_no_secret_in_work_order(self, service: AgentService) -> None:
        mission = service.create_mission("secret 检查")
        await service.send_turn(mission.mission_id, "委派任务。")
        rows = service.store.connection.execute("SELECT order_json FROM work_orders").fetchall()
        for row in rows:
            assert "sk-" not in row["order_json"]
            assert "api_key" not in row["order_json"]
        await service.close()

    async def test_worker_crash_honest(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        # Script: main decision, then worker call → script exhausted → crash.
        gateway = MockModelGateway(mock_profile(), [_hire_decision])
        service = AgentService(config, tmp_path, gateway=gateway)
        mission = service.create_mission("崩溃测试")
        result = await service.send_turn(mission.mission_id, "委派。")
        assert "未通过" in result.reply or "无法" in result.reply
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        assert orders[0].status == "FAILED"
        await service.close()


class TestWorkOrderBudget:
    async def test_worker_isolated_conversation(self, service: AgentService) -> None:
        mission = service.create_mission("隔离检查")
        await service.send_turn(mission.mission_id, "委派。")
        # The worker's model request is a fresh single-message conversation,
        # never the main agent's mutable conversation object.
        requests = service._gateway.requests
        worker_request = requests[-1]
        assert len(worker_request.messages) == 1
        assert "WorkOrder goal" in worker_request.messages[0]["content"]
        assert worker_request.tools == []
        await service.close()
