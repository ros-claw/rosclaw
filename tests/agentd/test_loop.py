"""AgentLoop tests against the scripted MockModelGateway (PR-NA-032 exits).

Covers: pure text → SIM tool → verify → reply closed loop; state machine
transitions; infinite tool loop cap; stale-context decision repair; user
cancel; crash recovery; model failure honesty; unavailable handlers degrade
honestly. No network, no real model.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from rosclaw.agentd.context import (
    BodyFacts,
    CapabilityInfo,
    ConsentFacts,
    ContextCompiler,
    OrgFacts,
    SelfFacts,
    SourceBundle,
    load_prompt,
)
from rosclaw.agentd.loop import AgentLoop
from rosclaw.agentd.mission import MissionStore
from rosclaw.agentd.models.gateway import MockModelGateway, ModelGatewayError, StrictTool
from rosclaw.agentd.models.profiles import kimi_k3_profile, mock_profile
from rosclaw.contracts.agent.mission import (
    BodyBinding,
    Goal,
    MissionSessionV1,
    MissionState,
)
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1, ToolCall
from tests.agentd.conftest import LOCAL_PRINCIPAL

NOW = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
ACTOR = "agent:rosclaw-native:sim_ur5e_01"


class FakeBody:
    def get_body(self, body_id: str):
        return BodyFacts(body_id=body_id, effective_body_hash="body_abc", summary="UR5e sim")


class FakeSelf:
    def __init__(self) -> None:
        self.seq = 1

    def get_self(self, body_id: str):
        return SelfFacts(
            self_snapshot_hash=f"selfsnap_{self.seq}",
            sequence=self.seq,
            observed_at=NOW - timedelta(milliseconds=50),
        )


class FakeCaps:
    def list_capabilities(self, query: str, limit: int):
        return [CapabilityInfo(name="get_robot_state")]


class FakeMemory:
    def retrieve(self, query: str, limit: int):
        return []


class FakeOrg:
    def get_org(self):
        return OrgFacts()


class FakeConsent:
    def get_consent(self, mission_id: str):
        return ConsentFacts(policy_hash="pol_1")


class MockTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def strict_tools(self, names: list[str]) -> list[StrictTool]:
        return [
            StrictTool(
                name="get_robot_state",
                description="Read current robot state.",
                parameters={
                    "type": "object",
                    "properties": {"verbose": {"type": "boolean"}},
                    "required": ["verbose"],
                    "additionalProperties": False,
                },
            )
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if name != "get_robot_state":
            raise ValueError(f"tool {name} not allowlisted")
        return json.dumps({"joints": [0.0] * 6, "fresh": True})


def _turn(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    assistant_message: dict | None = None,
) -> ModelTurnResultV1:
    calls = tool_calls or []
    message = assistant_message
    if message is None:
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = [
                {
                    "id": c.call_id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments_json},
                }
                for c in calls
            ]
            message["content"] = None
    return ModelTurnResultV1(
        turn_id="turn_x",
        provider="mock",
        model="mock-model",
        content=content,
        tool_calls=calls,
        assistant_message=message,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},  # type: ignore[arg-type]
    )


def _decision_block(request, intent: str = "ANSWER", summary: str = "done") -> str:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "dec_1",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": intent,
        "summary": summary,
        "evidence_refs": [],
    }
    return f"结论如下。\n```json\n{json.dumps(decision)}\n```"


@pytest.fixture
def store(tmp_path: Path) -> MissionStore:
    return MissionStore(tmp_path / "missions.db")


@pytest.fixture
def mission(store: MissionStore) -> MissionSessionV1:
    return store.create_mission(
        owner_principal=LOCAL_PRINCIPAL,
        goal=Goal(text="检查机器人状态"),
        body_binding=BodyBinding(body_id="sim_ur5e_01", effective_body_hash="body_abc"),
        actor_id=ACTOR,
    )


def _loop(store: MissionStore, gateway: MockModelGateway, tools=None) -> AgentLoop:
    compiler = ContextCompiler(
        SourceBundle(
            constitution_text="CONST",
            body=FakeBody(),
            self_source=FakeSelf(),
            capabilities=FakeCaps(),
            memory=FakeMemory(),
            organization=FakeOrg(),
            consent=FakeConsent(),
        )
    )
    return AgentLoop(
        store=store,
        compiler=compiler,
        gateway=gateway,
        prompt=load_prompt("native_agent_v1.md"),
        tools=tools,
        actor_id=ACTOR,
        max_tool_rounds=5,
    )


class TestClosedLoop:
    async def test_text_tool_verify_reply(self, store, mission) -> None:
        tools = MockTools()
        gateway = MockModelGateway(
            mock_profile(),
            [
                _turn(
                    tool_calls=[
                        ToolCall(
                            call_id="c1", name="get_robot_state", arguments_json='{"verbose": true}'
                        )
                    ]
                ),
                lambda req: _turn(content=_decision_block(req)),
            ],
        )
        loop = _loop(store, gateway, tools)
        result = await loop.run_user_turn(mission, "机器人现在状态如何？", now=NOW)
        assert result.tool_rounds == 1
        assert tools.calls == [("get_robot_state", {"verbose": True})]
        assert result.state is MissionState.IDLE
        assert "结论如下" in result.reply
        assert "```" not in result.reply  # decision block stripped
        assert result.decisions and result.decisions[0].next_intent == "ANSWER"
        # Full state path is journaled and consistent after the closed loop.
        store.verify_consistency(mission.mission_id)
        states = [
            e["to_state"]
            for e in store.events(mission.mission_id)
            if e["event_type"] == "rosclaw.agent.mission.transition.v1"
        ]
        assert states[:3] == ["UNDERSTAND", "GROUND", "PLAN"]
        # K3 continuity: assistant message + tool result appended verbatim.
        roles = [m["role"] for m in gateway.requests[-1].messages]
        assert "assistant" in roles and "tool" in roles

    async def test_answer_only_mission_completes(self, store, mission) -> None:
        gateway = MockModelGateway(
            mock_profile(), [lambda req: _turn(content=_decision_block(req))]
        )
        loop = _loop(store, gateway)
        result = await loop.run_user_turn(mission, "你能做什么？", now=NOW)
        assert result.state is MissionState.IDLE
        assert result.model_turns == 1


class TestFailureSemantics:
    async def test_infinite_tool_loop_capped(self, store, mission) -> None:
        tools = MockTools()
        gateway = MockModelGateway(
            mock_profile(),
            [
                _turn(
                    tool_calls=[
                        ToolCall(
                            call_id=f"c{i}",
                            name="get_robot_state",
                            arguments_json='{"verbose": false}',
                        )
                    ]
                )
                for i in range(20)
            ],
        )
        loop = _loop(store, gateway, tools)
        result = await loop.run_user_turn(mission, "一直读状态", now=NOW)
        assert result.degraded == "tool_rounds_exhausted"
        assert result.tool_rounds <= 6  # max_tool_rounds=5 (+1 boundary)

    async def test_stale_context_decision_repaired(self, store, mission) -> None:
        def bad_decision(req):
            block = _decision_block(req)
            payload = json.loads(block.split("```json\n")[1].split("\n```")[0])
            payload["context_revision"] = 999
            return _turn(content=f"```json\n{json.dumps(payload)}\n```")

        gateway = MockModelGateway(
            mock_profile(),
            [bad_decision, lambda req: _turn(content=_decision_block(req))],
        )
        loop = _loop(store, gateway)
        result = await loop.run_user_turn(mission, "状态？", now=NOW)
        assert result.state is MissionState.IDLE
        assert result.model_turns == 2  # one repair round used

    async def test_unrepairable_decision_fails_closed(self, store, mission) -> None:
        def bad_decision(req):
            payload = json.loads(_decision_block(req).split("```json\n")[1].split("\n```")[0])
            payload["context_revision"] = 999
            return _turn(content=f"```json\n{json.dumps(payload)}\n```")

        gateway = MockModelGateway(mock_profile(), [bad_decision] * 5)
        loop = _loop(store, gateway)
        result = await loop.run_user_turn(mission, "状态？", now=NOW)
        assert result.degraded is not None and "decision_rejected" in result.degraded
        assert result.state is MissionState.FAILED

    async def test_model_error_honest_no_fabrication(self, store, mission) -> None:
        class FailingGateway(MockModelGateway):
            async def complete(self, request):
                raise ModelGatewayError("http_error", "HTTP 500")

        loop = _loop(store, FailingGateway(mock_profile(), []))
        result = await loop.run_user_turn(mission, "状态？", now=NOW)
        assert result.degraded == "model_error: http_error"
        assert "失败" in result.reply

    async def test_hire_worker_without_fabric_is_honest(self, store, mission) -> None:
        def hire_decision(req):
            decision = {
                "schema_version": "rosclaw.decision.v1",
                "decision_id": "dec_h",
                "mission_id": req.mission_id,
                "context_id": req.context_id,
                "context_revision": req.context_revision,
                "next_intent": "HIRE_WORKER",
                "summary": "delegate analysis",
                "evidence_refs": ["artifact://logs/1"],
                "proposed_operation": {"type": "create_work_order"},
                "verification": {
                    "schema_version": "rosclaw.decision_verification.v1",
                    "verifiers": ["deterministic:schema"],
                },
            }
            return _turn(content=f"```json\n{json.dumps(decision)}\n```")

        gateway = MockModelGateway(mock_profile(), [hire_decision])
        loop = _loop(store, gateway)
        result = await loop.run_user_turn(mission, "找个人帮你分析", now=NOW)
        assert "Worker Fabric 尚未启用" in result.reply
        assert "已委派" not in result.reply  # never fabricate delegation

    async def test_request_action_without_channel_fails_closed(self, store, mission) -> None:
        def action_decision(req):
            decision = {
                "schema_version": "rosclaw.decision.v1",
                "decision_id": "dec_a",
                "mission_id": req.mission_id,
                "context_id": req.context_id,
                "context_revision": req.context_revision,
                "next_intent": "REQUEST_ACTION",
                "summary": "move",
                "evidence_refs": ["artifact://scene/1"],
                "proposed_operation": {"type": "request_action"},
                "verification": {
                    "schema_version": "rosclaw.decision_verification.v1",
                    "verifiers": ["deterministic:bounds"],
                },
            }
            return _turn(content=f"```json\n{json.dumps(decision)}\n```")

        gateway = MockModelGateway(mock_profile(), [action_decision])
        loop = _loop(store, gateway)
        result = await loop.run_user_turn(mission, "移动机械臂", now=NOW)
        assert "没有动作通道" in result.reply
        assert "已执行" not in result.reply


class TestLifecycle:
    async def test_cancel_between_turns(self, store, mission) -> None:
        tools = MockTools()

        def cancel_on_first(req):
            loop.request_cancel()
            return _turn(
                tool_calls=[
                    ToolCall(
                        call_id="c1", name="get_robot_state", arguments_json='{"verbose": true}'
                    )
                ]
            )

        gateway = MockModelGateway(mock_profile(), [cancel_on_first])
        loop = _loop(store, gateway, tools)
        result = await loop.run_user_turn(mission, "开始吧", now=NOW)
        assert "取消" in result.reply
        assert result.state is MissionState.FAILED

    async def test_crash_recovery_from_store(self, store, mission, tmp_path: Path) -> None:
        gateway = MockModelGateway(
            mock_profile(), [lambda req: _turn(content=_decision_block(req))]
        )
        loop = _loop(store, gateway)
        await loop.run_user_turn(mission, "你好", now=NOW)
        store.close()
        # "Crash": brand new handles, state comes only from the journal.
        store2 = MissionStore(tmp_path / "missions.db")
        loaded = store2.get_mission(mission.mission_id)
        assert loaded is not None
        assert loaded.state is MissionState.IDLE
        store2.verify_consistency(mission.mission_id)


class TestKimiProfile:
    def test_kimi_k3_profile_fields(self) -> None:
        profile = kimi_k3_profile()
        assert profile.base_url == "https://api.moonshot.cn/v1"
        assert profile.model == "kimi-k3"
        assert profile.api_key_ref == "env:MOONSHOT_API_KEY"
        assert "sk-" not in profile.api_key_ref
        assert profile.vendor_parameters == {"reasoning_effort": "high"}
        assert "llm.structured_decision" in profile.capabilities

    def test_strict_tool_schema_enforced(self) -> None:
        with pytest.raises(Exception, match="additionalProperties"):
            StrictTool(
                name="bad",
                description="x",
                parameters={"type": "object", "properties": {}},
            ).validate()
        with pytest.raises(Exception, match="required"):
            StrictTool(
                name="bad2",
                description="x",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": [],
                    "additionalProperties": False,
                },
            ).validate()

    async def test_mock_gateway_binds_context(self, mission) -> None:
        gateway = MockModelGateway(mock_profile(), [_turn(content="hi")])
        from rosclaw.agentd.models.gateway import ModelTurnRequest

        result = await gateway.complete(
            ModelTurnRequest(
                system_prompt="s",
                messages=[],
                mission_id=mission.mission_id,
                context_id="ctx_9",
                context_revision=3,
            )
        )
        assert result.context_id == "ctx_9"
        assert result.context_revision == 3
