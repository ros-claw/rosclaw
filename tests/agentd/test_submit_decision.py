"""rosclaw_submit_decision protocol tests (PR-06 exits).

- 模型经协议工具提交 DecisionV1：服务端补齐 context 绑定并校验通过
- 未知字段被拒绝（repair 回执，不静默通过）
- 禁用 legacy fenced JSON：fenced 块算 malformed attempt 进入修复
- 启用 fallback（默认）：fenced JSON 仍工作
- 协议工具不在 UI 显示为普通工具执行
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.decisions.submit_tool import (
    DECISION_SUBMIT_SCHEMA,
    SUBMIT_DECISION_TOOL,
    build_decision_payload,
    submit_decision_tool,
)
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1, ToolCall
from rosclaw.contracts.common import ValidationError


def _submit_turn(request, decision_args: dict, text: str = "") -> ModelTurnResultV1:
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content=text,
        tool_calls=[
            ToolCall(
                call_id="call_submit_1",
                name=SUBMIT_DECISION_TOOL,
                arguments_json=json.dumps(decision_args, ensure_ascii=False),
            )
        ],
        assistant_message={
            "role": "assistant",
            "content": text or None,
            "tool_calls": [
                {
                    "id": "call_submit_1",
                    "type": "function",
                    "function": {
                        "name": SUBMIT_DECISION_TOOL,
                        "arguments": json.dumps(decision_args, ensure_ascii=False),
                    },
                }
            ],
        },
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


def _fenced_turn(request) -> ModelTurnResultV1:
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
    )


class TestSubmitToolContract:
    def test_tool_is_strict(self) -> None:
        tool = submit_decision_tool()
        tool.validate()  # additionalProperties:false + required 完整
        assert tool.name == SUBMIT_DECISION_TOOL

    def test_payload_binds_context_server_side(self) -> None:
        payload = build_decision_payload(
            {"next_intent": "ANSWER", "summary": "s", "evidence_refs": []},
            mission_id="mis_x",
            context_id="ctx_x",
            context_revision=7,
        )
        assert payload["context_id"] == "ctx_x"
        assert payload["context_revision"] == 7
        assert payload["schema_version"] == "rosclaw.decision.v1"
        assert payload["decision_id"].startswith("dec_")

    def test_unknown_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown fields"):
            build_decision_payload(
                {
                    "next_intent": "ANSWER",
                    "summary": "s",
                    "evidence_refs": [],
                    "context_id": "ctx_forged",
                },
                mission_id="mis_x",
                context_id="ctx_x",
                context_revision=7,
            )

    def test_intent_enum_in_schema(self) -> None:
        intents = DECISION_SUBMIT_SCHEMA["properties"]["next_intent"]["enum"]
        assert "REQUEST_ACTION" in intents and "FAIL_SAFE" in intents


class TestProtocolRoundTrip:
    async def test_submit_tool_decision_applied(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        gateway = MockModelGateway(
            mock_profile(),
            [
                lambda req: _submit_turn(
                    req,
                    {
                        "next_intent": "ANSWER",
                        "summary": "协议提交的回答",
                        "evidence_refs": [],
                        "assumptions": [],
                        "uncertainty": {"level": "LOW", "reasons": []},
                        "proposed_operation": None,
                        "verification": None,
                        "on_failure": None,
                        "public_rationale": "",
                    },
                    text="直接回答。",
                )
            ],
        )
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            mission = service.create_mission("协议测试")
            result = await service.send_turn(mission.mission_id, "你好")
            assert result.state.value == "IDLE"
            assert result.decisions
            assert result.decisions[0].summary == "协议提交的回答"
            # 决策的 context 绑定由服务端补齐。
            assert result.decisions[0].context_id.startswith("ctx_mis_")
        finally:
            await service.close()

    async def test_invalid_submission_gets_error_receipt(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        gateway = MockModelGateway(
            mock_profile(),
            [
                lambda req: _submit_turn(
                    req, {"next_intent": "TELEPORT", "summary": "x", "evidence_refs": []}
                ),
                lambda req: _fenced_turn(req),
            ],
        )
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            mission = service.create_mission("修复测试")
            result = await service.send_turn(mission.mission_id, "你好")
            history = service.conversation(mission.mission_id)
            error_receipts = [m for m in history if "invalid DecisionV1" in str(m.get("content"))]
            assert error_receipts
            assert result.state.value == "IDLE"  # fallback 回合完成
        finally:
            await service.close()

    async def test_fallback_disabled_fenced_is_malformed(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        config.legacy_fenced_json_fallback = False
        gateway = MockModelGateway(mock_profile(), [_fenced_turn, _fenced_turn, _fenced_turn])
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            mission = service.create_mission("禁用 fallback")
            result = await service.send_turn(mission.mission_id, "你好")
            # fenced JSON 不再被接受：修复耗尽后 fail closed。
            assert result.degraded is not None
            assert "malformed" in result.degraded or "decision_rejected" in result.degraded
        finally:
            await service.close()

    async def test_fallback_enabled_fenced_still_works(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        gateway = MockModelGateway(mock_profile(), [_fenced_turn])
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            mission = service.create_mission("fallback 默认")
            result = await service.send_turn(mission.mission_id, "你好")
            assert result.state.value == "IDLE"
        finally:
            await service.close()
