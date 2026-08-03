"""``rosclaw_submit_decision`` — the internal decision protocol tool (PR-06, 大纲 §6).

它不是普通工具，也不执行任何动作：模型在推理结束后通过它提交
DecisionV1。AgentLoop 拦截该 tool call，服务端补齐 context 绑定字段并
校验；modeld 永远不能执行它（架构测试约束——它只能把 tool call 原样
返回给 Python）。
"""

from __future__ import annotations

from typing import Any

from rosclaw.agentd.models.gateway import StrictTool
from rosclaw.contracts.agent.decision import NextIntent
from rosclaw.contracts.common import ValidationError, new_id

SUBMIT_DECISION_TOOL = "rosclaw_submit_decision"

_INTENTS = [intent.value for intent in NextIntent]

#: DecisionV1 的严格提交 schema（context 绑定字段由服务端补齐，
#: 模型不伪造 context_id/revision——只声明 intent 与载荷）。
DECISION_SUBMIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "next_intent": {"type": "string", "enum": _INTENTS},
        "summary": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "assumptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "evidence_ref": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["claim", "evidence_ref", "confidence"],
                "additionalProperties": False,
            },
        },
        "uncertainty": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["LOW", "MODERATE", "HIGH"]},
                "reasons": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["level", "reasons"],
            "additionalProperties": False,
        },
        "proposed_operation": {
            "type": ["object", "null"],
            "properties": {
                "type": {"type": "string"},
                "payload_ref": {"type": "string"},
                "payload": {"type": "object"},
            },
            "required": ["type"],
            "additionalProperties": False,
        },
        "verification": {
            "type": ["object", "null"],
            "properties": {
                "schema_ref": {"type": "string"},
                "verifiers": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["schema_ref", "verifiers"],
            "additionalProperties": False,
        },
        "on_failure": {
            "type": ["object", "null"],
            "properties": {
                "intent": {"type": "string", "enum": _INTENTS},
                "reason": {"type": "string"},
            },
            "required": ["intent", "reason"],
            "additionalProperties": False,
        },
        "public_rationale": {"type": "string"},
    },
    "required": [
        "next_intent",
        "summary",
        "evidence_refs",
        "assumptions",
        "uncertainty",
        "proposed_operation",
        "verification",
        "on_failure",
        "public_rationale",
    ],
    "additionalProperties": False,
}


def submit_decision_tool() -> StrictTool:
    return StrictTool(
        name=SUBMIT_DECISION_TOOL,
        description=(
            "INTERNAL PROTOCOL — conclude this cognitive step by submitting a "
            "ROSClaw DecisionV1. This is not an action and executes nothing: "
            "the host validates the decision. Call this exactly once when you "
            "have finished reasoning; do not call physical tools with it."
        ),
        parameters=DECISION_SUBMIT_SCHEMA,
    )


def build_decision_payload(
    arguments: dict[str, Any],
    *,
    mission_id: str,
    context_id: str,
    context_revision: int,
) -> dict[str, Any]:
    """服务端补齐 context 绑定（模型不可自报）。其余字段按提交透传。"""
    if not isinstance(arguments, dict):
        raise ValidationError("decision arguments must be an object")
    unknown = set(arguments) - set(DECISION_SUBMIT_SCHEMA["properties"])
    if unknown:
        raise ValidationError(f"decision arguments contain unknown fields: {sorted(unknown)}")
    payload = dict(arguments)
    payload.update(
        {
            "schema_version": "rosclaw.decision.v1",
            "decision_id": new_id("dec"),
            "mission_id": mission_id,
            "context_id": context_id,
            "context_revision": context_revision,
        }
    )
    return payload
