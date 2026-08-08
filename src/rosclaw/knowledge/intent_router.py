"""Deterministic boundary router for Native Agent knowledge-related intents."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from .contracts import StrictWireModel

KnowledgeDestination = Literal["body", "memory", "know", "how", "skill", "action"]

_ACTION = re.compile(
    r"(?:让|请|马上|立即).*(?:回充|移动|前进|后退|转向|抓取|拿起|放下|停下|执行)|"
    r"^(?:move|grab|dock|stop|turn|send)\b",
    re.IGNORECASE,
)
_SKILL = re.compile(
    r"(?:技能|skill|mcp|工具|可执行能力|capabilit)|(?:会不会|能不能|能否).*(?:抓|导航|拍照)",
    re.IGNORECASE,
)
_MEMORY = re.compile(
    r"(?:上次|以前|之前|过去|曾经|历史|记得|我(?:们)?经历过|last time|previous|history|remember)",
    re.IGNORECASE,
)
_BODY = re.compile(
    r"(?:当前|这台|本机|现在绑定|眼前|current|this robot).*(?:机器人|型号|关节|传感器|执行器|电量|状态|限制|body|joint|sensor|actuator)",
    re.IGNORECASE,
)
_CURRENT_HOW = re.compile(
    r"(?:当前|现在|正在|这个).*(?:怎么办|怎么修|如何处理|如何适配|如何选择|诊断)|"
    r"(?:which route should|what should i do).*(?:current|now)",
    re.IGNORECASE,
)
_GENERAL_HOW = re.compile(
    r"(?:怎么办|怎么修|如何处理|如何排查|怎么适配|which route should|what should i do)",
    re.IGNORECASE,
)
_KNOW = re.compile(
    r"(?:官方|上游|论文|开源|别人|业界|资料|文档|规格|版本|release|official|upstream|paper|repository|project|api)",
    re.IGNORECASE,
)
_RESEARCH_TRIGGER = re.compile(
    r"(?:查资料|调研|类似项目|论文|上游|官方|api|版本|未知.*(?:错误|error)|比较.*方案|"
    r"research|look up|latest|unknown error|similar project)",
    re.IGNORECASE,
)


class KnowledgeIntentRouteV1(StrictWireModel):
    intent: str = Field(min_length=1, max_length=20_000)
    destination: KnowledgeDestination
    rationale_codes: list[str] = Field(min_length=1)
    requires_runtime_context: bool = False
    auto_research: bool = False
    action_authority: Literal[False] = False


class KnowledgeIntentRouter:
    """Route by ownership boundary, not by probabilistic topic similarity."""

    def route(self, intent: str) -> KnowledgeIntentRouteV1:
        text = intent.strip()
        if not text:
            raise ValueError("intent must not be empty")
        if _ACTION.search(text):
            destination: KnowledgeDestination = "action"
            codes = ["explicit_real_action_request", "handoff_to_action_safety_chain"]
        elif _SKILL.search(text):
            destination = "skill"
            codes = ["current_executable_capability_query"]
        elif _MEMORY.search(text):
            destination = "memory"
            codes = ["prior_local_experience_query"]
        elif _BODY.search(text):
            destination = "body"
            codes = ["current_bound_body_fact_query"]
        elif _CURRENT_HOW.search(text):
            destination = "how"
            codes = ["current_context_adaptation_or_diagnosis"]
        elif _KNOW.search(text):
            destination = "know"
            codes = ["external_world_or_primary_source_query"]
        elif _GENERAL_HOW.search(text):
            destination = "how"
            codes = ["current_context_adaptation_or_diagnosis"]
        else:
            destination = "know"
            codes = [
                "general_world_knowledge_query"
            ]
        return KnowledgeIntentRouteV1(
            intent=text,
            destination=destination,
            rationale_codes=codes,
            requires_runtime_context=destination in {"body", "how", "skill", "action"},
            auto_research=destination == "know" and bool(_RESEARCH_TRIGGER.search(text)),
        )


__all__ = ["KnowledgeDestination", "KnowledgeIntentRouteV1", "KnowledgeIntentRouter"]
