"""Native Agent-facing read-only/advisory tool definitions and dispatcher."""

from __future__ import annotations

from typing import Any

from .contracts import HowAdviceRequestV2, ReferenceContextV2, ResearchRequestV2

TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "rosclaw_know_research",
        "description": "Run bounded external-world research; never executes repository code.",
        "inputSchema": ResearchRequestV2.model_json_schema(),
        "read_only": True,
        "advisory": False,
    },
    {
        "name": "rosclaw_know_build_reference_pack",
        "description": "Retrieve a pinned, evidence-cited reference pack.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "context": ReferenceContextV2.model_json_schema(),
                "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
                "token_budget": {"type": "integer", "minimum": 1},
            },
            "required": ["query", "context"],
            "additionalProperties": False,
        },
        "read_only": True,
        "advisory": False,
    },
    {
        "name": "rosclaw_know_open_reference_pack",
        "description": "Open a previously built Reference Pack by opaque id.",
        "inputSchema": {
            "type": "object",
            "properties": {"reference_pack_id": {"type": "string", "minLength": 1}},
            "required": ["reference_pack_id"],
            "additionalProperties": False,
        },
        "read_only": True,
        "advisory": False,
    },
    {
        "name": "rosclaw_how_advice",
        "description": "Get evidence-cited DISCOVER/CONSULT/DIAGNOSE/CATALYZE advice.",
        "inputSchema": HowAdviceRequestV2.model_json_schema(),
        "read_only": True,
        "advisory": True,
    },
)


class KnowledgeAgentTools:
    def __init__(self, facade: Any) -> None:
        self.facade = facade

    def definitions(self) -> list[dict[str, Any]]:
        return [dict(item) for item in TOOL_DEFINITIONS]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "rosclaw_know_research":
            return self.facade.research(arguments)
        if name == "rosclaw_know_build_reference_pack":
            pack = self.facade.reference_pack(
                query=arguments["query"],
                context=arguments["context"],
                top_k=arguments.get("top_k", 10),
                token_budget=arguments.get("token_budget", 8_000),
            )
            return pack.model_dump(mode="json")
        if name == "rosclaw_know_open_reference_pack":
            pack = self.facade.get_reference_pack(arguments["reference_pack_id"])
            return (
                pack.model_dump(mode="json")
                if pack
                else {"status": "not_found", "reference_pack_id": arguments["reference_pack_id"]}
            )
        if name == "rosclaw_how_advice":
            return self.facade.advise(arguments).model_dump(mode="json")
        raise KeyError(f"unknown knowledge tool: {name}")
