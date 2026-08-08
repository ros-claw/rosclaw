"""MCP projection of knowledge tools. All tools are read-only or advisory."""

from __future__ import annotations

from typing import Any

from .agent_tools import KnowledgeAgentTools


class KnowledgeMCPTools:
    def __init__(self, agent_tools: KnowledgeAgentTools) -> None:
        self.agent_tools = agent_tools

    def specs(self) -> list[dict[str, Any]]:
        return self.agent_tools.definitions()

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self.agent_tools.call, name, arguments)
