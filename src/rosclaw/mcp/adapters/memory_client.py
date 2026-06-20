"""Thin adapter around the ROSClaw memory interface for MCP tools."""

from __future__ import annotations

from typing import Any


class MemoryClient:
    """Read-only client that queries experiences from MemoryInterface."""

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    def find_similar_experiences(
        self,
        instruction: str,
        *,
        limit: int = 5,
        outcome_filter: str | None = None,
    ) -> dict[str, Any]:
        """Return experiences similar to the given instruction."""
        results = self._memory.find_similar_experiences(
            instruction=instruction,
            limit=limit,
            outcome_filter=outcome_filter,
        )
        return {
            "experiences": results,
            "count": len(results),
            "mode": "live",
        }
