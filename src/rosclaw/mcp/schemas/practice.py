"""Schema for the ``practice_query`` MCP tool."""

from __future__ import annotations

from typing import Any, TypedDict


class PracticeQueryResponse(TypedDict):
    """Envelope payload returned by ``practice_query``."""

    episodes: list[Any]
    count: int
    mode: str
