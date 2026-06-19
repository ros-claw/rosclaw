"""Schema for the ``query_memory`` MCP tool."""

from __future__ import annotations

from typing import Any, TypedDict


class QueryMemoryResponse(TypedDict):
    """Envelope payload returned by ``query_memory``."""

    experiences: list[Any]
    count: int
    mode: str
