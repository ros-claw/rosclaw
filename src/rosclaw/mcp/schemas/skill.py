"""Schema for the ``list_skills`` MCP tool."""

from __future__ import annotations

from typing import Any, TypedDict


class ListSkillsResponse(TypedDict):
    """Envelope payload returned by ``list_skills``."""

    skills: list[Any]
    count: int
    mode: str
