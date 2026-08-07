"""Bounded research delegation adapter; grants no shell or physical tools."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .contracts import ResearchRequestV2, StrictWireModel


class BoundedResearchWorkV1(StrictWireModel):
    schema_version: str = "rosclaw.knowledge.research_work.v1"
    research: ResearchRequestV2
    deadline_seconds: int = Field(default=300, ge=1, le=3600)
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["rosclaw_know_research"], max_length=1
    )


def execute_bounded_research(work: BoundedResearchWorkV1, facade: Any) -> dict[str, Any]:
    if work.allowed_tools != ["rosclaw_know_research"]:
        raise ValueError("research worker may only use rosclaw_know_research")
    return facade.research(work.research)
