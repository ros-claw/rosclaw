"""Core-owned routing budgets; retrieval/ranking algorithms remain in Know."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .contracts import ResearchRequestV2

ResearchDepth = Literal["shallow", "standard", "deep"]


@dataclass(frozen=True)
class ResearchBudget:
    max_sources: int
    max_tokens: int


RESEARCH_BUDGETS: dict[ResearchDepth, ResearchBudget] = {
    "shallow": ResearchBudget(max_sources=8, max_tokens=20_000),
    "standard": ResearchBudget(max_sources=20, max_tokens=60_000),
    "deep": ResearchBudget(max_sources=50, max_tokens=150_000),
}


def bounded_research_request(
    value: ResearchRequestV2 | dict[str, Any], *, agent_default: bool = False
) -> ResearchRequestV2:
    """Apply hard source/token ceilings and the Native Agent's shallow default."""

    if isinstance(value, ResearchRequestV2):
        request = value
    else:
        payload = dict(value)
        if agent_default:
            payload.setdefault("depth", "shallow")
        depth = payload.get("depth", "standard")
        budget = RESEARCH_BUDGETS[depth]
        payload.setdefault("max_sources", budget.max_sources)
        payload.setdefault("token_budget", budget.max_tokens)
        request = ResearchRequestV2.model_validate(payload)
    budget = RESEARCH_BUDGETS[request.depth]
    return request.model_copy(
        update={
            "max_sources": min(request.max_sources, budget.max_sources),
            "token_budget": min(request.token_budget, budget.max_tokens),
        }
    )


__all__ = ["RESEARCH_BUDGETS", "ResearchBudget", "bounded_research_request"]
