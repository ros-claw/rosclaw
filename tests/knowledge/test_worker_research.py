from __future__ import annotations

import pytest

from rosclaw.knowledge.contracts import ResearchRequestV2
from rosclaw.knowledge.worker import BoundedResearchWorkV1, execute_bounded_research


class _Facade:
    def research(self, request):
        return {"request_id": request.request_id, "status": "completed"}


def test_bounded_worker_grants_only_research_tool():
    work = BoundedResearchWorkV1(
        research=ResearchRequestV2(
            request_id="research_1", topic="G1 football", goal="find prior work"
        )
    )
    assert execute_bounded_research(work, _Facade())["status"] == "completed"
    with pytest.raises(ValueError):
        execute_bounded_research(work.model_copy(update={"allowed_tools": ["shell"]}), _Facade())
