from __future__ import annotations

from rosclaw.knowledge.agent_tools import KnowledgeAgentTools
from rosclaw.knowledge.facade import KnowledgeFacade
from rosclaw.knowledge.policy import bounded_research_request
from rosclaw.knowledge.service_manager import KnowledgeServiceConfig, KnowledgeServiceManager

from .conftest import FakeHow, FakeKnow


def test_research_budget_profiles_and_hard_caps() -> None:
    base = {"request_id": "r1", "topic": "topic", "goal": "goal"}
    shallow = bounded_research_request(base, agent_default=True)
    standard = bounded_research_request({**base, "depth": "standard"})
    deep = bounded_research_request(
        {**base, "depth": "deep", "max_sources": 200, "token_budget": 2_000_000}
    )
    assert (shallow.max_sources, shallow.token_budget) == (8, 20_000)
    assert (standard.max_sources, standard.token_budget) == (20, 60_000)
    assert (deep.max_sources, deep.token_budget) == (50, 150_000)


def test_agent_research_defaults_to_shallow(reference_pack) -> None:
    class CapturingKnow(FakeKnow):
        request = None

        def research(self, request):
            self.request = request
            return super().research(request)

    know = CapturingKnow(reference_pack)
    manager = KnowledgeServiceManager(
        KnowledgeServiceConfig(mode="disabled"),
        know_client=know,
        how_client=FakeHow(reference_pack, know),
    )
    tools = KnowledgeAgentTools(KnowledgeFacade(manager))
    tools.call(
        "rosclaw_know_research",
        {"request_id": "r1", "topic": "G1", "goal": "find primary sources"},
    )
    assert know.request.depth == "shallow"
    assert (know.request.max_sources, know.request.token_budget) == (8, 20_000)


def test_active_workspace_retains_ids_and_warnings_not_document_bodies(reference_pack) -> None:
    know = FakeKnow(reference_pack)
    facade = KnowledgeFacade(
        KnowledgeServiceManager(
            KnowledgeServiceConfig(mode="disabled"),
            know_client=know,
            how_client=FakeHow(reference_pack, know),
        )
    )
    facade.reference_pack(query="camera error", context={"task": "diagnose"})
    snapshot = facade.active_references()
    assert snapshot["reference_pack_ids"] == ["pack_1"]
    assert snapshot["project_ids"] == ["project_1"]
    assert snapshot["open_evidence_ids"] == ["ev_1"]
    serialized = repr(snapshot)
    assert "Pinned evidence about" not in serialized
    assert "docs/guide.md" not in serialized
