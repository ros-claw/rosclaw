from __future__ import annotations

from rosclaw.knowledge.context_adapter import build_how_context
from rosclaw.knowledge.contracts import HowAdviceRequestV2, ResearchRequestV2
from rosclaw.knowledge.facade import KnowledgeFacade
from rosclaw.knowledge.feedback_adapter import build_usage_feedback
from rosclaw.knowledge.service_manager import KnowledgeServiceConfig, KnowledgeServiceManager
from tests.knowledge.conftest import FakeHow, FakeKnow, make_reference_pack


def test_fixture_research_reference_advice_feedback_loop():
    reference_pack = make_reference_pack()
    know = FakeKnow(reference_pack)
    how = FakeHow(reference_pack, know)
    facade = KnowledgeFacade(
        KnowledgeServiceManager(
            KnowledgeServiceConfig(mode="disabled"), know_client=know, how_client=how
        )
    )
    research = facade.research(
        ResearchRequestV2(
            request_id="research_1", topic="camera error", goal="find pinned upstream evidence"
        )
    )
    assert research["snapshot_count"] == 1
    pack = facade.reference_pack(query="camera error", context={"task": "diagnose camera"})
    context = build_how_context(task="diagnose camera", current_failure="camera error")
    advice = facade.advise(
        HowAdviceRequestV2(
            request_id="advice_request_1",
            mode="diagnose",
            query="camera error",
            context=context,
        )
    )
    feedback = build_usage_feedback(
        reference_pack_id=pack.reference_pack_id,
        advice_id=advice.advice_id,
        knowledge_unit_id=pack.items[0].knowledge_unit_ids[0],
        context_hash=context.context_hash(),
        verdict="useful",
        receipt_ref="receipt_fixture_1",
    )
    assert facade.feedback(feedback)
    assert know.feedback == [feedback]
