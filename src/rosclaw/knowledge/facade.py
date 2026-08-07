"""Core orchestration facade over versioned Know and How protocols."""

from __future__ import annotations

from typing import Any

from .contracts import (
    HowAdviceBundleV2,
    HowAdviceRequestV2,
    KnowledgeUsageFeedbackV1,
    ReferenceContextV2,
    ReferencePackV2,
    ResearchRequestV2,
)
from .event_adapter import KnowledgeEventAdapter


class KnowledgeFacade:
    def __init__(self, manager: Any, *, event_bus: Any | None = None) -> None:
        self.manager = manager
        self.events = KnowledgeEventAdapter(event_bus)

    def health(self) -> dict[str, Any]:
        return self.manager.health()

    def research(self, request: ResearchRequestV2 | dict[str, Any]) -> dict[str, Any]:
        request = (
            request
            if isinstance(request, ResearchRequestV2)
            else ResearchRequestV2.model_validate(request)
        )
        self.events.publish("know.research.requested", {"request_id": request.request_id})
        try:
            result = self.manager.know.research(request)
        except Exception as exc:
            self.events.publish(
                "know.research.failed",
                {"request_id": request.request_id, "error_type": type(exc).__name__},
            )
            raise
        self.events.publish(
            "know.research.completed",
            {
                "request_id": request.request_id,
                "status": str(result.get("status", "completed")),
            },
        )
        return result

    def reference_pack(
        self,
        *,
        query: str,
        context: ReferenceContextV2 | dict[str, Any],
        top_k: int = 10,
        token_budget: int = 8_000,
    ) -> ReferencePackV2:
        context = (
            context
            if isinstance(context, ReferenceContextV2)
            else ReferenceContextV2.model_validate(context)
        )
        pack = self.manager.know.reference_pack(
            query=query, context=context, top_k=top_k, token_budget=token_budget
        )
        self.events.publish(
            "know.reference_pack.created",
            {
                "reference_pack_id": pack.reference_pack_id,
                "index_version": pack.index_version,
                "count": len(pack.items),
            },
        )
        return pack

    def get_reference_pack(self, reference_pack_id: str) -> ReferencePackV2 | None:
        return self.manager.know.get_reference_pack(reference_pack_id)

    def advise(self, request: HowAdviceRequestV2 | dict[str, Any]) -> HowAdviceBundleV2:
        request = (
            request
            if isinstance(request, HowAdviceRequestV2)
            else HowAdviceRequestV2.model_validate(request)
        )
        advice = self.manager.how.advise(request)
        topic = "how.advice.abstained" if advice.abstained else "how.advice.created"
        self.events.publish(
            topic,
            {
                "request_id": request.request_id,
                "advice_id": advice.advice_id,
                "reference_pack_id": advice.reference_pack_id,
                "mode": advice.mode,
                "status": "abstained" if advice.abstained else "created",
            },
        )
        return advice

    def feedback(
        self, feedback: KnowledgeUsageFeedbackV1 | dict[str, Any], *, via_how: bool = True
    ) -> bool:
        feedback = (
            feedback
            if isinstance(feedback, KnowledgeUsageFeedbackV1)
            else KnowledgeUsageFeedbackV1.model_validate(feedback)
        )
        created = (
            self.manager.how.submit_feedback(feedback)
            if via_how
            else self.manager.know.submit_feedback(feedback)
        )
        self.events.publish(
            "how.feedback.recorded",
            {
                "feedback_id": feedback.feedback_id,
                "reference_pack_id": feedback.reference_pack_id,
                "knowledge_unit_id": feedback.knowledge_unit_id,
                "verdict": feedback.verdict,
                "status": "created" if created else "duplicate",
            },
        )
        return created
