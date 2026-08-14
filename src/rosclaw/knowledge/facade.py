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
from .intent_router import KnowledgeIntentRouter, KnowledgeIntentRouteV1
from .policy import bounded_research_request
from .workspace import ActiveReferenceWorkspace


class KnowledgeFacade:
    def __init__(self, manager: Any, *, event_bus: Any | None = None) -> None:
        self.manager = manager
        self.events = KnowledgeEventAdapter(event_bus)
        self.intent_router = KnowledgeIntentRouter()
        self.workspace = ActiveReferenceWorkspace()

    def health(self) -> dict[str, Any]:
        return self.manager.health()

    def research(self, request: ResearchRequestV2 | dict[str, Any]) -> dict[str, Any]:
        validated = (
            request
            if isinstance(request, ResearchRequestV2)
            else ResearchRequestV2.model_validate(request)
        )
        request = bounded_research_request(validated)
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
        self.workspace.observe_pack(pack)
        self.events.publish(
            "know.reference_pack.created",
            {
                "reference_pack_id": pack.reference_pack_id,
                "index_version": pack.index_version,
                "count": len(pack.items),
                # PR-DF-10: the usage tracker needs unit ids to attribute
                # feedback to the knowledge actually presented.
                "knowledge_unit_ids": [
                    unit_id for item in pack.items for unit_id in item.knowledge_unit_ids
                ],
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
        self.workspace.observe_advice(advice)
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

    def route_intent(self, intent: str) -> KnowledgeIntentRouteV1:
        return self.intent_router.route(intent)

    def active_references(self) -> dict[str, Any]:
        return self.workspace.snapshot().model_dump(mode="json")

    def know_doctor(self) -> dict[str, Any]:
        return self.manager.know.doctor()

    def know_explain(
        self, *, query: str, context: ReferenceContextV2 | dict[str, Any], top_k: int = 10
    ) -> dict[str, Any]:
        validated = (
            context
            if isinstance(context, ReferenceContextV2)
            else ReferenceContextV2.model_validate(context)
        )
        return self.manager.know.explain(query=query, context=validated, top_k=top_k)

    def know_diff(
        self, *, project_id: str, from_snapshot: str, to_snapshot: str
    ) -> dict[str, Any]:
        return self.manager.know.project_diff(
            project_id=project_id,
            from_snapshot=from_snapshot,
            to_snapshot=to_snapshot,
        )

    def know_refresh(self, *, source_id: str, apply: bool = False) -> dict[str, Any]:
        return self.manager.know.refresh_source(source_id=source_id, apply=apply)

    def know_freeze(self, *, label: str) -> dict[str, Any]:
        return self.manager.know.freeze(label=label)

    def how_doctor(self) -> dict[str, Any]:
        return self.manager.how.doctor()

    def how_explain(self, advice_id: str) -> dict[str, Any]:
        return self.manager.how.explain(advice_id)

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
