"""ROSClaw orchestration adapters for optional rosclaw-know/how v2 services."""

from .contracts import (
    HowAdviceBundleV2,
    HowAdviceRequestV2,
    KnowledgeUsageFeedbackV1,
    ReferencePackV2,
    ResearchRequestV2,
)
from .facade import KnowledgeFacade
from .intent_router import KnowledgeIntentRouter, KnowledgeIntentRouteV1
from .policy import RESEARCH_BUDGETS, bounded_research_request
from .service_manager import KnowledgeServiceConfig, KnowledgeServiceManager

__all__ = [
    "HowAdviceBundleV2",
    "HowAdviceRequestV2",
    "KnowledgeFacade",
    "KnowledgeIntentRouteV1",
    "KnowledgeIntentRouter",
    "KnowledgeServiceConfig",
    "KnowledgeServiceManager",
    "KnowledgeUsageFeedbackV1",
    "ReferencePackV2",
    "ResearchRequestV2",
    "RESEARCH_BUDGETS",
    "bounded_research_request",
]
