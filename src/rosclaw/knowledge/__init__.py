"""ROSClaw orchestration adapters for optional rosclaw-know/how v2 services."""

from .contracts import (
    HowAdviceBundleV2,
    HowAdviceRequestV2,
    KnowledgeUsageFeedbackV1,
    ReferencePackV2,
    ResearchRequestV2,
)
from .facade import KnowledgeFacade
from .service_manager import KnowledgeServiceConfig, KnowledgeServiceManager

__all__ = [
    "HowAdviceBundleV2",
    "HowAdviceRequestV2",
    "KnowledgeFacade",
    "KnowledgeServiceConfig",
    "KnowledgeServiceManager",
    "KnowledgeUsageFeedbackV1",
    "ReferencePackV2",
    "ResearchRequestV2",
]
