"""Evidence and consolidation contracts for bounded ROSClaw growth."""

from rosclaw.growth.contracts import (
    ConsolidationDecision,
    ConsolidationManifest,
    EvidenceLevel,
    EvidenceUsePolicy,
    GateName,
    GateResult,
    GateStatus,
    GrowthMetricSpec,
    MetricDirection,
    SkillGrowthSpec,
    TrainingEligibility,
)
from rosclaw.growth.experience import (
    ActionTraceCommitment,
    DerivedExperienceLineage,
    ExperienceSegment,
    FailureSignature,
    PhysicalAdvantageLabel,
)
from rosclaw.growth.generalization import (
    CoreGeneralizationGate,
    DomainValidationProof,
    GeneralizationDecision,
    GeneralizationEvidence,
    GeneralizationStatus,
)
from rosclaw.growth.plan import CandidateManifest, LearningJob, LearningPlan
from rosclaw.growth.registry import (
    GROWTH_ADAPTER_GROUP,
    GROWTH_LEARNER_GROUP,
    GrowthAdapter,
    GrowthDiscoveryReport,
    GrowthExtensionRegistry,
    LearnerDescriptor,
)
from rosclaw.growth.routing import (
    DataProfile,
    GrowthProblemSignals,
    LearnerRoute,
    RouteDisposition,
    route_learners,
)

__all__ = [
    "ActionTraceCommitment",
    "CandidateManifest",
    "ConsolidationDecision",
    "ConsolidationManifest",
    "CoreGeneralizationGate",
    "DerivedExperienceLineage",
    "DomainValidationProof",
    "EvidenceLevel",
    "EvidenceUsePolicy",
    "ExperienceSegment",
    "FailureSignature",
    "GROWTH_ADAPTER_GROUP",
    "GROWTH_LEARNER_GROUP",
    "GateName",
    "GateResult",
    "GateStatus",
    "GeneralizationDecision",
    "GeneralizationEvidence",
    "GeneralizationStatus",
    "GrowthMetricSpec",
    "GrowthAdapter",
    "GrowthDiscoveryReport",
    "GrowthExtensionRegistry",
    "GrowthProblemSignals",
    "DataProfile",
    "LearningJob",
    "LearningPlan",
    "LearnerDescriptor",
    "LearnerRoute",
    "MetricDirection",
    "PhysicalAdvantageLabel",
    "RouteDisposition",
    "SkillGrowthSpec",
    "TrainingEligibility",
    "route_learners",
]
