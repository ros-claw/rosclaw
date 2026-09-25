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
    "GateName",
    "GateResult",
    "GateStatus",
    "GeneralizationDecision",
    "GeneralizationEvidence",
    "GeneralizationStatus",
    "GrowthMetricSpec",
    "LearningJob",
    "LearningPlan",
    "MetricDirection",
    "PhysicalAdvantageLabel",
    "SkillGrowthSpec",
    "TrainingEligibility",
]
