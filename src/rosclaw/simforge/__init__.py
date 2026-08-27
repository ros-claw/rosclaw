"""CoreSimBench contracts for reproducible, fail-closed simulation evolution."""

from rosclaw.simforge.budget import DataBudgetManager, DataBudgetSpec
from rosclaw.simforge.distribution import ScenarioSampler
from rosclaw.simforge.models import (
    EvidenceRequirements,
    HumanInvolvement,
    Partition,
    ScenarioConstraint,
    ScenarioDistributionSpec,
    ScenarioSample,
    ScenarioVariable,
    SimForgeTaskSpec,
)
from rosclaw.simforge.monitors import (
    RobustnessAggregator,
    SafetyPredicateMonitor,
    TemporalPredicateMonitor,
)
from rosclaw.simforge.reproducibility import (
    ArtifactBinding,
    CrossProcessReplayVerdict,
    ReproducibilityClosure,
    RuntimeProcessContract,
    SourceTreeBinding,
    bind_source_tree,
    build_reproducibility_closure,
    canonical_json_hash,
    evaluate_cross_process_replays,
    file_sha256,
)
from rosclaw.simforge.seed_ledger import SeedLedger

__all__ = [
    "DataBudgetManager",
    "DataBudgetSpec",
    "EvidenceRequirements",
    "ArtifactBinding",
    "CrossProcessReplayVerdict",
    "HumanInvolvement",
    "Partition",
    "RobustnessAggregator",
    "ReproducibilityClosure",
    "RuntimeProcessContract",
    "SafetyPredicateMonitor",
    "ScenarioConstraint",
    "ScenarioDistributionSpec",
    "ScenarioSample",
    "ScenarioSampler",
    "ScenarioVariable",
    "SeedLedger",
    "SimForgeTaskSpec",
    "SourceTreeBinding",
    "TemporalPredicateMonitor",
    "bind_source_tree",
    "build_reproducibility_closure",
    "canonical_json_hash",
    "evaluate_cross_process_replays",
    "file_sha256",
]
