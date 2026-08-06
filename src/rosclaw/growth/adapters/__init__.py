"""Task adapters that turn executed traces into governed Growth evidence."""

from rosclaw.growth.adapters.g1_coupled import (
    FieldProvenance,
    FieldTruthStatus,
    FootballPhase,
    G1CoupledTriageReport,
    PhaseSegment,
    measure_g1_coupled_recovery_quality,
    triage_g1_coupled_trajectory,
    verified_coupled_evidence_context,
)

__all__ = [
    "FieldProvenance",
    "FieldTruthStatus",
    "FootballPhase",
    "G1CoupledTriageReport",
    "PhaseSegment",
    "measure_g1_coupled_recovery_quality",
    "triage_g1_coupled_trajectory",
    "verified_coupled_evidence_context",
]
