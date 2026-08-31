from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from rosclaw.continual.champion_registry import (
    CanonicalChampionRegistry,
    ChampionRecordKind,
    ChampionRegistryRecord,
    DominanceMetricRole,
    PairedDominanceEvidence,
    PairedDominanceMetric,
    PromotionAuthority,
)
from tests.continual.helpers import digest


def _baseline(*, track_id: str = "global") -> ChampionRegistryRecord:
    return ChampionRegistryRecord(
        agent_id="agent.goalkeeper",
        track_id=track_id,
        artifact_hash=digest(f"{track_id}:baseline"),
        evidence_hash=digest(f"{track_id}:evidence"),
        scenario_suite_hash=digest("suite"),
        authority=(
            PromotionAuthority.BASELINE_STRICT_EXAM
            if track_id == "global"
            else PromotionAuthority.SEALED_SPECIALIST_EXAM
        ),
        kind=(
            ChampionRecordKind.GLOBAL_BASELINE
            if track_id == "global"
            else ChampionRecordKind.SPECIALIST_BASELINE
        ),
        evidence_valid=True,
        promotion_passed=True,
        generation=0,
    )


def _child(parent: ChampionRegistryRecord, *, promoted: bool) -> ChampionRegistryRecord:
    return ChampionRegistryRecord(
        agent_id=parent.agent_id,
        track_id=parent.track_id,
        artifact_hash=digest(f"candidate:{promoted}"),
        evidence_hash=digest(f"candidate-evidence:{promoted}"),
        scenario_suite_hash=parent.scenario_suite_hash,
        authority=PromotionAuthority.PAIRED_DOMINANCE,
        kind=(
            ChampionRecordKind.TRACK_REPLACEMENT
            if promoted
            else ChampionRecordKind.CANDIDATE_ARCHIVED
        ),
        evidence_valid=True,
        promotion_passed=promoted,
        generation=parent.generation + 1,
        parent_record_hash=parent.record_hash,
        parent_artifact_hash=parent.artifact_hash,
    )


def test_archived_candidate_does_not_move_the_active_head() -> None:
    baseline = _baseline()
    archive = _child(baseline, promoted=False)
    registry = CanonicalChampionRegistry().append(baseline).append(archive)

    assert registry.active_head("global") == baseline
    assert registry.audit_parent_claim(
        track_id="global", claimed_parent_artifact_hash=baseline.artifact_hash
    ).valid


def test_stale_rejected_parent_cannot_spawn_a_new_global_candidate() -> None:
    baseline = _baseline()
    archive = _child(baseline, promoted=False)
    registry = CanonicalChampionRegistry((baseline, archive))
    stale_child = replace(
        _child(archive, promoted=True),
        artifact_hash=digest("stale-descendant"),
        generation=archive.generation + 1,
    )

    audit = registry.audit_parent_claim(
        track_id="global", claimed_parent_artifact_hash=archive.artifact_hash
    )
    assert not audit.valid
    assert audit.reasons == ("parent_not_active_track_head",)
    with pytest.raises(ValueError, match="active track head"):
        registry.append(stale_child)


def test_valid_replacement_advances_track_and_specialist_is_isolated() -> None:
    baseline = _baseline()
    replacement = _child(baseline, promoted=True)
    specialist = replace(
        _baseline(track_id="goalkeeper.arm.elite"),
        source_parent_artifact_hash=baseline.artifact_hash,
    )
    registry = CanonicalChampionRegistry().append(baseline).append(replacement).append(specialist)

    assert registry.active_head("global") == replacement
    assert registry.active_head("goalkeeper.arm.elite") == specialist
    assert registry.registry_hash.startswith("sha256:")


def test_registry_rejects_unqualified_baseline_and_duplicate_artifact() -> None:
    with pytest.raises(ValueError, match="valid, passing"):
        replace(_baseline(), promotion_passed=False)
    baseline = _baseline()
    with pytest.raises(ValueError, match="already recorded"):
        CanonicalChampionRegistry((baseline, replace(baseline, evidence_hash=digest("other"))))


def _dominance(*, acquisition: float, retention: float) -> PairedDominanceEvidence:
    return PairedDominanceEvidence(
        incumbent_artifact_hash=digest("incumbent"),
        challenger_artifact_hash=digest(f"challenger:{acquisition}:{retention}"),
        scenario_suite_hash=digest("paired-suite"),
        metrics=(
            PairedDominanceMetric(
                metric_id="acquisition_success_count",
                incumbent_value=27.0,
                challenger_value=acquisition,
                higher_is_better=True,
                role=DominanceMetricRole.OBJECTIVE,
                minimum_improvement=1.0,
            ),
            PairedDominanceMetric(
                metric_id="retention_success_count",
                incumbent_value=94.0,
                challenger_value=retention,
                higher_is_better=True,
                role=DominanceMetricRole.GUARDRAIL,
                maximum_regression=1.0,
            ),
        ),
    )


def test_paired_dominance_requires_growth_without_forgetting() -> None:
    tied = _dominance(acquisition=27.0, retention=94.0)
    improved = _dominance(acquisition=29.0, retention=93.0)
    forgetting = _dominance(acquisition=30.0, retention=92.0)

    assert not tied.promotion_passed
    assert improved.promotion_passed
    assert not forgetting.promotion_passed
    assert improved.evidence_hash.startswith("sha256:")


def test_paired_dominance_rejects_invalid_objective_and_duplicate_metric() -> None:
    with pytest.raises(ValueError, match="positive minimum"):
        PairedDominanceMetric(
            metric_id="acquisition",
            incumbent_value=1.0,
            challenger_value=1.0,
            higher_is_better=True,
            role=DominanceMetricRole.OBJECTIVE,
        )
    metric = _dominance(acquisition=29.0, retention=94.0).metrics[0]
    with pytest.raises(ValueError, match="invalid"):
        PairedDominanceEvidence(
            incumbent_artifact_hash=digest("incumbent"),
            challenger_artifact_hash=digest("challenger"),
            scenario_suite_hash=digest("suite"),
            metrics=(metric, metric),
        )


def test_paired_dominance_rejects_runtime_type_coercion() -> None:
    with pytest.raises(ValueError, match="direction or role"):
        PairedDominanceMetric(
            metric_id="acquisition",
            incumbent_value=1.0,
            challenger_value=2.0,
            higher_is_better=cast(Any, 1),
            role=DominanceMetricRole.OBJECTIVE,
            minimum_improvement=1.0,
        )
    metric = _dominance(acquisition=29.0, retention=94.0).metrics[0]
    with pytest.raises(ValueError, match="immutable typed tuple"):
        PairedDominanceEvidence(
            incumbent_artifact_hash=digest("incumbent"),
            challenger_artifact_hash=digest("challenger"),
            scenario_suite_hash=digest("suite"),
            metrics=cast(Any, [metric]),
        )
