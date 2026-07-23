from __future__ import annotations

from pathlib import Path

from rosclaw.simforge.candidates import CandidateCompiler, CandidateGenerator, ParameterBound
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.shield_reach import (
    RISK_THRESHOLD_PATH,
    compile_automatic_candidate,
    generate_shield_reach_1k,
    generate_shield_reach_cases,
    run_shield_reach_evaluation,
)


def _candidate():
    compiler = CandidateCompiler(
        parent_policy={RISK_THRESHOLD_PATH: 0.82},
        allowed_bounds={RISK_THRESHOLD_PATH: ParameterBound(0.1, 0.9)},
    )
    return compiler.compile(
        {RISK_THRESHOLD_PATH: 0.5},
        failure_signature_id="MIDPATH_COLLISION",
        generator=CandidateGenerator(type="search", algorithm="cross_entropy"),
    )


def test_shield_reach_1k_has_exact_public_and_hidden_partition_counts() -> None:
    ledger = SeedLedger(task_id="shield_reach_v1", secret=b"shield-reach-suite-test-secret")
    suite = generate_shield_reach_1k(ledger=ledger)
    visible = (
        suite[Partition.DISCOVERY] + suite[Partition.DEVELOPMENT] + suite[Partition.VALIDATION]
    )

    assert sum(map(len, suite.values())) == 1000
    assert len(suite[Partition.HOLDOUT]) == 200
    assert sum(case.category == "safe" for case in visible) == 300
    assert sum(case.category == "unsafe" for case in visible) == 300
    assert sum(case.category == "boundary" for case in visible) == 200
    assert "seed" not in ledger.public_manifest()["partitions"]["holdout"][0]


def test_shield_reach_cpu_oracle_executes_safe_and_collision_physics(tmp_path: Path) -> None:
    ledger = SeedLedger(task_id="shield_reach_oracle", secret=b"shield-reach-oracle-secret")
    cases = generate_shield_reach_cases(
        ledger=ledger,
        partition=Partition.DEVELOPMENT,
        count=2,
        root_seed=11,
        category_counts=(1, 1, 0),
    )
    bundle, receipts = run_shield_reach_evaluation(
        cases=cases,
        candidate=_candidate(),
        artifact_root=tmp_path / "external-evidence",
        source_checkout=Path.cwd(),
    )

    assert bundle.attestation.physics_complete
    assert bundle.attestation.independently_verified
    assert bundle.attestation.strict_replay
    assert bundle.attestation.artifact_hashes_valid
    assert {receipt["is_safe"] for receipt in receipts} == {True, False}
    assert all(receipt["valid_for_promotion"] is True for receipt in receipts)
    assert bundle.metrics.candidate_unsafe_allow_rate == 0


def test_candidate_threshold_is_selected_from_labeled_cases_without_human_patch() -> None:
    ledger = SeedLedger(task_id="shield_reach_search", secret=b"shield-reach-search-secret")
    cases = generate_shield_reach_cases(
        ledger=ledger,
        partition=Partition.DISCOVERY,
        count=20,
        root_seed=20260723,
        category_counts=(8, 8, 4),
    )
    labeled = tuple((case, case.pose == "safe") for case in cases)
    candidate, trace = compile_automatic_candidate(labeled, budget=60)
    threshold = float(candidate.changes[0].new)

    assert candidate.human_involvement.fully_autonomous
    assert len(trace) == 60
    assert all((case.risk <= threshold) is physically_safe for case, physically_safe in labeled)
