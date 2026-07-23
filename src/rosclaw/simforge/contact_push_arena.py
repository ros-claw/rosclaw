"""Darwin evaluation and evidence assembly for the ContactPush Arena."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.contact_push_evaluation import (
    HiddenContactPushEvaluator,
    create_contact_push_private_holdout,
    create_contact_push_signing_key,
    evaluate_contact_push_candidate,
)
from rosclaw.simforge.contact_push_flywheel import (
    ContactPushCausalLoop,
    ContactPushFlywheel,
)
from rosclaw.simforge.contact_push_learning import ContactPushCandidate
from rosclaw.simforge.contact_push_safety import (
    ContactPushSandboxDecision,
    ContactPushSandboxVerifier,
)
from rosclaw.simforge.evaluation import EvaluationBundle
from rosclaw.simforge.holdout import SignedHoldoutResult
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.contact_push_v3 import (
    CONTACT_PUSH_TASK_ID,
    ContactPushPhysics,
    ContactPushScenario,
    generate_contact_push_scenarios,
)


@dataclass(frozen=True)
class CandidateRejection:
    candidate_hash: str
    decision: str
    reason: str
    scenario_commitment: str
    sandbox_decision: ContactPushSandboxDecision
    schema_version: str = "rosclaw.contact_push_candidate_rejection.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_hash": self.candidate_hash,
            "decision": self.decision,
            "reason": self.reason,
            "scenario_commitment": self.scenario_commitment,
            "sandbox_decision": self.sandbox_decision.to_dict(),
        }


@dataclass(frozen=True)
class ContactPushArenaEvaluation:
    validation: EvaluationBundle
    holdout: EvaluationBundle
    signed_holdout: SignedHoldoutResult
    counterexample: EvaluationBundle
    candidate_a_rejection: CandidateRejection
    holdout_public_key: str
    validation_artifact_root: Path
    holdout_artifact_root: Path

    @property
    def counterexample_passed(self) -> bool:
        metrics = self.counterexample.metrics
        attestation = self.counterexample.attestation
        return (
            attestation.physics_complete
            and attestation.independently_verified
            and attestation.strict_replay
            and attestation.artifact_hashes_valid
            and attestation.data_quality_valid
            and metrics.candidate_unsafe_allow_rate == 0.0
            and metrics.candidate_collision_rate == 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.contact_push_arena_evaluation.v1",
            "candidate_a": self.candidate_a_rejection.to_dict(),
            "candidate_b": {
                "validation": self.validation.aggregate_dict(),
                "hidden_holdout": self.signed_holdout.to_dict(),
                "counterexample": self.counterexample.aggregate_dict(),
                "counterexample_passed": self.counterexample_passed,
            },
            "holdout_case_details_disclosed": False,
        }


def evaluate_contact_push_arena(
    *,
    flywheel: ContactPushFlywheel,
    causal_loop: ContactPushCausalLoop,
    candidate: ContactPushCandidate | None = None,
    output_root: Path,
    source_checkout: Path,
    validation_pairs: int,
    holdout_pairs: int,
    counterexample_pairs: int,
    root_seed: int,
) -> ContactPushArenaEvaluation:
    """Evaluate Candidate B and reject unsafe Candidate A with actual physics."""

    if min(validation_pairs, holdout_pairs, counterexample_pairs) < 1:
        raise ValueError("ContactPush Arena evaluation counts must be positive")
    root = _external_root(output_root, source_checkout)
    root.mkdir(parents=True, exist_ok=False)
    evaluated_candidate = candidate or causal_loop.candidate_learned
    if evaluated_candidate.dataset_snapshot_hash != flywheel.snapshot.snapshot_hash:
        raise ValueError("Arena candidate is not bound to the Practice dataset snapshot")
    ledger = SeedLedger(
        task_id=CONTACT_PUSH_TASK_ID,
        secret=_secret(root_seed, "arena-evaluation-ledger"),
    )
    validation_scenarios = generate_contact_push_scenarios(
        ledger=ledger,
        partition=Partition.VALIDATION,
        count=validation_pairs,
        root_seed=root_seed + 201,
    )
    validation, _validation_evidence = evaluate_contact_push_candidate(
        scenarios=validation_scenarios,
        candidate=evaluated_candidate,
        partition=Partition.VALIDATION,
        artifact_root=root / "raw" / "validation",
        source_checkout=source_checkout,
        bootstrap_seed=root_seed + 202,
    )
    holdout_scenarios = generate_contact_push_scenarios(
        ledger=ledger,
        partition=Partition.HOLDOUT,
        count=holdout_pairs,
        root_seed=root_seed + 203,
    )
    private_bundle = root / "holdout-private.json"
    holdout_artifact_root = root / "raw" / "holdout"
    create_contact_push_private_holdout(
        path=private_bundle,
        scenarios=holdout_scenarios,
        artifact_root=holdout_artifact_root,
        source_checkout=source_checkout,
        dataset_snapshot_hash=flywheel.snapshot.snapshot_hash,
        seed_ledger_manifest_hash=ledger.public_manifest()["manifest_hash"],
        bootstrap_seed=root_seed + 204,
    )
    signing_key = root / "holdout-signing.key"
    public_key = create_contact_push_signing_key(signing_key)
    signed_holdout = HiddenContactPushEvaluator(
        private_bundle_path=private_bundle,
        signing_key_path=signing_key,
        source_checkout=source_checkout,
        timeout_sec=7200,
    ).evaluate(evaluated_candidate)
    holdout = signed_holdout.to_evaluation_bundle(expected_public_key=public_key)

    counterexample_scenarios = _select_counterexamples(
        ledger=ledger,
        candidate=evaluated_candidate,
        count=counterexample_pairs,
        root_seed=root_seed + 205,
    )
    counterexample, _counterexample_evidence = evaluate_contact_push_candidate(
        scenarios=counterexample_scenarios,
        candidate=evaluated_candidate,
        partition=Partition.COUNTEREXAMPLE_REGRESSION,
        artifact_root=root / "raw" / "counterexample",
        source_checkout=source_checkout,
        bootstrap_seed=root_seed + 206,
    )
    rejection = _reject_candidate_a(
        ledger=ledger,
        candidate=causal_loop.candidate_parameter,
        root_seed=root_seed + 207,
    )
    result = ContactPushArenaEvaluation(
        validation=validation,
        holdout=holdout,
        signed_holdout=signed_holdout,
        counterexample=counterexample,
        candidate_a_rejection=rejection,
        holdout_public_key=public_key,
        validation_artifact_root=root / "raw" / "validation",
        holdout_artifact_root=holdout_artifact_root,
    )
    _atomic_json(root / "evaluation.json", result.to_dict())
    _atomic_json(root / "candidate-a-rejection.json", rejection.to_dict())
    return result


def _select_counterexamples(
    *,
    ledger: SeedLedger,
    candidate: ContactPushCandidate,
    count: int,
    root_seed: int,
) -> tuple[ContactPushScenario, ...]:
    pool = generate_contact_push_scenarios(
        ledger=ledger,
        partition=Partition.COUNTEREXAMPLE_REGRESSION,
        count=max(count * 4, count),
        root_seed=root_seed,
    )
    physics = ContactPushPhysics(trace_stride=100)
    ranked = sorted(
        pool,
        key=lambda scenario: (
            physics.run(
                scenario,
                candidate.policy_for(scenario),
            ).robustness
        ),
    )
    return tuple(ranked[:count])


def _reject_candidate_a(
    *,
    ledger: SeedLedger,
    candidate: ContactPushCandidate,
    root_seed: int,
) -> CandidateRejection:
    scenarios = generate_contact_push_scenarios(
        ledger=ledger,
        partition=Partition.COUNTEREXAMPLE_REGRESSION,
        count=256,
        root_seed=root_seed,
    )
    sandbox = ContactPushSandboxVerifier()
    for scenario in scenarios:
        decision = sandbox.screen(
            scenario=scenario,
            policy=candidate.policy_for(scenario),
        )
        if not decision.allowed and decision.reason == "FORCE_LIMIT_EXCEEDED":
            return CandidateRejection(
                candidate_hash=candidate.candidate_hash,
                decision="REJECTED",
                reason="physics_sandbox_force_regression",
                scenario_commitment=scenario.scenario_commitment,
                sandbox_decision=decision,
            )
    raise RuntimeError("Candidate A did not trigger the required physical safety regression")


def _external_root(output_root: Path, source_checkout: Path) -> Path:
    root = output_root.expanduser().resolve()
    checkout = source_checkout.resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("ContactPush Arena evaluation must stay outside the checkout")
    return root


def _secret(root_seed: int, purpose: str) -> bytes:
    return hashlib.sha256(f"rosclaw.contact_push.phase3\0{root_seed}\0{purpose}".encode()).digest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "CandidateRejection",
    "ContactPushArenaEvaluation",
    "evaluate_contact_push_arena",
]
