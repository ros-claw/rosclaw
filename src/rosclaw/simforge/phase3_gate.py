"""Phase 3 causal promotion gate layered on Statistical Gate V3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw.simforge.contact_push_learning import ContactPushCandidate
from rosclaw.simforge.evaluation import EvaluationBundle, StressEvidence
from rosclaw.simforge.models import Partition
from rosclaw.simforge.promotion_v3 import (
    GateCheck,
    GateDecision,
    GateV3Policy,
    StatisticalGateV3,
)
from rosclaw.simforge.proof import ModuleEvidenceLevel, ProofBundle

_CORE_CAUSAL_MODULES = (
    "body",
    "provider",
    "failure_router",
    "sandbox",
    "practice",
    "memory",
    "know",
    "how",
    "auto",
    "darwin",
)
_REPLAY_MODULES = ("sandbox", "practice", "darwin")


class Phase3Decision(StrEnum):
    SIM_CHAMPION = "SIM_CHAMPION"
    REJECTED = "REJECTED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"


@dataclass(frozen=True)
class Phase3Check:
    gate: str
    passed: bool
    missing: bool
    detail: str


@dataclass(frozen=True)
class ContactPushPromotionRecord:
    candidate_hash: str
    body_snapshot_hash: str
    dataset_snapshot_hash: str
    validation_bundle_hash: str
    holdout_bundle_hash: str
    proof_bundle_hash: str
    stress_candidate_hash: str
    stress_attestation_hash: str
    statistical_gate: tuple[GateCheck, ...]
    causal_gate: tuple[Phase3Check, ...]
    decision: Phase3Decision
    schema_version: str = "rosclaw.contact_push_promotion.v1"

    @property
    def passed(self) -> bool:
        return self.decision is Phase3Decision.SIM_CHAMPION

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "candidate_hash": self.candidate_hash,
            "body_snapshot_hash": self.body_snapshot_hash,
            "dataset_snapshot_hash": self.dataset_snapshot_hash,
            "validation_bundle_hash": self.validation_bundle_hash,
            "holdout_bundle_hash": self.holdout_bundle_hash,
            "proof_bundle_hash": self.proof_bundle_hash,
            "stress_candidate_hash": self.stress_candidate_hash,
            "stress_attestation_hash": self.stress_attestation_hash,
            "statistical_gate": [asdict(check) for check in self.statistical_gate],
            "causal_gate": [asdict(check) for check in self.causal_gate],
            "decision": self.decision.value,
        }
        value["promotion_hash"] = _hash_json(value)
        return value

    @property
    def promotion_hash(self) -> str:
        return str(self.to_dict()["promotion_hash"])

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ContactPushPromotionRecord:
        statistical = tuple(GateCheck(**item) for item in value["statistical_gate"])
        causal = tuple(Phase3Check(**item) for item in value["causal_gate"])
        record = cls(
            candidate_hash=str(value["candidate_hash"]),
            body_snapshot_hash=str(value["body_snapshot_hash"]),
            dataset_snapshot_hash=str(value["dataset_snapshot_hash"]),
            validation_bundle_hash=str(value["validation_bundle_hash"]),
            holdout_bundle_hash=str(value["holdout_bundle_hash"]),
            proof_bundle_hash=str(value["proof_bundle_hash"]),
            stress_candidate_hash=str(value["stress_candidate_hash"]),
            stress_attestation_hash=str(value["stress_attestation_hash"]),
            statistical_gate=statistical,
            causal_gate=causal,
            decision=Phase3Decision(str(value["decision"])),
            schema_version=str(value.get("schema_version", "rosclaw.contact_push_promotion.v1")),
        )
        claimed_hash = value.get("promotion_hash")
        if claimed_hash is not None and claimed_hash != record.promotion_hash:
            raise ValueError("ContactPush promotion artifact hash mismatch")
        return record


class ContactPushPhase3Gate:
    def __init__(self, statistical_policy: GateV3Policy | None = None) -> None:
        self._statistical = StatisticalGateV3(statistical_policy)

    def evaluate(
        self,
        *,
        candidate: ContactPushCandidate,
        body_snapshot_hash: str,
        dataset_snapshot_hash: str | None,
        validation: EvaluationBundle | None,
        holdout: EvaluationBundle | None,
        proof_bundle: ProofBundle | None,
        stress: StressEvidence | None,
        stress_attestation_hash: str | None,
        counterexample_regression: EvaluationBundle | None,
        same_seed_retry_passed: bool | None,
        memory_attempts_saved: int | None,
        know_invalid_candidates_reduced: int | None,
        know_safety_override_count: int | None,
    ) -> ContactPushPromotionRecord:
        statistical = self._statistical.evaluate(
            validation=validation,
            holdout=holdout,
            stress=stress,
            counterexample_regression=counterexample_regression,
        )
        learned_binding = (
            candidate.dataset_snapshot_hash is not None
            and dataset_snapshot_hash is not None
            and candidate.dataset_snapshot_hash == dataset_snapshot_hash
        )
        candidate_identity = (
            validation is not None
            and holdout is not None
            and validation.candidate_hash == candidate.candidate_hash
            and holdout.candidate_hash == candidate.candidate_hash
            and validation.partition is Partition.VALIDATION
            and holdout.partition is Partition.HOLDOUT
        )
        proof_identity = (
            proof_bundle is not None
            and proof_bundle.task_id == "contact_push_v3"
            and proof_bundle.body_snapshot_hash == body_snapshot_hash
            and proof_bundle.candidate_hash == candidate.candidate_hash
        )
        core_levels = _proof_levels_pass(
            proof_bundle,
            ModuleEvidenceLevel.DECISION_IMPACT,
            _CORE_CAUSAL_MODULES,
        )
        replay_levels = _proof_levels_pass(
            proof_bundle,
            ModuleEvidenceLevel.REPLAY_VERIFIED,
            _REPLAY_MODULES,
        )
        causal = (
            _check(
                "P3-STRESS-IDENTITY",
                stress is not None
                and stress.candidate_hash == candidate.candidate_hash
                and bool(stress_attestation_hash)
                and str(stress_attestation_hash).startswith("sha256:"),
                stress is None or stress_attestation_hash is None,
                "four-GPU stress attestation is bound to the promoted candidate",
            ),
            _check(
                "P3-DATASET",
                learned_binding,
                dataset_snapshot_hash is None or candidate.dataset_snapshot_hash is None,
                "learned candidate is bound to the immutable Practice dataset snapshot",
            ),
            _check(
                "P3-IDENTITY",
                candidate_identity and proof_identity,
                validation is None or holdout is None or proof_bundle is None,
                "candidate/evaluation/task/body identities match",
            ),
            _check(
                "P3-CAUSAL",
                core_levels,
                proof_bundle is None,
                "Body, Provider, Failure Router, Sandbox, Practice, Memory, Know, "
                "How, Auto, and Darwin reach E3",
            ),
            _check(
                "P3-REPLAY",
                replay_levels,
                proof_bundle is None,
                "Sandbox, Practice, and Darwin reach E5 before activation",
            ),
            _check(
                "P3-SAME-SEED",
                same_seed_retry_passed is True,
                same_seed_retry_passed is None,
                "the executable How patch succeeds on the exact failed seed",
            ),
            _check(
                "P3-MEMORY",
                memory_attempts_saved is not None and memory_attempts_saved > 0,
                memory_attempts_saved is None,
                "Memory ON saves at least one matched search attempt",
            ),
            _check(
                "P3-KNOW",
                know_invalid_candidates_reduced is not None
                and know_invalid_candidates_reduced > 0
                and know_safety_override_count == 0,
                know_invalid_candidates_reduced is None or know_safety_override_count is None,
                "Know removes invalid candidates and admits zero safety overrides",
            ),
            _check(
                "P3-AUTONOMY",
                candidate.human_involvement.fully_autonomous,
                False,
                "candidate generation and promotion contain no human-selection flags",
            ),
        )
        missing = statistical.decision is GateDecision.NEED_MORE_EVIDENCE or any(
            check.missing for check in causal
        )
        passed = statistical.passed and all(check.passed for check in causal)
        decision = (
            Phase3Decision.NEED_MORE_EVIDENCE
            if missing
            else Phase3Decision.SIM_CHAMPION
            if passed
            else Phase3Decision.REJECTED
        )
        return ContactPushPromotionRecord(
            candidate_hash=candidate.candidate_hash,
            body_snapshot_hash=body_snapshot_hash,
            dataset_snapshot_hash=dataset_snapshot_hash or "",
            validation_bundle_hash=_bundle_hash(validation),
            holdout_bundle_hash=_bundle_hash(holdout),
            proof_bundle_hash=proof_bundle.bundle_hash if proof_bundle is not None else "",
            stress_candidate_hash=stress.candidate_hash if stress is not None else "",
            stress_attestation_hash=stress_attestation_hash or "",
            statistical_gate=statistical.checks,
            causal_gate=causal,
            decision=decision,
        )


def _proof_levels_pass(
    bundle: ProofBundle | None,
    minimum: ModuleEvidenceLevel,
    modules: tuple[str, ...],
) -> bool:
    if bundle is None:
        return False
    try:
        bundle.require_levels(minimum=minimum, modules=modules)
    except ValueError:
        return False
    return True


def _check(gate: str, passed: bool, missing: bool, detail: str) -> Phase3Check:
    return Phase3Check(gate=gate, passed=passed, missing=missing, detail=detail)


def _bundle_hash(bundle: EvaluationBundle | None) -> str:
    return _hash_json(bundle.aggregate_dict()) if bundle is not None else ""


def _hash_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "ContactPushPhase3Gate",
    "ContactPushPromotionRecord",
    "Phase3Check",
    "Phase3Decision",
]
