"""Task-neutral arithmetic for a complete seen-bank milestone, never permission.

The caller must authenticate evidence and freeze the contract before evaluation.
Hashes bind declarations; this module cannot prove receipt truth or chronology.
There is deliberately no training, promotion, sealed-bank or executor integration.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw.feedback.contracts import canonical_hash


def _hash(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("content-bound SHA-256 required")


def _case(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,127}", value) is None:
        raise ValueError("normalized bounded case identity required")


@dataclass(frozen=True)
class SeenMilestoneContract:
    case_ids: tuple[str, ...]
    reference_positive_ids: tuple[str, ...]
    minimum_safe_successes: int
    reference_execution_hash: str
    candidate_execution_hash: str
    bank_hash: str
    evaluation_hash: str

    def __post_init__(self) -> None:
        for values in (self.case_ids, self.reference_positive_ids):
            if type(values) is not tuple or len(values) > 4096:
                raise ValueError("bounded immutable case list required")
            for value in values:
                _case(value)
            if tuple(sorted(set(values))) != values:
                raise ValueError("case lists must be unique and sorted")
        if not self.case_ids or not set(self.reference_positive_ids).issubset(self.case_ids):
            raise ValueError("positive anchors must belong to a nonempty declared bank")
        if type(self.minimum_safe_successes) is not int or not len(
            self.reference_positive_ids
        ) <= self.minimum_safe_successes <= len(self.case_ids):
            raise ValueError("milestone must preserve anchors within declared bank size")
        for value in (
            self.reference_execution_hash,
            self.candidate_execution_hash,
            self.bank_hash,
            self.evaluation_hash,
        ):
            _hash(value)

    @property
    def contract_hash(self) -> str:
        self.__post_init__()
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class SeenPairedOutcome:
    case_id: str
    reference_success: bool
    reference_safe: bool
    candidate_success: bool
    candidate_safe: bool
    intervention_applied: bool
    independent_replay_verified: bool
    reference_execution_hash: str
    candidate_execution_hash: str
    evaluation_hash: str
    bank_hash: str
    evidence_hash: str

    def __post_init__(self) -> None:
        _case(self.case_id)
        for value in (
            self.reference_success,
            self.reference_safe,
            self.candidate_success,
            self.candidate_safe,
            self.intervention_applied,
            self.independent_replay_verified,
        ):
            if type(value) is not bool:
                raise ValueError(
                    "explicit outcome, safety, intervention and replay booleans required"
                )
        for value in (
            self.reference_execution_hash,
            self.candidate_execution_hash,
            self.evaluation_hash,
            self.bank_hash,
            self.evidence_hash,
        ):
            _hash(value)


def evaluate_seen_milestone(
    contract: SeenMilestoneContract, outcomes: tuple[SeenPairedOutcome, ...]
) -> dict[str, Any]:
    """One row per declared source case; repeats cannot inflate sample count."""
    if type(contract) is not SeenMilestoneContract or type(outcomes) is not tuple:
        raise ValueError("typed frozen milestone contract and outcome tuple required")
    contract.__post_init__()
    if len(outcomes) != len(contract.case_ids):
        raise ValueError("complete declared bank required")
    for row in outcomes:
        if type(row) is not SeenPairedOutcome:
            raise ValueError("typed paired outcomes required")
        row.__post_init__()
        if not row.independent_replay_verified:
            raise ValueError("each source case requires independent replay")
        if (
            row.reference_execution_hash != contract.reference_execution_hash
            or row.candidate_execution_hash != contract.candidate_execution_hash
            or row.evaluation_hash != contract.evaluation_hash
            or row.bank_hash != contract.bank_hash
        ):
            raise ValueError("mixed execution or scoring lineage rejected")
        if (
            not row.intervention_applied
            or row.reference_execution_hash == row.candidate_execution_hash
        ) and (
            (row.reference_success, row.reference_safe)
            != (row.candidate_success, row.candidate_safe)
        ):
            raise ValueError("unchanged fallback must reproduce reference outcomes")
    ordered = tuple(sorted(outcomes, key=lambda row: row.case_id))
    if tuple(row.case_id for row in ordered) != contract.case_ids:
        raise ValueError("duplicate, missing or undeclared source case")
    positives = tuple(
        row.case_id for row in ordered if row.reference_success and row.reference_safe
    )
    if positives != contract.reference_positive_ids:
        raise ValueError("reference positive identities differ from frozen anchors")
    unsafe = tuple(row.case_id for row in ordered if not row.candidate_safe)
    lost = tuple(
        row.case_id
        for row in ordered
        if row.case_id in contract.reference_positive_ids
        and not (row.candidate_success and row.candidate_safe)
    )
    successes = sum(row.candidate_success and row.candidate_safe for row in ordered)
    passed = successes >= contract.minimum_safe_successes and not unsafe and not lost
    report = {
        "schema": "rosclaw.seen_milestone.v1",
        "contract_hash": contract.contract_hash,
        "status": "SEEN_MILESTONE_MET" if passed else "SEEN_MILESTONE_NOT_MET",
        "case_count": len(ordered),
        "reference_safe_successes": len(positives),
        "candidate_safe_successes": successes,
        "rescued_case_ids": tuple(
            row.case_id
            for row in ordered
            if not (row.reference_success and row.reference_safe)
            and row.candidate_success
            and row.candidate_safe
        ),
        "lost_case_ids": lost,
        "unsafe_case_ids": unsafe,
        "intervention_cases": sum(row.intervention_applied for row in ordered),
        "unchanged_fallback_cases": sum(not row.intervention_applied for row in ordered),
        "outcomes_hash": canonical_hash(tuple(asdict(row) for row in ordered)),
        "evidence_domain": "SEEN_ONLY",
        "evidence_authenticated_by_this_function": False,
        "teacher_qualified": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "blind_exam_evaluated": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["manifest_hash"] = canonical_hash(report)
    return report
