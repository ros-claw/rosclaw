from dataclasses import replace

import pytest

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.seen_milestone import (
    SeenMilestoneContract,
    SeenPairedOutcome,
    evaluate_seen_milestone,
)


def fixture():
    contract = SeenMilestoneContract(
        ("case.a", "case.b", "case.c", "case.d"),
        ("case.a",),
        2,
        canonical_hash("reference-execution"),
        canonical_hash("candidate-execution"),
        canonical_hash("frozen-seen-bank"),
        canonical_hash("unchanged-exam"),
    )
    rows = tuple(
        SeenPairedOutcome(
            case,
            case == "case.a",
            True,
            case in ("case.a", "case.b"),
            True,
            case != "case.d",
            True,
            contract.reference_execution_hash,
            contract.candidate_execution_hash,
            contract.evaluation_hash,
            contract.bank_hash,
            canonical_hash({"evidence": case}),
        )
        for case in contract.case_ids
    )
    return contract, rows


def test_complete_seen_improvement_has_no_learning_or_execution_authority():
    contract, rows = fixture()
    result = evaluate_seen_milestone(contract, rows)
    assert result["status"] == "SEEN_MILESTONE_MET"
    assert result["candidate_safe_successes"] == 2
    assert result["rescued_case_ids"] == ("case.b",)
    assert result["unchanged_fallback_cases"] == 1
    assert result["evidence_domain"] == "SEEN_ONLY"
    assert result["activation_ceiling"] == "SIM_ONLY"
    for field in (
        "teacher_qualified",
        "training_authorized",
        "promotion_authorized",
        "blind_exam_evaluated",
        "evidence_authenticated_by_this_function",
    ):
        assert result[field] is False
    assert evaluate_seen_milestone(contract, tuple(reversed(rows))) == result


def test_more_successes_cannot_hide_lost_anchor():
    contract, rows = fixture()
    changed = tuple(
        replace(r, candidate_success=r.case_id != "case.a", intervention_applied=True) for r in rows
    )
    result = evaluate_seen_milestone(contract, changed)
    assert result["candidate_safe_successes"] == 3
    assert result["status"] == "SEEN_MILESTONE_NOT_MET"
    assert result["lost_case_ids"] == ("case.a",)


def test_unsafe_failed_case_still_blocks_milestone():
    contract, rows = fixture()
    result = evaluate_seen_milestone(
        contract, (*rows[:2], replace(rows[2], candidate_safe=False), rows[3])
    )
    assert result["candidate_safe_successes"] == 2
    assert result["status"] == "SEEN_MILESTONE_NOT_MET"
    assert result["unsafe_case_ids"] == ("case.c",)


def test_below_threshold_is_not_met():
    contract, rows = fixture()
    result = evaluate_seen_milestone(
        contract, (rows[0], replace(rows[1], candidate_success=False), *rows[2:])
    )
    assert result["status"] == "SEEN_MILESTONE_NOT_MET"


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "foreign"])
def test_only_complete_distinct_declared_cases_count(change):
    contract, rows = fixture()
    if change == "missing":
        rows = rows[:-1]
    if change == "duplicate":
        rows = (*rows[:-1], rows[0])
    if change == "extra":
        rows = (*rows, rows[0])
    if change == "foreign":
        rows = (*rows[:-1], replace(rows[-1], case_id="other"))
    with pytest.raises(ValueError):
        evaluate_seen_milestone(contract, rows)


@pytest.mark.parametrize(
    "field",
    ["reference_execution_hash", "candidate_execution_hash", "evaluation_hash", "bank_hash"],
)
def test_mixed_lineage_is_rejected(field):
    contract, rows = fixture()
    with pytest.raises(ValueError, match="lineage"):
        evaluate_seen_milestone(
            contract, (replace(rows[0], **{field: canonical_hash("foreign")}), *rows[1:])
        )


def test_reference_anchor_identity_not_just_count_is_bound():
    contract, rows = fixture()
    changed = (
        replace(rows[0], reference_success=False),
        replace(rows[1], reference_success=True),
        *rows[2:],
    )
    with pytest.raises(ValueError, match="anchors"):
        evaluate_seen_milestone(contract, changed)


def test_missing_replay_and_changed_fallback_rejected():
    contract, rows = fixture()
    with pytest.raises(ValueError, match="replay"):
        evaluate_seen_milestone(
            contract, (replace(rows[0], independent_replay_verified=False), *rows[1:])
        )
    with pytest.raises(ValueError, match="fallback"):
        evaluate_seen_milestone(contract, (*rows[:-1], replace(rows[-1], candidate_success=True)))


@pytest.mark.parametrize(
    "field",
    [
        "reference_success",
        "candidate_success",
        "reference_safe",
        "candidate_safe",
        "intervention_applied",
        "independent_replay_verified",
    ],
)
@pytest.mark.parametrize("value", [1, "true", None, float("nan")])
def test_labels_must_be_real_booleans(field, value):
    _, rows = fixture()
    with pytest.raises(ValueError, match="booleans"):
        replace(rows[0], **{field: value})


@pytest.mark.parametrize("value", [True, -1, 0, 5, 2.0, float("inf")])
def test_threshold_bound(value):
    contract, _ = fixture()
    with pytest.raises(ValueError):
        replace(contract, minimum_safe_successes=value)


def test_invalid_and_mutated_contract_rejected_at_use():
    contract, rows = fixture()
    for ids in ((), ("case.b", "case.a"), ("case.a", "case.a"), ["case.a"], ("bad name",)):
        with pytest.raises(ValueError):
            replace(contract, case_ids=ids)
    object.__setattr__(contract, "evaluation_hash", "bad")
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_seen_milestone(contract, rows)


def test_same_execution_hash_cannot_hide_changed_outcome():
    contract, rows = fixture()
    same = replace(contract, candidate_execution_hash=contract.reference_execution_hash)
    rows = tuple(replace(r, candidate_execution_hash=same.candidate_execution_hash) for r in rows)
    with pytest.raises(ValueError, match="fallback"):
        evaluate_seen_milestone(same, rows)


def test_evidence_and_threshold_change_report_identity():
    contract, rows = fixture()
    original = evaluate_seen_milestone(contract, rows)
    changed = evaluate_seen_milestone(
        contract, (replace(rows[0], evidence_hash=canonical_hash("different")), *rows[1:])
    )
    assert original["manifest_hash"] != changed["manifest_hash"]
    assert replace(contract, minimum_safe_successes=3).contract_hash != contract.contract_hash
