"""Final-closeout budgets and missing-input gates do not reuse RSK allocation."""

import pytest
from pydantic import ValidationError

from benchmarks.ros_expert_ablation.closeout_protocol import (
    SCENARIOS,
    CloseoutProtocolV1,
    physical_gate,
    stop_after_sentinels,
)


def registration():
    return {
        "harness_source": "a" * 40,
        "code_inventory_sha256": "b" * 64,
        "provider_inventory_sha256": "c" * 64,
        "case_registry_sha256": "d" * 64,
        "model_identity": "registered-model",
        "environment_seeds": [101, 202],
        "cases": [
            {
                "instance_id": f"{i:024x}",
                "scenario": scenario,
                "environment_seed": seed,
                "facts_sha256": f"{i:064x}",
                "evidence_kind": "ORIGINAL_CAPTURE",
            }
            for i, (scenario, seed) in enumerate(
                ((scenario, seed) for seed in (101, 202) for scenario in SCENARIOS), 1
            )
        ],
    }


def test_exact_matrix_fixed_sentinel_order_and_independent_protocol_hash():
    p = CloseoutProtocolV1.model_validate(registration())
    rows = p.allocations()
    assert [r.scenario for r in rows[:4]] == ["P1", "N1", "N2", "N4"]
    assert len({r.instance_id for r in rows}) == 14
    assert p.original_entry_gate() == "ELIGIBLE_FOR_ORIGINAL_INPUT_REVIEW"
    assert p.permission == "NONE" and p.model_sampler_seed == "NOT_CONTROLLED"
    changed = registration()
    changed["harness_source"] = "e" * 40
    assert CloseoutProtocolV1.model_validate(changed).protocol_hash != p.protocol_hash


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_model_decisions", 5),
        ("max_physical_worlds", 7),
        ("environment_seeds", [101, 101]),
        ("environment_seeds", [True, 202]),
        ("max_total_model_invocations", 57),
    ],
)
def test_budget_or_seed_expansion_is_rejected(field, value):
    value_dict = registration()
    value_dict[field] = value
    with pytest.raises(ValidationError):
        CloseoutProtocolV1.model_validate(value_dict)


@pytest.mark.parametrize("kind", ["REPLAY", "SYNTHETIC", "UNAVAILABLE"])
def test_derived_or_missing_originals_cannot_authorize_trials(kind):
    value = registration()
    value["cases"][0]["evidence_kind"] = kind
    if kind == "UNAVAILABLE":
        value["cases"][0]["facts_sha256"] = None
    p = CloseoutProtocolV1.model_validate(value)
    assert p.original_entry_gate() == "NO_GO_ORIGINAL_CASE_MATRIX_INCOMPLETE"


@pytest.mark.parametrize(
    "fault", ["missing_cell", "duplicate_id", "duplicate_hash", "false_original"]
)
def test_registry_integrity_and_missing_evidence(fault):
    value = registration()
    if fault == "missing_cell":
        value["cases"].pop()
    elif fault == "duplicate_id":
        value["cases"][1]["instance_id"] = value["cases"][0]["instance_id"]
    elif fault == "duplicate_hash":
        value["cases"][1]["facts_sha256"] = value["cases"][0]["facts_sha256"]
    else:
        value["cases"][0]["facts_sha256"] = None
    with pytest.raises(ValidationError):
        CloseoutProtocolV1.model_validate(value)


def test_host_refusal_is_not_model_abstention_and_proposals_are_not_motion():
    assert stop_after_sentinels(["BUDGET_EXHAUSTED"] * 4).startswith("NO_GO")
    assert stop_after_sentinels(["HOST_REJECTED_UNSAFE"] * 4) is None
    assert stop_after_sentinels([], unsafe_accepted=True).startswith("NO_GO")
    assert (
        physical_gate(original_p1_admitted=False, valid_p1_proposals=2, safety_passed=True)
        == "NOT_RUN_NO_GO"
    )
    assert (
        physical_gate(original_p1_admitted=True, valid_p1_proposals=0, safety_passed=True)
        == "NOT_RUN_NO_GO"
    )
    assert (
        physical_gate(original_p1_admitted=True, valid_p1_proposals=1, safety_passed=False)
        == "NOT_RUN_NO_GO"
    )
