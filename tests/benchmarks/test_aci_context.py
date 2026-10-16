"""State frames do not grow with transcript; Memory remains a historical prior."""

import json
from copy import deepcopy

import pytest

from benchmarks.ros_expert_ablation.aci_context import EngineeringEpisodeContext
from tests.benchmarks.test_physical_aci import fixture, repair


def retrieval(rows=()):
    return {
        "status": "ACTUAL_PROVENANCE_FILTERED_TRAINING_RETRIEVAL",
        "authority": "NONE",
        "retrieval_event_sha256": "e" * 64,
        "admitted": list(rows),
        "admitted_count": len(rows),
    }


def memory_row(index=1):
    return {
        "memory_ref": "memory_" + f"{index:024x}",
        "original_episode_sha256": "c" * 64,
        "original_training_source": "d" * 40,
        "evidence_scope": "HISTORICAL_VERIFIED_SIM_TRAINING; NOT_CURRENT_BODY_OR_EXECUTION_AUTHORITY",
        "diagnosis_issue_codes": ["NAV2_COSTMAP_002"],
        "instruction": "NEVER put this unbounded historical text into the current frame" * 2000,
    }


def test_frame_is_bounded_and_journal_remains_complete_after_repeated_results():
    context = EngineeringEpisodeContext(fixture())
    first = context.frame()
    assert len(json.dumps(first, ensure_ascii=False)) < 3500
    for _ in range(100):
        context.call({"operation": "ros_engineer.inspect_task", "arguments": {}})
    last = context.frame()
    assert len(json.dumps(last)) < 4000
    assert last["state_revision"] == 100
    assert len(context.aci.trace) == len(context.causality_trace) == 100
    assert last["state_hash"] != first["state_hash"]
    assert context.frame()["state_hash"] == last["state_hash"]
    assert context.frame_metrics[-1]["utf8_bytes"] > 0
    assert context.frame_metrics[-1]["token_measurement"] == "ESTIMATE_NOT_PROVIDER_USAGE"


def test_proposal_and_finish_update_phase_without_hidden_validation():
    calls = []
    context = EngineeringEpisodeContext(fixture(evaluator=lambda *a, **kw: calls.append(a)))
    context.call(
        {
            "operation": "ros_engineer.propose_repair",
            "arguments": {
                "case_hash": context.aci.case.case_hash,
                "repair": {"kind": "cover_urdf_envelope", "radius_m": 0.4},
            },
        }
    )
    assert context.frame()["phase"] == "PROPOSED"
    context.call({"operation": "finish", "arguments": {}})
    assert context.frame()["phase"] == "FINISHED"
    assert calls == [] and context.aci.validation_count == 0


def test_c_automatic_empty_admission_removes_model_query_requirement_only_in_new_adapter():
    aci = fixture()
    aci.tools.arm = "C"
    queries = []
    aci.tools.memory_query = lambda query: queries.append(query) or retrieval()
    context = EngineeringEpisodeContext(aci)
    assert len(queries) == 1 and len(aci.tools.memory_trace) == 1
    assert context.frame()["memory_candidates"] == []
    assert repair(aci, radius_m=0.4)["status"] == "TYPED_STAGING_PROPOSAL_NOT_EXECUTED"
    assert context.stages["host_memory_retrieval"] == 1


def test_legacy_c_still_requires_query_and_a_b_never_retrieve():
    for arm in ["A", "B", "C"]:
        aci = fixture()
        aci.tools.arm = arm
        aci.tools.memory_query = lambda query: pytest.fail("A/B must not retrieve")
        if arm != "C":
            assert EngineeringEpisodeContext(aci).memory_candidates == []
        else:
            assert repair(aci, radius_m=0.4)["status"] == "REJECTED_TYPED_ACI_REQUEST"


def test_candidates_have_max_two_real_refs_and_citations_do_not_establish_benefit():
    aci = fixture()
    aci.tools.arm = "C"
    rows = [memory_row(index) for index in range(1, 4)]
    aci.tools.memory_query = lambda query: retrieval(rows)
    context = EngineeringEpisodeContext(aci)
    frame = context.frame()
    assert len(frame["memory_candidates"]) == 2
    assert "NEVER put" not in json.dumps(frame)
    context.call({"operation": "finish", "arguments": {}}, memory_refs_used=[rows[0]["memory_ref"]])
    assert context.causality_trace[-1]["memory_refs_declared"] == [rows[0]["memory_ref"]]
    assert context.causality_trace[-1]["causal_benefit"] == "NOT_ESTABLISHED_BY_CITATION"
    with pytest.raises(ValueError, match="actually admitted"):
        context.call(
            {"operation": "finish", "arguments": {}}, memory_refs_used=[rows[2]["memory_ref"]]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("authority", "REAL"),
        ("status", "MODEL_SAYS_VERIFIED"),
        ("retrieval_event_sha256", "bad"),
        ("admitted_count", 99),
    ],
)
def test_bad_memory_receipt_fails_closed(field, value):
    aci = fixture()
    aci.tools.arm = "C"
    bad = retrieval([memory_row()])
    bad[field] = value
    aci.tools.memory_query = lambda query: deepcopy(bad)
    with pytest.raises(ValueError, match="provenance admission"):
        EngineeringEpisodeContext(aci)


def test_changed_case_cannot_be_reused_as_current_state():
    context = EngineeringEpisodeContext(fixture())
    context.aci.case.constraints[0].evidence_refs.append("invented")
    with pytest.raises(ValueError, match="hash mismatch"):
        context.frame()


def test_invalid_arguments_keep_actionable_error_but_no_unbounded_result_in_frame():
    context = EngineeringEpisodeContext(fixture())
    context.call({"operation": {}, "arguments": {"unexpected": "x" * 5000}})
    result = context.frame()["recent_result"]
    assert result["status"] == "REJECTED_TYPED_ACI_REQUEST"
    assert result["legal_example"]["operation"] == "ros_engineer.inspect_task"
    assert len(result["error_excerpt"]) <= 400
    assert context.stages["INVALID"] == 1
    assert context.causality_trace[-1]["original_result_utf8_bytes"] > 0


def test_failed_verification_remains_review_phase_and_records_gate_details():
    aci = fixture(evaluator=lambda *a, **kw: {"verified_success": False})
    context = EngineeringEpisodeContext(aci)
    proposal = repair(aci, radius_m=0.4)["proposal"]
    context.call(
        {
            "operation": "ros_engineer.validate_proposal",
            "arguments": {
                "case_hash": aci.case.case_hash,
                "proposal_id": proposal["proposal_id"],
            },
        },
        remaining_sec=100,
    )
    frame = context.frame()
    assert frame["phase"] == "VALIDATION_REVIEW"
    assert frame["recent_result"]["verification"]["accepted_success"] is False
    assert frame["current_l1"] == "NOT_EVALUATED" and frame["current_authority"] == "NONE"
