"""The new E tool path exposes concepts and staging, never implicit validation."""

from copy import deepcopy

import pytest

from benchmarks.ros_expert_ablation.physical_aci import PhysicalACI
from benchmarks.ros_expert_ablation.rsk_physical_aci import RskPhysicalACI
from tests.benchmarks.test_cross_body_memory_trace import query_fixture


def facade(tmp_path, arm="E"):
    query, _, record, aci, case = query_fixture(tmp_path)
    aci.tools.minimum_radius_m = case.derived_documents["navigation_observation_review"][
        "candidate"
    ]["geometry"]["physical_radius_m"]
    base = PhysicalACI(case, aci.model, aci.tools, max_validations=1)
    return RskPhysicalACI(base, arm=arm, memory_query=query if arm == "E" else None), record


def request(aci, operation, **arguments):
    return aci.call(
        {
            "operation": "ros_engineer." + operation,
            "arguments": {"case_hash": aci.base.case.case_hash, **arguments},
        }
    )


def test_actual_retrieval_context_and_typed_selection_leave_staging_and_authority_unchanged(
    tmp_path,
):
    aci, record = facade(tmp_path)
    before = deepcopy(aci.base.tools.staging)
    assert (
        request(aci, "propose_cross_body_memory_repair", memory_ref=record.experience_hash)[
            "status"
        ]
        == "REJECTED_TYPED_RSK_ACI_REQUEST"
    )
    result = request(aci, "query_cross_body_memory", query="disabled localization transform")
    assert result["concepts"][0]["historical_numeric_parameters"] == "OMITTED"
    selected = request(aci, "propose_cross_body_memory_repair", memory_ref=record.experience_hash)
    assert selected["status"] == "TYPED_CROSS_BODY_STAGING_PROPOSAL_NOT_EXECUTED"
    assert selected["proposal"]["proposal_id"] in aci.base.proposals
    assert selected["current_authority"] == "NONE"
    assert aci.base.validation_count == 0 and aci.base.tools.staging == before
    assert [e["stage"] for e in aci.memory_query.trace] == [
        "RETRIEVAL",
        "CONTEXT_INSERTION",
        "CONDITIONAL_SELECTION",
    ]
    failed = aci.call(
        {
            "operation": "ros_engineer.validate_proposal",
            "arguments": {
                "case_hash": aci.base.case.case_hash,
                "proposal_id": selected["proposal"]["proposal_id"],
            },
        },
        remaining_sec=10,
    )
    assert failed["status"] == "REJECTED_TYPED_ACI_REQUEST" and "unavailable" in failed["error"]


@pytest.mark.parametrize("arm", ["B", "C"])
def test_only_new_e_can_query_memory_and_old_g5_memory_operation_is_not_exposed(tmp_path, arm):
    aci, _ = facade(tmp_path, arm)
    assert (
        request(aci, "query_cross_body_memory", query="tf")["status"]
        == "REJECTED_TYPED_RSK_ACI_REQUEST"
    )
    schema = aci.request_schema()["discriminator"]["mapping"]
    assert "ros_engineer.query_cross_body_memory" not in schema
    assert "ros_engineer.propose_memory_repair" not in schema


@pytest.mark.parametrize("fault", ["permission", "stale_case", "changed_original", "finished"])
def test_injected_authority_and_stale_context_do_not_enter_memory(tmp_path, fault):
    aci, _ = facade(tmp_path)
    args = {"case_hash": aci.base.case.case_hash, "query": "tf"}
    if fault == "permission":
        args["permit"] = "invented"
    elif fault == "stale_case":
        args["case_hash"] = "stale"
    elif fault == "changed_original":
        aci.base.tools.files["robot.urdf"] += " "
    else:
        aci.base.finished = True
    result = aci.call({"operation": "ros_engineer.query_cross_body_memory", "arguments": args})
    assert result["status"] == "REJECTED_TYPED_RSK_ACI_REQUEST" and not aci.memory_query.trace


def test_valid_call_schema_has_consistent_shared_definitions_and_legacy_rejects_new_operation(
    tmp_path,
):
    aci, _ = facade(tmp_path)
    schema = aci.request_schema()
    for row in schema["oneOf"]:
        assert row["$ref"].split("/")[-1] in schema["$defs"]
    assert (
        aci.base.call(
            {
                "operation": "ros_engineer.query_cross_body_memory",
                "arguments": {"case_hash": aci.base.case.case_hash, "query": "tf"},
            }
        )["status"]
        == "REJECTED_TYPED_ACI_REQUEST"
    )


def test_independent_proposal_mismatch_cannot_leave_a_registered_candidate(tmp_path, monkeypatch):
    aci, record = facade(tmp_path)
    request(aci, "query_cross_body_memory", query="tf")
    select = aci.memory_query.select

    def mismatch(*args, **kwargs):
        result = select(*args, **kwargs)
        result["proposal"]["current_authority"] = "FORGED"
        return result

    monkeypatch.setattr(aci.memory_query, "select", mismatch)
    result = request(aci, "propose_cross_body_memory_repair", memory_ref=record.experience_hash)
    assert result["status"] == "REJECTED_TYPED_RSK_ACI_REQUEST"
    assert aci.base.proposals == {} and aci.base.validation_count == 0


def test_provider_schema_binds_current_case_and_only_offers_retrieved_context_references(tmp_path):
    aci, record = facade(tmp_path)
    schema = aci.decision_schema()
    assert "CrossBodyProposalCall" not in {
        v["$ref"].split("/")[-1] for v in schema["properties"]["decision"]["anyOf"]
    }
    result = aci.call_model_decision(
        {
            "decision": {
                "operation": "ros_engineer.query_cross_body_memory",
                "arguments": {"case_hash": aci.base.case.case_hash, "query": "tf"},
            },
            "decision_summary": "Read checked historical concepts",
            "memory_refs_used": [],
        }
    )
    assert result["current_authority"] == "NONE"
    schema = aci.decision_schema()
    assert schema["$defs"]["CrossBodyProposalArgs"]["properties"]["memory_ref"]["enum"] == [
        record.experience_hash
    ]
    assert schema["$defs"]["CrossBodyProposalArgs"]["properties"]["case_hash"]["enum"] == [
        aci.base.case.case_hash
    ]
    with pytest.raises(ValueError, match="admitted"):
        aci.call_model_decision(
            {
                "decision": {"operation": "finish", "arguments": {}},
                "decision_summary": "",
                "memory_refs_used": ["invented"],
            }
        )


def test_sdk_callback_requires_original_pilot_entry_before_facade_creation(tmp_path):
    aci, _ = facade(tmp_path)
    calls = []
    aci.base.evaluator = lambda *a, **kw: calls.append("SDK")
    with pytest.raises(ValueError, match="entry proof"):
        RskPhysicalACI(aci.base, arm="C")
    assert calls == []


def test_no_go_original_entry_cannot_attach_callback_even_with_valid_protocol(tmp_path):
    from benchmarks.ros_expert_ablation.rsk_pilot_protocol import GATES
    from tests.benchmarks.test_rsk_pilot_protocol import protocol

    aci, _ = facade(tmp_path)
    aci.base.evaluator = lambda *a, **kw: pytest.fail("no SDK callback may run")
    with pytest.raises(ValueError, match="NO_GO"):
        RskPhysicalACI(
            aci.base,
            arm="C",
            pilot_protocol=protocol(),
            pilot_evidence=dict.fromkeys(GATES),
            evidence_root=tmp_path,
        )


def test_eligible_gate_cannot_attach_sdk_from_another_harness_source(tmp_path):
    from tests.benchmarks.test_rsk_pilot_protocol import evidence, protocol

    query, _, _, aci, case = query_fixture(tmp_path)
    base = PhysicalACI(
        case,
        aci.model,
        aci.tools,
        max_validations=1,
        evaluator=lambda *a, **kw: pytest.fail("not called"),
        operator_source="b" * 40,
        original_run_id="f" * 32,
    )
    p = protocol()
    with pytest.raises(ValueError, match="operator source differs"):
        RskPhysicalACI(
            base,
            arm="B",
            pilot_protocol=p,
            pilot_evidence=evidence(tmp_path, p),
            evidence_root=tmp_path,
        )


def test_repair_validation_cannot_promote_reused_read_only_probe_world(tmp_path):
    from tests.benchmarks.test_rsk_application_aci import complete_result
    from tests.benchmarks.test_rsk_pilot_protocol import evidence, protocol

    query, _, _, aci, case = query_fixture(tmp_path)
    aci.tools.minimum_radius_m = case.derived_documents["navigation_observation_review"][
        "candidate"
    ]["geometry"]["physical_radius_m"]
    base = PhysicalACI(
        case,
        aci.model,
        aci.tools,
        max_validations=1,
        evaluator=lambda *a, **kw: complete_result(),
        operator_source="b" * 40,
        original_run_id="f" * 32,
    )
    p = protocol().model_copy(update={"harness_source": "b" * 40})
    typed = RskPhysicalACI(
        base,
        arm="B",
        pilot_protocol=p,
        pilot_evidence=evidence(tmp_path, p),
        evidence_root=tmp_path,
    )
    base.probe_results = [{"fresh_probe_performed": True, "new_world_run": "c" * 32}]
    proposed = request(typed, "propose_repair", repair={"kind": "enable_localization_tf"})
    result = typed.call(
        {
            "operation": "ros_engineer.validate_proposal",
            "arguments": {
                "case_hash": case.case_hash,
                "proposal_id": proposed["proposal"]["proposal_id"],
            },
        },
        remaining_sec=10,
    )
    assert result["verification"]["accepted_success"] is False
    assert result["verification"]["acceptance_status"] == "REJECTED_REUSED_READ_ONLY_PROBE_WORLD"
    assert not next(iter(base.results.values()))["accepted_success"]
