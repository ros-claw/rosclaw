"""Conditional staging is separate from current physics and permission."""

from copy import deepcopy

import pytest

from benchmarks.ros_expert_ablation.aci_context import EngineeringEpisodeContext
from benchmarks.ros_expert_ablation.aci_schema import decision_schema
from rosclaw.connectors.ros.engineering.staging_options import StagingOptionV1
from tests.benchmarks.test_compound_staging import compound
from tests.benchmarks.test_physical_aci import fixture


def select(aci, refs):
    return aci.call(
        {
            "operation": "ros_engineer.select_staging_options",
            "arguments": {"case_hash": aci.case.case_hash, "option_refs": refs},
        }
    )


@pytest.mark.parametrize(
    "fault,kind",
    [
        ("tf", "enable_localization_tf"),
        ("scan", "resume_scan_bridge"),
        ("life", "activate_managed_nodes"),
        ("footprint", "cover_urdf_envelope"),
    ],
)
def test_original_evidence_compiles_an_unexecuted_conditional_option(fault, kind):
    aci = fixture(fault)
    before = deepcopy(aci.tools.staging)
    options = aci.conditional_staging_options()["options"]
    assert len(options) == 1
    option = options[0]
    assert (
        option.intent.kind == kind
        and option.current_authority == "NONE"
        and option.current_case_l1 == "NOT_EVALUATED"
    )
    assert len(option.required_new_world_proofs) == 4
    if fault != "life":
        assert option.original_candidate_preconditions
    assert not aci.proposals and aci.validation_count == 0 and aci.tools.staging == before
    result = select(aci, [option.option_ref])
    assert result["proposal"]["proposal_id"] == option.proposal_id
    assert (
        result["selected_staging_option_refs"] == [option.option_ref]
        and aci.tools.staging == before
    )


def test_duplicate_or_fabricated_refs_never_register_a_proposal():
    aci = fixture("tf")
    ref = aci.conditional_staging_options()["options"][0].option_ref
    for refs in [[ref, ref], ["invented-ref"]]:
        assert select(aci, refs)["status"] == "REJECTED_TYPED_ACI_REQUEST"
    assert not aci.proposals and aci.validation_count == 0


def test_changed_option_payload_or_base_is_rejected():
    aci = fixture("tf")
    option = aci.conditional_staging_options()["options"][0]
    payload = option.model_dump(mode="json")
    payload["changes"][0]["value"] = False
    with pytest.raises(ValueError, match="hash mismatch"):
        StagingOptionV1.model_validate(payload)
    aci.tools.staging["nav2.yaml"]["/sample"]["amcl"]["ros__parameters"]["tf_broadcast"] = True
    assert (
        select(aci, [option.option_ref])["status"] == "REJECTED_TYPED_ACI_REQUEST"
        and not aci.proposals
    )


def test_multiple_refs_recompile_one_original_baseline_and_exact_rollback():
    aci = compound()
    before = deepcopy(aci.tools.staging)
    options = aci.conditional_staging_options()["options"]
    assert len(options) == 2
    result = select(aci, [r.option_ref for r in options])
    p = result["proposal"]
    assert len(p["changes"]) == 2 and len(p["rollback"]) == 2
    assert (
        p["intent"]["kind"] == "compound_repair"
        and aci.tools.staging == before
        and aci.validation_count == 0
    )


def test_common_arm_gets_no_expert_precompiled_options():
    aci = fixture("tf")
    ref = aci.conditional_staging_options()["options"][0].option_ref
    aci.tools.arm = "A"
    assert select(aci, [ref])["status"] == "REJECTED_TYPED_ACI_REQUEST"
    assert (
        "staging_options"
        not in aci.call({"operation": "ros_engineer.inspect_task", "arguments": {}})["task"]
    )
    assert "SelectOptionsCall" not in decision_schema("A")["$defs"]


def test_healthy_and_unknown_owned_route_have_no_supported_option():
    assert fixture("healthy").conditional_staging_options()["options"] == []
    aci = fixture("route")
    result = aci.conditional_staging_options()
    assert result["options"] == [] and result["blocked"]


def test_phase_schema_binds_actual_option_refs_and_context_keeps_remaining_proof():
    aci = fixture("tf")
    options = aci.conditional_staging_options()["options"]
    refs = [r.option_ref for r in options]
    schema = decision_schema(
        "B", case_hash=aci.case.case_hash, proposal_ids=[], verification_refs=[], option_refs=refs
    )
    assert (
        schema["$defs"]["SelectOptionsArgs"]["properties"]["option_refs"]["items"]["enum"] == refs
    )
    frame = EngineeringEpisodeContext(aci).frame()
    option = frame["task"]["task"]["staging_options"][0]
    assert option["issue_id"] == "F-TF-001"
    assert frame["task"]["task"]["staging_phase"] == "STATIC_DIFF_CHECKED_CONDITIONAL_SELECTION"
    assert len(frame["task"]["task"]["proof_obligations"]) == 4
    assert frame["task"]["task"]["conditional_candidates"][0]["preconditions"]
    assert (
        frame["current_l1"] == "NOT_EVALUATED"
        and frame["current_authority"] == "NONE"
        and not frame["proposals"]
    )


def test_runner_selects_a_conditional_hypothesis_but_finish_cannot_verify(tmp_path):
    import json

    from benchmarks.ros_expert_ablation.aci_runner import run_typed_episode
    from tests.benchmarks.test_typed_aci_runner import TASK, client_factory

    aci = fixture("tf")
    ref = aci.conditional_staging_options()["options"][0].option_ref
    request = {
        "operation": "ros_engineer.select_staging_options",
        "arguments": {"case_hash": aci.case.case_hash, "option_refs": [ref]},
    }
    out = tmp_path / "episode"
    result = run_typed_episode(
        TASK,
        aci,
        out,
        model_client=client_factory([request, {"operation": "finish", "arguments": {}}], []),
        tool_budget=2,
    )
    assert result["first_valid_proposal"]["turn"] == 1 and result["validation_calls"] == 0
    assert (
        result["status"] == "FINISHED_WITHOUT_ACCEPTED_SDK_RESULT"
        and result["verified_success"] is False
    )
    full = json.loads((out / "operator-original-staging-options.json").read_text())
    assert (
        full["options"][0]["option_ref"] == ref
        and full["options"][0]["source_urdf_sha256"] == aci.case.source_urdf_sha256
    )


def test_stale_case_and_mismatched_source_do_not_offer_repair_options():
    from rosclaw.contracts.common import content_hash

    aci = fixture("tf")
    data = aci.case.model_dump(mode="json")
    data["source_status"] = "MISMATCH"
    data["case_hash"] = content_hash(
        "roscase", {key: value for key, value in data.items() if key != "case_hash"}
    )
    aci.case = type(aci.case).from_dict(data)
    assert not aci.conditional_staging_options()["options"]
    aci = fixture("tf")
    ref = aci.conditional_staging_options()["options"][0].option_ref
    node = aci.requirements["lifecycle_nodes"]["localization"]
    aci.model.observations["node_parameters"][node]["tf_broadcast"] = True
    assert select(aci, [ref])["status"] == "REJECTED_TYPED_ACI_REQUEST"
    assert not aci.proposals


def test_scan_symptom_with_running_bridge_has_no_invented_resume_option():
    import json

    from benchmarks.ros_expert_ablation.physical_aci import PhysicalACI
    from benchmarks.ros_expert_ablation.task_tools import TaskTools

    original = fixture("scan")
    files = deepcopy(original.tools.files)
    files["bringup.json"] = json.dumps({"scan_bridge_state": "running"})
    tools = TaskTools(
        files,
        arm="B",
        reviewed_at=original.tools.reviewed_at,
        minimum_radius_m=original.tools.minimum_radius_m,
    )
    aci = PhysicalACI(original.case, original.model, tools)
    compiled = aci.conditional_staging_options()
    assert not compiled["options"] and compiled["blocked"][0]["issue_id"] == "F-SCAN-001"
    card = aci.call({"operation": "ros_engineer.inspect_task", "arguments": {}})["task"]
    assert card["staging_phase"] == "NO_SUPPORTED_STAGING_OPTION" and not aci.proposals


def test_malformed_original_managed_node_list_returns_no_option():
    from rosclaw.connectors.ros.engineering import compile_engineering_case
    from rosclaw.connectors.ros.engineering.staging_options import compile_staging_options
    from tests.connectors.ros.test_engineering_case_compiler import NOW

    aci = fixture("life")
    manager = "/sample/lifecycle_manager_navigation"
    aci.model.observations["node_parameters"][manager]["node_names"] = [123]
    aci.model.seal()
    case = compile_engineering_case(
        aci.model,
        aci.tools.files["robot.urdf"].encode(),
        aci.requirements,
        source_ref="fixture:malformed-managed-nodes",
        now=NOW,
        evidence_domain="SEALED_REPLAY",
    )
    result = compile_staging_options(
        case,
        aci.model,
        aci.requirements,
        aci.tools.staging,
        minimum_radius_m=aci.tools.minimum_radius_m,
    )
    assert not result["options"] and "string list required" in result["blocked"][0]["reason"]
