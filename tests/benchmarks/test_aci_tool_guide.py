"""Guide/schema alignment only; synthetic clients never count as model evidence."""

import json

import pytest

from benchmarks.ros_expert_ablation.aci_runner import run_typed_episode
from benchmarks.ros_expert_ablation.aci_schema import decision_schema
from benchmarks.ros_expert_ablation.aci_tool_guide import build_tool_guide
from tests.benchmarks.test_physical_aci import fixture
from tests.benchmarks.test_typed_aci_runner import TASK, client_factory


def operations(schema):
    return {
        schema["$defs"][choice["$ref"].rsplit("/", 1)[1]]["properties"]["operation"]["enum"][0]
        for choice in schema["properties"]["decision"]["anyOf"]
    }


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_guide_describes_only_actual_schema_operations_and_stays_bounded(arm):
    schema = decision_schema(arm, memory_available=arm == "C")
    guide = build_tool_guide(schema)
    rows = guide["operations_available_this_turn"]
    assert {row["operation"] for row in rows} == operations(schema)
    assert len(json.dumps(guide, separators=(",", ":"))) <= 6500
    assert "NOT_NATIVE_ROS_OR_ACTION_AUTHORITY" in guide["scope"]
    if arm == "A":
        assert not any(row["operation"] == "ros_engineer.select_staging_options" for row in rows)


def test_empty_reference_phase_has_no_validation_result_or_memory_method():
    schema = decision_schema(
        "C",
        memory_available=False,
        proposal_ids=[],
        verification_refs=[],
        memory_refs=[],
        option_refs=[],
    )
    names = {row["operation"] for row in build_tool_guide(schema)["operations_available_this_turn"]}
    assert "ros_engineer.validate_proposal" not in names
    assert "ros_engineer.explain_result" not in names
    assert "ros_engineer.propose_memory_repair" not in names
    assert "ros_engineer.select_staging_options" not in names


def test_registration_validation_probe_and_finish_have_distinct_effects():
    rows = {
        row["operation"]: row
        for row in build_tool_guide(decision_schema("C", memory_available=True))[
            "operations_available_this_turn"
        ]
    }
    assert rows["ros_engineer.select_staging_options"]["host_effect"] == "REGISTER_STAGING_ONLY"
    assert (
        rows["ros_engineer.validate_proposal"]["host_effect"]
        == "EXPLICIT_HIGH_COST_FRESH_SDK_VALIDATION"
    )
    assert "ZERO_GOAL_REPLICA" in rows["ros_engineer.investigate"]["host_effect"]
    assert rows["finish"]["host_effect"] == "END_REPORT_ONLY"
    assert rows["ros_engineer.select_staging_options"]["argument_fields"] == [
        "case_hash",
        "option_refs",
    ]


def test_guide_supplies_no_body_value_case_label_or_selected_option_answer():
    aci = fixture("tf")
    option = aci.conditional_staging_options()["options"][0]
    guide = json.dumps(
        build_tool_guide(
            decision_schema("B", case_hash=aci.case.case_hash, option_refs=[option.option_ref])
        )
    )
    assert aci.case.case_hash not in guide and option.option_ref not in guide
    assert "enable_localization_tf" not in guide and "radius_m" not in guide
    assert "PRIVATE_GOLD_LABEL" not in guide


def test_runner_sends_guide_and_finish_still_does_not_start_validation(tmp_path):
    aci = fixture()
    prompts = []
    result = run_typed_episode(
        TASK,
        aci,
        tmp_path / "episode",
        model_client=client_factory([{"operation": "finish", "arguments": {}}], prompts),
        tool_budget=1,
    )
    assert "HOST_OPERATION_GUIDE" in prompts[0]
    assert "REGISTER_STAGING_ONLY" in prompts[0]
    assert "EXPLICIT_HIGH_COST_FRESH_SDK_VALIDATION" not in prompts[0]
    assert aci.validation_count == 0 and result["verified_success"] is False
