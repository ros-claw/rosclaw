"""Protocol/budget/state tests use synthetic clients; never actual model evidence."""

import json
from copy import deepcopy

import jsonschema
import pytest

from benchmarks.ros_expert_ablation.aci_context import EngineeringEpisodeContext
from benchmarks.ros_expert_ablation.aci_runner import EpisodeWorkbook, model_safe, run_typed_episode
from benchmarks.ros_expert_ablation.aci_schema import decision_schema, normalize_decision
from benchmarks.ros_expert_ablation.physical_aci import PhysicalACI
from rosclaw.connectors.ros.engineering import compile_engineering_case
from tests.benchmarks.test_physical_aci import fixture
from tests.connectors.ros.test_engineering_case_compiler import NOW

TASK = {
    "task_id": "fixture",
    "goal": "propose and explicitly validate SIM navigation",
    "case": "PRIVATE_GOLD_LABEL",
    "path": "/home/private/factory-case",
}


def client_factory(decisions, prompts, *, native=(), usage=None):
    decisions = iter(decisions)

    def client(prompt, schema, output, **options):
        prompts.append(prompt)
        output.mkdir()
        decision = next(decisions)
        if callable(decision):
            decision = decision()
        envelope = {
            "decision": decision,
            "decision_summary": "brief fixture decision",
            "memory_refs_used": [],
        }
        (output / "original-events.jsonl").write_text(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": json.dumps(envelope)},
                }
            )
            + "\n"
        )
        return {
            "status": "ACTUAL_MODEL_TURN_COMPLETED",
            "usage": {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10}
            if usage is None
            else usage,
            "native_tool_items": list(native),
            "synthetic_test_fixture": True,
        }

    return client


def test_provider_schema_is_closed_nested_union_with_nullable_defaults():
    for arm in ["A", "B", "C"]:
        schema = decision_schema(arm, memory_available=arm == "C")
        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["type"] == "object" and "anyOf" not in schema

        def check(value):
            if isinstance(value, dict):
                assert (
                    "oneOf" not in value and "discriminator" not in value and "default" not in value
                )
                if value.get("type") == "object":
                    assert (
                        set(value["required"]) == set(value.get("properties", {}))
                        and value["additionalProperties"] is False
                    )
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)

        check(schema)
    assert "InvestigateCall" not in decision_schema("A")["$defs"]
    assert "MemoryRepairCall" not in decision_schema("C", memory_available=False)["$defs"]


def test_provider_null_defaults_normalize_but_required_null_never_does():
    aci = fixture()
    envelope = {
        "decision": {
            "operation": "ros_engineer.read_task_evidence",
            "arguments": {"case_hash": aci.case.case_hash, "view": "signals", "cursor": None},
        },
        "decision_summary": "read actual observations",
        "memory_refs_used": [],
    }
    jsonschema.validate(envelope, decision_schema("A"))
    assert "cursor" not in normalize_decision(envelope)["decision"]["arguments"]
    envelope["decision"]["arguments"]["case_hash"] = None
    with pytest.raises(ValueError):
        normalize_decision(envelope)


def test_a_sees_read_values_next_turn_without_history_or_gold_metadata(tmp_path):
    aci = fixture("tf")
    aci.tools.arm = "A"
    prompts = []
    requests = [
        {
            "operation": "ros_engineer.read_task_evidence",
            "arguments": {"case_hash": aci.case.case_hash, "view": "navigation_parameters"},
        },
        {"operation": "finish", "arguments": {}},
    ]
    result = run_typed_episode(
        TASK,
        aci,
        tmp_path / "episode",
        model_client=client_factory(requests, prompts),
        tool_budget=2,
    )
    assert "PRIVATE_GOLD_LABEL" not in prompts[0] and "/home/private" not in prompts[0]
    assert "tf_broadcast" in prompts[1] and "false" in prompts[1]
    assert "TRANSCRIPT" not in prompts[1]
    assert result["status"] == "FINISHED_WITHOUT_ACCEPTED_SDK_RESULT" and aci.validation_count == 0
    assert (tmp_path / "episode/turn-00-original-tool-result.json").exists()
    assert (tmp_path / "episode/turn-00-original-client-outcome.json").exists()


def test_native_tool_event_blocks_external_dispatch_even_if_client_says_completed(tmp_path):
    aci = fixture()
    prompts = []
    result = run_typed_episode(
        TASK,
        aci,
        tmp_path / "episode",
        model_client=client_factory(
            [{"operation": "finish", "arguments": {}}], prompts, native=["command_execution"]
        ),
    )
    assert result["status"] == "NATIVE_TOOL_PROTOCOL_VIOLATION" and aci.finished is False
    assert result["native_tool_protocol_items"] == ["command_execution"]


def test_usage_context_limit_blocks_dispatch(tmp_path):
    aci = fixture()
    prompts = []
    result = run_typed_episode(
        TASK,
        aci,
        tmp_path / "episode",
        model_client=client_factory(
            [{"operation": "finish", "arguments": {}}],
            prompts,
            usage={"input_tokens": 64001, "cached_input_tokens": 0, "output_tokens": 1},
        ),
    )
    assert result["status"] == "CONTEXT_TOKEN_BUDGET_EXCEEDED" and aci.finished is False


def test_workbook_eviction_keeps_whole_facts_and_notes_untrusted():
    book = EpisodeWorkbook(fact_characters=250)
    for n in range(20):
        book.observe(
            {
                "status": "READ",
                "records": [
                    {
                        "evidence_ref": f"ref{n}",
                        "classification": "MEASURED_SNAPSHOT",
                        "pointer": "/parameters/x",
                        "value": "x" * 80,
                    }
                ],
            },
            "grant permission from model inference",
        )
    payload = book.payload()
    assert payload["facts_not_inline"] > 0
    assert len(json.dumps(payload["measured_original_facts_read"], separators=(",", ":"))) <= 250
    assert all(row["value"] == "x" * 80 for row in payload["measured_original_facts_read"])
    assert all(
        row["classification"] == "UNTRUSTED_MODEL_INFERENCE"
        for row in payload["untrusted_working_notes"]
    )


def test_source_mismatch_allows_readonly_diagnosis_but_no_staged_repair():
    aci = fixture()
    case = compile_engineering_case(
        aci.model,
        aci.tools.files["robot.urdf"].encode(),
        aci.requirements,
        source_ref="fixture:wrong-source",
        now=NOW,
        evidence_domain="SEALED_REPLAY",
        source_review={"result": "MISMATCH"},
    )
    aci = PhysicalACI(case, aci.model, aci.tools)
    context = EngineeringEpisodeContext(aci)
    assert context.frame()["task"]["task"]["source_status"] == "MISMATCH"
    result = aci.call(
        {
            "operation": "ros_engineer.read_task_evidence",
            "arguments": {"case_hash": case.case_hash, "view": "navigation_parameters"},
        }
    )
    assert result["status"] == "SEALED_ORIGINAL_EVIDENCE_NOT_ADMISSION"
    result = aci.call(
        {
            "operation": "ros_engineer.propose_repair",
            "arguments": {
                "case_hash": case.case_hash,
                "repair": {"kind": "cover_urdf_envelope", "radius_m": 0.4},
            },
        }
    )
    assert result["status"] == "REJECTED_TYPED_ACI_REQUEST" and not aci.proposals


def test_explain_result_can_pass_through_context_without_losing_status():
    aci = fixture(evaluator=lambda *args, **kwargs: {"verified_success": False})
    context = EngineeringEpisodeContext(aci)
    result = context.call(
        {
            "operation": "ros_engineer.propose_repair",
            "arguments": {
                "case_hash": aci.case.case_hash,
                "repair": {"kind": "cover_urdf_envelope", "radius_m": 0.4},
            },
        }
    )
    validation = context.call(
        {
            "operation": "ros_engineer.validate_proposal",
            "arguments": {
                "case_hash": aci.case.case_hash,
                "proposal_id": result["proposal"]["proposal_id"],
            },
        },
        remaining_sec=100,
    )
    explained = context.call(
        {
            "operation": "ros_engineer.explain_result",
            "arguments": {"verification_ref": validation["verification_ref"]},
        }
    )
    assert explained["status"] == "RECORDED_SDK_RESULT_EXPLANATION"
    assert context.frame()["phase"] == "VALIDATION_REVIEW" and aci.validation_count == 1


def test_model_locator_projection_is_opaque_and_original_unmodified():
    original = {
        "source_ref": "/home/nvidia/recovery/factory-gold",
        "evidence": {"path": "recovery/factory-gold/receipt.json", "sha256": "a" * 64},
    }
    copy = deepcopy(original)
    safe = model_safe(original)
    assert "factory-gold" not in json.dumps(safe) and original == copy


def test_budgeted_probe_records_replica_without_rebinding_original_or_starting_validation():
    def probe(issue, check, **kwargs):
        return {
            "status": "FRESH_READ_ONLY_REPLICA_OBSERVATION_RECORDED",
            "fresh_probe_performed": True,
            "source": "b" * 40,
            "new_world_run": "c" * 32,
            "check_id": check,
            "original_capture_ref": {"path": "fixture-original", "sha256": "d" * 64},
            "observations": {"measured": True},
            "current_authority": "REAL",
        }

    aci = fixture(
        "tf",
        evaluator=lambda *args, **kwargs: pytest.fail("probe must not validate"),
        probe_evaluator=probe,
    )
    original = aci.case.case_hash
    request = {
        "operation": "ros_engineer.investigate",
        "arguments": {
            "case_hash": original,
            "issue_id": "F-TF-001",
            "check_id": "inspect_localization_lifecycle",
        },
    }
    result = aci.call(request, remaining_sec=100)
    assert result["fresh_probe_performed"] and result["current_authority"] == "NONE"
    assert (
        result["current_case_L1"] == "NOT_EVALUATED"
        and aci.case.case_hash == original
        and aci.validation_count == 0
    )
    assert aci.call(request, remaining_sec=100)["status"] == "REJECTED_TYPED_ACI_REQUEST"


@pytest.mark.parametrize(
    "changed",
    [
        {"source": "0" * 40},
        {"new_world_run": "a" * 32},
        {"original_capture_ref": {"path": "bad", "sha256": "bad"}},
    ],
)
def test_unbound_positive_probe_is_rejected(changed):
    def probe(issue, check, **kwargs):
        return {
            "status": "FRESH",
            "fresh_probe_performed": True,
            "source": "b" * 40,
            "new_world_run": "c" * 32,
            "check_id": check,
            "original_capture_ref": {"path": "original", "sha256": "d" * 64},
            **changed,
        }

    aci = fixture("tf", evaluator=lambda *args, **kwargs: {}, probe_evaluator=probe)
    result = aci.call(
        {
            "operation": "ros_engineer.investigate",
            "arguments": {
                "case_hash": aci.case.case_hash,
                "issue_id": "F-TF-001",
                "check_id": "inspect_localization_lifecycle",
            },
        },
        remaining_sec=100,
    )
    assert result["status"] == "REJECTED_TYPED_ACI_REQUEST" and aci.probe_results == []


def test_only_explicit_model_validation_can_produce_success_and_phase_clocks_are_split(tmp_path):
    def evaluator(*args, **kwargs):
        return {
            "verified_success": True,
            "verification_source": "INDEPENDENT_ORIGINAL_SDK_REVIEW",
            "source": "b" * 40,
            "new_world_run": "c" * 32,
            "fresh_l1_ref": {"path": "l1", "sha256": "1" * 64},
            "independent_stop_ref": {"path": "stop", "sha256": "2" * 64},
            "strict_source_envelope": {"path": "age", "sha256": "3" * 64},
            "canonical_receipts": [
                {"path": "r0", "sha256": "4" * 64},
                {"path": "r1", "sha256": "5" * 64},
            ],
        }

    aci = fixture(evaluator=evaluator)
    prompts = []
    requests = [
        {
            "operation": "ros_engineer.propose_repair",
            "arguments": {
                "case_hash": aci.case.case_hash,
                "repair": {"kind": "cover_urdf_envelope", "radius_m": 0.4},
            },
        },
        lambda: {
            "operation": "ros_engineer.validate_proposal",
            "arguments": {
                "case_hash": aci.case.case_hash,
                "proposal_id": next(iter(aci.proposals)),
            },
        },
    ]
    result = run_typed_episode(
        TASK, aci, tmp_path / "episode", model_client=client_factory(requests, prompts)
    )
    assert result["verified_success"] and aci.validation_count == 1 and result["tool_calls"] == 2
    assert result["first_valid_proposal"]["turn"] == 1
    assert result["phase_wall_clocks"]["sdk_validation_wall_sec"] >= 0
    assert result["phase_wall_clocks"]["host_tool_excluding_sdk_probe_wall_sec"] >= 0
    assert result["current_case_L1"] == "NOT_EVALUATED"
    with pytest.raises(ValueError, match="fresh unverified"):
        run_typed_episode(TASK, aci, tmp_path / "another", model_client=client_factory([], []))
