"""Synthetic driver-only application contracts; no ROS/World/model evidence."""

import copy
import hashlib
import json

import pytest
import yaml

from benchmarks.ros_expert_ablation.physical_aci import PhysicalACI
from benchmarks.ros_expert_ablation.rsk_application_aci import ApplicationPhysicalACI
from benchmarks.ros_expert_ablation.rsk_tool_guide import build_rsk_tool_guide
from benchmarks.ros_expert_ablation.task_tools import TaskTools
from rosclaw.connectors.ros.engineering import compile_engineering_case
from rosclaw.connectors.ros.engineering.application_bootstrap import compile_driver_bootstrap
from rosclaw.contracts.common import content_hash
from tests.benchmarks.test_rsk_pilot_protocol import evidence, protocol
from tests.connectors.ros.test_application_bootstrap import inputs
from tests.connectors.ros.test_navigation_body_workspace import NOW


def fixture(tmp_path, *, namespace="/anonymous/base", validator=None, gates=False, fault=None):
    graph, model, raw, bringup, template, manifests, request = inputs(namespace)
    compiled = compile_driver_bootstrap(graph, model, raw, bringup, template, manifests, request)
    requirements = compiled["requirements"]
    if fault is not None:
        from rosclaw.connectors.ros.intelligence.operational_graph import compile_operational_graph

        if fault == "missing_driver":
            model.graph["topics"] = [
                t for t in model.graph["topics"] if t["name"] != request["control"]["command_topic"]
            ]
        elif fault == "missing_component":
            model.observations["package_inventory"].remove("nav2_amcl")
        else:
            raise ValueError("unsupported synthetic fault")
        model.seal()
        graph = compile_operational_graph(model, source_ref="fixture:changed-original", now=NOW)

    original = {
        "nav2.yaml": yaml.safe_dump({namespace: template}),
        "bringup.json": json.dumps(bringup),
        "requirements.json": json.dumps(requirements),
        "navigation-policy.json": json.dumps(
            {
                "schema_version": "rosclaw.sdk_navigation_policy_input.v1",
                "scope": "OPERATOR_OWNED_DISPOSABLE_SIM_NEW_BODY",
                "policy_id": "original-policy",
                "body_id": request["body_id"],
                "source_urdf_sha256": hashlib.sha256(raw).hexdigest(),
                "navigation_requirements_hash": content_hash("navrequirements", requirements),
                "navigation_limits": request["navigation_limits"],
            }
        ),
    }
    case = compile_engineering_case(
        model,
        raw,
        requirements,
        source_ref="fixture:driver-only",
        now=NOW,
        evidence_domain="SEALED_REPLAY",
    )
    files = {**original, "robot.urdf": raw.decode(), "native.json": "{}"}
    tools = TaskTools(
        files, arm="B", reviewed_at=NOW.isoformat(), minimum_radius_m=compiled["physical_radius_m"]
    )
    base = PhysicalACI(
        case,
        model,
        tools,
        max_validations=1,
        operator_source="b" * 40 if validator else None,
        original_run_id="a" * 32 if validator else None,
    )
    data = {
        "graph": graph.model_dump(mode="json"),
        "driver_bringup": bringup,
        "software_template": template,
        "component_manifests": manifests,
        "request": request,
        "original_files": original,
    }
    options = {}
    if gates:
        p = protocol().model_copy(update={"harness_source": "b" * 40})
        options = {
            "pilot_protocol": p,
            "pilot_evidence": evidence(tmp_path, p),
            "evidence_root": tmp_path,
        }
    aci = ApplicationPhysicalACI(
        base, application_inputs=data, application_validator=validator, arm="B", **options
    )
    return aci, compiled, data


def call(aci, operation="propose_application_bootstrap", **args):
    return aci.call(
        {
            "operation": "ros_engineer." + operation,
            "arguments": {"case_hash": aci.base.case.case_hash, **args},
        },
        remaining_sec=10,
    )


@pytest.mark.parametrize("namespace", ["/anonymous/base", "/renamed/current_body"])
def test_current_driver_only_observations_compile_exact_core_files_without_nav2_or_authority(
    tmp_path, namespace
):
    aci, expected, _ = fixture(tmp_path, namespace=namespace)
    baseline = copy.deepcopy(aci.base.tools.staging)
    assert not aci.base.model.graph["actions"] and not aci.base.model.lifecycle
    result = call(aci)
    assert result["status"] == "CONDITIONAL_APPLICATION_STAGING_PROPOSAL_NOT_EXECUTED"
    p = result["proposal"]
    assert p["files"] == expected["files"] and p["file_sha256"] == expected["file_sha256"]
    assert p["current_authority"] == "NONE" and p["admission"] == "NOT_ADMITTED"
    assert p["interface_status"] == "PLANNED_NOT_OBSERVED_OR_ADMITTED"
    assert aci.base.tools.staging == baseline and aci.base.validation_count == 0
    assert p["rollback"] and all(
        c["operation"] == "REPLACE_WHOLE_ORIGINAL_FILE" for c in p["changes"]
    )
    assert p["proposal_id"] in aci.base.proposals


def test_application_validation_without_operator_callback_is_rejected_without_consuming_budget(
    tmp_path,
):
    aci, _, _ = fixture(tmp_path)
    p = call(aci)["proposal"]
    result = call(aci, "validate_proposal", proposal_id=p["proposal_id"])
    assert (
        result["status"] == "REJECTED_TYPED_APPLICATION_REQUEST"
        and "unavailable" in result["error"]
    )
    assert aci.base.validation_count == 0 and not aci.base.results


def test_actual_sdk_callback_requires_original_pilot_gate_before_constructor_admits_it(tmp_path):
    with pytest.raises(ValueError, match="pilot entry"):
        fixture(tmp_path, validator=lambda *a, **kw: pytest.fail("not called"))


def test_typed_model_envelope_and_operation_guide_support_application_call(tmp_path):
    aci, _, _ = fixture(tmp_path)
    schema = aci.decision_schema()
    guide = build_rsk_tool_guide(schema)
    assert any(
        r["operation"] == "ros_engineer.propose_application_bootstrap"
        for r in guide["operations_available_this_turn"]
    )
    result = aci.call_model_decision(
        {
            "decision": {
                "operation": "ros_engineer.propose_application_bootstrap",
                "arguments": {"case_hash": aci.base.case.case_hash},
            },
            "decision_summary": "compile current original driver application",
            "memory_refs_used": [],
        }
    )
    assert result["status"] == "CONDITIONAL_APPLICATION_STAGING_PROPOSAL_NOT_EXECUTED"


def test_wrong_case_or_forged_original_input_bundle_cannot_register_proposal(tmp_path):
    aci, _, _ = fixture(tmp_path)
    request = {
        "operation": "ros_engineer.propose_application_bootstrap",
        "arguments": {"case_hash": "forged"},
    }
    assert (
        aci.call(request)["status"] == "REJECTED_TYPED_APPLICATION_REQUEST"
        and not aci.base.proposals
    )
    aci._inputs_json = aci._inputs_json.replace("original-policy", "changed-policy")
    assert call(aci)["status"] == "REJECTED_TYPED_APPLICATION_REQUEST" and not aci.base.proposals


@pytest.mark.parametrize(
    "fault,expected",
    [("missing_driver", "NEEDS_DRIVER"), ("missing_component", "NEEDS_DEPENDENCIES")],
)
def test_honestly_missing_prerequisites_retain_explicit_abstention_without_partial_files(
    tmp_path, fault, expected
):
    aci, _, _ = fixture(tmp_path, fault=fault)
    result = call(aci)
    assert result["status"] == expected and result["proposal_registered"] is False
    assert result["application_plan"]["files"] is None and not aci.base.proposals


def test_source_case_original_files_cannot_change_after_binding(tmp_path):
    aci, _, _ = fixture(tmp_path)
    aci.base.tools.files["robot.urdf"] += " "
    assert call(aci)["status"] == "REJECTED_TYPED_APPLICATION_REQUEST" and not aci.base.proposals


def test_body_limits_in_host_request_cannot_replace_original_policy(tmp_path):
    aci, _, _ = fixture(tmp_path)
    data = json.loads(aci._inputs_json)
    data["request"]["navigation_limits"]["max_speed_mps"] *= 2
    aci._inputs_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
    aci._inputs_hash = content_hash("rskappinputs", data)
    assert call(aci)["status"] == "REJECTED_TYPED_APPLICATION_REQUEST" and not aci.base.proposals


def test_shared_application_validation_budget_and_incomplete_callback_evidence_do_not_promote(
    tmp_path,
):
    calls = []

    def validator(files, interfaces, index, **options):
        calls.append((files, interfaces, index))
        return {"verified_success": True, "verification_source": "INDEPENDENT_ORIGINAL_SDK_REVIEW"}

    aci, expected, _ = fixture(tmp_path, validator=validator, gates=True)
    p = call(aci)["proposal"]
    result = call(aci, "validate_proposal", proposal_id=p["proposal_id"])
    assert result["status"] == "EXTERNAL_SDK_APPLICATION_VERIFICATION_RECORDED"
    assert (
        not result["verification"]["accepted_success"]
        and not result["verification"]["evidence_complete"]
    )
    assert len(calls) == 1 and calls[0][0] == expected["files"] and calls[0][2] == 0
    assert aci.base.validation_count == 1
    assert (
        call(aci, "validate_proposal", proposal_id=p["proposal_id"])["status"]
        == "REJECTED_TYPED_APPLICATION_REQUEST"
    )
    assert len(calls) == 1


def test_tampered_registered_application_cannot_reach_existing_sdk_callback(tmp_path):
    aci, _, _ = fixture(tmp_path, validator=lambda *a, **kw: pytest.fail("not called"), gates=True)
    p = call(aci)["proposal"]
    aci.base.proposals[p["proposal_id"]]["files"]["bringup.json"] = "{}"
    result = call(aci, "validate_proposal", proposal_id=p["proposal_id"])
    assert (
        result["status"] == "REJECTED_TYPED_APPLICATION_REQUEST" and aci.base.validation_count == 0
    )


def complete_result(run_id="c" * 32, source="b" * 40):
    ref = {"path": "fixture://synthetic-original", "sha256": "f" * 64}
    return {
        "verified_success": True,
        "verification_source": "INDEPENDENT_ORIGINAL_SDK_REVIEW",
        "source": source,
        "new_world_run": run_id,
        "fresh_l1_ref": ref,
        "independent_stop_ref": ref,
        "strict_source_envelope": ref,
        "canonical_receipts": [ref, ref],
    }


@pytest.mark.parametrize("fault", ["old_run", "probe_run", "wrong_source"])
def test_application_result_identity_rejects_original_probe_or_other_source(tmp_path, fault):
    result = complete_result(
        run_id="a" * 32 if fault == "old_run" else "c" * 32,
        source="d" * 40 if fault == "wrong_source" else "b" * 40,
    )
    aci, _, _ = fixture(tmp_path, validator=lambda *a, **kw: result, gates=True)
    if fault == "probe_run":
        aci.base.probe_results = [{"fresh_probe_performed": True, "new_world_run": "c" * 32}]
    proposal = call(aci)["proposal"]
    reviewed = call(aci, "validate_proposal", proposal_id=proposal["proposal_id"])
    assert not reviewed["verification"]["accepted_success"]
    assert aci.base.validation_count == 1


def test_callback_exception_is_recorded_once_and_cannot_retry_world(tmp_path):
    calls = []

    def fail(*args, **options):
        calls.append(1)
        raise TimeoutError("synthetic callback failure, no actual World")

    aci, _, _ = fixture(tmp_path, validator=fail, gates=True)
    proposal = call(aci)["proposal"]
    result = call(aci, "validate_proposal", proposal_id=proposal["proposal_id"])
    assert result["status"] == "REJECTED_TYPED_APPLICATION_REQUEST"
    assert "without retry" in result["error"] and aci.base.validation_count == 1
    assert (
        call(aci, "validate_proposal", proposal_id=proposal["proposal_id"])["status"]
        == "REJECTED_TYPED_APPLICATION_REQUEST"
    )
    assert len(calls) == 1


def test_registered_model_loop_can_select_current_application_and_finish_without_sdk(tmp_path):
    from benchmarks.ros_expert_ablation.rsk_episode_runner import (
        EpisodeFactsV1,
        FrozenProviderV1,
        run_rsk_typed_episode,
    )
    from benchmarks.ros_expert_ablation.rsk_pilot_protocol import allocate_pilot
    from tests.benchmarks.test_rsk_episode_runner import put

    aci, expected, _ = fixture(tmp_path)
    task = {
        "task_id": "SYNTHETIC-application-contract",
        "goal": "build proposed navigation application",
    }
    episode = put(tmp_path, "episode.json", {"scope": "SYNTHETIC_CONTRACT_ONLY"})
    refs = {
        key: put(tmp_path, key + ".json", value)
        for key, value in [
            ("case", aci.base.case.model_dump(mode="json")),
            ("model", aci.base.model.to_dict()),
            ("requirements", aci.base.requirements),
            ("staging", aci.base.tools.initial_staging),
            ("task", task),
        ]
    }
    refs["run_settings"] = put(tmp_path, "run.json", {"run_id": "f" * 32})
    facts = put(
        tmp_path,
        "facts.json",
        EpisodeFactsV1(**refs, evaluation_episode_sha256=episode.sha256).model_dump(mode="json"),
    )
    p = protocol()
    provider = put(
        tmp_path,
        "provider.json",
        FrozenProviderV1(
            execution_kind="SYNTHETIC_CONTRACT_CLIENT",
            model_identity=p.model_identity,
            effort="fixture",
            client_code_sha256="0" * 64,
            client_settings_code_sha256="0" * 64,
            executable_sha256="0" * 64,
        ).model_dump(mode="json"),
    )
    cases = [c.model_dump(mode="json") for c in p.cases]
    cases[3].update(facts_manifest_sha256=facts.sha256, evaluation_episode_sha256=episode.sha256)
    policy = put(
        tmp_path,
        "policy.json",
        {
            "schema_version": "rosclaw.rsk_memory_policy_manifest.v1",
            "cases": {c["case_id"]: {} for c in cases},
        },
    )
    p = type(p).model_validate(
        {
            **p.model_dump(mode="json"),
            "cases": cases,
            "provider_settings_sha256": provider.sha256,
            "memory_policy_sha256": policy.sha256,
        }
    )
    ordinal = next(
        row["ordinal"]
        for row in allocate_pilot(p)
        if row["case_id"] == cases[3]["case_id"] and row["arm"] == "B"
    )
    decisions = iter(
        [
            {
                "operation": "ros_engineer.propose_application_bootstrap",
                "arguments": {"case_hash": aci.base.case.case_hash},
            },
            {"operation": "finish", "arguments": {}},
        ]
    )
    prompts = []

    def client(prompt, schema, output, **options):
        prompts.append(prompt)
        output.mkdir()
        payload = {
            "decision": next(decisions),
            "decision_summary": "synthetic proposal only",
            "memory_refs_used": [],
        }
        (output / "original-events.jsonl").write_text(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": json.dumps(payload)},
                }
            )
            + "\n"
        )
        return {
            "status": "ACTUAL_MODEL_TURN_COMPLETED",
            "usage": {"input_tokens": 1},
            "native_tool_items": [],
        }

    result = run_rsk_typed_episode(
        task,
        aci,
        tmp_path / "output",
        protocol=p,
        pilot_evidence=evidence(tmp_path, p),
        evidence_root=tmp_path,
        ordinal=ordinal,
        facts_manifest=facts,
        evaluation_episode=episode,
        provider_settings=provider,
        memory_policy=policy,
        contract_client=client,
    )
    assert result["status"] == "FINISHED_WITHOUT_ACCEPTED_SDK_RESULT"
    assert (
        result["first_valid_proposal"]["decision"] == 1
        and result["actual_model_calls"] == result["new_Worlds_requested"] == 0
    )
    assert result["tool_calls"][0]["result"]["proposal"]["files"] == expected["files"]
    assert "application_onboarding" in prompts[0] and "NOT_DRIVER_ACCEPTANCE" in prompts[0]
