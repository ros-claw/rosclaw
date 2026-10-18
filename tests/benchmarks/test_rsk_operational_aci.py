"""Synthetic graph/probe contracts only; zero SDK/model/physical evidence."""

import copy
import json

import pytest
import yaml

from benchmarks.ros_expert_ablation.physical_aci import PhysicalACI
from benchmarks.ros_expert_ablation.rsk_operational_aci import OperationalRskACI
from benchmarks.ros_expert_ablation.rsk_physical_aci import RskPhysicalACI
from benchmarks.ros_expert_ablation.task_tools import TaskTools
from rosclaw.connectors.ros.engineering import compile_engineering_case
from tests.connectors.ros.test_operational_graph_contracts import NOW, native_fixture


def facade(arm="C"):
    model, urdf, requirements = native_fixture()
    case = compile_engineering_case(
        model,
        urdf,
        requirements,
        source_ref="fixture:unit-original",
        now=NOW,
        evidence_domain="SEALED_REPLAY",
    )
    files = {
        "native.json": "{}",
        "robot.urdf": urdf.decode(),
        "requirements.json": json.dumps(requirements),
        "nav2.yaml": yaml.safe_dump(
            {"/sample": {"amcl": {"ros__parameters": {"tf_broadcast": False}}}}
        ),
        "bringup.json": json.dumps({"namespace": "/sample"}),
    }
    tools = TaskTools(files, arm="B", reviewed_at=NOW.isoformat(), minimum_radius_m=0.5)
    base = PhysicalACI(case, model, tools, max_validations=1)
    return OperationalRskACI(RskPhysicalACI(base, arm=arm))


def req(aci, operation, **args):
    return {
        "operation": "ros_engineer." + operation,
        "arguments": {"case_hash": aci.base.case.case_hash, **args},
    }


def qos_id(aci):
    return next(
        f.id
        for f in aci._graph().findings
        if f.invariant == "qos.observed_requested_offered_policies" and f.status == "VIOLATED"
    )


def plan(aci):
    return aci.call(req(aci, "plan_diagnostic_experiment", issue_ids=[qos_id(aci)]))


def test_operational_qos_finding_and_plan_go_beyond_five_mechanism_checks():
    aci = facade()
    key = qos_id(aci)
    assert key not in {c.id for c in aci.base.case.constraints}
    result = plan(aci)
    assert result["status"] == "DIAGNOSTIC_PLAN_PROPOSED_NOT_EXECUTED"
    hypotheses = result["experiment"]["hypotheses"]
    assert any(
        h["predicate"] == "QOS_CONFLICT" and h["status"] == "SUPPORTED_OBSERVATION"
        for h in hypotheses
    )
    assert any(h["predicate"] == "UNRESOLVED_PATH" and h["status"] == "UNKNOWN" for h in hypotheses)
    assert not aci.base.proposals and aci.base.probe_count == aci.base.validation_count == 0
    assert result["experiment"]["current_l1"] == "NOT_EVALUATED"
    assert result["experiment"]["execution_authority"] == "NONE"


def test_graph_pages_are_complete_by_cursor_and_explicitly_bound_to_source():
    aci = facade()
    seen, cursor = [], 0
    while cursor is not None:
        result = aci.call(req(aci, "inspect_operational_graph", cursor=cursor, limit=2))
        seen.extend(f["id"] for f in result["findings"])
        assert result["graph_hash"] == aci._graph().graph_hash
        assert result["current_authority"] == "NONE"
        cursor = result["next_cursor"]
    assert seen == [f.id for f in aci._graph().findings]


def test_b_can_plan_but_cannot_execute_probe_even_with_direct_host_request():
    aci = facade("B")
    result = plan(aci)["experiment"]
    assert (
        "ros_engineer.run_diagnostic_probe" not in aci.request_schema()["discriminator"]["mapping"]
    )
    rejected = aci.call(
        req(
            aci,
            "run_diagnostic_probe",
            experiment_hash=result["experiment_hash"],
            probe_id=result["probes"][0]["request"]["id"],
        ),
        remaining_sec=60,
    )
    assert rejected["status"] == "REJECTED_OPERATIONAL_DIAGNOSTIC_REQUEST"
    assert aci.base.probe_count == 0


def test_missing_provider_keeps_plan_unexecuted_without_consuming_budget():
    aci = facade()
    experiment = plan(aci)["experiment"]
    result = aci.call(
        req(
            aci,
            "run_diagnostic_probe",
            experiment_hash=experiment["experiment_hash"],
            probe_id=experiment["probes"][0]["request"]["id"],
        ),
        remaining_sec=60,
    )
    assert result["status"] == "NEEDS_OPERATOR_OWNED_DIAGNOSTIC_PROVIDER"
    assert not result["probe_performed"] and aci.base.probe_count == 0
    assert not aci.base.results


@pytest.mark.parametrize(
    "args",
    [
        {"issue_ids": ["invented"]},
        {"issue_ids": []},
        {"issue_ids": ["F-TF-001", "F-TF-001"]},
        {"issue_ids": ["F-TF-001"], "max_probes": 9},
    ],
)
def test_invented_duplicate_or_over_budget_scope_never_registers(args):
    aci = facade()
    result = aci.call(req(aci, "plan_diagnostic_experiment", **args))
    assert result["status"] == "REJECTED_OPERATIONAL_DIAGNOSTIC_REQUEST"
    assert not aci.diagnostic_plans and aci.base.probe_count == 0


def test_tampered_graph_is_rejected_before_observation_or_plan():
    aci = facade()
    graph = json.loads(aci._graph_json)
    graph["findings"][0]["status"] = (
        "SATISFIED" if graph["findings"][0]["status"] != "SATISFIED" else "UNKNOWN"
    )
    aci._graph_json = json.dumps(graph)
    result = aci.call(req(aci, "inspect_operational_graph"))
    assert result["status"] == "REJECTED_OPERATIONAL_DIAGNOSTIC_REQUEST"
    assert not aci.diagnostic_plans


def test_probe_plan_hash_and_original_predictions_cannot_be_rewritten():
    aci = facade()
    original = plan(aci)["experiment"]
    changed = copy.deepcopy(original)
    changed["probes"][0]["predictions"][0]["if_supported"] = "invented"
    aci.diagnostic_plans[original["experiment_hash"]] = changed
    result = aci.call(
        req(
            aci,
            "run_diagnostic_probe",
            experiment_hash=original["experiment_hash"],
            probe_id=original["probes"][0]["request"]["id"],
        ),
        remaining_sec=60,
    )
    assert result["status"] == "REJECTED_OPERATIONAL_DIAGNOSTIC_REQUEST"
    assert aci.base.probe_count == 0


def test_provider_schema_nullable_defaults_follow_the_same_host_contract():
    aci = facade()
    value = {
        "decision": req(aci, "inspect_operational_graph", cursor=None, limit=None),
        "decision_summary": "synthetic",
        "memory_refs_used": [],
    }
    result = aci.call_model_decision(value)
    assert (
        result["cursor"] == 0 and result["status"] == "SEALED_OPERATIONAL_GRAPH_PAGE_NOT_ADMISSION"
    )


def test_graph_original_input_binding_contains_no_motion_permission():
    aci = facade()
    original = aci.registered_diagnostic_inputs()
    assert original["graph"]["snapshot_hash"] == aci.base.case.snapshot_hash
    assert original["provider_policy"] is None
    assert original["graph"]["current_authority"] == "NONE"
    assert not aci.base.finished and not aci.base.proposals


def provider_fixture(tmp_path, aci):
    import hashlib

    from benchmarks.ros_expert_ablation.rsk_diagnostic_provider import (
        DiagnosticProbePolicyV1,
        DiagnosticProbeProvider,
    )
    from benchmarks.ros_expert_ablation.rsk_episode_runner import OriginalRef
    from benchmarks.ros_expert_ablation.rsk_native_diagnostic_operator import (
        SensorOnlyDiagnosticOperator,
    )
    from rosclaw.contracts.common import content_hash
    from tests.benchmarks.test_rsk_pilot_protocol import evidence, protocol

    config = {
        "schema_version": "rosclaw.rsk_sensor_observation_config.v1",
        "namespace": "/rsk_fixture_contract1",
        "topic": "/rsk_fixture_contract1/sensor/measurement",
        "offered_reliability": "BEST_EFFORT",
        "requested_reliability": "RELIABLE",
        "producer_schedule": "CONTINUOUS",
    }
    path = tmp_path / "private-sensor-config.json"
    path.write_text(json.dumps(config))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    aci.base.operator_source = "b" * 40
    aci.base.original_run_id = "a" * 32
    policy = DiagnosticProbePolicyV1(
        source_revision="b" * 40,
        original_run_id="a" * 32,
        original_snapshot_hash=aci.base.case.snapshot_hash,
        source_urdf_sha256=aci.base.case.source_urdf_sha256,
        requirements_hash=content_hash("navrequirements", aci.base.requirements),
        case_staging_hash=content_hash("acistaging", aci.base.tools.initial_staging),
        execution_config_sha256=digest,
        operations=("ENDPOINT_QOS",),
        minimum_wall_sec=12,
    )
    operator = SensorOnlyDiagnosticOperator(
        policy,
        worktree=tmp_path,
        evidence_root=tmp_path,
        execution_config=OriginalRef(path=path.name, sha256=digest),
        output_parent=tmp_path,
    )
    provider = DiagnosticProbeProvider(
        policy, operator=operator, worktree=tmp_path, evidence_root=tmp_path
    )
    p = protocol().model_copy(update={"harness_source": "b" * 40})
    return provider, {
        "pilot_protocol": p,
        "pilot_evidence": evidence(tmp_path, p),
        "evidence_root": tmp_path,
    }


def test_real_provider_binding_requires_original_four_entry_gates(tmp_path):
    aci = facade()
    provider, _ = provider_fixture(tmp_path, aci)
    with pytest.raises(ValueError, match="pilot entry"):
        OperationalRskACI(aci.delegate, diagnostic_probe_provider=provider)
    assert aci.base.probe_evaluator is None and aci.base.probe_count == 0


def test_wrong_provider_arm_or_frozen_source_cannot_bind(tmp_path):
    aci = facade("B")
    provider, args = provider_fixture(tmp_path, aci)
    with pytest.raises(ValueError, match="C/E"):
        OperationalRskACI(aci.delegate, diagnostic_probe_provider=provider, **args)


def test_single_probe_slot_is_shared_and_insufficient_budget_does_not_consume_it(
    tmp_path, monkeypatch
):
    original = facade()
    provider, args = provider_fixture(tmp_path, original)
    aci = OperationalRskACI(original.delegate, diagnostic_probe_provider=provider, **args)
    calls = []
    monkeypatch.setattr(provider, "check_context", lambda base: None)
    monkeypatch.setattr(
        provider,
        "run_and_review",
        lambda *a, **kw: (
            calls.append(True)
            or {"status": "SYNTHETIC_PROVIDER_CONTRACT_NO_SDK", "new_world_run": "c" * 32}
        ),
    )
    experiment = plan(aci)["experiment"]
    selected = next(p for p in experiment["probes"] if p["request"]["operation"] == "ENDPOINT_QOS")
    request = req(
        aci,
        "run_diagnostic_probe",
        experiment_hash=experiment["experiment_hash"],
        probe_id=selected["request"]["id"],
    )
    low = aci.call(request, remaining_sec=5)
    assert low["status"].startswith("REJECTED") and not calls and aci.base.probe_count == 0
    result = aci.call(request, remaining_sec=60)
    assert result["status"] == "SYNTHETIC_PROVIDER_CONTRACT_NO_SDK"
    assert len(calls) == aci.base.probe_count == 1 and len(aci.base.probe_results) == 1
    assert aci.call(request, remaining_sec=60)["status"].startswith("REJECTED") and len(calls) == 1


def test_an_existing_legacy_probe_consumes_the_same_operational_budget(tmp_path, monkeypatch):
    original = facade()
    provider, args = provider_fixture(tmp_path, original)
    aci = OperationalRskACI(original.delegate, diagnostic_probe_provider=provider, **args)
    aci.base.probe_count = 1
    called = []
    monkeypatch.setattr(provider, "run_and_review", lambda *a, **kw: called.append(True))
    experiment = plan(aci)["experiment"]
    result = aci.call(
        req(
            aci,
            "run_diagnostic_probe",
            experiment_hash=experiment["experiment_hash"],
            probe_id=experiment["probes"][0]["request"]["id"],
        ),
        remaining_sec=60,
    )
    assert result["status"].startswith("REJECTED") and not called


def test_operational_guide_advertises_proposals_without_hiding_execution_costs():
    from benchmarks.ros_expert_ablation.rsk_tool_guide import build_rsk_tool_guide

    aci = facade()
    guide = build_rsk_tool_guide(aci.decision_schema())
    operations = {row["operation"]: row for row in guide["operations_available_this_turn"]}
    assert (
        operations["ros_engineer.plan_diagnostic_experiment"]["host_effect"]
        == "REGISTER_READ_ONLY_DIAGNOSTIC_PLAN_NOT_EXECUTED"
    )


def test_callback_failure_consumes_one_slot_and_persists_without_retry(tmp_path, monkeypatch):
    original = facade()
    provider, args = provider_fixture(tmp_path, original)
    aci = OperationalRskACI(original.delegate, diagnostic_probe_provider=provider, **args)
    calls = []
    monkeypatch.setattr(provider, "check_context", lambda base: None)

    def broken(*a, **kw):
        calls.append(True)
        raise RuntimeError("SYNTHETIC SDK callback exception")

    monkeypatch.setattr(provider.operator, "collect", broken)
    experiment = plan(aci)["experiment"]
    selected = next(p for p in experiment["probes"] if p["request"]["operation"] == "ENDPOINT_QOS")
    request = req(
        aci,
        "run_diagnostic_probe",
        experiment_hash=experiment["experiment_hash"],
        probe_id=selected["request"]["id"],
    )
    result = aci.call(request, remaining_sec=60)
    assert result["status"] == "DIAGNOSTIC_SDK_FAILED_OR_INCOMPLETE_NO_RETRY"
    assert result["error_type"] == "RuntimeError" and result["original_manifest_ref"] is None
    assert len(calls) == aci.base.probe_count == len(provider.trace) == 1
    assert aci.call(request, remaining_sec=60)["status"].startswith("REJECTED") and len(calls) == 1
    assert not aci.base.results


def test_diagnostic_only_provider_does_not_charge_legacy_probe_request(tmp_path):
    original = facade()
    provider, args = provider_fixture(tmp_path, original)
    aci = OperationalRskACI(original.delegate, diagnostic_probe_provider=provider, **args)
    assert "ros_engineer.investigate" not in aci.request_schema()["discriminator"]["mapping"]
    result = aci.call(
        req(aci, "investigate", issue_id="F-TF-001", check_id="inspect_transform_authorities"),
        remaining_sec=60,
    )
    assert result["status"] == "NEEDS_LEGACY_READ_ONLY_PROVIDER_NOT_CONSUMED"
    assert aci.base.probe_count == provider.operator.calls == 0


def test_model_loop_can_register_graph_plan_without_faking_a_repair_or_sdk_result(tmp_path):
    from tests.benchmarks.test_rsk_episode_runner import bundle, decision, run

    delegate, _, kwargs = bundle(tmp_path, arm="B")
    aci = OperationalRskACI(delegate)
    result, prompts = run(
        tmp_path,
        aci,
        kwargs,
        [
            decision(
                aci,
                "plan_diagnostic_experiment",
                issue_ids=["F-TF-001"],
                budget_sec=20,
                max_probes=1,
            ),
            decision(aci, "finish"),
        ],
    )
    assert result["status"] == "FINISHED_WITHOUT_ACCEPTED_SDK_RESULT" and len(prompts) == 2
    assert "TF_TIME_CONFLICT" in prompts[1] and "experiment_hash" in prompts[1]
    assert result["first_valid_proposal"] is None and not aci.base.proposals
    assert result["actual_model_calls"] == result["new_Worlds_requested"] == 0
    assert result["tool_calls"][0]["result"]["experiment"]["hypotheses"]
    assert result["physical_validation_phase"] == "NOT_REQUESTED"


def test_synthetic_model_cannot_acquire_or_run_closed_diagnostic_provider(tmp_path):
    from tests.benchmarks.test_rsk_episode_runner import bundle, refreeze, run

    delegate, _, kwargs = bundle(tmp_path, arm="C")
    temporary = OperationalRskACI(delegate)
    provider, _ = provider_fixture(tmp_path, temporary)
    refreeze(tmp_path, kwargs, harness_source="b" * 40)
    aci = OperationalRskACI(
        delegate,
        diagnostic_probe_provider=provider,
        pilot_protocol=kwargs["protocol"],
        pilot_evidence=kwargs["pilot_evidence"],
        evidence_root=tmp_path,
    )
    result, prompts = run(tmp_path, aci, kwargs, [])
    assert result["status"] == "NO_GO" and "synthetic" in result["error"]
    assert not prompts and aci.base.probe_count == provider.operator.calls == 0


def test_corrupt_registered_diagnostic_input_is_rejected_before_model_or_tools(tmp_path):
    from tests.benchmarks.test_rsk_episode_runner import bundle, put, refreeze, run

    delegate, _, kwargs = bundle(tmp_path, arm="B")
    aci = OperationalRskACI(delegate)
    reference = put(tmp_path, "diagnostic-inputs.json", aci.registered_diagnostic_inputs())
    facts = json.loads((tmp_path / "facts-manifest.json").read_text())
    facts["diagnostic_inputs"] = reference.model_dump(mode="json")
    manifest = put(tmp_path, "diagnostic-facts.json", facts)
    kwargs["facts_manifest"] = manifest
    cases = [c.model_dump(mode="json") for c in kwargs["protocol"].cases]
    cases[0]["facts_manifest_sha256"] = manifest.sha256
    refreeze(tmp_path, kwargs, cases=cases)
    (tmp_path / "diagnostic-inputs.json").write_text("{}")
    result, prompts = run(tmp_path, aci, kwargs, [])
    assert result["status"] == "NO_GO" and not prompts and not aci.trace


def test_native_only_context_does_not_invent_urdf_or_navigation_bindings():
    from benchmarks.ros_expert_ablation.rsk_diagnostic_provider import (
        DiagnosticProbePolicyV1,
        NativeDiagnosticContext,
    )

    model, _, _ = native_fixture()
    context = NativeDiagnosticContext(model, "b" * 40, "a" * 32)
    assert context.checked_model().snapshot_hash == model.snapshot_hash
    assert not hasattr(context, "case") and not hasattr(context, "tools")
    policy = DiagnosticProbePolicyV1(
        source_revision="b" * 40,
        original_run_id="a" * 32,
        original_snapshot_hash=model.snapshot_hash,
        source_urdf_sha256=None,
        requirements_hash=None,
        case_staging_hash=None,
        execution_config_sha256="c" * 64,
        operations=("ENDPOINT_QOS",),
        minimum_wall_sec=12,
    )
    assert policy.source_urdf_sha256 is policy.requirements_hash is policy.case_staging_hash is None
    with pytest.raises(ValueError, match="partial invented navigation"):
        type(policy).model_validate(
            {**policy.model_dump(), "requirements_hash": "invented-requirements"}
        )
