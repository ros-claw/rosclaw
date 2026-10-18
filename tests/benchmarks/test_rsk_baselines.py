"""Synthetic contract/failure tests only: no model, ROS or World evidence."""

import copy
import json

import pytest

from benchmarks.ros_expert_ablation.physical_aci import PhysicalACI
from benchmarks.ros_expert_ablation.rsk_deterministic_baseline import (
    current_deterministic_request,
    run_rsk_deterministic_episode,
)
from benchmarks.ros_expert_ablation.rsk_generic_baseline import GenericRskACI
from benchmarks.ros_expert_ablation.rsk_pilot_protocol import allocate_pilot
from benchmarks.ros_expert_ablation.rsk_registered_inputs import registered_episode_inputs
from benchmarks.ros_expert_ablation.rsk_tool_guide import build_rsk_tool_guide
from benchmarks.ros_expert_ablation.task_tools import TaskTools
from rosclaw.connectors.ros.engineering.repair_compiler import proposal_staging
from tests.benchmarks.test_rsk_application_aci import fixture as application_fixture
from tests.benchmarks.test_rsk_episode_runner import (
    TASK,
    bundle,
    decision,
    put,
    refreeze,
    run,
)


def generic(base, **options):
    tools = TaskTools(
        copy.deepcopy(base.tools.files),
        arm="A",
        reviewed_at=base.tools.reviewed_at,
        minimum_radius_m=base.tools.minimum_radius_m,
    )
    physical = PhysicalACI(base.case, base.model, tools, max_validations=1)
    return GenericRskACI(physical, **options)


def ordinal(k, arm):
    k["ordinal"] = next(
        row["ordinal"]
        for row in allocate_pilot(k["protocol"])
        if row["arm"] == arm and row["case_id"] == k["protocol"].cases[0].case_id
    )


def request(aci, op, **args):
    return {"operation": op, "arguments": {"case_hash": aci.base.case.case_hash, **args}}


def tf_change(aci, value=True):
    name, path = next(
        (name, path) for name, path in aci.base.tools.allowed if path[-1] == "tf_broadcast"
    )
    return {
        "file": name,
        "pointer": "/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in path),
        "value_json": json.dumps(value),
    }


def test_generic_has_same_common_facts_and_no_expert_shortcuts(tmp_path):
    old, _, _ = bundle(tmp_path, arm="B")
    aci = generic(old.base)
    assert aci.base.common_evidence.catalog() == old.base.common_evidence.catalog()
    schema = aci.request_schema()["discriminator"]["mapping"]
    assert {"files.list", "files.read", "staging.propose_patch"} <= set(schema)
    assert not (
        {
            "ros_engineer.propose_repair",
            "ros_engineer.select_staging_options",
            "ros_engineer.investigate",
            "ros_engineer.query_cross_body_memory",
        }
        & set(schema)
    )
    text = json.dumps(aci.context_projection())
    assert "F-TF-001" not in text and "enable_localization_tf" not in text
    assert not aci.base.proposals
    assert len(build_rsk_tool_guide(aci.decision_schema())["operations_available_this_turn"]) > 3


@pytest.mark.parametrize(
    "operation",
    [
        "ros_engineer.propose_repair",
        "ros_engineer.select_staging_options",
        "ros_engineer.investigate",
        "ros_engineer.query_cross_body_memory",
    ],
)
def test_generic_host_dispatch_also_rejects_expert_requests(tmp_path, operation):
    old, _, _ = bundle(tmp_path, arm="B")
    aci = generic(old.base)
    result = aci.call(request(aci, operation))
    assert result["status"] == "REJECTED_GENERIC_BASELINE_REQUEST"
    assert not aci.base.proposals and aci.base.validation_count == 0


def test_generic_model_supplies_patch_and_preserves_originals_and_rollback(tmp_path):
    old, _, _ = bundle(tmp_path, arm="B")
    aci = generic(old.base)
    before = copy.deepcopy(aci.base.tools.staging)
    result = aci.call(request(aci, "staging.propose_patch", changes=[tf_change(aci)]))
    assert result["status"] == "GENERIC_STAGING_PROPOSAL_NOT_EXECUTED"
    p = result["proposal"]
    assert p["changes"][0]["before"] is False and p["changes"][0]["value"] is True
    assert p["rollback"][0]["value"] is False
    assert proposal_staging(p, before) != before and aci.base.tools.staging == before
    assert aci.base.validation_count == 0


@pytest.mark.parametrize(
    "change",
    [
        {"file": "bringup.json", "pointer": "/namespace", "value_json": '"/invented"'},
        {"file": "nav2.yaml", "pointer": "/bad/robot_radius", "value_json": "-1"},
    ],
)
def test_generic_unsafe_or_unknown_patch_never_registers(tmp_path, change):
    old, _, _ = bundle(tmp_path, arm="B")
    aci = generic(old.base)
    result = aci.call(request(aci, "staging.propose_patch", changes=[change]))
    assert result["status"] == "REJECTED_GENERIC_BASELINE_REQUEST" and not aci.base.proposals


def test_generic_noop_and_duplicate_changes_are_not_valid_proposals(tmp_path):
    old, _, _ = bundle(tmp_path, arm="B")
    aci = generic(old.base)
    for changes in [[tf_change(aci, False)], [tf_change(aci), tf_change(aci)]]:
        result = aci.call(request(aci, "staging.propose_patch", changes=changes))
        assert result["status"] == "REJECTED_GENERIC_BASELINE_REQUEST"
    assert not aci.base.proposals


def test_original_file_slices_are_bounded_and_cannot_read_host_paths(tmp_path):
    old, _, _ = bundle(tmp_path, arm="B")
    aci = generic(old.base)
    result = aci.call(request(aci, "files.read", file="robot.urdf", offset=3, limit=11))
    assert result["text"] == aci.base.tools.files["robot.urdf"][3:14]
    for name in ["../../etc/passwd", "/etc/passwd"]:
        assert aci.call(request(aci, "files.read", file=name))["status"].startswith("REJECTED")
    assert aci.call(request(aci, "files.read", file="robot.urdf", limit=16001))[
        "status"
    ].startswith("REJECTED")
    envelope = {
        "decision": request(aci, "files.read", file="robot.urdf", offset=None, limit=None),
        "decision_summary": "test",
        "memory_refs_used": [],
    }
    assert aci.call_model_decision(envelope)["offset"] == 0


def test_new_generic_loop_uses_registered_a_without_counting_synthetic_as_model(tmp_path):
    old, _, k = bundle(tmp_path, arm="B")
    aci = generic(old.base)
    ordinal(k, "A")
    first = {
        "decision": request(aci, "staging.propose_patch", changes=[tf_change(aci)]),
        "decision_summary": "synthetic",
        "memory_refs_used": [],
    }
    r, prompts = run(tmp_path, aci, k, [first, decision(aci, "finish")])
    assert r["status"] == "FINISHED_WITHOUT_ACCEPTED_SDK_RESULT" and len(prompts) == 2
    assert r["actual_model_calls"] == r["new_Worlds_requested"] == 0
    assert r["first_valid_proposal"]["decision"] == 1
    assert "enable_localization_tf" not in prompts[0]


def test_generic_application_mismatch_does_not_return_or_register_compiler_answer(tmp_path):
    old, expected, data = application_fixture(tmp_path)
    aci = generic(old.base, application_inputs=data)
    args = {
        "nav2_yaml": "wrong",
        "bringup_json": "{}",
        "requirements_json": "{}",
        "navigation_policy_json": "{}",
    }
    result = aci.call(request(aci, "staging.propose_application", **args))
    assert result["status"] == "REJECTED_GENERIC_BASELINE_REQUEST" and not aci.base.proposals
    assert "files" not in result and expected["files"]["nav2.yaml"] not in json.dumps(result)


def test_generic_and_typed_application_have_identical_original_access_and_output_contract(tmp_path):
    old, expected, data = application_fixture(tmp_path)
    aci = generic(old.base, application_inputs=data)
    assert all(
        aci._original_files()[name] == text for name, text in old.original_input_files().items()
    )
    for name in old.original_input_files():
        r = old.call(
            request(old, "ros_engineer.read_application_input", file=name, offset=0, limit=50)
        )
        assert r["text"] == aci._original_files()[name][:50]
    f = expected["files"]
    r = aci.call(
        request(
            aci,
            "staging.propose_application",
            nav2_yaml=f["nav2.yaml"],
            bringup_json=f["bringup.json"],
            requirements_json=f["requirements.json"],
            navigation_policy_json=f["navigation-policy.json"],
        )
    )
    assert r["generic_caller_supplied_files"] and r["proposal"]["files"] == f
    no_sdk = aci.call(
        request(aci, "ros_engineer.validate_proposal", proposal_id=r["proposal"]["proposal_id"]),
        remaining_sec=10,
    )
    assert no_sdk["status"].startswith("REJECTED") and aci.base.validation_count == 0


def test_d_selects_current_supported_options_without_gold_labels(tmp_path):
    aci, _, _ = bundle(tmp_path, arm="B")
    req = current_deterministic_request(aci)
    assert req["operation"] == "ros_engineer.select_staging_options"
    result = aci.call(req)
    assert "proposal" in result and not aci.base.validation_count


def test_d_registered_zero_model_baseline_cannot_claim_unexecuted_success(tmp_path):
    aci, _, k = bundle(tmp_path, arm="B")
    ordinal(k, "D")
    r = run_rsk_deterministic_episode(TASK, aci, tmp_path / "D-output", **k)
    assert r["status"] == "CONDITIONAL_DETERMINISTIC_PROPOSAL_NOT_EXECUTED"
    assert (
        r["actual_model_calls"]
        == r["actual_model_client_invocations"]
        == r["new_Worlds_requested"]
        == 0
    )
    assert not r["verified_success"] and r["model_decisions"] == 0


def test_d_no_go_precedes_all_compiler_and_sdk_dispatch(tmp_path):
    aci, _, k = bundle(tmp_path, arm="B")
    ordinal(k, "D")
    k["pilot_evidence"] = dict.fromkeys(k["pilot_evidence"])
    r = run_rsk_deterministic_episode(TASK, aci, tmp_path / "D-output", **k)
    assert r["status"] == "NO_GO" and not aci.base.proposals and not aci.trace


def test_d_uses_application_compiler_for_driver_only_originals(tmp_path):
    aci, expected, _ = application_fixture(tmp_path)
    req = current_deterministic_request(aci)
    assert req["operation"] == "ros_engineer.propose_application_bootstrap"
    assert aci.call(req)["proposal"]["files"] == expected["files"]


def test_actual_common_inputs_require_original_task_files_manifest(tmp_path):
    aci, _, k = bundle(tmp_path, arm="B")
    with pytest.raises(ValueError, match="all original task file bytes"):
        registered_episode_inputs(
            TASK,
            aci,
            root=tmp_path,
            arm="B",
            require_complete_originals=True,
            **{
                key: value
                for key, value in k.items()
                if key not in {"pilot_evidence", "evidence_root"}
            },
        )


def test_common_original_files_hash_is_checked_for_synthetic_and_actual(tmp_path):
    aci, _, k = bundle(tmp_path, arm="B")
    facts = json.loads((tmp_path / "facts-manifest.json").read_text())
    originals = put(tmp_path, "raw-original-files.json", aci.base.tools.files)
    facts["original_files"] = originals.model_dump(mode="json")
    ref = put(tmp_path, "all-facts-manifest.json", facts)
    k["facts_manifest"] = ref
    cases = [c.model_dump(mode="json") for c in k["protocol"].cases]
    cases[0]["facts_manifest_sha256"] = ref.sha256
    refreeze(tmp_path, k, cases=cases)
    (tmp_path / "raw-original-files.json").write_text("{}")
    r, prompts = run(tmp_path, aci, k, [])
    assert r["status"] == "NO_GO" and not prompts and not aci.trace


def test_generic_endpoint_disable_requires_same_original_ownership_proof(tmp_path):
    old, _, _ = bundle(tmp_path, arm="B")
    aci = generic(old.base)
    result = aci.call(
        request(
            aci,
            "staging.propose_patch",
            changes=[
                {
                    "file": "bringup.json",
                    "pointer": "/additional_command_endpoint/enabled",
                    "value_json": "false",
                }
            ],
        )
    )
    assert result["status"] == "REJECTED_GENERIC_BASELINE_REQUEST"
    assert "ownership" in result["error"] and not aci.base.proposals


def test_synthetic_d_cannot_acquire_sdk_callback_during_compiler_selection(tmp_path, monkeypatch):
    import benchmarks.ros_expert_ablation.rsk_deterministic_baseline as module

    aci, _, k = bundle(tmp_path, arm="B")
    ordinal(k, "D")
    invoked = []
    original = module.current_deterministic_request

    def changed(current):
        proposed = original(current)
        current.base.evaluator = lambda *a, **kw: invoked.append(True)
        return proposed

    monkeypatch.setattr(module, "current_deterministic_request", changed)
    r = module.run_rsk_deterministic_episode(TASK, aci, tmp_path / "D-output", **k)
    assert r["status"] == "NO_GO" and not invoked and not aci.base.proposals
    assert r["actual_model_calls"] == r["new_Worlds_requested"] == 0


def test_d_unsupported_current_constraints_abstain_without_onboarding_fallback(
    tmp_path, monkeypatch
):
    aci, _, _ = bundle(tmp_path, arm="B")
    monkeypatch.setattr(aci.base, "conditional_staging_options", lambda: {"options": []})
    assert current_deterministic_request(aci) is None
    assert not aci.base.proposals


def test_generic_invalid_model_envelope_is_rejected_without_memory_or_tools(tmp_path):
    old, _, _ = bundle(tmp_path, arm="B")
    aci = generic(old.base)
    with pytest.raises(ValueError, match="Memory"):
        aci.call_model_decision(
            {
                "decision": request(aci, "files.list"),
                "decision_summary": "test",
                "memory_refs_used": ["invented"],
            }
        )
    assert not aci.trace


def test_host_callback_exception_is_persisted_without_retry_in_model_loop(tmp_path, monkeypatch):
    aci, _, k = bundle(tmp_path, arm="B")
    observed = []

    def broken(value, **kwargs):
        observed.append(value)
        raise RuntimeError("synthetic host fault")

    monkeypatch.setattr(aci, "call_model_decision", broken)
    r, prompts = run(tmp_path, aci, k, [decision(aci, "inspect_task")])
    assert r["status"] == "HOST_TOOL_CALLBACK_EXCEPTION_NO_AUTOMATIC_RETRY"
    assert len(prompts) == len(observed) == 1 and r["actual_model_calls"] == 0
    artifact = tmp_path / "output" / "turn-00-tool-result.json"
    assert json.loads(artifact.read_text())["result"]["error_type"] == "RuntimeError"
    assert (tmp_path / "output" / "original-rsk-episode-result.json").is_file()


def test_generic_application_accepts_semantics_without_hash_calculation_or_serializer_guessing(
    tmp_path,
):
    import yaml

    old, expected, data = application_fixture(tmp_path)
    aci = generic(old.base, application_inputs=data)
    f = expected["files"]
    policy = json.loads(f["navigation-policy.json"])
    policy.pop("navigation_requirements_hash")
    result = aci.call(
        request(
            aci,
            "staging.propose_application",
            nav2_yaml=yaml.safe_dump(yaml.safe_load(f["nav2.yaml"]), sort_keys=True),
            bringup_json=json.dumps(json.loads(f["bringup.json"]), sort_keys=True),
            requirements_json=json.dumps(json.loads(f["requirements.json"]), sort_keys=True),
            navigation_policy_json=json.dumps(policy, sort_keys=True),
        )
    )
    assert result["host_sealed_metadata_and_serialization_only"]
    assert result["proposal"]["files"] == f and aci.base.validation_count == 0


@pytest.mark.parametrize("invalid", ["a: &a [*a]", "duplicate: 1\nduplicate: 2", "[bad: yaml"])
def test_generic_application_rejects_aliases_duplicate_keys_or_malformed_yaml(tmp_path, invalid):
    old, expected, data = application_fixture(tmp_path)
    aci = generic(old.base, application_inputs=data)
    f = expected["files"]
    result = aci.call(
        request(
            aci,
            "staging.propose_application",
            nav2_yaml=invalid,
            bringup_json=f["bringup.json"],
            requirements_json=f["requirements.json"],
            navigation_policy_json=f["navigation-policy.json"],
        )
    )
    assert result["status"] == "REJECTED_GENERIC_BASELINE_REQUEST" and not aci.base.proposals


def test_generic_and_expert_can_read_identical_complete_raw_native_and_urdf(tmp_path):
    expert, _, _ = bundle(tmp_path, arm="B")
    aci = generic(expert.base)
    assert aci.original_files_catalog() == expert.original_files_catalog()
    for name in expert.base.tools.files:
        a = aci.call(request(aci, "files.read", file=name, offset=0, limit=333))
        b = expert.call(request(expert, "files.read", file=name, offset=0, limit=333))
        assert a == b
    value = {
        "decision": request(expert, "files.read", file="robot.urdf", offset=None, limit=None),
        "decision_summary": "test",
        "memory_refs_used": [],
    }
    assert expert.call_model_decision(value)["offset"] == 0


def test_application_raw_file_catalog_and_readback_equal_for_generic_and_expert(tmp_path):
    expert, _, data = application_fixture(tmp_path)
    aci = generic(expert.base, application_inputs=data)
    assert aci.original_files_catalog() == expert.original_files_catalog()
    for name in expert._original_files():
        assert aci.call(request(aci, "files.read", file=name, offset=3, limit=97)) == expert.call(
            request(expert, "files.read", file=name, offset=3, limit=97)
        )


def test_provider_friendly_application_read_defaults_do_not_reject_null_defaults(tmp_path):
    expert, _, _ = application_fixture(tmp_path)
    value = {
        "decision": request(
            expert,
            "ros_engineer.read_application_input",
            file="software-template.json",
            offset=None,
            limit=None,
        ),
        "decision_summary": "test",
        "memory_refs_used": [],
    }
    assert expert.call_model_decision(value)["offset"] == 0


def test_original_read_budget_is_shared_with_expert_evidence_reads(tmp_path):
    aci, _, k = bundle(tmp_path, arm="B")
    refreeze(tmp_path, k, max_read_only_probes=1)
    read = {
        "decision": request(aci, "files.read", file="robot.urdf", offset=0, limit=25),
        "decision_summary": "test",
        "memory_refs_used": [],
    }
    second = decision(aci, "read_task_evidence", view="signals", cursor=0)
    r, prompts = run(tmp_path, aci, k, [read, second])
    assert r["status"] == "READ_ONLY_PROBE_BUDGET_EXHAUSTED" and len(prompts) == 2
    assert len(r["tool_calls"]) == 1 and r["read_only_investigation_requests"] == 1
    assert r["new_Worlds_requested"] == 0


def test_ordered_application_manifest_binds_actual_original_input_order(tmp_path):
    _, _, k = bundle(tmp_path, arm="B")
    aci, _, data = application_fixture(tmp_path)
    facts = json.loads((tmp_path / "facts-manifest.json").read_text())
    for key, value in [
        ("case", aci.base.case.model_dump(mode="json")),
        ("model", aci.base.model.to_dict()),
        ("requirements", aci.base.requirements),
        ("staging", aci.base.tools.initial_staging),
        ("original_files", aci.base.tools.files),
    ]:
        facts[key] = put(tmp_path, "app-bound-" + key + ".json", value).model_dump(mode="json")
    original = put(tmp_path, "app-original-inputs.json", data)
    facts["application_inputs"] = original.model_dump(mode="json")
    manifest = put(tmp_path, "app-complete-facts.json", facts)
    k["facts_manifest"] = manifest
    cases = [c.model_dump(mode="json") for c in k["protocol"].cases]
    cases[0]["facts_manifest_sha256"] = manifest.sha256
    refreeze(tmp_path, k, cases=cases)
    aci.base.original_run_id = "f" * 32
    arguments = {
        key: value for key, value in k.items() if key not in {"pilot_evidence", "evidence_root"}
    }
    registered_episode_inputs(
        TASK, aci, root=tmp_path, arm="B", require_complete_originals=True, **arguments
    )
    # Equal dictionaries with reordered original JSON inputs must not bind a
    # differently serialized compiler template to the same registered input.
    aci._inputs_json = json.dumps(dict(reversed(list(data.items()))), separators=(",", ":"))
    import hashlib

    aci._inputs_bytes_sha256 = hashlib.sha256(aci._inputs_json.encode()).hexdigest()
    with pytest.raises(ValueError, match="inputs/order"):
        registered_episode_inputs(
            TASK, aci, root=tmp_path, arm="B", require_complete_originals=True, **arguments
        )


def test_generic_semantic_application_check_keeps_nav2_scalar_types_strict(tmp_path):
    import yaml

    old, expected, data = application_fixture(tmp_path)
    aci = generic(old.base, application_inputs=data)
    f = expected["files"]
    nav2 = yaml.safe_load(f["nav2.yaml"])
    group = next(iter(nav2.values()))
    value = group["controller_server"]["ros__parameters"]["controller_frequency"]
    assert type(value) is float and value.is_integer()
    group["controller_server"]["ros__parameters"]["controller_frequency"] = int(value)
    r = aci.call(
        request(
            aci,
            "staging.propose_application",
            nav2_yaml=yaml.safe_dump(nav2),
            bringup_json=f["bringup.json"],
            requirements_json=f["requirements.json"],
            navigation_policy_json=f["navigation-policy.json"],
        )
    )
    assert r["status"] == "REJECTED_GENERIC_BASELINE_REQUEST" and not aci.base.proposals
