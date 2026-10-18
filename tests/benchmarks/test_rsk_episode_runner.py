"""All clients/gates here are synthetic contracts: zero model/World evidence."""

import hashlib
import json
from copy import deepcopy

import pytest

from benchmarks.ros_expert_ablation.physical_aci import PhysicalACI
from benchmarks.ros_expert_ablation.rsk_episode_runner import (
    EpisodeFactsV1,
    FrozenProviderV1,
    OriginalRef,
    run_rsk_typed_episode,
)
from benchmarks.ros_expert_ablation.rsk_physical_aci import RskPhysicalACI
from benchmarks.ros_expert_ablation.rsk_pilot_protocol import allocate_pilot
from rosclaw.contracts.common import content_hash
from tests.benchmarks.test_cross_body_memory_trace import query_fixture
from tests.benchmarks.test_rsk_pilot_protocol import evidence, protocol

TASK = {
    "task_id": "SYNTHETIC-contract",
    "goal": "inspect and propose; no SDK",
    "case": "PRIVATE_GOLD_LABEL",
    "path": "/home/private/fixture",
}


def put(root, name, value):
    p = root / name
    p.write_text(json.dumps(value, indent=2) + "\n")
    return OriginalRef(path=name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def bundle(tmp_path, arm="E"):
    episode = put(tmp_path, "evaluation-episode.json", {"scope": "SYNTHETIC_CONTRACT_ONLY"})
    query, _, record, old, case = query_fixture(tmp_path, excluded_episode_sha256=[episode.sha256])
    old.tools.minimum_radius_m = case.derived_documents["navigation_observation_review"][
        "candidate"
    ]["geometry"]["physical_radius_m"]
    base = PhysicalACI(case, old.model, old.tools, max_validations=1)
    aci = RskPhysicalACI(base, arm=arm, memory_query=query if arm == "E" else None)
    refs = {
        key: put(tmp_path, key + ".json", value)
        for key, value in [
            ("case", case.model_dump(mode="json")),
            ("model", base.model.to_dict()),
            ("requirements", base.requirements),
            ("staging", base.tools.initial_staging),
            ("task", TASK),
        ]
    }
    refs["run_settings"] = put(tmp_path, "settings.json", {"run_id": "f" * 32})
    facts = put(
        tmp_path,
        "facts-manifest.json",
        EpisodeFactsV1(**refs, evaluation_episode_sha256=episode.sha256).model_dump(mode="json"),
    )
    frozen = protocol()
    provider = put(
        tmp_path,
        "provider.json",
        FrozenProviderV1(
            execution_kind="SYNTHETIC_CONTRACT_CLIENT",
            model_identity=frozen.model_identity,
            effort="fixture",
            client_code_sha256="0" * 64,
            client_settings_code_sha256="0" * 64,
            executable_sha256="0" * 64,
        ).model_dump(mode="json"),
    )
    cases = [row.model_dump(mode="json") for row in frozen.cases]
    cases[0].update(facts_manifest_sha256=facts.sha256, evaluation_episode_sha256=episode.sha256)
    policy = put(
        tmp_path,
        "Memory-policy.json",
        {
            "schema_version": "rosclaw.rsk_memory_policy_manifest.v1",
            "cases": {row["case_id"]: query.policy for row in cases},
        },
    )
    frozen = type(frozen).model_validate(
        {
            **frozen.model_dump(mode="json"),
            "cases": cases,
            "provider_settings_sha256": provider.sha256,
            "memory_policy_sha256": policy.sha256,
        }
    )
    gates = evidence(tmp_path, frozen)
    ordinal = next(
        row["ordinal"]
        for row in allocate_pilot(frozen)
        if row["case_id"] == cases[0]["case_id"] and row["arm"] == arm
    )
    kwargs = {
        "protocol": frozen,
        "pilot_evidence": gates,
        "evidence_root": tmp_path,
        "ordinal": ordinal,
        "facts_manifest": facts,
        "evaluation_episode": episode,
        "provider_settings": provider,
        "memory_policy": policy,
    }
    return aci, record, kwargs


def client(decisions, prompts, *, native=(), usage=True):
    decisions = iter(decisions)

    def invoke(prompt, schema, output, **options):
        prompts.append(prompt)
        output.mkdir()
        decision = next(decisions)
        if callable(decision):
            decision = decision()
        (output / "original-events.jsonl").write_text(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": json.dumps(decision)},
                }
            )
            + "\n"
        )
        return {
            "status": "ACTUAL_MODEL_TURN_COMPLETED",
            "usage": {"input_tokens": 100, "output_tokens": 5} if usage else None,
            "native_tool_items": list(native),
            "synthetic_test_fixture": True,
        }

    return invoke


def decision(aci, op, refs=None, **arguments):
    return {
        "decision": {
            "operation": "finish" if op == "finish" else "ros_engineer." + op,
            "arguments": {}
            if op == "finish"
            else {"case_hash": aci.base.case.case_hash, **arguments},
        },
        "decision_summary": "synthetic public summary",
        "memory_refs_used": refs or [],
    }


def run(tmp_path, aci, kwargs, decisions, **options):
    prompts = []
    result = run_rsk_typed_episode(
        TASK,
        aci,
        tmp_path / "output",
        contract_client=client(decisions, prompts, **options),
        **kwargs,
    )
    return result, prompts


def test_no_go_precedes_provider_and_all_host_tools(tmp_path):
    aci, _, k = bundle(tmp_path)
    k["pilot_evidence"] = dict.fromkeys(k["pilot_evidence"])
    r, p = run(tmp_path, aci, k, [])
    assert r["status"] == "NO_GO" and not p and not aci.trace
    assert (
        r["actual_model_calls"] == r["synthetic_contract_calls"] == r["new_Worlds_requested"] == 0
    )


def test_new_e_query_then_current_proposal_uses_actual_context_refs_without_sdk(tmp_path):
    aci, record, k = bundle(tmp_path)
    before = deepcopy(aci.base.tools.staging)
    r, p = run(
        tmp_path,
        aci,
        k,
        [
            decision(aci, "query_cross_body_memory", query="disabled TF"),
            decision(
                aci,
                "propose_cross_body_memory_repair",
                refs=[record.experience_hash],
                memory_ref=record.experience_hash,
            ),
            decision(aci, "finish"),
        ],
    )
    assert r["synthetic_contract_calls"] == 3 and r["actual_model_calls"] == 0
    assert r["status"] == "FINISHED_WITHOUT_ACCEPTED_SDK_RESULT"
    assert r["new_Worlds_requested"] == 0 and not r["verified_success"]
    assert (
        r["first_valid_proposal"]["decision"] == 2 and r["known_sanity_proposal_within_3_decisions"]
    )
    assert [row["stage"] for row in r["Memory_trace"]] == [
        "RETRIEVAL",
        "CONTEXT_INSERTION",
        "CONDITIONAL_SELECTION",
    ]
    assert aci.base.tools.staging == before and "PRIVATE_GOLD_LABEL" not in "".join(p)
    assert "/home/private/fixture" not in "".join(p)
    payload = dict(r)
    digest = payload.pop("result_hash")
    assert digest == content_hash("rskepisode", payload)
    assert r["artifact_scope"] == "SYNTHETIC_CONTRACT_ONLY_NOT_MODEL_OR_PHYSICAL_EVIDENCE"


@pytest.mark.parametrize(
    "slot", ["facts_manifest", "evaluation_episode", "provider_settings", "memory_policy"]
)
def test_all_registered_hashes_checked_before_any_client_call(tmp_path, slot):
    aci, _, k = bundle(tmp_path)
    k[slot] = OriginalRef(path=k[slot].path, sha256="e" * 64)
    r, p = run(tmp_path, aci, k, [])
    assert r["status"] == "NO_GO" and not p and not aci.trace


def test_mutated_original_fact_bytes_are_rejected_before_model(tmp_path):
    aci, _, k = bundle(tmp_path)
    path = tmp_path / "staging.json"
    path.write_text("{}")
    r, p = run(tmp_path, aci, k, [])
    assert r["status"] == "NO_GO" and "SHA" in r["error"] and not p


def test_task_text_cannot_differ_between_paired_conditions(tmp_path):
    aci, _, k = bundle(tmp_path)
    r = run_rsk_typed_episode(
        {**TASK, "goal": "different prompt"},
        aci,
        tmp_path / "output",
        contract_client=client([], []),
        **k,
    )
    assert r["status"] == "NO_GO" and r["actual_model_calls"] == r["synthetic_contract_calls"] == 0


def test_wrong_registered_arm_never_calls_model(tmp_path):
    aci, _, k = bundle(tmp_path)
    k["ordinal"] = next(
        row["ordinal"] for row in allocate_pilot(k["protocol"]) if row["arm"] == "B"
    )
    r, p = run(tmp_path, aci, k, [])
    assert r["status"] == "NO_GO" and not p


def test_synthetic_contract_cannot_invoke_any_sdk_callback(tmp_path):
    aci, _, k = bundle(tmp_path)
    aci.base.evaluator = lambda *a, **kw: pytest.fail("SDK must not be called")
    r, p = run(tmp_path, aci, k, [])
    assert r["status"] == "NO_GO" and "synthetic" in r["error"] and not p


def test_native_tool_attempt_stops_before_host_dispatch(tmp_path):
    aci, _, k = bundle(tmp_path)
    r, p = run(
        tmp_path,
        aci,
        k,
        [decision(aci, "query_cross_body_memory", query="tf")],
        native=["command_execution"],
    )
    assert r["status"] == "NATIVE_TOOL_PROTOCOL_VIOLATION" and len(p) == 1 and not aci.trace
    assert r["actual_model_calls"] == 0 and r["new_Worlds_requested"] == 0


def test_missing_usage_is_failure_not_a_free_retry(tmp_path):
    aci, _, k = bundle(tmp_path)
    r, p = run(
        tmp_path, aci, k, [decision(aci, "query_cross_body_memory", query="tf")], usage=False
    )
    assert r["status"] == "MODEL_CLIENT_OR_USAGE_FAILURE" and len(p) == 1 and not aci.trace


def test_memory_context_cannot_be_injected_before_registered_episode(tmp_path):
    aci, _, k = bundle(tmp_path)
    aci.memory_query("disabled TF")
    r, p = run(tmp_path, aci, k, [])
    assert r["status"] == "NO_GO" and not p


def test_current_episode_exclusion_is_mandatory_even_with_frozen_policy_hash(tmp_path):
    aci, _, k = bundle(tmp_path)
    aci.memory_query.excluded_episodes = frozenset({"0" * 64})
    r, p = run(tmp_path, aci, k, [])
    assert r["status"] == "NO_GO" and "Memory policy" in r["error"] and not p


def test_invalid_memory_citation_stops_without_proposal_or_sdk(tmp_path):
    aci, _, k = bundle(tmp_path)
    r, p = run(
        tmp_path,
        aci,
        k,
        [
            decision(
                aci, "propose_cross_body_memory_repair", refs=["invented"], memory_ref="invented"
            )
        ],
    )
    assert r["status"] == "INVALID_MODEL_DECISION_OR_ORIGINAL_INPUT" and not aci.base.proposals
    assert len(p) == 1 and r["actual_model_calls"] == 0 and r["new_Worlds_requested"] == 0


def refreeze(tmp_path, kwargs, **changes):
    kwargs["protocol"] = type(kwargs["protocol"]).model_validate(
        {**kwargs["protocol"].model_dump(mode="json"), **changes}
    )
    kwargs["pilot_evidence"] = evidence(tmp_path, kwargs["protocol"])


def test_decision_budget_has_no_automatic_model_retry_or_world(tmp_path):
    aci, _, k = bundle(tmp_path, "B")
    refreeze(tmp_path, k, max_model_decisions=3)
    inspect = {
        "decision": {"operation": "ros_engineer.inspect_task", "arguments": {}},
        "decision_summary": "inspect original card",
        "memory_refs_used": [],
    }
    r, prompts = run(tmp_path, aci, k, [inspect] * 4)
    assert r["status"] == "MODEL_DECISION_BUDGET_EXHAUSTED"
    assert len(prompts) == r["synthetic_contract_calls"] == 3
    assert r["actual_model_calls"] == r["new_Worlds_requested"] == 0


def test_readonly_request_budget_does_not_trigger_a_second_probe(tmp_path):
    aci, _, k = bundle(tmp_path, "C")
    refreeze(tmp_path, k, max_read_only_probes=1)
    issue = next(row for row in aci.base.case.constraints if row.next_checks)
    check = decision(aci, "investigate", issue_id=issue.id, check_id=issue.next_checks[0])
    r, prompts = run(tmp_path, aci, k, [check, check])
    assert r["status"] == "READ_ONLY_PROBE_BUDGET_EXHAUSTED" and len(prompts) == 2
    assert r["read_only_investigation_requests"] == 1 and r["new_Worlds_requested"] == 0


def test_expired_wall_budget_never_calls_provider(tmp_path):
    aci, _, k = bundle(tmp_path)
    ticks = iter([0.0, 301.0, 301.0])
    r = run_rsk_typed_episode(
        TASK,
        aci,
        tmp_path / "output",
        contract_client=client([], []),
        clock=lambda: next(ticks),
        **k,
    )
    assert r["status"] == "WALL_TIME_BUDGET_EXHAUSTED"
    assert r["actual_model_calls"] == r["synthetic_contract_calls"] == 0


def test_synthetic_callable_cannot_impersonate_real_provider(tmp_path):
    aci, _, k = bundle(tmp_path)
    value = json.loads((tmp_path / "provider.json").read_text())
    value["execution_kind"] = "ACTUAL_MODEL_CLIENT"
    provider = put(tmp_path, "provider-actual.json", value)
    k["provider_settings"] = provider
    refreeze(tmp_path, k, provider_settings_sha256=provider.sha256)
    r, prompts = run(tmp_path, aci, k, [])
    assert r["status"] == "NO_GO" and "impersonate" in r["error"] and not prompts
    assert r["actual_model_client_invocations"] == 0


def test_raw_native_event_cannot_be_hidden_by_empty_client_summary(tmp_path):
    aci, _, k = bundle(tmp_path)

    def dishonest(prompt, schema, output, **options):
        output.mkdir()
        (output / "original-events.jsonl").write_text(
            json.dumps({"type": "item.started", "item": {"type": "command_execution"}}) + "\n"
        )
        return {
            "status": "ACTUAL_MODEL_TURN_COMPLETED",
            "usage": {"input_tokens": 1},
            "native_tool_items": [],
        }

    r = run_rsk_typed_episode(TASK, aci, tmp_path / "output", contract_client=dishonest, **k)
    assert r["status"] == "NATIVE_TOOL_PROTOCOL_VIOLATION" and not aci.trace
    assert r["actual_model_calls"] == r["new_Worlds_requested"] == 0


def test_original_reference_symlink_is_not_a_valid_fact_manifest(tmp_path):
    aci, _, k = bundle(tmp_path)
    (tmp_path / "alias.json").symlink_to(tmp_path / k["facts_manifest"].path)
    k["facts_manifest"] = OriginalRef(path="alias.json", sha256=k["facts_manifest"].sha256)
    r, prompts = run(tmp_path, aci, k, [])
    assert r["status"] == "NO_GO" and "symlink" in r["error"] and not prompts


def test_original_mutation_during_provider_call_blocks_host_dispatch(tmp_path):
    aci, _, k = bundle(tmp_path)

    def mutate():
        (tmp_path / "staging.json").write_text("{}")
        return decision(aci, "query_cross_body_memory", query="tf")

    r, prompts = run(tmp_path, aci, k, [mutate])
    assert r["status"] == "INVALID_MODEL_DECISION_OR_ORIGINAL_INPUT"
    assert len(prompts) == 1 and not aci.trace and r["new_Worlds_requested"] == 0


def test_callback_added_during_contract_client_call_cannot_execute(tmp_path):
    aci, _, k = bundle(tmp_path)

    def attach():
        aci.base.evaluator = lambda *a, **kw: pytest.fail("callback must not run")
        return decision(aci, "query_cross_body_memory", query="tf")

    r, prompts = run(tmp_path, aci, k, [attach])
    assert r["status"] == "INVALID_MODEL_DECISION_OR_ORIGINAL_INPUT"
    assert "callback" in r["error"] and not aci.trace and len(prompts) == 1
