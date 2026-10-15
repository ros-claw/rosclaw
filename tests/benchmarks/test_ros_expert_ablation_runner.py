"""Synthetic runner budget/provenance checks, not actual model/World evidence."""

import json
from types import SimpleNamespace

from benchmarks.ros_expert_ablation.runner import run_episode


class Tools:
    operations = ["finish", "list_files", "run_tests"]
    staging = {}
    selected_interfaces = None
    patch_trace = []
    memory_trace = []
    files = {}

    def call(self, op, args):
        return {"synthetic": "original tool result"}


def client(decisions, usage=None):
    def call(prompt, schema, path, **kwargs):
        path.mkdir()
        decision = decisions.pop(0)
        (path / "original-events.jsonl").write_text(
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
            "usage": usage or {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10},
        }

    return call


def decision(op):
    return {
        "operation": op,
        "arguments": "{}",
        "decision_summary": "synthetic contract",
        "memory_refs_used": [],
    }


def good_evaluation(*args, **kwargs):
    return {
        "verified_success": True,
        "verification_source": "INDEPENDENT_ORIGINAL_SDK_REVIEW",
        "new_world_run": "synthetic",
        "fresh_l1_ref": "synthetic",
        "canonical_receipts": ["synthetic"],
        "independent_stop_ref": "synthetic",
    }


def test_model_finish_never_substitutes_for_physics(tmp_path):
    result = run_episode(
        {},
        Tools(),
        tmp_path / "episode",
        lambda *a, **k: {"verified_success": True},
        model_client=client([decision("finish")]),
    )
    assert (
        not result["verified_success"]
        and result["status"] == "SUCCESS_REJECTED_MISSING_INDEPENDENT_PHYSICS_EVIDENCE"
    )


def test_original_evaluation_is_required_even_on_model_finish(tmp_path):
    result = run_episode(
        {},
        Tools(),
        tmp_path / "episode",
        good_evaluation,
        model_client=client([decision("finish")]),
    )
    assert result["verified_success"] and len(result["tests"]) == 1
    assert result["actual_usage"]["input_tokens"] == 100


def test_tool_budget_has_no_extra_hidden_round(tmp_path):
    result = run_episode(
        {},
        Tools(),
        tmp_path / "episode",
        good_evaluation,
        model_client=client([decision("list_files")]),
        tool_budget=1,
    )
    assert (
        not result["verified_success"]
        and len(result["raw_model_calls"]) == result["tool_calls"] == 1
    )


def test_unavailable_usage_is_not_converted_to_zero(tmp_path):
    def failed(*a, **kw):
        return {"status": "MODEL_CLIENT_FAILED_OR_USAGE_MISSING", "usage": None}

    result = run_episode({}, Tools(), tmp_path / "episode", good_evaluation, model_client=failed)
    assert result["actual_usage"] is None and not result["verified_success"]


def test_late_positive_physics_does_not_pass_wall_time_budget(tmp_path):
    now = SimpleNamespace(value=0)

    def late(*a, **kw):
        now.value = 301
        return good_evaluation()

    result = run_episode(
        {},
        Tools(),
        tmp_path / "episode",
        late,
        model_client=client([decision("run_tests")]),
        clock=lambda: now.value,
    )
    assert (
        not result["verified_success"]
        and result["status"] == "FRESH_WORLD_VALIDATION_EXCEEDED_EPISODE_BUDGET"
    )


def test_model_cannot_claim_unadmitted_memory_or_trigger_validation(tmp_path):
    claimed = decision("finish")
    claimed["memory_refs_used"] = ["invented-ref"]

    def must_not_evaluate(*args, **kwargs):
        raise AssertionError("unadmitted memory claim must stop before evaluation")

    result = run_episode(
        {}, Tools(), tmp_path / "episode", must_not_evaluate, model_client=client([claimed])
    )
    assert result["status"] == "UNADMITTED_MEMORY_REFERENCE_IN_MODEL_DECISION"
    assert not result["verified_success"] and not result["tests"]


def test_failed_client_preserves_reported_actual_usage(tmp_path):
    actual = {"input_tokens": 321, "cached_input_tokens": 200, "output_tokens": 12}

    def failed(*args, **kwargs):
        return {"status": "NATIVE_TOOL_PROTOCOL_VIOLATION", "usage": actual}

    result = run_episode({}, Tools(), tmp_path / "episode", good_evaluation, model_client=failed)
    assert result["actual_usage"] == actual and not result["verified_success"]
