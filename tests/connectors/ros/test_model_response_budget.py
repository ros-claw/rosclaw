"""Budget release gates must stop actual tool responses on unverifiable usage."""

import json

import pytest

from rosclaw.connectors.ros.verification.model_budget import (
    ModelResponseBudget,
    RegisteredModelBudgetV1,
)


def registered(**changes):
    return RegisteredModelBudgetV1(
        provider="openai-codex",
        model="gpt-6.1-sol",
        reasoning_effort="low",
        max_requests=3,
        max_output_tokens_per_request=32,
        max_total_tokens=100,
        wall_budget_sec=60,
        **changes,
    )


def payload(**changes):
    return json.dumps(
        {
            "model": "gpt-6.1-sol",
            "store": False,
            "stream": True,
            "input": [{"role": "user", "content": "test"}],
            "reasoning": {"effort": "low", "summary": "auto"},
            "text": {"verbosity": "low"},
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "include": ["reasoning.encrypted_content"],
            **changes,
        }
    ).encode()


def response(
    *,
    input_tokens=10,
    output_tokens=5,
    total_tokens=None,
    event="response.completed",
    cap=32,
    **changes,
):
    value = {
        "model": "gpt-6.1-sol",
        "max_output_tokens": cap,
        "status": "completed" if event == "response.completed" else "incomplete",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens if total_tokens is None else total_tokens,
        },
        **changes,
    }
    return b"data: " + json.dumps({"type": event, "response": value}).encode() + b"\n\n"


def test_actual_usage_allows_one_response_and_preserves_native_payload():
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    pending = budget.admit(payload(), now=11)
    upstream = json.loads(pending.upstream_body)
    assert upstream == {**json.loads(payload()), "max_output_tokens": 32}
    result = budget.finish(response(), http_status=200, now=12)
    assert result["allow_response"] is True
    assert result["actual_usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert budget.snapshot()["actual_total_tokens"] == 15
    assert budget.snapshot()["budget_scope"] == "RESPONSE_RELEASE_GATE_NOT_PREBILLING_COST_CAP"


@pytest.mark.parametrize(
    "change",
    [
        {"model": "gpt-5.5"},
        {"store": True},
        {"stream": False},
        {"reasoning": {"effort": "high", "summary": "auto"}},
        {"text": {"verbosity": "high"}},
        {"temperature": 0.5},
        {"parallel_tool_calls": False},
        {"max_output_tokens": 999},
        {"input": []},
        {"unknown_inference_override": True},
        {"tool_choice": "required"},
        {"include": []},
        {"tools": [{"type": "web_search"}]},
    ],
)
def test_fair_inference_configuration_cannot_drift(change):
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    with pytest.raises(ValueError):
        budget.admit(payload(**change), now=11)
    assert budget.snapshot()["requests_admitted"] == 0


def test_duplicate_json_keys_are_not_a_model_selection_override():
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    with pytest.raises(ValueError):
        budget.admit(b'{"model":"gpt-6.1-sol","model":"gpt-5.5"}', now=11)


def test_unknown_usage_halts_without_releasing_or_silently_retrying():
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    budget.admit(payload(), now=11)
    result = budget.finish(response(usage=None), http_status=200, now=12)
    assert result["allow_response"] is False
    assert result["usage_verified"] is False
    with pytest.raises(ValueError):
        budget.admit(payload(), now=13)


@pytest.mark.parametrize(
    "changes",
    [
        {"input_tokens": True},
        {"output_tokens": -1},
        {"total_tokens": 99},
        {"model": "gpt-5.5"},
        {"cap": 64},
        {"status": "incomplete"},
    ],
)
def test_forged_or_inconsistent_response_fails_closed(changes):
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    budget.admit(payload(), now=11)
    assert budget.finish(response(**changes), http_status=200, now=12)["allow_response"] is False


def test_actual_single_request_overshoot_is_recorded_not_hidden():
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    budget.admit(payload(), now=11)
    result = budget.finish(response(input_tokens=120), http_status=200, now=12)
    assert result["allow_response"] is False
    assert result["status"] == "TOTAL_TOKEN_BUDGET_EXCEEDED"
    assert budget.snapshot()["actual_total_tokens"] == 125
    assert budget.snapshot()["overshoot_tokens"] == 25
    with pytest.raises(ValueError):
        budget.admit(payload(), now=13)


def test_remaining_budget_reduces_next_output_cap():
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    budget.admit(payload(), now=11)
    assert budget.finish(response(input_tokens=70, output_tokens=5), http_status=200, now=12)[
        "allow_response"
    ]
    pending = budget.admit(payload(), now=13)
    assert json.loads(pending.upstream_body)["max_output_tokens"] == 25
    assert budget.finish(response(cap=25), http_status=200, now=14)["allow_response"]


def test_incomplete_output_cap_counts_usage_but_withholds_partial_tool_arguments():
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    budget.admit(payload(), now=11)
    result = budget.finish(
        response(
            output_tokens=32,
            event="response.incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        ),
        http_status=200,
        now=12,
    )
    assert result["allow_response"] is False and result["status"] == "MODEL_OUTPUT_LIMIT"
    assert budget.snapshot()["actual_total_tokens"] == 42


@pytest.mark.parametrize("now", [float("nan"), float("inf"), 9, True, 71])
def test_deadline_and_clock_changes_cannot_extend_a_frozen_budget(now):
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    with pytest.raises(ValueError):
        budget.admit(payload(), now=now)


def test_response_after_wall_deadline_is_accounted_but_not_released():
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    budget.admit(payload(), now=11)
    result = budget.finish(response(), http_status=200, now=71)
    assert result["allow_response"] is False and result["status"] == "WALL_BUDGET_EXCEEDED"
    assert budget.snapshot()["actual_total_tokens"] == 15


def test_pending_request_blocks_races_and_unsolicited_responses():
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    with pytest.raises(ValueError):
        budget.finish(response(), http_status=200, now=11)
    budget.admit(payload(), now=11)
    with pytest.raises(ValueError):
        budget.admit(payload(), now=12)
    assert budget.finish(response(), http_status=200, now=13)["allow_response"]


def test_request_ceiling_counts_failed_requests():
    config = registered().model_copy(update={"max_requests": 1})
    budget = ModelResponseBudget(config, started_monotonic=10)
    budget.admit(payload(), now=11)
    result = budget.finish(b'{"error":"upstream refused"}', http_status=400, now=12)
    assert not result["allow_response"] and not result["usage_verified"]
    assert budget.snapshot()["requests_admitted"] == 1
    with pytest.raises(ValueError):
        budget.admit(payload(), now=13)


def test_duplicate_terminal_events_do_not_double_count_usage():
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    budget.admit(payload(), now=11)
    result = budget.finish(response() + response(), http_status=200, now=12)
    assert not result["allow_response"] and not result["usage_verified"]
    assert budget.snapshot()["actual_total_tokens"] == 0


def test_ledger_snapshots_and_returned_usage_cannot_mutate_retained_accounting():
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    budget.admit(payload(), now=11)
    result = budget.finish(response(), http_status=200, now=12)
    result["actual_usage"]["total_tokens"] = 0
    snapshot = budget.snapshot()
    snapshot["records"][0]["actual_usage"]["total_tokens"] = 999
    assert budget.snapshot()["records"][0]["actual_usage"]["total_tokens"] == 15


def test_request_ceiling_also_stops_after_successful_requests():
    budget = ModelResponseBudget(
        registered().model_copy(update={"max_requests": 1}), started_monotonic=10
    )
    budget.admit(payload(), now=11)
    assert budget.finish(response(), http_status=200, now=12)["allow_response"]
    with pytest.raises(ValueError):
        budget.admit(payload(), now=13)


def test_error_event_cannot_be_hidden_by_a_later_completed_event():
    budget = ModelResponseBudget(registered(), started_monotonic=10)
    budget.admit(payload(), now=11)
    raw = b'data: {"type":"error"}\n\n' + response()
    assert budget.finish(raw, http_status=200, now=12)["allow_response"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_requests", True),
        ("max_requests", 0),
        ("max_total_tokens", -1),
        ("max_output_tokens_per_request", 128001),
        ("unexpected_override", True),
    ],
)
def test_registration_is_typed_and_closed(field, value):
    data = registered().model_dump()
    data[field] = value
    with pytest.raises(ValueError):
        RegisteredModelBudgetV1.model_validate(data)
