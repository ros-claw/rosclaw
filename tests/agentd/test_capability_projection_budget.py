"""Large observations must not silently lose tail fields or artifact refs."""

import json
from types import SimpleNamespace

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError, _envelope_result


def envelope(value, refs=None):
    return SimpleNamespace(
        status=SimpleNamespace(value="SUCCEEDED"),
        capability_id="semantic.observe_candidates",
        value=value,
        artifact_refs=refs or [],
    )


def request():
    return SimpleNamespace(request_id="request_1")


def test_complete_small_observation_retains_tail_and_references():
    result = _envelope_result(
        request(),
        envelope(
            {"candidates": [{"id": "x"}], "return_to_initial_pose": {"proposal_id": "home"}},
            [{"artifact_id": "a", "path": "/review.json"}],
        ),
    )
    projection = json.loads(result.summary)
    assert result.ok and projection["value"]["return_to_initial_pose"]["proposal_id"] == "home"
    assert projection["artifact_refs"][0]["artifact_id"] == "a"


@pytest.mark.parametrize("payload", ["x" * 8100, "货架" * 4100, '\\"' * 4100])
def test_oversize_is_explicit_nonretryable_error_never_partial_json(payload):
    with pytest.raises(ToolBridgeError) as caught:
        _envelope_result(
            request(),
            envelope({"candidates": payload, "return_to_initial_pose": {"proposal_id": "home"}}),
        )
    assert caught.value.code == "CAPABILITY_OUTPUT_TOO_LARGE"
    assert caught.value.retryable is False
    assert "filtered/paginated" in str(caught.value)
    assert "No partial JSON" in str(caught.value)


def test_reference_overflow_also_refuses_complete_success():
    with pytest.raises(ToolBridgeError, match="limit 8000"):
        _envelope_result(
            request(), envelope({"ok": True}, [{"artifact_id": "a", "path": "x" * 8100}])
        )


def test_exact_limit_is_complete_and_next_character_is_refused():
    empty = envelope({"payload": ""})
    overhead = len(_envelope_result(request(), empty).summary)
    fit = envelope({"payload": "a" * (8000 - overhead)})
    result = _envelope_result(request(), fit)
    assert len(result.summary) == 8000 and json.loads(result.summary)["value"] == fit.value
    with pytest.raises(ToolBridgeError):
        _envelope_result(request(), envelope({"payload": "a" * (8001 - overhead)}))
