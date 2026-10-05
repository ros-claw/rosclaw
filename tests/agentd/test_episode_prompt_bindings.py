import json

import pytest

from rosclaw.agentd.episode_prompt_bindings import (
    PROMPT_BINDINGS_MARKER,
    EpisodePromptBindingsError,
    validate_episode_prompt_bindings,
)

EXPECTED = {
    "case_id": "ACTION_current",
    "nonce": "current",
    "namespace": "/action_current",
    "action_name": "/action_current/finite_sequence",
    "domain": 185,
    "goals": [40, 5],
}


def prompt(values: object) -> str:
    return "Use the admitted public inputs below.\n" + PROMPT_BINDINGS_MARKER + json.dumps(values)


def test_current_typed_inputs_accept_with_different_json_order_and_spacing():
    validate_episode_prompt_bindings(prompt(dict(reversed(list(EXPECTED.items())))), EXPECTED)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("namespace", "/action_old"),
        ("action_name", "/action_old/finite_sequence"),
        ("nonce", "old"),
        ("domain", "185"),
        ("domain", True),
        ("goals", [40, 6]),
    ],
)
def test_stale_or_wrong_type_inputs_reject_before_runtime(field, value):
    with pytest.raises(EpisodePromptBindingsError, match="MISMATCH"):
        validate_episode_prompt_bindings(prompt({**EXPECTED, field: value}), EXPECTED)


@pytest.mark.parametrize(
    "text",
    [
        "Connect to /action_old; current case metadata only.",
        prompt(EXPECTED) + "\n" + prompt(EXPECTED),
        PROMPT_BINDINGS_MARKER + '{"domain":185,"domain":186}',
        PROMPT_BINDINGS_MARKER + '{"domain":NaN}',
        PROMPT_BINDINGS_MARKER + '{"nested":{"key":1,"key":2}}',
        prompt([]),
        prompt({}),
        prompt({**EXPECTED, "extra": 1}),
    ],
)
def test_incomplete_ambiguous_and_nonfinite_public_bindings_reject(text):
    with pytest.raises(EpisodePromptBindingsError):
        validate_episode_prompt_bindings(text, EXPECTED)


def test_independent_expected_bindings_required():
    with pytest.raises(EpisodePromptBindingsError):
        validate_episode_prompt_bindings(prompt({}), {})


def test_boolean_and_integer_cannot_alias_even_inside_nested_inputs():
    with pytest.raises(EpisodePromptBindingsError, match="MISMATCH"):
        validate_episode_prompt_bindings(
            prompt({"options": {"enabled": 1}}), {"options": {"enabled": True}}
        )
