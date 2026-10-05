"""Read-only admission of public prompt inputs against independent authority."""

import json
from collections.abc import Mapping

PROMPT_BINDINGS_MARKER = "ROSCLAW_EPISODE_INPUT_JSON="


class EpisodePromptBindingsError(ValueError):
    """The public instructions do not bind to the admitted runtime inputs."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EpisodePromptBindingsError("DUPLICATE_PROMPT_BINDING_KEY")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise EpisodePromptBindingsError("NONFINITE_PROMPT_BINDING:" + value)


def validate_episode_prompt_bindings(prompt: str, expected: Mapping[str, object]) -> None:
    """Require one complete typed public JSON binding, before clock or worker.

    Expected values come from the admitted spec, never from the prompt itself.
    This checks the marked inputs, not arbitrary prose or execution permission.
    Callers must generate endpoint-bearing prose from those same inputs.
    """
    if not expected or any(not isinstance(key, str) or not key for key in expected):
        raise EpisodePromptBindingsError("INDEPENDENT_PROMPT_BINDINGS_REQUIRED")
    rows = [line for line in prompt.splitlines() if line.startswith(PROMPT_BINDINGS_MARKER)]
    if len(rows) != 1:
        raise EpisodePromptBindingsError("EXACTLY_ONE_PUBLIC_PROMPT_BINDING_REQUIRED")
    try:
        actual = json.loads(
            rows[0][len(PROMPT_BINDINGS_MARKER) :],
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
        wanted = json.dumps(dict(expected), allow_nan=False, sort_keys=True, separators=(",", ":"))
        found = json.dumps(actual, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        raise EpisodePromptBindingsError("INVALID_PUBLIC_PROMPT_BINDINGS") from exc
    if not isinstance(actual, dict) or wanted != found:
        raise EpisodePromptBindingsError("PUBLIC_PROMPT_BINDING_MISMATCH")
