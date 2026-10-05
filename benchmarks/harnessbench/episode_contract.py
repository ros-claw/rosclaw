"""Offline binding checks for prepared native episodes, before any unique GO.

Prompts must explicitly assign case_id and runtime_nonce (plain text or JSON).
This checks preparation metadata only, never task success or runtime authority.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class EpisodeContractError(ValueError):
    """A prepared episode must not start with conflicting bindings."""


def _binding(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item or not re.fullmatch(r"[A-Za-z0-9_-]+", item):
        raise EpisodeContractError(f"INVALID_BINDING:{key}")
    return item


def _prompt_values(prompt: str, field: str) -> list[str]:
    # Do not use substring membership: a correct ID embedded in a stale or
    # conflicting assignment must not rescue a different delivery directive.
    pattern = (
        rf"""(?<![\w])(?:["'`]?{field}["'`]?)\s*(?:[:=]\s*|\s+)["'`]?"""
        r"""([A-Za-z0-9_-]+)(?=$|[\s,;:'"`}\]]|\.(?=\s|$))"""
    )
    return re.findall(pattern, prompt)


def validate_episode_contract(
    spec: Mapping[str, Any],
    schema: Mapping[str, Any],
    prompt: str,
    go: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Check explicit prompt/spec/schema/optional-GO identifiers offline.

    Repeated prompt assignments are allowed only when every value agrees.
    Schema validation, hashes, process ownership, budgets and physical checks
    remain separate required gates; this function issues no authorization.
    """
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise EpisodeContractError("INVALID_GO_SCHEMA")
    bindings = {key: _binding(spec, key) for key in ("source", "nonce", "case_id")}
    for key, expected in bindings.items():
        rule = properties.get(key)
        if key not in required or not isinstance(rule, Mapping) or rule.get("const") != expected:
            raise EpisodeContractError(f"SPEC_SCHEMA_MISMATCH:{key}")
        if go is not None and go.get(key) != expected:
            raise EpisodeContractError(f"SPEC_GO_MISMATCH:{key}")
    phase_rule = properties.get("authorized_phase")
    if "authorized_phase" not in required or not isinstance(phase_rule, Mapping):
        raise EpisodeContractError("INVALID_AUTHORIZED_PHASE")
    phase = _binding(phase_rule, "const")
    if "authorized_phase" in spec and spec["authorized_phase"] != phase:
        raise EpisodeContractError("SPEC_SCHEMA_MISMATCH:authorized_phase")
    if go is not None and go.get("authorized_phase") != phase:
        raise EpisodeContractError("SPEC_GO_MISMATCH:authorized_phase")
    for field, key in (("case_id", "case_id"), ("runtime_nonce", "nonce")):
        values = _prompt_values(prompt, field)
        if not values or any(value != bindings[key] for value in values):
            raise EpisodeContractError(f"PROMPT_BINDING_MISMATCH:{field}")
    return {**bindings, "authorized_phase": phase}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("spec", "schema", "prompt"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--go", type=Path)
    args = parser.parse_args(argv)
    try:
        bindings = validate_episode_contract(
            json.loads(args.spec.read_text()),
            json.loads(args.schema.read_text()),
            args.prompt.read_text(),
            json.loads(args.go.read_text()) if args.go else None,
        )
    except EpisodeContractError as exc:
        print(
            json.dumps(
                {"status": "REFUSED_CONTRACT_BINDINGS", "code": str(exc), "authorized": False}
            )
        )
        return 2
    except (OSError, ValueError, TypeError, AttributeError):
        # No raw spec/error dump: prepared inputs may contain private paths.
        print(
            json.dumps(
                {
                    "status": "REFUSED_CONTRACT_BINDINGS",
                    "code": "INVALID_CONTRACT_INPUT",
                    "authorized": False,
                }
            )
        )
        return 2
    print(json.dumps({"status": "PASS_METADATA_BINDINGS_ONLY", "authorized": False, **bindings}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
