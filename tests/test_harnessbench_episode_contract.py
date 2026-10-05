"""Regressions for real pre-runtime case/nonce template mismatches."""

import copy
import json

import pytest

from benchmarks.harnessbench.episode_contract import (
    EpisodeContractError,
    main,
    validate_episode_contract,
)


def fixture():
    spec = {"source": "7c56dbea", "nonce": "528ace98", "case_id": "FILEUX_V4_528ace98"}
    schema = {
        "required": [*spec, "authorized_phase"],
        "properties": {key: {"const": value} for key, value in spec.items()},
    }
    schema["properties"]["authorized_phase"] = {"const": "FILEUX_V4"}
    go = {**spec, "authorized_phase": "FILEUX_V4"}
    prompt = "Deliver case_id FILEUX_V4_528ace98, runtime_nonce 528ace98."
    return spec, schema, prompt, go


def test_correct_explicit_text_and_json_bindings():
    spec, schema, prompt, go = fixture()
    assert validate_episode_contract(spec, schema, prompt, go) == go
    assert (
        validate_episode_contract(
            spec, schema, json.dumps({"case_id": spec["case_id"], "runtime_nonce": spec["nonce"]})
        )["case_id"]
        == spec["case_id"]
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "case_id FILEUX_528ace98 runtime_nonce 528ace98",  # actual V4 replacement-order bug
        "case_id FILEUX_V4_528ace98_old runtime_nonce 528ace98",  # prefix is not identity
        "case_id FILEUX_V4_528ace98.old runtime_nonce 528ace98",
        "case_id FILEUX_V4_528ace98 runtime_nonce stale",
        "case_id FILEUX_V4_528ace98",  # nonce hidden inside case ID is insufficient
        "FILEUX_V4_528ace98 528ace98",  # unlabelled mentions are insufficient
        "case_id FILEUX_V4_528ace98 runtime_nonce 528ace98 case_id FILEUX_528ace98",
    ],
)
def test_stale_or_conflicting_prompt_cannot_pass(prompt):
    spec, schema, _, go = fixture()
    with pytest.raises(EpisodeContractError, match="PROMPT_BINDING_MISMATCH"):
        validate_episode_contract(spec, schema, prompt, go)


@pytest.mark.parametrize("key", ["source", "nonce", "case_id", "authorized_phase"])
def test_schema_and_go_bindings_cannot_drift(key):
    spec, schema, prompt, go = fixture()
    altered = copy.deepcopy(schema)
    altered["properties"][key]["const"] = "old"
    if key == "authorized_phase":
        spec[key] = "FILEUX_V4"
    with pytest.raises(EpisodeContractError, match="SPEC_SCHEMA_MISMATCH"):
        validate_episode_contract(spec, altered, prompt, go)
    go[key] = "old"
    with pytest.raises(EpisodeContractError, match="SPEC_GO_MISMATCH"):
        validate_episode_contract(spec, schema, prompt, go)


def test_missing_schema_requirement_or_nonstring_binding_is_refused():
    spec, schema, prompt, _ = fixture()
    schema["required"].remove("nonce")
    with pytest.raises(EpisodeContractError, match="SPEC_SCHEMA_MISMATCH:nonce"):
        validate_episode_contract(spec, schema, prompt)
    spec["nonce"] = True
    with pytest.raises(EpisodeContractError, match="INVALID_BINDING:nonce"):
        validate_episode_contract(spec, schema, prompt)


def test_cli_checks_real_files_without_issuing_go(tmp_path, capsys):
    spec, schema, prompt, _ = fixture()
    paths = {key: tmp_path / key for key in ("spec", "schema", "prompt")}
    paths["spec"].write_text(json.dumps(spec))
    paths["schema"].write_text(json.dumps(schema))
    paths["prompt"].write_text(prompt)
    argv = [value for key, path in paths.items() for value in (f"--{key}", str(path))]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["authorized"] is False
    paths["prompt"].write_text(prompt.replace("FILEUX_V4_", "FILEUX_"))
    assert main(argv) == 2
    rejected = json.loads(capsys.readouterr().out)
    assert rejected["authorized"] is False
    assert rejected["code"] == "PROMPT_BINDING_MISMATCH:case_id"
