"""Mutation tests for external semantic oracle; no models or robot rollouts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.harnessbench.capability_cases import CASES
from benchmarks.harnessbench.capability_matrix import (
    CAPABILITY_TASKS,
    CASE_BY_ID,
    _equal,
    judge_capability,
)


def _stage(root: Path, task_id: str) -> dict:
    task = CAPABILITY_TASKS[task_id]
    content = task.staged_files["input.json"]
    (root / "input.json").write_text(content)
    answer = {
        "scope": "FIXTURE_ONLY",
        "result": copy.deepcopy(CASE_BY_ID[task_id][3]),
        "input_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "reason": "offline derivation",
    }
    (root / "answer.json").write_text(json.dumps(answer))
    return answer


@pytest.mark.parametrize("task_id", CAPABILITY_TASKS)
def test_good_witness_and_missing_semantic_field(tmp_path, task_id):
    answer = _stage(tmp_path, task_id)
    assert judge_capability(task_id, tmp_path)["verified_success"]
    del answer["result"][next(iter(answer["result"]))]
    (tmp_path / "answer.json").write_text(json.dumps(answer))
    assert not judge_capability(task_id, tmp_path)["verified_success"]


@pytest.mark.parametrize("task_id", CAPABILITY_TASKS)
def test_same_length_changed_fixture_rejected(tmp_path, task_id):
    _stage(tmp_path, task_id)
    content = (tmp_path / "input.json").read_bytes()
    (tmp_path / "input.json").write_bytes(content.replace(b"\n", b" ", 1))
    assert not judge_capability(task_id, tmp_path)["verified_success"]


@pytest.mark.parametrize("mutation", ["scope", "hash", "nan", "extra", "scalar_type", "reason"])
def test_adversarial_contract(tmp_path, mutation):
    answer = _stage(tmp_path, "C40")
    if mutation == "scope":
        answer["scope"] = "LIVE"
    elif mutation == "hash":
        answer["input_sha256"] = "0" * 64
    elif mutation == "nan":
        answer["result"]["u"] = float("nan")
    elif mutation == "extra":
        answer["result"]["success"] = True
    elif mutation == "scalar_type":
        answer["result"]["u"] = True
    else:
        answer["reason"] = ""
    (tmp_path / "answer.json").write_text(json.dumps(answer))
    assert not judge_capability("C40", tmp_path)["verified_success"]


@pytest.mark.parametrize("answer", ["[]", "null", "{", '{"result":false}'])
def test_corrupt_answers(tmp_path, answer):
    _stage(tmp_path, "C01")
    (tmp_path / "answer.json").write_text(answer)
    assert not judge_capability("C01", tmp_path)["verified_success"]


def test_numeric_shape_and_truth_distinction():
    assert not _equal([1], [1, 2])
    assert not _equal("1", 1)
    assert not _equal(1, True)
    assert not _equal(float("inf"), 1)
    assert _equal(0.666666666667, 2 / 3)


def test_sixty_distinct_directions_and_no_staged_answers():
    assert len(CASES) == len({case[0] for case in CASES}) == 60
    for task in CAPABILITY_TASKS.values():
        assert set(task.staged_files) == {"input.json"}
        assert task.oracle["kind"] == "offline_capability"
        assert "result JSON Schema" in task.prompt


def test_independent_analytic_witnesses():
    # Independently recompute selected nontrivial expected values, not production grader.
    values = {case[0]: (case[2], case[3]) for case in CASES}
    data, truth = values["tf_transform_direction"]
    p = [
        sum(row[j] * data["point_child"][j] for j in range(3)) + data["t"][i]
        for i, row in enumerate(data["R"])
    ]
    assert truth["point_parent"] == p
    data, truth = values["navigation_braking"]
    assert truth["distance_m"] == data["v"] ** 2 / (2 * data["a"]) + data["v"] * data["latency"]
    data, truth = values["resource_budget"]
    assert (
        truth["total_bytes"]
        == data["batch"]
        * data["channels"]
        * data["height"]
        * data["width"]
        * data["bytes_per_scalar"]
    )
    data, truth = values["artifact_sha_integrity"]
    assert truth["integrity_verified"] == (
        hashlib.sha256(bytes.fromhex(data["candidate_hex"])).hexdigest() == data["expected_sha256"]
    )


def test_typed_contract_no_reference_values_or_lengths():
    from benchmarks.harnessbench.capability_matrix import _schema

    assert _schema({"tau_B": [0, 0, 1]}) == {
        "type": "object",
        "properties": {"tau_B": {"type": "array", "items": {"type": "number"}}},
        "required": ["tau_B"],
        "additionalProperties": False,
    }
    assert _schema(True) == {"type": "boolean"}
    assert "minItems" not in str(_schema([1, 2, 3]))
    assert "const" not in str(_schema(123))
    assert "enum" not in str(_schema("reference_answer"))
