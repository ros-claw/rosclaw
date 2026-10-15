"""Protocol integrity tests; these are not model experiment results."""

import copy

import pytest

from benchmarks.ros_expert_ablation.protocol import (
    admit_memory,
    allocate,
    usage_from_events,
    validate_complete_results,
)


def test_exact_paired_randomized_allocation():
    rows = allocate()
    assert rows == allocate() and rows != allocate(1014)
    assert len(rows) == len({r["episode_id"] for r in rows}) == 54
    for offset in range(0, 54, 3):
        pair = rows[offset : offset + 3]
        assert len({(r["case"], r["seed"]) for r in pair}) == 1
        assert {r["arm"] for r in pair} == {"A", "B", "C"}


@pytest.mark.parametrize(
    "fault", ["same_case", "eval_run", "wrong_body", "unverified", "not_training"]
)
def test_memory_rejects_answers_and_provenance_mismatch(fault):
    record = {
        "independently_verified": True,
        "claim_scope": "SIMULATION_ONLY",
        "training_only": True,
        "fault_case_id": "scan_stale",
        "source_urdf_sha256": "body",
        "episode_sha256": "a" * 64,
        "world_runs": ["training"],
    }
    assert admit_memory(
        record, evaluation_case="map_to_odom_missing", evaluation_runs=["eval"], urdf_sha256="body"
    )
    if fault == "same_case":
        record["fault_case_id"] = "map_to_odom_missing"
    elif fault == "eval_run":
        record["world_runs"] = ["eval"]
    elif fault == "wrong_body":
        record["source_urdf_sha256"] = "other"
    elif fault == "unverified":
        record["independently_verified"] = False
    else:
        record["training_only"] = False
    assert not admit_memory(
        record, evaluation_case="map_to_odom_missing", evaluation_runs=["eval"], urdf_sha256="body"
    )


def test_missing_usage_is_not_zero():
    with pytest.raises(ValueError):
        usage_from_events([])
    with pytest.raises(ValueError):
        usage_from_events([{"type": "turn.completed", "usage": {}}])
    actual = {"input_tokens": 234, "cached_input_tokens": 100, "output_tokens": 27}
    assert usage_from_events([{"type": "turn.completed", "usage": actual}]) == actual


def test_duplicate_episode_cannot_replace_missing_episode():
    rows = allocate()
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="54 distinct"):
        validate_complete_results(rows)


def test_physicsless_success_is_rejected():
    rows = allocate()
    for r in rows:
        r.update(
            raw_model_calls=["original"],
            actual_usage={"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1},
            verified_success=False,
        )
    assert validate_complete_results(rows)
    rows[0]["verified_success"] = True
    with pytest.raises(ValueError, match="physics"):
        validate_complete_results(rows)
