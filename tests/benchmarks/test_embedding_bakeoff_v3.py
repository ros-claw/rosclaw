from __future__ import annotations

import pytest

from benchmarks.memory.generate_dataset_v3 import (
    _label_for,
    build_queries,
    validate_benchmark_shape,
    validate_queries,
)


def _failure(record_id: str, body_id: str, *, cjk: bool = True) -> dict:
    return {
        "id": record_id,
        "memory_type": "failure",
        "body_id": body_id,
        "gesture_name": "scissors",
        "failure_type": "joint_not_reached",
        "document": "剪刀未到位" if cjk else "scissors joint not reached",
        "title": "failure",
        "doc_is_cjk": cjk,
    }


def test_structured_labels_exclude_wrong_body() -> None:
    left = _failure("left", "rh56_left_01")
    right = _failure("right", "rh56_right_01")

    labels = _label_for(left, [left, right])

    assert labels == {"left": 3}


def test_query_validation_rejects_same_body_hard_negative() -> None:
    left_a = _failure("left-a", "rh56_left_01")
    left_b = _failure("left-b", "rh56_left_01")
    query = {
        "id": "q1",
        "kind": "hard_negative_body",
        "source_memory": "left-a",
        "labels": {"left-a": 3},
        "forbidden": ["left-b"],
    }

    with pytest.raises(ValueError, match="not from the opposite body"):
        validate_queries([left_a, left_b], [query])


def test_query_validation_rejects_unknown_forbidden_memory() -> None:
    left = _failure("left", "rh56_left_01")
    query = {
        "id": "q1",
        "kind": "hard_negative_body",
        "source_memory": "left",
        "labels": {"left": 3},
        "forbidden": ["missing"],
    }

    with pytest.raises(ValueError, match="unknown forbidden"):
        validate_queries([left], [query])


def test_benchmark_shape_rejects_incomplete_lane_distribution() -> None:
    with pytest.raises(ValueError, match="lane distribution"):
        validate_benchmark_shape([{"kind": "cjk_to_cjk"}])


def test_generator_refuses_to_mislabel_cjk_rows_as_english_lane() -> None:
    corpus = [_failure(f"left-{index}", "rh56_left_01") for index in range(60)]

    with pytest.raises(ValueError, match="cjk_to_en.*found 0"):
        build_queries(corpus)
