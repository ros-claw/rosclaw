import copy

import numpy as np
import pytest

from rosclaw.growth.anchor_consolidation import consolidate_unique
from rosclaw.growth.anchor_output_memory import AnchorOutputMemory


def memory(x=None, y=None):
    return AnchorOutputMemory(
        [[0.0, 1.0], [2.0, 3.0]] if x is None else x,
        [[4.0], [5.0]] if y is None else y,
        bandwidth=1e-4,
        encoder_hash="sha256:" + "a" * 64,
        parent_policy_hash="sha256:" + "b" * 64,
        evidence_hash="sha256:" + "c" * 64,
    )


def consolidate(old, x, y):
    return consolidate_unique(
        old, x, y, parent_policy_hash="sha256:" + "d" * 64, evidence_hash="sha256:" + "e" * 64
    )


def test_repeated_samples_keep_complete_mapping_and_preserve_parent():
    old = memory()
    before = copy.deepcopy(old.to_dict())
    result, proof = consolidate(old, [[-0.0, 1.0], [6.0, 7.0], [6.0, 7.0]], [[4.0], [8.0], [8.0]])
    assert old.to_dict() == before
    assert proof["sample_to_memory_row"] == [0, 2, 2]
    assert proof["exact_duplicate_samples"] == 2
    assert proof["appended_unique_rows"] == 1
    assert result.to_dict()["predecessor_memory_hash"] == before["memory_hash"]
    for x, y in zip([[0.0, 1.0], [2.0, 3.0], [6.0, 7.0]], [[4.0], [5.0], [8.0]], strict=True):
        assert np.array_equal(result.blend(x, [99.0], encoder_hash=result.encoder_hash), y)
    assert proof["hardware_authorized"] is False
    assert proof["evidence_independently_verified"] is False


def test_all_repeats_can_rebind_current_parent_without_new_rows():
    old = memory()
    result, proof = consolidate(old, [[0.0, 1.0]], [[4.0]])
    assert proof["appended_unique_rows"] == 0
    assert result.to_dict()["observations"] == old.to_dict()["observations"]
    assert result.parent_policy_hash != old.parent_policy_hash


def test_conflicting_prediction_never_overwrites_history():
    old = memory()
    before = old.to_dict()
    with pytest.raises(ValueError, match="conflicting"):
        consolidate(old, [[0.0, 1.0]], [[4.0 + 1e-14]])
    assert old.to_dict() == before


def test_almost_equal_states_are_distinct_not_rounded():
    result, proof = consolidate(memory(), [[1e-14, 1.0]], [[8.0]])
    assert proof["appended_unique_rows"] == 1
    assert len(result.to_dict()["observations"]) == 3


@pytest.mark.parametrize(
    "x,y",
    [([[0.0]], [[4.0]]), ([[0.0, 1.0]], [[4.0, 5.0]]), ([[float("nan"), 1.0]], [[4.0]]), ([], [])],
)
def test_bad_dimension_or_nonfinite_inputs_rejected(x, y):
    with pytest.raises(ValueError):
        consolidate(memory(), x, y)


def test_capacity_is_not_silently_maintained_by_dropping_old_states():
    old = memory(np.arange(32768.0)[:, None], np.zeros((32768, 1)))
    with pytest.raises(ValueError):
        consolidate(old, [[32768.0]], [[0.0]])


def test_full_capacity_still_accepts_exact_redundant_sample():
    old = memory(np.arange(32768.0)[:, None], np.zeros((32768, 1)))
    result, proof = consolidate(old, [[123.0]], [[0.0]])
    assert len(result.to_dict()["observations"]) == 32768
    assert proof["sample_to_memory_row"] == [123]
