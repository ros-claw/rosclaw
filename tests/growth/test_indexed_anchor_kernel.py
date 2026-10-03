import builtins
import copy

import numpy as np
import pytest

from rosclaw.growth.anchor_kernel import AnchorKernelGuard
from rosclaw.growth.indexed_anchor_kernel import IndexedAnchorKernelGuard


@pytest.mark.parametrize("dimension", [1, 12, 135, 512])
def test_private_index_preserves_logical_anchors_contract_and_exact_gates(dimension):
    rng = np.random.default_rng(433)
    anchors = np.repeat(rng.normal(size=(24, dimension)), 4, axis=0)
    fast = IndexedAnchorKernelGuard(anchors, bandwidth=0.1)
    reference = AnchorKernelGuard(anchors, bandwidth=0.1)
    before = copy.deepcopy(fast.to_dict())
    assert before == reference.to_dict()
    assert len(fast._anchors) == 96
    assert len(fast._first_logical_indices) == 24
    queries = np.concatenate((anchors[::4], anchors[::4] + 1e-5, rng.normal(size=(32, dimension))))
    np.testing.assert_array_equal(fast.gates(queries), reference.gates(queries))
    anchors[:] = 0
    assert fast.to_dict() == before
    assert not fast._anchors.flags.writeable
    assert not fast._first_logical_indices.flags.writeable
    restored = IndexedAnchorKernelGuard.from_dict(before)
    assert restored.to_dict() == before
    np.testing.assert_array_equal(restored.gates(queries), reference.gates(queries))


@pytest.mark.parametrize("epsilon", [0.0, 1e-15, 1e-13, 1e-9])
@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_coordinate_ties_keep_the_original_tree(epsilon, reverse):
    anchors = [[-1.0, 0.0]] * 4 + [[1.0, 0.0]] * 4
    if reverse:
        anchors.reverse()
    fast = IndexedAnchorKernelGuard(anchors, bandwidth=1)
    reference = AnchorKernelGuard(anchors, bandwidth=1)
    assert fast.gate([epsilon, 0.0]) == reference.gate([epsilon, 0.0])


def test_signed_zero_keys_do_not_rewrite_serialized_logical_rows():
    anchors = [[0.0, 0.0], [-0.0, 0.0], [0.0, -0.0]]
    fast = IndexedAnchorKernelGuard(anchors, bandwidth=1)
    reference = AnchorKernelGuard(anchors, bandwidth=1)
    assert len(fast._first_logical_indices) == 1
    assert fast.to_dict() == reference.to_dict()
    for query in ([0.0, 0.0], [1e-10, 0.0], [0.1, 0.1]):
        assert fast.gate(query) == reference.gate(query)


@pytest.mark.parametrize("accelerated", [False, True])
def test_optional_scipy_missing_and_explicit_reference_mode(monkeypatch, accelerated):
    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy.spatial":
            raise ImportError("optional dependency fixture")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    fast = IndexedAnchorKernelGuard([[0.0], [0.0], [2.0]], bandwidth=1, accelerated=accelerated)
    reference = AnchorKernelGuard([[0.0], [0.0], [2.0]], bandwidth=1, accelerated=False)
    assert fast._coordinate_tree is None
    np.testing.assert_array_equal(
        fast.gates([[0.0], [1.0], [2.0]]), reference.gates([[0.0], [1.0], [2.0]])
    )


@pytest.mark.parametrize("query", [[float("nan")], [float("inf")], [1e6 + 1], [0, 0]])
def test_acceleration_never_bypasses_input_validation(query):
    with pytest.raises(ValueError, match="observation"):
        IndexedAnchorKernelGuard([[0.0], [0.0]], bandwidth=1).gate(query)
