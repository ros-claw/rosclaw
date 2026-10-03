import builtins
import copy

import numpy as np
import pytest

from rosclaw.growth.anchor_output_memory import AnchorOutputMemory
from rosclaw.growth.indexed_anchor_output_memory import IndexedAnchorOutputMemory

IDENTITY = "sha256:" + "a" * 64
PARENT = "sha256:" + "b" * 64
EVIDENCE = "sha256:" + "c" * 64


def make(cls, x, y, *, accelerated=True):
    return cls(
        x,
        y,
        bandwidth=0.1,
        encoder_hash=IDENTITY,
        parent_policy_hash=PARENT,
        evidence_hash=EVIDENCE,
        accelerated=accelerated,
    )


@pytest.mark.parametrize("dimension", [1, 12, 135, 512])
def test_exact_duplicates_preserve_all_logical_rows_hash_and_blended_outputs(dimension):
    rng = np.random.default_rng(415)
    x = rng.normal(size=(32, dimension))
    y = rng.normal(size=(32, 12))
    x, y = np.repeat(x, 3, axis=0), np.repeat(y, 3, axis=0)
    fast, old = make(IndexedAnchorOutputMemory, x, y), make(AnchorOutputMemory, x, y)
    before = copy.deepcopy(fast.to_dict())
    assert fast.to_dict() == old.to_dict()
    assert len(fast._observations) == 96
    assert len(fast._first_logical_indices) == 32
    for query in np.concatenate((x[::3], x[::3] + 0.001, rng.normal(size=(50, dimension)))):
        assert fast._nearest(query) == old._nearest(query)
        np.testing.assert_array_equal(
            fast.blend(query, np.ones(12), encoder_hash=IDENTITY),
            old.blend(query, np.ones(12), encoder_hash=IDENTITY),
        )
    assert fast.to_dict() == before
    assert not fast._first_logical_indices.flags.writeable
    assert not fast._observations.flags.writeable
    assert not fast._predictions.flags.writeable
    assert IndexedAnchorOutputMemory.from_dict(before).to_dict() == before


@pytest.mark.parametrize("epsilon", [0.0, 1e-15, 1e-13, 1e-9])
@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_coordinate_ties_and_near_ties_preserve_original_lowest_index(epsilon, reverse):
    x = [[-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    y = [[1.0], [1.0], [2.0], [2.0]]
    if reverse:
        x, y = x[::-1], y[::-1]
    fast = make(IndexedAnchorOutputMemory, x, y)
    reference = make(AnchorOutputMemory, x, y, accelerated=False)
    query = np.array([epsilon, 0.0])
    assert fast._nearest(query) == reference._nearest(query)
    np.testing.assert_array_equal(
        fast.blend(query, [9.0], encoder_hash=IDENTITY),
        reference.blend(query, [9.0], encoder_hash=IDENTITY),
    )


def test_signed_zero_all_duplicates_and_conflicting_outputs():
    x = [[0.0, 0.0], [-0.0, 0.0], [0.0, -0.0]]
    fast = make(IndexedAnchorOutputMemory, x, [[1.0]] * 3)
    assert len(fast._first_logical_indices) == 1
    assert fast._nearest(np.array([123.0, 456.0])) == 0
    with pytest.raises(ValueError, match="causal context"):
        make(IndexedAnchorOutputMemory, x, [[1.0], [2.0], [1.0]])


def test_optional_scipy_missing_retains_original_numpy_path(monkeypatch):
    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy.spatial":
            raise ImportError("fixture optional SciPy unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    fast = make(IndexedAnchorOutputMemory, [[0.0], [0.0], [2.0]], [[1.0], [1.0], [3.0]])
    assert fast._tree is None
    assert fast._nearest(np.array([1.0])) == 0
    assert fast._nearest(np.array([2.0])) == 2


@pytest.mark.parametrize("fault", ["hash", "authority", "encoder", "nan", "proposal"])
def test_original_integrity_and_prediction_checks_are_not_bypassed(fault):
    memory = make(IndexedAnchorOutputMemory, [[0.0], [0.0]], [[1.0], [1.0]])
    blob = memory.to_dict()
    if fault == "hash":
        blob["observations"][0][0] = 2
        with pytest.raises(ValueError, match="unsealed"):
            IndexedAnchorOutputMemory.from_dict(blob)
    elif fault == "authority":
        from rosclaw.growth.anchor_kernel import _hash

        blob["hardware_authorized"] = True
        blob["memory_hash"] = _hash({k: v for k, v in blob.items() if k != "memory_hash"})
        with pytest.raises(ValueError, match="authority"):
            IndexedAnchorOutputMemory.from_dict(blob)
    else:
        with pytest.raises(ValueError):
            memory.blend(
                [float("nan") if fault == "nan" else 0.0],
                [float("nan") if fault == "proposal" else 0.0],
                encoder_hash=PARENT if fault == "encoder" else IDENTITY,
            )


def test_extension_is_logical_and_requires_explicit_recompilation():
    old = make(IndexedAnchorOutputMemory, [[0.0], [0.0]], [[1.0], [1.0]])
    before = old.to_dict()
    new = old.extend([[2.0]], [[3.0]], parent_policy_hash=PARENT, evidence_hash=EVIDENCE)
    assert type(new) is AnchorOutputMemory
    assert old.to_dict() == before
    indexed = IndexedAnchorOutputMemory.from_dict(new.to_dict())
    assert indexed.to_dict() == new.to_dict()
