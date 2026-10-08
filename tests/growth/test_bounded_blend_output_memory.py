import builtins

import numpy as np
import pytest

from rosclaw.growth.anchor_output_memory import AnchorOutputMemory
from rosclaw.growth.bounded_blend_output_memory import BoundedBlendOutputMemory
from tests.growth.test_indexed_anchor_output_memory import IDENTITY, make


@pytest.mark.parametrize("dimension", [1, 12, 135, 512])
def test_complete_bank_and_original_output_bits_retained(dimension):
    rng = np.random.default_rng(657)
    x = rng.normal(size=(12, dimension))
    y = rng.normal(size=(12, 12))
    x, y = np.repeat(x, 3, axis=0), np.repeat(y, 3, axis=0)
    fast, old = make(BoundedBlendOutputMemory, x, y), make(AnchorOutputMemory, x, y)
    before = fast.to_dict()
    assert before == old.to_dict()
    queries = np.concatenate((x, x + 1e-10, x + 1e-3, rng.normal(size=(20, dimension))))
    for query in queries:
        for proposal in (rng.normal(size=12), np.zeros(12), -np.zeros(12)):
            a = fast.blend(query, proposal, encoder_hash=IDENTITY)
            b = old.blend(query, proposal, encoder_hash=IDENTITY)
            assert np.array_equal(a.view(np.uint64), b.view(np.uint64))
    assert fast.to_dict() == before
    assert not fast._observations.flags.writeable
    assert BoundedBlendOutputMemory.from_dict(before).to_dict() == before


def test_saturated_nonzero_output_does_not_query_prediction_bank(monkeypatch):
    fast = make(BoundedBlendOutputMemory, [[0.0], [1.0]], [[-2.0], [2.0]])

    def forbidden(_):
        raise AssertionError("zero-weight reference must not be queried")

    monkeypatch.setattr(fast, "_nearest", forbidden)
    for value in (1.0, -1.0, np.nextafter(0.0, 1.0), np.nextafter(0.0, -1.0)):
        assert fast._guard.gate([100.0]) == 1.0
        result = fast.blend([100.0], [value], encoder_hash=IDENTITY)
        assert np.array_equal(result.view(np.uint64), np.array([value]).view(np.uint64))
        result[0] = 9
        assert fast.to_dict()["predictions"] == [[-2.0], [2.0]]


@pytest.mark.parametrize("proposal", [0.0, -0.0])
@pytest.mark.parametrize("reference", [0.0, -0.0, -2.0, 2.0])
def test_saturated_zero_preserves_original_signed_zero_bits(proposal, reference, monkeypatch):
    fast = make(BoundedBlendOutputMemory, [[0.0]], [[reference]])
    old = make(AnchorOutputMemory, [[0.0]], [[reference]])
    nearest = fast._nearest
    calls = []

    def record(query):
        calls.append(1)
        return nearest(query)

    monkeypatch.setattr(fast, "_nearest", record)
    a = fast.blend([100.0], [proposal], encoder_hash=IDENTITY)
    b = old.blend([100.0], [proposal], encoder_hash=IDENTITY)
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))
    assert calls == [1]


@pytest.mark.parametrize("distance", [0.0, 1e-10, 0.1, 0.84, 0.86, 0.88, 1.6])
def test_near_and_saturation_transition_matches_original_bits(distance):
    x, y = [[0.0], [10.0]], [[-2.0], [3.0]]
    fast, old = make(BoundedBlendOutputMemory, x, y), make(AnchorOutputMemory, x, y)
    a = fast.blend([distance], [0.713], encoder_hash=IDENTITY)
    b = old.blend([distance], [0.713], encoder_hash=IDENTITY)
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))


@pytest.mark.parametrize("fault", ["encoder", "nan_observation", "nan_proposal", "shape", "large"])
def test_checks_not_bypassed_on_saturated_path(fault):
    memory = make(BoundedBlendOutputMemory, [[0.0]], [[1.0]])
    with pytest.raises(ValueError):
        memory.blend(
            [float("nan") if fault == "nan_observation" else 100.0],
            [float("nan") if fault == "nan_proposal" else 1e7 if fault == "large" else 1.0]
            * (2 if fault == "shape" else 1),
            encoder_hash="wrong" if fault == "encoder" else IDENTITY,
        )


def test_optional_scipy_absent_and_logical_extension_preserved(monkeypatch):
    original = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy.spatial":
            raise ImportError("optional fixture")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    fast = make(BoundedBlendOutputMemory, [[0.0], [0.0]], [[1.0], [1.0]])
    old = make(AnchorOutputMemory, [[0.0], [0.0]], [[1.0], [1.0]])
    for query in ([0.0], [0.001], [100.0]):
        assert np.array_equal(
            fast.blend(query, [2.0], encoder_hash=IDENTITY).view(np.uint64),
            old.blend(query, [2.0], encoder_hash=IDENTITY).view(np.uint64),
        )
    extended = fast.extend(
        [[2.0]],
        [[3.0]],
        parent_policy_hash=old.parent_policy_hash,
        evidence_hash=old.evidence_hash,
    )
    assert type(extended) is AnchorOutputMemory
    assert len(fast.to_dict()["observations"]) == 2
