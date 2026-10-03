import builtins
import copy

import numpy as np
import pytest

from rosclaw.growth.anchor_kernel import AnchorKernelGuard
from rosclaw.growth.indexed_anchor_kernel import IndexedAnchorKernelGuard
from rosclaw.growth.radius_indexed_anchor_kernel import RadiusIndexedAnchorKernelGuard


@pytest.fixture(params=[IndexedAnchorKernelGuard, RadiusIndexedAnchorKernelGuard])
def guard_class(request):
    return request.param


@pytest.mark.parametrize("dimension", [1, 12, 135, 512])
def test_private_index_preserves_logical_anchors_contract_and_exact_gates(dimension, guard_class):
    rng = np.random.default_rng(433)
    anchors = np.repeat(rng.normal(size=(24, dimension)), 4, axis=0)
    fast = guard_class(anchors, bandwidth=0.1)
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
    restored = guard_class.from_dict(before)
    assert restored.to_dict() == before
    np.testing.assert_array_equal(restored.gates(queries), reference.gates(queries))


@pytest.mark.parametrize("epsilon", [0.0, 1e-15, 1e-13, 1e-9])
@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_coordinate_ties_keep_the_original_tree(epsilon, reverse, guard_class):
    anchors = [[-1.0, 0.0]] * 4 + [[1.0, 0.0]] * 4
    if reverse:
        anchors.reverse()
    fast = guard_class(anchors, bandwidth=1)
    reference = AnchorKernelGuard(anchors, bandwidth=1)
    assert fast.gate([epsilon, 0.0]) == reference.gate([epsilon, 0.0])


def test_signed_zero_keys_do_not_rewrite_serialized_logical_rows(guard_class):
    anchors = [[0.0, 0.0], [-0.0, 0.0], [0.0, -0.0]]
    fast = guard_class(anchors, bandwidth=1)
    reference = AnchorKernelGuard(anchors, bandwidth=1)
    assert len(fast._first_logical_indices) == 1
    assert fast.to_dict() == reference.to_dict()
    for query in ([0.0, 0.0], [1e-10, 0.0], [0.1, 0.1]):
        assert fast.gate(query) == reference.gate(query)


@pytest.mark.parametrize("accelerated", [False, True])
def test_optional_scipy_missing_and_explicit_reference_mode(monkeypatch, accelerated, guard_class):
    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy.spatial":
            raise ImportError("optional dependency fixture")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    fast = guard_class([[0.0], [0.0], [2.0]], bandwidth=1, accelerated=accelerated)
    reference = AnchorKernelGuard([[0.0], [0.0], [2.0]], bandwidth=1, accelerated=False)
    assert fast._coordinate_tree is None
    np.testing.assert_array_equal(
        fast.gates([[0.0], [1.0], [2.0]]), reference.gates([[0.0], [1.0], [2.0]])
    )


@pytest.mark.parametrize("query", [[float("nan")], [float("inf")], [1e6 + 1], [0, 0]])
def test_acceleration_never_bypasses_input_validation(query, guard_class):
    with pytest.raises(ValueError, match="observation"):
        guard_class([[0.0], [0.0]], bandwidth=1).gate(query)


@pytest.mark.parametrize("bandwidth", [1e-4, 0.1, 1, 1000])
def test_radius_keeps_exact_zero_smooth_gate_and_saturation_boundaries(bandwidth):
    assert float(np.expm1(-128.0)) == -1.0
    anchors = np.array([[0.0], [0.0], [200 * bandwidth]])
    fast = RadiusIndexedAnchorKernelGuard(anchors, bandwidth=bandwidth)
    reference = AnchorKernelGuard(anchors, bandwidth=bandwidth)
    distances = np.concatenate(
        (
            [0, 1e-11, 1e-10],
            np.linspace(0, 20 * bandwidth, 2048),
            [
                np.nextafter(16 * bandwidth, 0),
                16 * bandwidth,
                np.nextafter(16 * bandwidth, float("inf")),
            ],
        )
    )
    queries = distances[:, None]
    np.testing.assert_array_equal(fast.gates(queries), reference.gates(queries))
    assert fast.to_dict() == reference.to_dict()


def test_radius_nearby_distinct_coordinates_preserve_reference_ties():
    anchors = [[-1e-5, 0.0], [-1e-5, 0.0], [1e-5, 0.0], [1e-5, 0.0]]
    fast = RadiusIndexedAnchorKernelGuard(anchors, bandwidth=1e-4)
    reference = AnchorKernelGuard(anchors, bandwidth=1e-4)
    queries = [[e, 0.0] for e in [0.0, 1e-20, 1e-18, 1e-15, -1e-15]]
    np.testing.assert_array_equal(fast.gates(queries), reference.gates(queries))
