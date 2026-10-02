import copy

import numpy as np
import pytest

from rosclaw.growth.anchor_kernel import AnchorKernelGuard, _hash


@pytest.mark.parametrize("accelerated", [False, True])
def test_full_span_does_not_remove_novel_state_plasticity(accelerated):
    guard = AnchorKernelGuard(np.eye(5), bandwidth=0.2, accelerated=accelerated)
    for anchor in np.eye(5):
        assert guard.gate(anchor) == 0.0
        assert guard.gate(anchor) * 1e100 == 0.0
    assert 0 < guard.gate(np.ones(5)) <= 1
    assert 0 < guard.gate(np.eye(5)[0] + 0.01) < guard.gate(np.ones(5))


def test_reloading_and_external_mutation_preserve_the_frozen_metric():
    anchors = np.eye(3)
    guard = AnchorKernelGuard(anchors, bandwidth=0.1)
    anchors[:] = 999
    restored = AnchorKernelGuard.from_dict(guard.to_dict())
    assert np.array_equal(restored.gates(np.eye(3)), np.zeros(3))
    assert np.array_equal(restored.gates([[2, 3, 4]]), guard.gates([[2, 3, 4]]))


@pytest.mark.parametrize("field", ["hardware_authorized", "promotion_authorized"])
def test_resealed_authority_forgery_is_rejected(field):
    value = copy.deepcopy(AnchorKernelGuard([[0, 1]], bandwidth=0.1).to_dict())
    value[field] = True
    value["guard_hash"] = _hash({k: v for k, v in value.items() if k != "guard_hash"})
    with pytest.raises(ValueError):
        AnchorKernelGuard.from_dict(value)


@pytest.mark.parametrize(
    "anchors,bandwidth",
    [([], 0.1), ([[float("nan")]], 0.1), ([[0]], 0), ([[0]], True), (np.zeros((1, 513)), 0.1)],
)
def test_invalid_bank_or_metric_fails_closed(anchors, bandwidth):
    with pytest.raises(ValueError):
        AnchorKernelGuard(anchors, bandwidth=bandwidth)


def test_invalid_query_and_unbounded_batch_fail_closed():
    guard = AnchorKernelGuard([[0, 1]], bandwidth=0.1)
    for values in ([0], [0, float("inf")], [0, 1e7]):
        with pytest.raises(ValueError):
            guard.gate(values)
    with pytest.raises(ValueError):
        guard.gates([])
