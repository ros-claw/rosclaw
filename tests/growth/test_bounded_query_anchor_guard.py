import numpy as np
import pytest

from rosclaw.growth.anchor_kernel import AnchorKernelGuard
from rosclaw.growth.bounded_query_anchor_guard import (
    BoundedQueryAnchorGuard,
    BoundedQueryDomainAnchorGuard,
)
from rosclaw.growth.domain_anchor_bank import DomainAnchorGuard, build_domain_anchor_bank


@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize("bandwidth", [1e-4, 0.1, 1000.0])
def test_scalar_law_parity_at_anchors_nearby_saturation_and_ties(accelerated, bandwidth):
    rng = np.random.default_rng(483)
    anchors = rng.normal(size=(64, 135))
    anchors = np.vstack((anchors, anchors[:3], np.ones((1, 135)) * 999000))
    reference = AnchorKernelGuard(anchors, bandwidth=bandwidth, accelerated=accelerated)
    fast = BoundedQueryAnchorGuard(anchors, bandwidth=bandwidth, accelerated=accelerated)
    assert fast.to_dict() == reference.to_dict()
    restored = BoundedQueryAnchorGuard.from_dict(reference.to_dict(), accelerated=accelerated)
    assert isinstance(restored, BoundedQueryAnchorGuard)
    queries = [*anchors, *rng.normal(size=(30, 135))]
    for scale in (0, 1e-11, 0.1, 1, 8, 12, 15.999999, 16, 16.000001, 100):
        query = anchors[0].copy()
        query[0] += bandwidth * scale
        queries.append(query)
    queries += [(anchors[0] + anchors[1]) / 2, np.full(135, -999000.0)]
    np.testing.assert_array_equal(fast.gates(queries), reference.gates(queries))
    np.testing.assert_array_equal(restored.gates(queries), reference.gates(queries))


@pytest.mark.parametrize("query", [[0], [0, np.nan], [0, np.inf], [0, 1e7]])
def test_invalid_query_fails_closed(query):
    guard = BoundedQueryAnchorGuard([[0, 1]], bandwidth=1e-4)
    with pytest.raises(ValueError):
        guard.gate(query)


def test_all_rows_retained_and_extreme_small_difference_not_pruned():
    anchors = np.full((100, 512), 999999.0)
    reference = AnchorKernelGuard(anchors, bandwidth=1e-4)
    fast = BoundedQueryAnchorGuard(anchors, bandwidth=1e-4)
    query = anchors[0].copy()
    query[0] = np.nextafter(query[0], np.inf)
    assert fast.gate(query) == reference.gate(query)
    assert len(fast.to_dict()["anchors"]) == 100
    assert fast.to_dict()["promotion_authorized"] is False


def test_multidomain_bank_api_and_duplicate_provenance_are_unchanged():
    bank = build_domain_anchor_bank(
        [
            {
                "domain_id": d,
                "source_evidence_hash": "sha256:" + "a" * 64,
                "context_ids": ["same-context"],
                "context_rows": [2],
                "observations": [[0, 1], [0, 1]],
            }
            for d in ("cpu", "gpu")
        ],
        parent_hash="sha256:" + "b" * 64,
        encoder_hash="sha256:" + "c" * 64,
    )
    reference = DomainAnchorGuard(bank)
    fast = BoundedQueryDomainAnchorGuard(bank)
    assert fast.bank() == reference.bank() == bank
    assert fast.bank_hash == reference.bank_hash
    assert fast.to_dict() == reference.to_dict()
    np.testing.assert_array_equal(
        fast.gates([[0, 1], [0.1, 1], [1000, 1000]]),
        reference.gates([[0, 1], [0.1, 1], [1000, 1000]]),
    )
    fast._tree = None
    reference._tree = None
    np.testing.assert_array_equal(
        fast.gates([[0, 1], [0.1, 1], [1000, 1000]]),
        reference.gates([[0, 1], [0.1, 1], [1000, 1000]]),
    )
