import copy

import numpy as np
import pytest

from rosclaw.growth.domain_anchor_bank import (
    DomainAnchorGuard,
    build_domain_anchor_bank,
    validate_domain_anchor_bank,
)


def domains():
    return [
        {
            "domain_id": name,
            "source_evidence_hash": "sha256:" + digit * 64,
            "context_ids": ["same-context"],
            "context_rows": [2],
            "observations": [[0.0, 1.0], [value, 2.0]],
        }
        for name, digit, value in (("cpu", "a", 1.0), ("gpu", "b", 3.0))
    ]


def bank(values=None):
    return build_domain_anchor_bank(
        domains() if values is None else values,
        parent_hash="sha256:" + "c" * 64,
        encoder_hash="sha256:" + "d" * 64,
    )


def test_all_domains_and_duplicates_are_retained_with_owned_frozen_guard():
    values = domains()
    encoded = bank(values)
    guard = DomainAnchorGuard(encoded)
    assert encoded["row_count"] == 4
    assert encoded["context_count"] == 2
    assert np.array_equal(guard.gates([[0, 1], [1, 2], [3, 2]]), np.zeros(3))
    assert guard.gate([8, 8]) > 0
    values[0]["observations"][0][0] = 99
    encoded["domains"][0]["observations"][0][0] = 88
    returned = guard.bank()
    returned["domains"][0]["observations"][0][0] = 77
    assert guard.gate([0, 1]) == 0
    validate_domain_anchor_bank(guard.bank())


@pytest.mark.parametrize(
    "key,value",
    [
        ("hardware_authorized", True),
        ("physical_batch_verified", True),
        ("row_count", 3),
        ("all_declared_rows_retained", False),
        ("dimension", 1),
        ("parent_hash", "unknown"),
    ],
)
def test_tampering_rejected(key, value):
    encoded = bank()
    encoded[key] = value
    with pytest.raises(ValueError):
        DomainAnchorGuard(encoded)


@pytest.mark.parametrize(
    "mutation", ["duplicate-domain", "missing-row", "nonfinite", "boolean", "duplicate-context"]
)
def test_invalid_declarations_rejected_without_silent_repairs(mutation):
    values = domains()
    if mutation == "duplicate-domain":
        values[1]["domain_id"] = values[0]["domain_id"]
    elif mutation == "missing-row":
        values[0]["observations"].pop()
    elif mutation == "nonfinite":
        values[0]["observations"][0][0] = np.nan
    elif mutation == "boolean":
        values[0]["observations"] = [[True, False], [False, True]]
    else:
        values[0]["context_ids"] = ["same-context", "same-context"]
        values[0]["context_rows"] = [1, 1]
    with pytest.raises(ValueError):
        bank(values)


def test_cross_domain_order_is_bound_and_capacity_is_not_evicted():
    values = domains()
    assert bank(values)["bank_hash"] != bank(list(reversed(values)))["bank_hash"]
    huge = copy.deepcopy(values)
    huge[0]["context_rows"] = [32768]
    huge[0]["observations"] = np.zeros((32768, 2)).tolist()
    with pytest.raises(ValueError, match="capacity"):
        bank(huge)
