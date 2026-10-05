"""Canonical snapshots preserve data, never confer trust or authority."""

from dataclasses import FrozenInstanceError

import pytest

from rosclaw.growth.canonical_json_snapshot import CanonicalJSONSnapshot
from rosclaw.growth.shared_proof_payload import canonical_hash


def test_owned_exact_snapshot_and_hash():
    source = {"weights": [[0.0, -0.0, 1.25]], "文字": "球", "hardware_authorized": False}
    snapshot = CanonicalJSONSnapshot(source)
    assert snapshot.verify() == canonical_hash(source)
    result = snapshot.restore()
    assert result == source
    source["weights"][0][2] = 9
    result["weights"][0][2] = 8
    assert snapshot.restore()["weights"][0][2] == 1.25
    assert snapshot.restore()["hardware_authorized"] is False
    with pytest.raises(FrozenInstanceError):
        snapshot.content_hash = "changed"


@pytest.mark.parametrize(
    "value", [[], None, {1: "bad"}, {"x": (1, 2)}, {"x": float("nan")}, {"x": float("inf")}]
)
def test_invalid_or_lossy_input(value):
    with pytest.raises((ValueError, TypeError)):
        CanonicalJSONSnapshot(value)


def test_private_corruption_fails_closed():
    snapshot = CanonicalJSONSnapshot({"x": 1})
    object.__setattr__(snapshot, "_data", b'{"x":2}')
    for action in (snapshot.verify, snapshot.restore):
        with pytest.raises(ValueError, match="snapshot changed"):
            action()
