import copy

import pytest

from rosclaw.growth.frozen_payload_field import FrozenPayloadField
from rosclaw.growth.shared_proof_payload import MARKER, canonical_hash


def test_restore_has_full_canonical_identity_and_owns_every_returned_field():
    payload = {"weights": [1.0, -0.0], "label": "具身"}
    field = FrozenPayloadField(payload)
    envelope = field.envelope({"seed": 1, "flags": [False]}, "model")
    expected = {"model": copy.deepcopy(payload), "seed": 1, "flags": [False]}
    payload["weights"][0] = 99
    assert field.restore(envelope, "model") == expected
    restored = field.restore(envelope, "model")
    assert canonical_hash(restored) == envelope["logical_document_hash"]
    restored["model"]["weights"][0] = 77
    restored["flags"][0] = True
    assert field.restore(envelope, "model") == expected


def test_restore_rejects_resealed_foreign_payload_marker_or_document():
    field = FrozenPayloadField({"weights": [1.0]})
    original = field.envelope({"seed": 1}, "model")
    for key, value in (
        ("location", ["other"]),
        ("payload_hash", "sha256:" + "a" * 64),
        ("logical_document_hash", "sha256:" + "b" * 64),
        ("extra", False),
        ("stripped_document", {"seed": 1, "model": {MARKER: field.payload_hash, "extra": 0}}),
        ("stripped_document", {"seed": 2, "model": {MARKER: field.payload_hash}}),
    ):
        bad = copy.deepcopy(original)
        bad[key] = value
        bad["envelope_hash"] = canonical_hash(
            {k: v for k, v in bad.items() if k != "envelope_hash"}
        )
        with pytest.raises(ValueError):
            field.restore(bad, "model")
    field.payload_hash = "sha256:" + "c" * 64
    with pytest.raises(ValueError):
        field.restore(original, "model")


def test_restore_rejects_raw_documents_and_bad_seals():
    field = FrozenPayloadField({"weights": [1.0]})
    envelope = field.envelope({"seed": 1}, "model")
    for value in (None, [], {"model": {}}, dict(envelope, envelope_hash="wrong")):
        with pytest.raises(ValueError):
            field.restore(value, "model")
    for key in (True, "", "x" * 129):
        with pytest.raises(ValueError):
            field.restore(envelope, key)
