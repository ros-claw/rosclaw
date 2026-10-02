import copy

import pytest

from rosclaw.growth.frozen_payload_field import FrozenPayloadField
from rosclaw.growth.shared_proof_payload import canonical_hash, detach_payload, restore_payload


def test_full_canonical_hash_and_envelope_are_identical_not_trimmed():
    payload = {"浮点": [0.0, -0.0, 1.234567891234e-20] * 100, "nested": {"Ω": "你好"}}
    cache = FrozenPayloadField(payload)
    for index in range(100):
        fields = {"seed": index, "schema": "fixture", "a": "𐀀", "z": {"flag": False}}
        document = {**fields, "mean_model": payload}
        assert cache.document_hash(fields, "mean_model") == canonical_hash(document)
        envelope = cache.envelope(fields, "mean_model")
        expected, _ = detach_payload(document, ("mean_model",))
        assert envelope == expected
        assert restore_payload(envelope, payload) == document


def test_cached_payload_and_returned_fields_are_frozen_snapshots():
    payload = {"weight": [0.1, 0.2]}
    original = copy.deepcopy(payload)
    cache = FrozenPayloadField(payload)
    fields = {"nested": {"seed": 1}}
    envelope = cache.envelope(fields, "mean")
    payload["weight"][0] = 99
    fields["nested"]["seed"] = 99
    assert restore_payload(envelope, original) == {"nested": {"seed": 1}, "mean": original}
    with pytest.raises(ValueError):
        restore_payload(envelope, payload)


@pytest.mark.parametrize("payload", [{"bad": float("nan")}, {"$rosclaw_shared_payload": "x"}, []])
def test_invalid_payload_rejected(payload):
    with pytest.raises(ValueError):
        FrozenPayloadField(payload)


@pytest.mark.parametrize("fields,key", [({"mean": 1}, "mean"), ({}, ""), ({1: "x"}, "mean")])
def test_ambiguous_key_or_document_rejected(fields, key):
    with pytest.raises(ValueError):
        FrozenPayloadField({}).document_hash(fields, key)
