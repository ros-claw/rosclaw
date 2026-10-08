"""Complete hash parity and isolation only; never physical verification."""

import copy

import pytest

from rosclaw.growth.cached_shared_payload import CachedSharedPayload
from rosclaw.growth.shared_proof_payload import canonical_hash, detach_payload


def document(depth):
    payload = {"actor": [[1.0, -0.25]], "critic": {"bias": [2.0]}, "body": "全身"}
    location = tuple(f"层{i}" for i in range(depth))
    nested = payload
    for key in reversed(location):
        nested = {key: nested, "siblings": [False, None, 7.5]}
    nested["report_hash"] = canonical_hash(nested)
    return nested, location


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_full_and_seal_free_hashes_preserve_every_numeric_field(depth):
    original, location = document(depth)
    original_hash = canonical_hash(original)
    envelope, payload = detach_payload(original, location)
    cache = CachedSharedPayload(payload)
    assert cache.document_hash(envelope) == original_hash
    assert cache.document_hash(envelope, omit_root_keys=("report_hash",)) == original["report_hash"]
    assert cache.verified_fields(envelope) == envelope["stripped_document"]
    payload["critic"]["bias"][0] = 123
    assert cache.document_hash(envelope) == original_hash
    fields = cache.verified_fields(envelope)
    fields.clear()
    assert cache.verified_fields(envelope)


@pytest.mark.parametrize("change", ["seal", "logical", "payload", "marker", "path", "extra"])
def test_changed_envelope_or_resealed_logical_document_rejected(change):
    original, location = document(4)
    envelope, payload = detach_payload(original, location)
    cache = CachedSharedPayload(payload)
    bad = copy.deepcopy(envelope)
    if change == "seal":
        bad["envelope_hash"] = "sha256:" + "0" * 64
    elif change in ("logical", "payload"):
        key = "logical_document_hash" if change == "logical" else "payload_hash"
        bad[key] = "sha256:" + "0" * 64
    elif change == "marker":
        target = bad["stripped_document"]
        for key in location[:-1]:
            target = target[key]
        target[location[-1]]["additional"] = 1
    elif change == "path":
        bad["location"] = ["missing"]
    else:
        bad["unexpected"] = True
    if change != "seal":
        bad["envelope_hash"] = canonical_hash(
            {k: v for k, v in bad.items() if k != "envelope_hash"}
        )
    with pytest.raises(ValueError):
        cache.document_hash(bad)


def test_cached_bytes_changed_even_with_snapshot_hash_rewritten_rejected():
    original, location = document(2)
    envelope, payload = detach_payload(original, location)
    cache = CachedSharedPayload(payload)
    data = cache._payload._data + b" "
    object.__setattr__(cache._payload, "_data", data)
    with pytest.raises(ValueError):
        cache.document_hash(envelope)
    import hashlib

    object.__setattr__(cache._payload, "content_hash", "sha256:" + hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError):
        cache.document_hash(envelope)


@pytest.mark.parametrize(
    "omit", [("层0",), ("missing",), ("report_hash", "report_hash"), ["report_hash"], (True,)]
)
def test_bad_seal_omission_never_removes_payload_or_hides_changed_fields(omit):
    original, location = document(2)
    envelope, payload = detach_payload(original, location)
    with pytest.raises(ValueError):
        CachedSharedPayload(payload).document_hash(envelope, omit_root_keys=omit)


@pytest.mark.parametrize("payload", [{1: "lossy"}, {"values": (1, 2)}, {"nan": float("nan")}])
def test_lossy_or_nonfinite_payload_rejected(payload):
    with pytest.raises(ValueError):
        CachedSharedPayload(payload)


def test_hash_identity_does_not_change_or_approve_authority_flags():
    original, location = document(4)
    original.update(hardware_authorized=True, promotion_authorized=True)
    envelope, payload = detach_payload(original, location)
    cache = CachedSharedPayload(payload)
    assert cache.document_hash(envelope) == canonical_hash(original)
    fields = cache.verified_fields(envelope)
    assert fields["hardware_authorized"] is True
    assert fields["promotion_authorized"] is True
    # These are caller's original data, not a new authorization receipt.
    assert set(fields) == set(original)
