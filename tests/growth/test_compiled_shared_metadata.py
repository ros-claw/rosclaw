import copy

import pytest

from rosclaw.growth.compiled_shared_metadata import CompiledSharedMetadata
from rosclaw.growth.shared_proof_payload import canonical_hash, detach_payload, restore_payload


def source():
    document = {
        "évidence": [1, -0.0, True, None, "中文"],
        "policy": {"nested": {"matrix": [[1.1, 2.2], [3, 4]], "name": "模型"}},
        "marker_string": "$rosclaw_shared_payload",
    }
    document["report_hash"] = canonical_hash(document)
    envelope, payload = detach_payload(document, ("policy", "nested"))
    return document, envelope, payload


def test_exact_original_unicode_floats_and_root_seal_without_complete_payload():
    document, envelope, payload = source()
    verifier = CompiledSharedMetadata(payload)
    result = verifier.verify(envelope, sealed_field="report_hash")
    assert restore_payload(envelope, payload) == document
    assert result["logical_document_hash"] == canonical_hash(document)
    assert result["root_seal_verified"]
    assert result["complete_numerical_document_returned"] is False
    assert result["runtime_execution_authorized"] is False
    assert result["hardware_authorized"] is False
    assert result["promotion_authorized"] is False


def test_input_and_result_mutations_do_not_change_owned_payload_or_later_metadata():
    _, envelope, payload = source()
    verifier = CompiledSharedMetadata(payload)
    payload["matrix"][0][0] = 999
    result = verifier.verify(envelope, sealed_field="report_hash")
    result["metadata_with_payload_marker"]["évidence"][0] = 999
    assert verifier.verify(envelope)["metadata_with_payload_marker"]["évidence"][0] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("envelope_hash", "sha256:" + "0" * 64),
        ("payload_hash", "sha256:" + "0" * 64),
        ("logical_document_hash", "sha256:" + "0" * 64),
        ("location", ["missing"]),
    ],
)
def test_tampered_envelope_rejected_even_if_resealed(field, value):
    _, envelope, payload = source()
    envelope[field] = value
    if field != "envelope_hash":
        envelope["envelope_hash"] = canonical_hash(
            {k: v for k, v in envelope.items() if k != "envelope_hash"}
        )
    with pytest.raises(ValueError):
        CompiledSharedMetadata(payload).verify(envelope)


def test_logical_seal_is_not_a_substitute_for_root_report_seal():
    document, _, _ = source()
    document["report_hash"] = "sha256:" + "0" * 64
    envelope, payload = detach_payload(document, ("policy", "nested"))
    verifier = CompiledSharedMetadata(payload)
    assert verifier.verify(envelope)["root_seal_verified"] is False
    with pytest.raises(ValueError, match="root seal"):
        verifier.verify(envelope, sealed_field="report_hash")


def test_literal_marker_in_other_metadata_cannot_redirect_payload_insertion():
    document, _, _ = source()
    document["another"] = {"$rosclaw_shared_payload": "unrelated"}
    document["report_hash"] = canonical_hash(
        {k: v for k, v in document.items() if k != "report_hash"}
    )
    envelope, payload = detach_payload(document, ("policy", "nested"))
    assert CompiledSharedMetadata(payload).verify(envelope, sealed_field="report_hash")[
        "root_seal_verified"
    ]


def test_bad_marker_cache_bytes_and_missing_root_field_rejected():
    _, envelope, payload = source()
    wrong = copy.deepcopy(envelope)
    wrong["stripped_document"]["policy"]["nested"] = {}
    wrong["envelope_hash"] = canonical_hash(
        {k: v for k, v in wrong.items() if k != "envelope_hash"}
    )
    verifier = CompiledSharedMetadata(payload)
    with pytest.raises(ValueError, match="marker"):
        verifier.verify(wrong)
    with pytest.raises(ValueError, match="root seal"):
        verifier.verify(envelope, sealed_field="missing")
    verifier._payload_bytes = b"{}"
    with pytest.raises(ValueError, match="identity"):
        verifier.verify(envelope)


@pytest.mark.parametrize("payload", [{"value": float("nan")}, {"$rosclaw_shared_payload": "x"}, []])
def test_nonfinite_or_referenced_payload_rejected(payload):
    with pytest.raises(ValueError):
        CompiledSharedMetadata(payload)


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_complete_restore_exact_and_independently_owned(depth):
    payload = {
        "actor": [[-0.0, 1e-20, 1e20, 1.1]],
        "critic": {"bias": [2.0]},
        "name": '中文\\"\n',
    }
    location = tuple(f"层{i}" for i in range(depth))
    document = payload
    for key in reversed(location):
        document = {key: document, "sibling": [False, None, "$rosclaw_shared_payload"]}
    document["report_hash"] = canonical_hash(document)
    envelope, detached = detach_payload(document, location)
    verifier = CompiledSharedMetadata(detached)
    restored = verifier.restore(envelope, sealed_field="report_hash")
    assert restored == restore_payload(envelope, detached) == document
    assert canonical_hash(restored) == canonical_hash(document)
    restored.clear()
    detached["critic"]["bias"][0] = 999
    again = verifier.restore(envelope, sealed_field="report_hash")
    branch = again
    for key in location:
        branch = branch[key]
    assert branch["critic"]["bias"] == [2.0]
    assert branch["actor"][0][0] == 0.0
    assert canonical_hash(again) == envelope["logical_document_hash"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("envelope_hash", "sha256:" + "0" * 64),
        ("payload_hash", "sha256:" + "0" * 64),
        ("logical_document_hash", "sha256:" + "0" * 64),
        ("location", ["missing"]),
    ],
)
def test_full_restore_refuses_tampered_or_resealed_envelopes(field, value):
    _, envelope, payload = source()
    envelope[field] = value
    if field != "envelope_hash":
        envelope["envelope_hash"] = canonical_hash(
            {k: v for k, v in envelope.items() if k != "envelope_hash"}
        )
    with pytest.raises(ValueError):
        CompiledSharedMetadata(payload).restore(envelope, sealed_field="report_hash")


def test_full_restore_checks_root_seal_not_only_whole_logical_identity():
    document, _, _ = source()
    document["report_hash"] = "sha256:" + "0" * 64
    envelope, payload = detach_payload(document, ("policy", "nested"))
    verifier = CompiledSharedMetadata(payload)
    assert verifier.restore(envelope) == document
    with pytest.raises(ValueError, match="root seal"):
        verifier.restore(envelope, sealed_field="report_hash")


def test_full_restore_preserves_literal_markers_and_caller_flags_as_data():
    document, _, _ = source()
    document.update(
        literal={"$rosclaw_shared_payload": "unrelated"},
        hardware_authorized=True,
        promotion_authorized=True,
    )
    document["report_hash"] = canonical_hash(
        {k: v for k, v in document.items() if k != "report_hash"}
    )
    envelope, payload = detach_payload(document, ("policy", "nested"))
    restored = CompiledSharedMetadata(payload).restore(envelope, sealed_field="report_hash")
    assert restored == document
    # Restoring caller flags is not an authorization receipt or new trust grant.
    assert set(restored) == set(document)


@pytest.mark.parametrize("payload", [{1: "lossy"}, {"values": (1, 2)}])
def test_coercible_lossy_python_payload_rejected(payload):
    with pytest.raises(ValueError, match="lossless"):
        CompiledSharedMetadata(payload)


def test_payload_changed_after_verification_before_full_restore_rejected(monkeypatch):
    _, envelope, payload = source()
    verifier = CompiledSharedMetadata(payload)
    original = verifier.verify

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        verifier._payload_bytes = b"{}"
        return result

    monkeypatch.setattr(verifier, "verify", changing)
    with pytest.raises(ValueError, match="restoration payload"):
        verifier.restore(envelope)


def test_changed_complete_reconstruction_rejected(monkeypatch):
    from rosclaw.growth import compiled_shared_metadata as module

    _, envelope, payload = source()
    verifier = CompiledSharedMetadata(payload)
    original = module._chunks

    def changing(document, location, data):
        yield from original(document, location, data)
        yield b" "

    monkeypatch.setattr(module, "_chunks", changing)
    # Logical verification also uses chunks and rejects before returning data.
    with pytest.raises(ValueError, match="logical document"):
        verifier.restore(envelope)
