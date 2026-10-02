import copy
import json

import pytest

from rosclaw.growth.shared_proof_payload import (
    MARKER,
    canonical_hash,
    detach_payload,
    restore_payload,
)


def document():
    return {
        "receipt": "old-seal",
        "proof": {
            "model": {
                "weight": [0.0, -0.0, 1.2345678901234567e-12],
                "unicode": "全部数据",
            }
        },
        "observed": [1, 2, 3],
    }


def test_complete_lossless_roundtrip_and_no_original_mutation():
    original = document()
    before = json.dumps(original, sort_keys=True, ensure_ascii=False)
    envelope, payload = detach_payload(original, ("proof", "model"))
    result = restore_payload(envelope, payload)
    assert json.dumps(original, sort_keys=True, ensure_ascii=False) == before
    assert json.dumps(result, sort_keys=True, ensure_ascii=False) == before
    assert canonical_hash(result) == canonical_hash(original)
    assert envelope["stripped_document"]["proof"]["model"] == {MARKER: canonical_hash(payload)}
    assert set(envelope).isdisjoint({"activation", "promotion_authorized", "hardware_authorized"})


def test_changed_payload_rejected():
    envelope, payload = detach_payload(document(), ("proof", "model"))
    changed = copy.deepcopy(payload)
    changed["weight"][2] += 1e-20
    with pytest.raises(ValueError, match="payload identity"):
        restore_payload(envelope, changed)


def test_changed_envelope_rejected_before_reconstruction():
    envelope, payload = detach_payload(document(), ("proof", "model"))
    envelope["stripped_document"]["observed"] = [999]
    with pytest.raises(ValueError, match="seal changed"):
        restore_payload(envelope, payload)


def test_resealed_stripped_data_still_cannot_claim_original_logical_document():
    envelope, payload = detach_payload(document(), ("proof", "model"))
    envelope["stripped_document"]["observed"] = [999]
    envelope["envelope_hash"] = canonical_hash(
        {k: v for k, v in envelope.items() if k != "envelope_hash"}
    )
    with pytest.raises(ValueError, match="logical document"):
        restore_payload(envelope, payload)


@pytest.mark.parametrize("location", [(), ("missing",), ("observed", "0"), ("",), ("x",) * 9])
def test_missing_non_dictionary_or_unbounded_location_rejected(location):
    with pytest.raises(ValueError):
        detach_payload(document(), location)


def test_nonfinite_values_never_become_a_shared_proof():
    value = document()
    value["proof"]["model"]["weight"] = [float("nan")]
    with pytest.raises(ValueError):
        detach_payload(value, ("proof", "model"))


def test_reference_payload_cannot_recursively_redirect():
    value = document()
    value["proof"]["model"] = {MARKER: "sha256:" + "a" * 64}
    with pytest.raises(ValueError, match="nested references"):
        detach_payload(value, ("proof", "model"))


def test_file_path_is_never_a_payload_reference():
    envelope, payload = detach_payload(document(), ("proof", "model"))
    envelope["payload_hash"] = "../../secret"
    envelope["envelope_hash"] = canonical_hash(
        {k: v for k, v in envelope.items() if k != "envelope_hash"}
    )
    with pytest.raises(ValueError, match="not a file path"):
        restore_payload(envelope, payload)
