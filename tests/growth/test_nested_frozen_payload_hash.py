"""Full canonical byte equivalence, not policy or execution qualification."""

import copy
import hashlib
import json

import pytest

from rosclaw.growth.frozen_payload_field import FrozenPayloadField


def reference(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("depth", [1, 2, 3, 16])
@pytest.mark.parametrize("seed", [0, 17, 2**32 - 1])
def test_nested_hash_includes_every_field_and_full_owned_payload(depth, seed):
    payload = {"weights": [[0.25, -0.0, 1e-100]], "unicode": '球⚽"\\'}
    original = copy.deepcopy(payload)
    cache = FrozenPayloadField(payload)
    payload["weights"][0][0] = 999
    fields = {"seed": seed, "metadata": {"enabled": False, "null": None}}
    path = tuple(f"level-{i}" for i in range(depth))
    for key in reversed(path[:-1]):
        fields = {"sibling": [True, "{}"], key: fields}
    unchanged = copy.deepcopy(fields)
    expected = copy.deepcopy(fields)
    node = expected
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = original
    assert cache.document_hash_at_path(fields, path) == reference(expected)
    assert fields == unchanged
    assert cache.document_hash_at_path(fields, path) != reference(fields)


@pytest.mark.parametrize(
    "fields,path",
    [
        ({}, ()),
        ({}, ["payload"]),
        ({}, ("",)),
        ({}, (True,)),
        ({}, ("x" * 129,)),
        ({}, ("x",) * 17),
        ({}, ("missing", "payload")),
        ({"parent": []}, ("parent", "payload")),
        ({"parent": None}, ("parent", "payload")),
        ({"parent": {"payload": {}}}, ("parent", "payload")),
        ({"payload": None}, ("payload",)),
        ({"parent": {1: "lossy"}}, ("parent", "payload")),
        ({"parent": {"tuple": (1, 2)}}, ("parent", "payload")),
        ({"parent": {"nan": float("nan")}}, ("parent", "payload")),
        ({"parent": {"inf": float("inf")}}, ("parent", "payload")),
    ],
)
def test_ambiguous_missing_lossy_or_nonfinite_input_rejected(fields, path):
    with pytest.raises(ValueError):
        FrozenPayloadField({"whole": [1, 2, 3]}).document_hash_at_path(fields, path)


def test_root_equivalence_and_cached_byte_tampering_rejected():
    cache = FrozenPayloadField({"weights": [1, 2, 3]})
    fields = {"other": {"payload": "same name at a different path"}}
    assert cache.document_hash_at_path(fields, ("payload",)) == cache.document_hash(
        fields, "payload"
    )
    cache._payload_bytes = b"{}"
    with pytest.raises(ValueError, match="unchanged"):
        cache.document_hash_at_path(fields, ("payload",))


def test_full_payload_bytes_do_not_get_reencoded(monkeypatch):
    payload = {"weights": [0.5] * 1000}
    cache = FrozenPayloadField(payload)
    import rosclaw.growth.frozen_payload_field as module

    original = module._bytes

    def small_fields_only(value):
        assert value != payload
        return original(value)

    monkeypatch.setattr(module, "_bytes", small_fields_only)
    expected = {"parent": {"seed": 1, "payload": payload}}
    assert cache.document_hash_at_path({"parent": {"seed": 1}}, ("parent", "payload")) == reference(
        expected
    )
