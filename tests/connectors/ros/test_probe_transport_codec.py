"""Lossless transport cannot change original evidence or expand without bounds."""

import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from rosclaw.connectors.ros.probe.transport_codec import (
    MAX_BYTES,
    SCHEMA,
    decode_probe_snapshot,
    encode_probe_snapshot,
)


def original():
    return json.dumps(
        {
            "schema_version": "rosclaw.ros_probe.v1",
            "captured_at": "2026-10-18T07:00:00Z",
            "original_document": '网格\\"' * 12000,
        },
        ensure_ascii=False,
        indent=2,
    ).encode()


def test_exact_roundtrip_and_capture_original_clocks(tmp_path):
    raw = original()
    wire = encode_probe_snapshot(raw)
    assert len(wire) < len(raw)
    assert decode_probe_snapshot(wire) == (raw, json.loads(raw))
    path = (
        Path(__file__).resolve().parents[3]
        / "integrations/ros_probe/acceptance/simulation_native_capture.py"
    )
    spec = importlib.util.spec_from_file_location("capture", path)
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    capture.persist_native_capture(
        tmp_path, wire, index=1, received_monotonic_ns=12345, publish_pointer=True
    )
    pointer = json.loads((tmp_path / "live-native-pointer.json").read_text())
    assert (tmp_path / pointer["file"]).read_bytes() == raw
    assert pointer["sha256"] == hashlib.sha256(raw).hexdigest()
    assert pointer["received_monotonic_ns"] == 12345
    assert pointer["captured_at"] == json.loads(raw)["captured_at"]
    assert (tmp_path / "wire-native-001.json").read_bytes() == wire


def test_legacy_is_byte_identical_and_small_native_stays_raw():
    raw = b'{"schema_version": "rosclaw.ros_probe.v1", "original": [1, 2]}'
    assert encode_probe_snapshot(raw) == raw
    assert decode_probe_snapshot(raw) == (raw, json.loads(raw))


@pytest.mark.parametrize(
    "mutation",
    [
        {"original_bytes": True},
        {"original_bytes": MAX_BYTES + 1},
        {"original_bytes": 1},
        {"original_sha256": "0" * 64},
        {"encoding": "other"},
        {"extra": 1},
        {"data": "%%%"},
    ],
)
def test_malformed_transport_rejected(mutation):
    value = json.loads(encode_probe_snapshot(original()))
    value.update(mutation)
    with pytest.raises(ValueError):
        decode_probe_snapshot(json.dumps(value).encode())


@pytest.mark.parametrize(
    "suffix",
    [b"extra", gzip.compress(b"another member", mtime=0)],
    ids=["trailing-bytes", "concatenated-member"],
)
def test_concatenated_or_trailing_compressed_data_rejected(suffix):
    value = json.loads(encode_probe_snapshot(original()))
    data = base64.b64decode(value["data"])
    value["data"] = base64.b64encode(data + suffix).decode()
    with pytest.raises(ValueError):
        decode_probe_snapshot(json.dumps(value).encode())


def test_truncated_and_expansion_bomb_rejected():
    value = json.loads(encode_probe_snapshot(original()))
    data = base64.b64decode(value["data"])
    value["data"] = base64.b64encode(data[:-3]).decode()
    with pytest.raises(ValueError):
        decode_probe_snapshot(json.dumps(value).encode())
    value["original_bytes"] = 100
    value["data"] = base64.b64encode(gzip.compress(b"x" * (MAX_BYTES + 1))).decode()
    with pytest.raises(ValueError):
        decode_probe_snapshot(json.dumps(value).encode())


def test_nested_envelope_cannot_masquerade_as_native():
    raw = json.dumps({"schema_version": SCHEMA}).encode()
    value = json.loads(encode_probe_snapshot(original()))
    value.update(
        original_bytes=len(raw),
        original_sha256=hashlib.sha256(raw).hexdigest(),
        data=base64.b64encode(gzip.compress(raw)).decode(),
    )
    with pytest.raises(ValueError):
        decode_probe_snapshot(json.dumps(value).encode())


def test_duplicate_envelope_field_rejected():
    wire = encode_probe_snapshot(original())
    duplicate = b'{"encoding":"gzip-base64",' + wire[1:]
    with pytest.raises(ValueError, match="duplicate"):
        decode_probe_snapshot(duplicate)
