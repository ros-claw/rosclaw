"""Optional bounded, lossless transport; decoded observations keep their original clocks.

This envelope authenticates no source and grants no authority. Its digest only
detects corruption of the exact pre-transport JSON bytes.
"""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import re
import zlib

MAX_BYTES = 5_000_000
SCHEMA = "rosclaw.ros_probe_transport.v1"
_FIELDS = {"schema_version", "encoding", "original_sha256", "original_bytes", "data"}


def encode_probe_snapshot(raw: bytes) -> bytes:
    """Return the original bytes when the bounded envelope would not save space."""
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_BYTES:
        raise ValueError("bounded original probe bytes required")
    native = json.loads(raw)
    if not isinstance(native, dict) or native.get("schema_version") != "rosclaw.ros_probe.v1":
        raise ValueError("original native probe schema required")
    if len(raw) < 32_768:
        return raw
    wire = json.dumps(
        {
            "schema_version": SCHEMA,
            "encoding": "gzip-base64",
            "original_sha256": hashlib.sha256(raw).hexdigest(),
            "original_bytes": len(raw),
            "data": base64.b64encode(gzip.compress(raw, compresslevel=1, mtime=0)).decode("ascii"),
        },
        separators=(",", ":"),
    ).encode()
    return wire if len(wire) < len(raw) else raw


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, item in pairs:
        if key in result:
            raise ValueError("duplicate transport field")
        result[key] = item
    return result


def decode_probe_snapshot(wire: bytes) -> tuple[bytes, dict]:
    """Decode at most MAX_BYTES, rejecting truncation, concatenation and bombs."""
    if type(wire) is not bytes or not 0 < len(wire) <= MAX_BYTES:
        raise ValueError("bounded probe transport bytes required")
    value = json.loads(wire)
    if not isinstance(value, dict):
        raise ValueError("probe object required")
    if value.get("schema_version") != SCHEMA:
        return wire, value
    value = json.loads(wire, object_pairs_hook=_unique_fields)
    size = value.get("original_bytes")
    digest = value.get("original_sha256")
    data = value.get("data")
    if (
        set(value) != _FIELDS
        or value.get("encoding") != "gzip-base64"
        or type(size) is not int
        or not 0 < size <= MAX_BYTES
        or not isinstance(digest, str)
        or re.fullmatch("[0-9a-f]{64}", digest) is None
        or not isinstance(data, str)
    ):
        raise ValueError("invalid probe transport envelope")
    try:
        compressed = base64.b64decode(data, validate=True)
        if base64.b64encode(compressed).decode("ascii") != data:
            raise ValueError("noncanonical transport base64")
        decoder = zlib.decompressobj(wbits=31)
        raw = decoder.decompress(compressed, size + 1)
    except (ValueError, binascii.Error, zlib.error) as error:
        raise ValueError("invalid compressed probe bytes") from error
    if (
        len(raw) != size
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or hashlib.sha256(raw).hexdigest() != digest
    ):
        raise ValueError("probe transport length, digest or gzip boundary mismatch")
    native = json.loads(raw)
    if not isinstance(native, dict) or native.get("schema_version") != "rosclaw.ros_probe.v1":
        raise ValueError("decoded original native probe schema required")
    return raw, native
