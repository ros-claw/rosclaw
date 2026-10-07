"""Additive realistic-size binary point-cloud quality analysis (pure software).

This module is the Stage B companion to
:mod:`rosclaw.perception.cloud_quality`. It is fully additive: the legacy
16x16 / 17-count ``analyze_cloud`` API and its algorithm are unchanged; the
new entry points here accept realistic-size generated binary clouds (up to
307200 points, 8 MiB of binary data, 32 fields) as plain metadata dicts plus
exact ``bytes``.

Scope: pure generated binary software only. No ROS/DDS runtime, no neural
network, no hardware, no Artifact resolver/catalog integration. Both entry
points are deterministic pure functions over caller-supplied bytes (the file
variant only reads one bounded ordinary file the caller explicitly scopes).

Bounded admission vs quality verdict:

* Non-plain types (subclasses, custom objects with conversion hooks),
  unknown/missing metadata keys, and oversize envelopes are rejected with
  ``TypeError``/``ValueError`` *before* any conversion, copy, or decode.
  Rejection diagnostics are short, fixed, and never echo caller type names,
  keys, or values.
* Admitted-but-malformed layouts (duplicate fields, invalid xyz field
  layout, stride or data-length mismatch) are ordinary quality results:
  ``QUALITY_INVALID`` with ``total/0/total`` counts and null statistics.

Public API:

* ``analyze_binary_cloud(metadata, data) -> dict``
* ``analyze_cloud_file(metadata, data_path, *, allowed_root, expected_sha256) -> dict``

The result dict has exactly seven keys: ``status``, ``total_count``,
``valid_count``, ``invalid_count``, ``min_xyz``, ``max_xyz``,
``centroid_xyz``.
"""

import errno
import hashlib
import math
import os
import stat
import struct
from typing import Any

# Bounds (public contract).
MAX_DIMENSION = 2048
MAX_POINTS = 307200
MAX_BINARY_BYTES = 8 * 1024 * 1024
MAX_FIELDS = 32
MAX_FIELD_NAME_UTF8_BYTES = 64
MAX_POINT_STEP = 256

_METADATA_KEYS = frozenset(
    {"width", "height", "point_step", "row_step", "fields", "is_bigendian", "is_dense"}
)
_FIELD_KEYS = frozenset({"name", "offset", "datatype", "count"})

_FLOAT32 = 7  # sensor_msgs/msg/PointField.FLOAT32
_DATATYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}

# Offset/count admission ceiling: generous but keeps layout arithmetic small.
_MAX_INT_COMPONENT = 1 << 31


def _invalid_result(total: int) -> dict:
    return {
        "status": "QUALITY_INVALID",
        "total_count": total,
        "valid_count": 0,
        "invalid_count": total,
        "min_xyz": None,
        "max_xyz": None,
        "centroid_xyz": None,
    }


def _empty_result(status: str, total: int, valid: int) -> dict:
    return {
        "status": status,
        "total_count": total,
        "valid_count": valid,
        "invalid_count": total - valid,
        "min_xyz": None,
        "max_xyz": None,
        "centroid_xyz": None,
    }


def _require_plain_int(value: Any, lo: int, hi: int) -> int:
    # Exact ``int`` only: bool and int subclasses (and any custom object)
    # are rejected without touching conversion hooks.
    if type(value) is not int:
        raise TypeError("metadata integer component has a non-plain type")
    if not (lo <= value <= hi):
        raise ValueError("metadata integer component outside bounded envelope")
    return value


def _require_plain_bool(value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError("metadata boolean component has a non-plain type")
    return value


def _admit_metadata(metadata: Any) -> dict:
    """Bounded admission of the plain metadata dict; returns a clean copy."""
    if type(metadata) is not dict:
        raise TypeError("metadata must be a plain dict")
    # Exact key count before any set/copy of caller keys.
    if len(metadata) != len(_METADATA_KEYS):
        raise ValueError("metadata keys must be exactly the contracted set")
    if not _METADATA_KEYS.issuperset(metadata.keys()):
        raise ValueError("metadata keys must be exactly the contracted set")
    width = _require_plain_int(metadata["width"], 1, MAX_DIMENSION)
    height = _require_plain_int(metadata["height"], 1, MAX_DIMENSION)
    # Both dimensions are already bounded <= 2048, so this product is small.
    if width * height > MAX_POINTS:
        raise ValueError("declared point total exceeds bounded envelope")
    point_step = _require_plain_int(metadata["point_step"], 1, MAX_POINT_STEP)
    row_step = _require_plain_int(metadata["row_step"], 1, MAX_BINARY_BYTES)
    if height * row_step > MAX_BINARY_BYTES:
        raise ValueError("declared row envelope exceeds bounded byte budget")
    is_bigendian = _require_plain_bool(metadata["is_bigendian"])
    is_dense = _require_plain_bool(metadata["is_dense"])
    raw_fields = metadata["fields"]
    if type(raw_fields) is not list:
        raise TypeError("metadata fields must be a plain list")
    if len(raw_fields) > MAX_FIELDS:
        raise ValueError("metadata fields exceed bounded count")
    fields = []
    for entry in raw_fields:
        if type(entry) is not dict:
            raise TypeError("field entry must be a plain dict")
        if len(entry) != len(_FIELD_KEYS) or not _FIELD_KEYS.issuperset(entry.keys()):
            raise ValueError("field entry keys must be exactly the contracted set")
        name = entry["name"]
        if type(name) is not str:
            raise TypeError("field name must be a plain str")
        # Character count guard before UTF-8 encoding, then byte count.
        if len(name) > MAX_FIELD_NAME_UTF8_BYTES:
            raise ValueError("field name exceeds bounded UTF-8 length")
        if len(name.encode("utf-8")) > MAX_FIELD_NAME_UTF8_BYTES:
            raise ValueError("field name exceeds bounded UTF-8 length")
        offset = _require_plain_int(entry["offset"], 0, _MAX_INT_COMPONENT)
        datatype = _require_plain_int(entry["datatype"], 1, 8)
        count = _require_plain_int(entry["count"], 0, _MAX_INT_COMPONENT)
        fields.append({"name": name, "offset": offset, "datatype": datatype, "count": count})
    return {
        "width": width,
        "height": height,
        "point_step": point_step,
        "row_step": row_step,
        "fields": fields,
        "is_bigendian": is_bigendian,
        "is_dense": is_dense,
    }


def _admit_data(data: Any) -> bytes:
    if type(data) is not bytes:
        raise TypeError("data must be exact plain bytes")
    if len(data) > MAX_BINARY_BYTES:
        raise ValueError("data exceeds bounded byte budget")
    return data


def _layout_valid(meta: dict, data: bytes) -> bool:
    """Ordinary layout quality verdict over already-admitted plain values."""
    width = meta["width"]
    point_step = meta["point_step"]
    row_step = meta["row_step"]
    height = meta["height"]
    if row_step < width * point_step:
        return False
    if len(data) != height * row_step:
        return False
    fields = meta["fields"]
    names = [f["name"] for f in fields]
    if len(names) != len(set(names)):
        return False
    for f in fields:
        size = _DATATYPE_SIZE[f["datatype"]]
        if f["count"] < 1:
            return False
        if f["offset"] + size * f["count"] > point_step:
            return False
    for req in ("x", "y", "z"):
        matches = [f for f in fields if f["name"] == req]
        if len(matches) != 1:
            return False
        f = matches[0]
        if f["datatype"] != _FLOAT32 or f["count"] != 1 or f["offset"] + 4 > point_step:
            return False
    return True


def analyze_binary_cloud(metadata: Any, data: Any) -> dict:
    """Analyze a generated binary point cloud from plain metadata + bytes.

    Parameters:
        metadata: Plain ``dict`` with exactly the keys ``width``, ``height``,
            ``point_step``, ``row_step``, ``fields``, ``is_bigendian``,
            ``is_dense`` and plain bounded values.
        data: Exact ``bytes`` (``type(data) is bytes``), at most 8 MiB.

    Returns:
        The exact seven-key result dict.

    Raises:
        TypeError/ValueError: Bounded admission rejection for non-plain
            types, unknown keys, or oversize envelopes; raised before any
            conversion, copy, or decode of the offending input.
    """
    meta = _admit_metadata(metadata)
    payload = _admit_data(data)
    total = meta["width"] * meta["height"]
    if not _layout_valid(meta, payload):
        return _invalid_result(total)
    fmt = ">" if meta["is_bigendian"] else "<"
    offsets = {f["name"]: f["offset"] for f in meta["fields"] if f["name"] in ("x", "y", "z")}
    ps, rs, w, h = meta["point_step"], meta["row_step"], meta["width"], meta["height"]
    ox, oy, oz = offsets["x"], offsets["y"], offsets["z"]
    valid = 0
    mins = [math.inf] * 3
    maxs = [-math.inf] * 3
    xs: list = []
    ys: list = []
    zs: list = []
    unpack = struct.unpack_from
    for row in range(h):
        base = row * rs
        for col in range(w):
            pbase = base + col * ps
            x = unpack(fmt + "f", payload, pbase + ox)[0]
            y = unpack(fmt + "f", payload, pbase + oy)[0]
            z = unpack(fmt + "f", payload, pbase + oz)[0]
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                valid += 1
                xs.append(x)
                ys.append(y)
                zs.append(z)
                if x < mins[0]:
                    mins[0] = x
                if x > maxs[0]:
                    maxs[0] = x
                if y < mins[1]:
                    mins[1] = y
                if y > maxs[1]:
                    maxs[1] = y
                if z < mins[2]:
                    mins[2] = z
                if z > maxs[2]:
                    maxs[2] = z
    if valid == 0:
        return _empty_result("NO_VALID_RETURN", total, 0)
    # Exact finite centroid: math.fsum tracks partials, so cancellation-heavy
    # finite float32 coordinates match independent fsum/n within 1e-10.
    centroid = [math.fsum(axis) / valid for axis in (xs, ys, zs)]
    return {
        "status": "VALID_CLOUD",
        "total_count": total,
        "valid_count": valid,
        "invalid_count": total - valid,
        "min_xyz": mins,
        "max_xyz": maxs,
        "centroid_xyz": centroid,
    }


def analyze_cloud_file(
    metadata: Any,
    data_path: Any,
    *,
    allowed_root: Any,
    expected_sha256: Any,
) -> dict:
    """Analyze a bounded ordinary binary file under a caller-supplied root.

    ``allowed_root`` is an explicit caller scope, not an OS authority. The
    target must resolve inside that root, must not be a symlink itself and
    must have no symlink in any parent path component (even when the link
    destination stays inside the root), must be a regular file of at most
    8 MiB (size checked via ``fstat`` before a bounded read), and must match
    ``expected_sha256`` exactly for its current contents. No adversarial
    atomic-snapshot guarantee is promised.

    Raises:
        TypeError/ValueError/OSError: Scoped rejection; diagnostics are
            short and never echo caller type names, keys, or values.
    """
    if type(data_path) is not str:
        raise TypeError("data path must be a plain str")
    if type(allowed_root) is not str:
        raise TypeError("allowed root must be a plain str")
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ValueError("expected sha256 must be 64 lowercase hex characters")
    # Metadata admission fully precedes any payload os.read.
    meta = _admit_metadata(metadata)
    root_real = os.path.realpath(allowed_root)
    target_real = os.path.realpath(data_path)
    rel = os.path.relpath(target_real, root_real)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        raise ValueError("data path escapes the caller-supplied root")
    # No-symlink admission: realpath/normalized resolution hides links, so
    # walk every RAW caller path component (before any dot/dotdot lexical
    # collapse) and lstat each one. Any symlink in the leaf or any parent
    # directory component rejects the path before the payload read, even
    # when the destination stays inside the root (e.g. ``link/../file``
    # where the final canonical target is in-root). Ordinary ``..`` between
    # ordinary directories stays valid: only symlink components reject.
    raw_path = data_path if os.path.isabs(data_path) else os.path.join(os.getcwd(), data_path)
    cursor = os.path.sep
    for component in raw_path.split(os.path.sep):
        if not component or component == os.curdir:
            continue
        cursor = os.path.join(cursor, component)
        try:
            component_stat = os.lstat(cursor)
        except OSError:
            # Missing/unresolvable raw component: reject before any payload
            # read; never fall through to trust the canonicalized path.
            raise OSError(errno.ENOENT, "data file not readable") from None
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError("symbolic links are not admitted")
    # Pre-open envelope guards: reject symlinks and non-regular files (FIFO,
    # socket, device) promptly via lstat so no blocking open ever happens.
    try:
        lst = os.lstat(target_real)
    except OSError:
        raise OSError(errno.ENOENT, "data file not readable") from None
    if stat.S_ISLNK(lst.st_mode):
        raise ValueError("symbolic links are not admitted")
    if not stat.S_ISREG(lst.st_mode):
        raise OSError(errno.EINVAL, "not a regular file")
    if lst.st_size > MAX_BINARY_BYTES:
        raise ValueError("file exceeds bounded byte budget")
    try:
        fd = os.open(target_real, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        raise OSError(errno.EACCES, "data file not readable") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(errno.EINVAL, "not a regular file")
        if st.st_size > MAX_BINARY_BYTES:
            raise ValueError("file exceeds bounded byte budget")
        payload = os.read(fd, MAX_BINARY_BYTES + 1)
        # Drain any remainder within the bounded budget.
        while len(payload) <= MAX_BINARY_BYTES:
            chunk = os.read(fd, MAX_BINARY_BYTES + 1 - len(payload))
            if not chunk:
                break
            payload += chunk
    finally:
        os.close(fd)
    if len(payload) != st.st_size or len(payload) > MAX_BINARY_BYTES:
        raise ValueError("file size changed or exceeds bounded byte budget")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("file contents do not match the expected sha256")
    return analyze_binary_cloud(meta, payload)


__all__ = ["analyze_binary_cloud", "analyze_cloud_file"]
