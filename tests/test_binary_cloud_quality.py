"""Native tests for rosclaw.perception.binary_cloud_quality.

Pure generated-binary software tests: synthetic clouds are built with
``struct``/``math`` only. No ROS/DDS/NN/hardware, no mock capabilities, no
Artifact resolver/catalog. Numeric expectations are computed from the
generated coordinates, not from the decoder under test.
"""

import copy
import hashlib
import math
import os
import struct
import time

import pytest

from rosclaw.perception.binary_cloud_quality import (
    MAX_BINARY_BYTES,
    analyze_binary_cloud,
    analyze_cloud_file,
)

RESULT_KEYS = {
    "status",
    "total_count",
    "valid_count",
    "invalid_count",
    "min_xyz",
    "max_xyz",
    "centroid_xyz",
}


def _meta(width, height, point_step=12, row_step=None, bigendian=False, fields=None):
    if row_step is None:
        row_step = width * point_step
    if fields is None:
        fields = [
            {"name": "x", "offset": 0, "datatype": 7, "count": 1},
            {"name": "y", "offset": 4, "datatype": 7, "count": 1},
            {"name": "z", "offset": 8, "datatype": 7, "count": 1},
        ]
    return {
        "width": width,
        "height": height,
        "point_step": point_step,
        "row_step": row_step,
        "fields": fields,
        "is_bigendian": bigendian,
        "is_dense": True,
    }


def _points(width, height, invalid=None):
    """Deterministic generated grid; invalid set holds (row, col) NaN points."""
    invalid = invalid or set()
    pts = []
    for r in range(height):
        for c in range(width):
            if (r, c) in invalid:
                pts.append((math.nan, math.inf, 0.0))
            else:
                pts.append((c * 0.5 - 1.0, r * 0.25, (c - r) * 0.125))
    return pts


def _pack(meta, points):
    fmt = ">" if meta["is_bigendian"] else "<"
    row = bytearray(meta["row_step"])
    rows = []
    idx = 0
    for _r in range(meta["height"]):
        rowbuf = bytearray(row)
        for _c in range(meta["width"]):
            x, y, z = points[idx]
            idx += 1
            struct.pack_into(fmt + "fff", rowbuf, _c * meta["point_step"], x, y, z)
        rows.append(bytes(rowbuf))
    return b"".join(rows)


def _expected_stats(points):
    valid = [p for p in points if all(math.isfinite(v) for v in p)]
    if not valid:
        return "NO_VALID_RETURN", None, None, None
    mins = [min(p[a] for p in valid) for a in range(3)]
    maxs = [max(p[a] for p in valid) for a in range(3)]
    centroid = [sum(p[a] for p in valid) / len(valid) for a in range(3)]
    return "VALID_CLOUD", mins, maxs, centroid


def _assert_close(got, expected):
    assert set(got) == RESULT_KEYS
    for key in ("min_xyz", "max_xyz", "centroid_xyz"):
        if expected[key] is None:
            assert got[key] is None
        else:
            for g, e in zip(got[key], expected[key], strict=True):
                assert math.isclose(g, e, rel_tol=1e-10, abs_tol=1e-10)


def test_valid_small_cloud_exact_counts_and_stats():
    meta = _meta(8, 4)
    points = _points(8, 4)
    status, mins, maxs, centroid = _expected_stats(points)
    got = analyze_binary_cloud(meta, _pack(meta, points))
    assert got["status"] == status == "VALID_CLOUD"
    assert got["total_count"] == 32
    assert got["valid_count"] == 32
    assert got["invalid_count"] == 0
    _assert_close(got, {"min_xyz": mins, "max_xyz": maxs, "centroid_xyz": centroid})


def test_bigendian_with_padding_matches_little_endian():
    points = _points(8, 4)
    le = _meta(8, 4)
    be = _meta(8, 4, point_step=16, row_step=8 * 16 + 5, bigendian=True)
    assert analyze_binary_cloud(le, _pack(le, points)) == analyze_binary_cloud(
        be, _pack(be, points)
    )


def test_mixed_nan_inf_invalid_points():
    invalid = {(0, 0), (1, 3), (3, 7)}
    meta = _meta(8, 4)
    meta["is_dense"] = False
    points = _points(8, 4, invalid)
    status, mins, maxs, centroid = _expected_stats(points)
    got = analyze_binary_cloud(meta, _pack(meta, points))
    assert got["status"] == status == "VALID_CLOUD"
    assert got["total_count"] == 32
    assert got["valid_count"] == 29
    assert got["invalid_count"] == 3
    _assert_close(got, {"min_xyz": mins, "max_xyz": maxs, "centroid_xyz": centroid})


def test_all_invalid_points_no_valid_return():
    meta = _meta(8, 4)
    meta["is_dense"] = False
    points = _points(8, 4, {(r, c) for r in range(4) for c in range(8)})
    got = analyze_binary_cloud(meta, _pack(meta, points))
    assert got["status"] == "NO_VALID_RETURN"
    assert (got["total_count"], got["valid_count"], got["invalid_count"]) == (32, 0, 32)
    assert got["min_xyz"] is None and got["max_xyz"] is None and got["centroid_xyz"] is None


def test_max_realistic_size_cloud_boundary_accepted():
    # 640x480 = 307200 points, exactly the admitted maximum.
    meta = _meta(640, 480)
    points = _points(640, 480)
    got = analyze_binary_cloud(meta, _pack(meta, points))
    assert got["status"] == "VALID_CLOUD"
    assert got["total_count"] == 307200
    assert got["valid_count"] == 307200
    _, mins, maxs, centroid = _expected_stats(points)
    _assert_close(got, {"min_xyz": mins, "max_xyz": maxs, "centroid_xyz": centroid})


def test_input_metadata_not_mutated():
    meta = _meta(4, 4)
    before = copy.deepcopy(meta)
    analyze_binary_cloud(meta, _pack(meta, _points(4, 4)))
    assert meta == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["fields"].append(dict(m["fields"][0])),  # duplicate fields
        lambda m: m.update(row_step=1),  # stride too short
        lambda m: m["fields"][0].update(datatype=8),  # x not FLOAT32
        lambda m: m["fields"][0].update(offset=m["point_step"]),  # offset outside
        lambda m: m["fields"][0].update(count=0),  # zero count
        lambda m: m["fields"].pop(2),  # missing z field
    ],
)
def test_malformed_layout_is_quality_invalid(mutate):
    meta = _meta(8, 4)
    points = _points(8, 4)
    data = _pack(meta, points)
    mutate(meta)
    got = analyze_binary_cloud(meta, data)
    assert got["status"] == "QUALITY_INVALID"
    assert (got["total_count"], got["valid_count"], got["invalid_count"]) == (32, 0, 32)
    assert got["min_xyz"] is None


@pytest.mark.parametrize("delta", [-1, 1])
def test_data_length_mismatch_is_quality_invalid(delta):
    meta = _meta(8, 4)
    data = _pack(meta, _points(8, 4))
    data = data[:-1] if delta < 0 else data + b"0"
    got = analyze_binary_cloud(meta, data)
    assert got["status"] == "QUALITY_INVALID"
    assert got["total_count"] == 32


@pytest.mark.parametrize(
    "patch",
    [
        {"width": 2049},
        {"width": 641, "height": 480},  # product over 307200
        {"width": 1 << 4096},
        {"width": True},
        {"height": 0},
        {"point_step": 0},
        {"point_step": 257},
        {"row_step": MAX_BINARY_BYTES + 1},
        {"row_step": MAX_BINARY_BYTES},  # height*row_step over 8 MiB
        {"is_bigendian": 1},
        {"is_dense": None},
    ],
)
def test_admission_rejects_bad_metadata(patch):
    meta = _meta(8, 4)
    meta.update(patch)
    with pytest.raises((TypeError, ValueError)):
        analyze_binary_cloud(meta, b"")


def test_admission_rejects_unknown_and_missing_keys():
    meta = _meta(8, 4)
    with pytest.raises(ValueError):
        analyze_binary_cloud({**meta, "extra": 1}, b"")
    del meta["width"]
    with pytest.raises(ValueError):
        analyze_binary_cloud(meta, b"")


def test_admission_rejects_oversize_fields_and_long_name():
    meta = _meta(8, 4)
    meta["fields"] = meta["fields"] * 11
    with pytest.raises(ValueError):
        analyze_binary_cloud(meta, b"")
    meta = _meta(8, 4)
    meta["fields"][0]["name"] = "x" * 65
    with pytest.raises(ValueError):
        analyze_binary_cloud(meta, b"")
    meta = _meta(8, 4)
    meta["fields"] = "not-a-list"
    with pytest.raises(TypeError):
        analyze_binary_cloud(meta, b"")


class _HookTrap:
    def __bytes__(self):
        raise AssertionError("CUSTOM_CONVERSION_MUST_NOT_RUN")

    def __int__(self):
        raise AssertionError("CUSTOM_INT_MUST_NOT_RUN")

    def __iter__(self):
        raise AssertionError("CUSTOM_ITERATION_MUST_NOT_RUN")


@pytest.mark.parametrize(
    "meta,data",
    [
        (_HookTrap(), b""),
        ({"width": _HookTrap()}, b""),
        (_meta(8, 4), _HookTrap()),
        (_meta(8, 4), bytearray(8 * 4 * 12)),
    ],
)
def test_admission_rejects_non_plain_without_side_effects(meta, data):
    try:
        analyze_binary_cloud(meta, data)
    except (TypeError, ValueError) as exc:
        assert len(str(exc).encode("utf-8")) <= 512
    else:
        raise AssertionError("MALFORMED_INPUT_ACCEPTED")


def test_admission_rejects_oversize_data():
    with pytest.raises(ValueError):
        analyze_binary_cloud(_meta(8, 4), b"0" * (MAX_BINARY_BYTES + 1))


def test_analyze_cloud_file_ordinary_owned_file(tmp_path):
    meta = _meta(8, 4)
    data = _pack(meta, _points(8, 4))
    owned = tmp_path / "owned"
    owned.mkdir()
    path = owned / "cloud.bin"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    got = analyze_cloud_file(meta, str(path), allowed_root=str(owned), expected_sha256=digest)
    assert got["status"] == "VALID_CLOUD"
    assert got["total_count"] == 32


def test_analyze_cloud_file_rejections(tmp_path):
    meta = _meta(8, 4)
    data = _pack(meta, _points(8, 4))
    owned = tmp_path / "owned"
    owned.mkdir()
    path = owned / "cloud.bin"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(data)
    link = owned / "link.bin"
    link.symlink_to(outside)
    sparse = owned / "oversize.bin"
    with sparse.open("wb") as f:
        f.truncate(MAX_BINARY_BYTES + 1)
    for bad_path, expected in [
        (outside, digest),
        (link, digest),
        (owned / "absent.bin", digest),
        (owned, digest),
        (sparse, digest),
        (path, "0" * 64),
    ]:
        with pytest.raises((TypeError, ValueError, OSError)):
            analyze_cloud_file(
                meta, str(bad_path), allowed_root=str(owned), expected_sha256=expected
            )
    with pytest.raises(ValueError):
        analyze_cloud_file(meta, str(path), allowed_root=str(owned), expected_sha256="ZZ" * 32)


def _finite_meta_data(coords):
    meta = _meta(len(coords), 1)
    points = [(c, 0.0, float(i)) for i, c in enumerate(coords)]
    return meta, _pack(meta, points)


@pytest.mark.parametrize(
    "coords",
    [
        [2.0**80, 1.0, -(2.0**80)],
        [2.0**80, 1.0, 1.0, -(2.0**80)],
        [1.0, 2.0**80, -(2.0**80)],
        [-(2.0**80), 2.0**80, 1.0, 1.0],
    ],
)
def test_centroid_matches_independent_fsum_under_cancellation(coords):
    meta, data = _finite_meta_data(coords)
    got = analyze_binary_cloud(meta, data)
    assert got["status"] == "VALID_CLOUD"
    expected = math.fsum(struct.unpack("<f", struct.pack("<f", c))[0] for c in coords) / len(coords)
    assert math.isclose(got["centroid_xyz"][0], expected, rel_tol=1e-10, abs_tol=1e-10)


def test_centroid_cancellation_with_invalid_mix():
    coords = [2.0**80, 1.0, -(2.0**80)]
    meta, data = _finite_meta_data(coords)
    # Append one NaN point by widening to 4 columns.
    meta4 = _meta(4, 1)
    row = data[: 4 * 12]
    nan_pt = struct.pack("<fff", math.nan, 0.0, 9.0)
    data4 = row + nan_pt
    got = analyze_binary_cloud(meta4, data4)
    assert got["status"] == "VALID_CLOUD"
    assert (got["valid_count"], got["invalid_count"]) == (3, 1)
    fcoords = [struct.unpack("<f", struct.pack("<f", c))[0] for c in coords]
    assert math.isclose(
        got["centroid_xyz"][0], math.fsum(fcoords) / 3, rel_tol=1e-10, abs_tol=1e-10
    )


def test_leaf_symlink_inside_root_rejected_before_payload_read(tmp_path):
    # Leaf link whose destination stays inside allowed_root must still
    # reject, even with the exact correct digest.
    meta = _meta(8, 4)
    data = _pack(meta, _points(8, 4))
    owned = tmp_path / "owned"
    owned.mkdir()
    real_file = owned / "cloud.bin"
    real_file.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    link = owned / "link.bin"
    link.symlink_to(real_file)
    with pytest.raises((ValueError, OSError)) as excinfo:
        analyze_cloud_file(meta, str(link), allowed_root=str(owned), expected_sha256=digest)
    assert len(str(excinfo.value).encode("utf-8")) <= 512
    assert str(link) not in str(excinfo.value)


def test_parent_directory_symlink_inside_root_rejected(tmp_path):
    # A symlink in any parent directory component must reject even when the
    # resolved destination stays inside allowed_root.
    meta = _meta(8, 4)
    data = _pack(meta, _points(8, 4))
    owned = tmp_path / "owned"
    owned.mkdir()
    real_dir = owned / "real_dir"
    real_dir.mkdir()
    real_file = real_dir / "cloud.bin"
    real_file.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    dir_link = owned / "dir_link"
    dir_link.symlink_to(real_dir, target_is_directory=True)
    via_link = dir_link / "cloud.bin"
    with pytest.raises((ValueError, OSError)) as excinfo:
        analyze_cloud_file(meta, str(via_link), allowed_root=str(owned), expected_sha256=digest)
    assert len(str(excinfo.value).encode("utf-8")) <= 512


def test_nested_ordinary_path_inside_root_still_accepted(tmp_path):
    meta = _meta(8, 4)
    data = _pack(meta, _points(8, 4))
    owned = tmp_path / "owned"
    nested = owned / "a" / "b"
    nested.mkdir(parents=True)
    path = nested / "cloud.bin"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    got = analyze_cloud_file(meta, str(path), allowed_root=str(owned), expected_sha256=digest)
    assert got["status"] == "VALID_CLOUD"
    assert got["total_count"] == 32


def test_missing_long_path_bounded_diagnostic_no_echo(tmp_path):
    meta = _meta(8, 4)
    owned = tmp_path / "owned"
    owned.mkdir()
    missing = str(owned / ("absent_" + "x" * 900 + ".bin"))
    with pytest.raises(OSError) as excinfo:
        analyze_cloud_file(meta, missing, allowed_root=str(owned), expected_sha256="0" * 64)
    msg = str(excinfo.value)
    assert len(msg.encode("utf-8")) <= 512
    assert "absent_" not in msg
    assert missing not in msg


def test_parent_symlink_dotdot_rejected_before_payload_read(tmp_path):
    # ``dir_link/../cloud.bin``: the canonical target stays inside the root,
    # but the raw parent component is a symlink and must reject. A lexical
    # ``..`` collapse before the component walk would hide the link.
    meta = _meta(8, 4)
    data = _pack(meta, _points(8, 4))
    owned = tmp_path / "owned"
    owned.mkdir()
    real_dir = owned / "real_dir"
    real_dir.mkdir()
    real_file = owned / "cloud.bin"
    real_file.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    dir_link = owned / "dir_link"
    dir_link.symlink_to(real_dir, target_is_directory=True)
    via_dotdot = owned / "dir_link" / ".." / "cloud.bin"
    with pytest.raises((ValueError, OSError)) as excinfo:
        analyze_cloud_file(meta, str(via_dotdot), allowed_root=str(owned), expected_sha256=digest)
    assert len(str(excinfo.value).encode("utf-8")) <= 512
    assert str(via_dotdot) not in str(excinfo.value)


def test_ordinary_dotdot_path_inside_root_still_accepted(tmp_path):
    # ``ordinary/../ordinary/cloud.bin`` with no symlink anywhere must stay
    # a valid positive: dotdot alone is not a rejection cause.
    meta = _meta(8, 4)
    data = _pack(meta, _points(8, 4))
    owned = tmp_path / "owned"
    ordinary = owned / "ordinary"
    ordinary.mkdir(parents=True)
    path = ordinary / "cloud.bin"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    via_dotdot = str(ordinary / ".." / "ordinary" / "cloud.bin")
    got = analyze_cloud_file(meta, via_dotdot, allowed_root=str(owned), expected_sha256=digest)
    assert got["status"] == "VALID_CLOUD"
    assert got["total_count"] == 32


def test_missing_raw_component_dotdot_rejected_before_payload_read(tmp_path):
    # ``missing/../cloud.bin``: the canonical target exists inside the root,
    # but a raw caller component is unreadable/unresolvable and must reject
    # before the payload read; the walk must not break out and trust the
    # normalized destination.
    meta = _meta(8, 4)
    data = _pack(meta, _points(8, 4))
    owned = tmp_path / "owned"
    owned.mkdir()
    real_file = owned / "cloud.bin"
    real_file.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    via_missing = str(owned / "missing" / ".." / "cloud.bin")
    with pytest.raises((ValueError, OSError)) as excinfo:
        analyze_cloud_file(meta, via_missing, allowed_root=str(owned), expected_sha256=digest)
    assert len(str(excinfo.value).encode("utf-8")) <= 512
    assert via_missing not in str(excinfo.value)


def test_nonregular_fifo_rejected_promptly(tmp_path):
    meta = _meta(8, 4)
    owned = tmp_path / "owned"
    owned.mkdir()
    fifo = owned / "pipe"
    os.mkfifo(fifo)
    start = time.monotonic()
    with pytest.raises((OSError, ValueError)) as excinfo:
        analyze_cloud_file(meta, str(fifo), allowed_root=str(owned), expected_sha256="0" * 64)
    assert time.monotonic() - start < 1.0
    assert len(str(excinfo.value).encode("utf-8")) <= 512
