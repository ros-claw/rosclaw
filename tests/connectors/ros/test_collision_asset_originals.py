"""Original mesh bounds are offline evidence, never actual loaded geometry."""

import base64
import hashlib
import math
import struct

import pytest

from rosclaw.connectors.ros.context.collision_assets import review_collision_assets, stl_bounds
from rosclaw.connectors.ros.context.geometry import derive_collision_envelope

VERTICES = ((1.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.0, 4.0, 2.0))
URI = "package://anonymous_description/meshes/chassis.stl"


def binary():
    return (
        b"solid binary header".ljust(80, b" ")
        + struct.pack("<I", 1)
        + struct.pack("<12fH", 0.0, 0.0, 1.0, *(x for v in VERTICES for x in v), 0)
    )


def ascii_stl():
    vertices = "\n".join("vertex " + " ".join(str(x) for x in v) for v in VERTICES)
    return (
        "solid original\nfacet normal 0 0 1\nouter loop\n"
        + vertices
        + "\nendloop\nendfacet\nendsolid original\n"
    ).encode()


def asset(raw=None):
    raw = binary() if raw is None else raw
    return {
        "uri": URI,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content_base64": base64.b64encode(raw).decode(),
    }


@pytest.mark.parametrize("raw,encoding", [(binary(), "BINARY_STL"), (ascii_stl(), "ASCII_STL")])
def test_original_extrema_cover_all_vertices_without_modifying_physics(raw, encoding):
    result = stl_bounds(raw)
    assert result["encoding"] == encoding and result["triangle_count"] == 1
    assert result["local_min"] == [1.0, 2.0, 0.0] and result["local_max"] == [3.0, 4.0, 2.0]
    assert result["original_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (
        result["physical_loaded_geometry"] == "NOT_VERIFIED"
        and result["execution_authority"] == "NONE"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "truncated",
        "trailing",
        "zero_count",
        "large_count",
        "nan",
        "inf",
        "empty",
        "ascii_missing_end",
        "ascii_truncated",
        "ascii_executable",
        "noncanonical_base64",
        "wrong_sha",
        "duplicate",
        "traversal",
        "network",
        "extra_fields",
    ],
)
def test_invalid_or_unbound_originals_fail_closed(fault):
    raw = binary()
    if fault == "truncated":
        raw = raw[:-1]
    elif fault == "trailing":
        raw += b"x"
    elif fault == "zero_count":
        raw = raw[:80] + struct.pack("<I", 0)
    elif fault == "large_count":
        raw = raw[:80] + struct.pack("<I", 100001) + raw[84:]
    elif fault in {"nan", "inf"}:
        raw = raw[:96] + struct.pack("<f", float(fault)) + raw[100:]
    elif fault == "empty":
        raw = b""
    elif fault == "ascii_missing_end":
        raw = ascii_stl().replace(b"endsolid original", b"missing")
    elif fault == "ascii_truncated":
        raw = ascii_stl().replace(b"vertex 1.0 2.0 0.0\n", b"")
    elif fault == "ascii_executable":
        raw = ascii_stl().replace(b"vertex 1.0", b"vertex __import__('os')")
    row = asset(raw)
    rows = [row]
    if fault == "noncanonical_base64":
        row["content_base64"] += "\n"
    elif fault == "wrong_sha":
        row["sha256"] = "0" * 64
    elif fault == "duplicate":
        rows.append(dict(row))
    elif fault == "traversal":
        row["uri"] = URI.replace("/meshes/", "/../")
    elif fault == "network":
        row["uri"] = "https://unbound.invalid/chassis.stl"
    elif fault == "extra_fields":
        row["verified"] = True
    with pytest.raises(ValueError):
        review_collision_assets(rows)


def mesh_urdf(scale="1 1 1", origin="0 0 0"):
    return f'<robot name="anonymous"><link name="base"><collision><origin xyz="{origin}"/><geometry><mesh filename="{URI}" scale="{scale}"/></geometry></collision></link></robot>'.encode()


def test_mesh_requires_exact_originals_and_preserves_original_urdf_and_offset_scale():
    raw = mesh_urdf("2 3 1", "-1 -2 0")
    unknown = derive_collision_envelope(raw, base_frame="base")
    assert unknown["complete"] is False and "original_mesh_assets" not in unknown
    result = derive_collision_envelope(raw, base_frame="base", mesh_originals=[asset()])
    assert result["complete"] is True
    assert result["physical_radius_m"] == pytest.approx(math.hypot(5, 10))
    assert result["source_urdf_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["physical_mesh_loading"] == "NOT_VERIFIED_NOT_RUNTIME_OR_BODY_ADMISSION"
    assert (
        result["capabilities_granted"] == []
        and result["collision_count"] == result["resolved_collision_count"] == 1
    )
    assert unknown == derive_collision_envelope(raw, base_frame="base")


@pytest.mark.parametrize("scale", ["0 1 1", "-1 1 1", "nan 1 1", "1 1", "1e309 1 1"])
def test_unsupported_scale_never_synthesizes_a_default_radius(scale):
    result = derive_collision_envelope(
        mesh_urdf(scale), base_frame="base", mesh_originals=[asset()]
    )
    assert result["complete"] is False and result["physical_radius_m"] is None


def test_wrong_mesh_uri_does_not_bind_by_filename_or_nearest_package():
    row = asset()
    row["uri"] = URI.replace("anonymous_description", "other_description")
    result = derive_collision_envelope(mesh_urdf(), base_frame="base", mesh_originals=[row])
    assert result["complete"] is False


def test_mesh_inspection_cli_preserves_originals_and_refuses_overwrite_and_alias(tmp_path, capsys):
    import json

    from rosclaw.connectors.ros.cli.ros_cli import dispatch_ros_command
    from tests.connectors.ros.test_intervention_cli import parser

    raw = tmp_path / "original.urdf"
    raw.write_bytes(mesh_urdf())
    data = tmp_path / "original-assets.json"
    data.write_text(json.dumps([asset()]))
    before = raw.read_bytes(), data.read_bytes()
    output = tmp_path / "derived.json"
    argv = [
        "ros",
        "inspect-collision-assets",
        "--urdf",
        str(raw),
        "--base-frame",
        "base",
        "--asset-originals",
        str(data),
        "--json",
        "--output",
        str(output),
    ]
    assert dispatch_ros_command(parser().parse_args(argv)) == 0
    result = json.loads(capsys.readouterr().out)
    assert (
        result["collision_envelope"]["complete"] is True
        and result["physical_loaded_geometry"] == "NOT_VERIFIED"
    )
    assert result["execution_authority"] == "NONE" and result["current_l1"] == "NOT_EVALUATED"
    saved = output.read_bytes()
    assert dispatch_ros_command(parser().parse_args(argv)) != 0
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert output.read_bytes() == saved and before == (raw.read_bytes(), data.read_bytes())
    link = tmp_path / "alias.json"
    link.symlink_to(data)
    argv[argv.index("--asset-originals") + 1] = str(link)
    argv = argv[:-2]
    assert dispatch_ros_command(parser().parse_args(argv)) != 0
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_cached_original_bytes_are_exact_bounded_and_caller_mutation_isolated():
    from rosclaw.connectors.ros.context.collision_assets import _stl_bounds_json

    _stl_bounds_json.cache_clear()
    raw = binary()
    original = stl_bounds(raw)
    first = _stl_bounds_json.cache_info()
    modified = stl_bounds(raw)
    modified["local_min"][0] = -999
    modified["original_sha256"] = "forged"
    assert stl_bounds(raw) == original
    assert _stl_bounds_json.cache_info().hits == first.hits + 2
    changed = b"different header".ljust(80, b" ") + raw[80:]
    assert stl_bounds(changed)["original_sha256"] != original["original_sha256"]
    for i in range(12):
        stl_bounds(str(i).encode().ljust(80, b" ") + raw[80:])
    assert _stl_bounds_json.cache_info().currsize == 8
    with pytest.raises(ValueError):
        stl_bounds(bytearray(raw))
    with pytest.raises(ValueError):
        review_collision_assets([dict(asset(changed), sha256=original["original_sha256"])])
    _stl_bounds_json.cache_clear()
