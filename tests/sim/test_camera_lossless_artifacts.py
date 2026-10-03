"""Actual renderer fixtures verify metric depth and high-id segmentation without stepping."""

import io

import numpy as np
import pytest

from rosclaw.sim.backends.mujoco.backend import MujocoBackend


def _camera_fixture(tmp_path, *, many=False):
    hidden = (
        "".join(f'<geom name="off{i}" pos="100 0 0" size=".01"/>' for i in range(260))
        if many
        else ""
    )
    xml = (
        '<mujoco><statistic extent="2"/><visual><global offwidth="640" offheight="480"/></visual>'
        "<worldbody>" + hidden + '<geom name="target" type="box" size=".2 .2 .05" pos="0 0 0"/>'
        '<camera name="front" pos="0 0 1" quat="1 0 0 0" fovy="60"/>'
        "</worldbody></mujoco>"
    )
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model_xml(xml, source={"kind": "renderer_fixture"}).model_ref
    state = backend.initial_state_v2(ref)
    return backend, ref, state


def _observe(backend, ref, state, kind):
    try:
        return backend.observe(ref, state, [kind + ":front"]).values[kind + ":front"]
    except ValueError as exc:
        if str(exc).startswith("SIM_RENDER_UNAVAILABLE"):
            pytest.skip(str(exc))  # Honest environment skip, never a simulated pass.
        raise


def test_actual_metric_depth_raw_is_lossless_and_not_per_frame_normalized(tmp_path):
    backend, ref, state = _camera_fixture(tmp_path)
    value = _observe(backend, ref, state, "camera_depth")
    raw = np.load(io.BytesIO(backend.store.get(value["raw_artifact_ref"])), allow_pickle=False)
    assert raw.shape == (480, 640) and raw.dtype == np.float32
    assert raw[240, 320] == pytest.approx(0.95, abs=0.001)
    assert value["raw_units"] == "metre"
    assert value["artifact_semantics"] == "visualization_only"
    assert value["model_ref"] == ref and value["state_ref"] == state
    assert value["raw_shape"] == [480, 640]
    assert value["extrinsics"]["pos"] == [0, 0, 1]
    assert value["raw_artifact_ref"] != value["artifact_ref"]
    manifest = backend.store.get(value["observation_manifest_ref"])
    assert manifest["schema_version"] == "rosclaw.sim.camera.v2"
    assert manifest["raw_artifact_ref"] == value["raw_artifact_ref"]
    assert manifest["model_ref"] == ref and manifest["state_ref"] == state
    assert manifest["intrinsics"] == value["intrinsics"]
    assert manifest["extrinsics"] == value["extrinsics"]


def test_actual_segmentation_preserves_ids_above255_type_and_background(tmp_path):
    backend, ref, state = _camera_fixture(tmp_path, many=True)
    value = _observe(backend, ref, state, "camera_segmentation")
    raw = np.load(io.BytesIO(backend.store.get(value["raw_artifact_ref"])), allow_pickle=False)
    assert raw.shape == (480, 640, 2) and raw.dtype == np.int32
    assert raw[240, 320].tolist() == [260, 5]  # mjOBJ_GEOM is 5 in installed MuJoCo
    assert np.any(np.all(raw == [-1, -1], axis=2))
    assert value["raw_channels"] == ["object_id", "object_type"]
    assert value["background"] == [-1, -1]
    assert value["raw_units"] == "categorical"


def test_raw_depth_retains_absolute_units_across_camera_distances(tmp_path):
    backend, ref, state = _camera_fixture(tmp_path)
    first = _observe(backend, ref, state, "camera_depth")
    source = backend.store.get(ref)["mjcf_xml"].replace('pos="0 0 1"', 'pos="0 0 2"')
    other = backend.load_model_xml(source, source={"kind": "renderer_fixture"}).model_ref
    second = _observe(backend, other, backend.initial_state_v2(other), "camera_depth")
    a, b = [
        np.load(io.BytesIO(backend.store.get(v["raw_artifact_ref"])), allow_pickle=False)
        for v in (first, second)
    ]
    assert a[240, 320] == pytest.approx(0.95, abs=0.001)
    assert b[240, 320] == pytest.approx(1.95, abs=0.001)
    assert b[240, 320] - a[240, 320] == pytest.approx(1, abs=0.002)
