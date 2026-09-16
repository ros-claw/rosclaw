"""多模态观测测试（MH13，0916 优化 §十六，红→绿）。

camera_rgb/depth/segmentation 返回 artifact_ref + intrinsics +
extrinsics + simulation_time——不把图像数组塞进 JSON。
"""

from __future__ import annotations

import pytest

CAM_MODEL = """<mujoco model="cam_world">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.05">
      <freejoint name="bf"/>
      <geom name="box_g" type="box" size="0.05 0.05 0.05" mass="0.5" rgba="1 0 0 1"/>
    </body>
    <camera name="top" pos="0 0 1.5" fovy="60"/>
    <camera name="side" pos="1 0 0.2" fovy="45"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "cam.xml").write_text(CAM_MODEL, encoding="utf-8")
    b = MujocoBackend(tmp_path)
    return b, b.load_model("cam.xml")


def _observe_or_skip(backend, model_ref, state_ref, channels):
    try:
        return backend.observe(model_ref, state_ref, channels)
    except ValueError as exc:
        if "SIM_RENDER_UNAVAILABLE" in str(exc):
            pytest.skip(f"GL backend unavailable on this machine: {exc}")
        raise


def test_camera_rgb_returns_artifact(backend) -> None:
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)
    result = _observe_or_skip(b, ref.model_ref, state, ["camera_rgb:top"])
    value = result.values["camera_rgb:top"]

    assert value["artifact_ref"].startswith("simrnd_")
    assert value["width"] == 640 and value["height"] == 480
    assert value["dtype"] == "uint8"
    assert value["camera"] == "top"
    assert value["intrinsics"]["fovy_deg"] == pytest.approx(60.0)
    assert value["intrinsics"]["focal_px"] > 0
    assert value["intrinsics"]["principal_point"] == [320.0, 240.0]
    assert len(value["extrinsics"]["pos"]) == 3
    assert value["extrinsics"]["pos"] == pytest.approx([0, 0, 1.5])
    assert value["simulation_time"] == pytest.approx(0.0)
    assert value["renderer_backend"] in ("egl", "osmesa")  # 实际后端，不是环境声明

    blob = b.store.get(value["artifact_ref"])
    assert isinstance(blob, bytes)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_camera_depth_dtype_and_artifact(backend) -> None:
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)
    result = _observe_or_skip(b, ref.model_ref, state, ["camera_depth:side"])
    value = result.values["camera_depth:side"]
    assert value["dtype"] == "uint16"
    assert value["artifact_ref"].startswith("simrnd_")
    assert b.store.get(value["artifact_ref"])[:8] == b"\x89PNG\r\n\x1a\n"


def test_camera_segmentation_artifact(backend) -> None:
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)
    result = _observe_or_skip(b, ref.model_ref, state, ["camera_segmentation:top"])
    value = result.values["camera_segmentation:top"]
    assert value["dtype"] == "uint8"
    assert value["artifact_ref"].startswith("simrnd_")


def test_camera_unknown_channel_rejected(backend) -> None:
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)
    with pytest.raises(ValueError, match="OBSERVE_CHANNEL_UNKNOWN"):
        b.observe(ref.model_ref, state, ["camera_rgb:ghost"])


def test_mixed_channels_in_one_call(backend) -> None:
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)
    result = _observe_or_skip(
        b, ref.model_ref, state, ["joint_positions", "camera_rgb:top", "contact_summary"]
    )
    assert "joint_positions" in result.values
    assert "camera_rgb:top" in result.values
    assert "contact_summary" in result.values
