"""模型身份测试（PR-MH9，0915 优化文档 §四，红→绿）。

model_digest 必须是真正的"物理模型身份"：canonical MJCF +
全部资产 digest。相同 XML、不同 mesh 字节 → 不同 model_digest
→ 跨模型状态必须拒绝。
"""

from __future__ import annotations

import struct

import pytest


def _binary_stl(scale: float) -> bytes:
    """最小合法二进制 STL（四面体，4 顶点 4 面）。"""
    verts = [
        (0.0, 0.0, 0.0),
        (scale, 0.0, 0.0),
        (0.0, scale, 0.0),
        (0.0, 0.0, scale),
    ]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    body = b""
    for a, b, c in faces:
        body += struct.pack(
            "<12fH",
            0.0,
            0.0,
            1.0,
            *verts[a],
            *verts[b],
            *verts[c],
            0,
        )
    return b"\0" * 80 + struct.pack("<I", len(faces)) + body


STL_A = _binary_stl(1.0)
STL_B = _binary_stl(2.0)

MESH_MODEL = """<mujoco model="mesh_bot">
  <asset>
    <mesh name="part" file="part.stl"/>
  </asset>
  <worldbody>
    <body name="base" pos="0 0 0.2">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom name="g" type="mesh" mesh="part" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def mesh_backend(tmp_path):
    """同一 task_root 下两个"XML 相同、mesh 字节不同"的模型目录。"""
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    for sub, blob in (("a", STL_A), ("b", STL_B)):
        subdir = tmp_path / sub
        subdir.mkdir()
        (subdir / "model.xml").write_text(MESH_MODEL, encoding="utf-8")
        (subdir / "part.stl").write_bytes(blob)
    return MujocoBackend(tmp_path)


def test_same_xml_different_mesh_different_digest(mesh_backend) -> None:
    ref_a = mesh_backend.load_model("a/model.xml")
    ref_b = mesh_backend.load_model("b/model.xml")

    # XML 逐字节相同，manifest 不同——但 model_digest 也必须不同。
    xml_a = mesh_backend.store.get(ref_a.model_ref)["mjcf_xml"]
    xml_b = mesh_backend.store.get(ref_b.model_ref)["mjcf_xml"]
    assert xml_a == xml_b
    assert ref_a.model_digest != ref_b.model_digest


def test_cross_model_state_rejected_on_asset_difference(mesh_backend) -> None:
    ref_a = mesh_backend.load_model("a/model.xml")
    ref_b = mesh_backend.load_model("b/model.xml")

    # XML 相同、资产不同 = 不同物理模型——跨模型状态 fail closed。
    state_ref = mesh_backend.initial_state(ref_a.model_ref)
    with pytest.raises(ValueError, match="CROSS_MODEL_REF"):
        mesh_backend.restore_state(ref_b.model_ref, state_ref)


def test_digest_stable_across_reloads(mesh_backend) -> None:
    ref1 = mesh_backend.load_model("a/model.xml")
    ref2 = mesh_backend.load_model("a/model.xml")
    assert ref1.model_digest == ref2.model_digest  # 幂等不漂
