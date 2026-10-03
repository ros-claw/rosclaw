"""模型身份测试（PR-MH9，0915 优化文档 §四，红→绿）。

model_digest 必须是真正的"物理模型身份"：canonical MJCF +
全部资产 digest。相同 XML、不同 mesh 字节 → 不同 model_digest
→ 跨模型状态必须拒绝。
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

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


@pytest.mark.parametrize("asset_tag", ["mesh", "hfield", "texture", "skin"])
def test_uncaptured_external_file_rejected_before_compilation(mesh_backend, tmp_path, asset_tag):
    external = tmp_path / "part.stl"
    external.write_bytes(STL_A)
    xml = f'<mujoco><asset><{asset_tag} name="outside" file="{external}"/></asset></mujoco>'
    with pytest.raises(ValueError, match="MODEL_ASSET_UNBOUND"):
        mesh_backend.load_model_xml(xml, source={"kind": "fixture"})


def test_external_include_cannot_escape_model_identity(mesh_backend, tmp_path):
    external = tmp_path / "include.xml"
    external.write_text('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match="MODEL_ASSET_UNBOUND"):
        mesh_backend.load_model_xml(
            f'<mujoco><include file="{external}"/></mujoco>', source={"kind": "fixture"}
        )


def test_existing_uncaptured_manifest_rejected_in_fresh_process(mesh_backend, tmp_path):
    import subprocess
    import sys

    external = tmp_path / "part.stl"
    external.write_bytes(STL_B)
    ref = mesh_backend.store.put(
        "models",
        {
            "kind": "model_manifest",
            "mjcf_xml": MESH_MODEL.replace("part.stl", str(external)),
            "assets": {},
            "backend": "mujoco",
            "backend_version": "3.13.0",
        },
    )
    code = """import sys
from rosclaw.sim.backends.mujoco.backend import MujocoBackend
try:
    MujocoBackend(sys.argv[1]).initial_state(sys.argv[2])
except ValueError as exc:
    assert 'MODEL_ASSET_UNBOUND' in str(exc), str(exc)
else:
    raise AssertionError('old manifest used mutable filesystem asset')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), ref],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_captured_mesh_survives_source_mutation_in_fresh_process(mesh_backend, tmp_path):
    import json
    import subprocess
    import sys

    original = mesh_backend.load_model("a/model.xml")
    original_state = mesh_backend.initial_state(original.model_ref)
    code = """import json, sys
from rosclaw.sim.backends.mujoco.backend import MujocoBackend
b = MujocoBackend(sys.argv[1])
manifest = b.store.get(sys.argv[2])
model, _ = b._compile_smoke(b._spec_from_manifest(manifest))
b.restore_state(sys.argv[2], sys.argv[3])
print(json.dumps({'digest': b._model_digest(manifest), 'inertia': model.body_inertia.tolist()}))
"""
    args = [sys.executable, "-c", code, str(tmp_path), original.model_ref, original_state]
    before = json.loads(
        subprocess.check_output(
            args,
            text=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        )
    )
    (tmp_path / "a" / "part.stl").write_bytes(STL_B)
    after = json.loads(
        subprocess.check_output(
            args,
            text=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        )
    )
    assert before == after
    changed = mesh_backend.load_model("a/model.xml")
    assert changed.model_digest != original.model_digest
    changed_state = mesh_backend.initial_state(changed.model_ref)
    changed_args = [sys.executable, "-c", code, str(tmp_path), changed.model_ref, changed_state]
    actual_changed = json.loads(
        subprocess.check_output(
            changed_args,
            text=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        )
    )
    assert actual_changed["inertia"] != before["inertia"]
    with pytest.raises(ValueError, match="CROSS_MODEL_REF"):
        mesh_backend.restore_state(changed.model_ref, original_state)


@pytest.mark.parametrize("directory_setting", ["meshdir", "assetdir"])
def test_mesh_directory_captured_bytes_compile_without_source(tmp_path, directory_setting):
    import subprocess
    import sys

    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    assetdir = tmp_path / "assets"
    assetdir.mkdir()
    (assetdir / "part.stl").write_bytes(STL_A)
    xml = MESH_MODEL.replace("<asset>", f'<compiler {directory_setting}="assets"/><asset>')
    (tmp_path / "model.xml").write_text(xml)
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model("model.xml")
    (assetdir / "part.stl").unlink()
    code = "from rosclaw.sim.backends.mujoco.backend import MujocoBackend; import sys; MujocoBackend(sys.argv[1]).inspect_model(sys.argv[2])"
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), ref.model_ref],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_texture_directory_captured_bytes_compile_without_source(tmp_path):
    import subprocess
    import sys

    from PIL import Image

    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    assetdir = tmp_path / "assets"
    assetdir.mkdir()
    Image.new("RGB", (2, 2), color="red").save(assetdir / "surface.png")
    (tmp_path / "model.xml").write_text("""<mujoco>
      <compiler texturedir="assets"/>
      <asset><texture name="tex" type="2d" file="surface.png"/>
      <material name="mat" texture="tex"/></asset>
      <worldbody><geom type="sphere" size=".1" material="mat"/></worldbody>
    </mujoco>""")
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model("model.xml")
    (assetdir / "surface.png").unlink()
    code = "from rosclaw.sim.backends.mujoco.backend import MujocoBackend; import sys; MujocoBackend(sys.argv[1]).inspect_model(sys.argv[2])"
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), ref.model_ref],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_task_local_include_requires_flattened_closure(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "child.xml").write_text(
        '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
    )
    (tmp_path / "model.xml").write_text('<mujoco><include file="child.xml"/></mujoco>')
    with pytest.raises(ValueError, match="MODEL_ASSET_UNBOUND"):
        MujocoBackend(tmp_path).load_model("model.xml")


def test_hfield_captured_bytes_compile_without_source(tmp_path):
    import subprocess
    import sys

    from PIL import Image

    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    Image.new("L", (3, 3), color=128).save(tmp_path / "terrain.png")
    (tmp_path / "model.xml").write_text("""<mujoco>
      <asset><hfield name="terrain" file="terrain.png" size="1 1 1 .1"/></asset>
      <worldbody><geom type="hfield" hfield="terrain"/></worldbody>
    </mujoco>""")
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model("model.xml")
    (tmp_path / "terrain.png").unlink()
    code = "from rosclaw.sim.backends.mujoco.backend import MujocoBackend; import sys; MujocoBackend(sys.argv[1]).inspect_model(sys.argv[2])"
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), ref.model_ref],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert result.returncode == 0, result.stderr
