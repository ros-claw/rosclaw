"""transplant_state 结构签名测试（PR-MH9，0915 优化 §八，红→绿）。

维度一致 ≠ 语义兼容：两台 nq/nv/nu 相同的机器人，qpos[0] 可以
一个是 shoulder 一个是 wrist。默认只允许 lineage-compatible +
structural-signature-compatible 的移植。
"""

from __future__ import annotations

import pytest

TINY_MJCF = """<mujoco model="tiny_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
      <geom name="base_geom" type="capsule" size="0.05 0.2" mass="1.0"/>
      <body name="forearm" pos="0 0 0.4">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="forearm_geom" type="capsule" size="0.03 0.2" mass="0.5"/>
        <site name="tool0" pos="0 0 0.3"/>
      </body>
    </body>
    <camera name="top" pos="0 0 2"/>
  </worldbody>
  <actuator>
    <position name="shoulder_servo" joint="shoulder" kp="10" ctrlrange="-1.57 1.57" forcerange="-50 50"/>
    <position name="elbow_servo" joint="elbow" kp="5"/>
  </actuator>
  <sensor>
    <jointpos name="shoulder_pos" joint="shoulder"/>
  </sensor>
</mujoco>
"""

SIBLING_ARM = """<mujoco model="sibling_arm">
  <compiler autolimits="true"/>
  <worldbody>
    <body name="hip" pos="0 0 0">
      <joint name="wrist_roll" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
      <geom name="s0" type="capsule" size="0.05 0.2" mass="1.0"/>
      <body name="segment" pos="0 0 0.4">
        <joint name="wrist_pitch" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="s1" type="capsule" size="0.03 0.2" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="srv_a" joint="wrist_roll" kp="10"/>
    <position name="srv_b" joint="wrist_pitch" kp="5"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "arm.xml").write_text(TINY_MJCF, encoding="utf-8")
    (tmp_path / "sibling.xml").write_text(SIBLING_ARM, encoding="utf-8")
    return MujocoBackend(tmp_path)


def test_transplant_within_lineage_ok(backend) -> None:
    ref = backend.load_model("arm.xml")
    state_ref = backend.initial_state(ref.model_ref)
    patched = backend.patch_model(
        ref.model_ref,
        [{"op": "set", "target": {"type": "joint", "name": "shoulder"}, "field": "damping", "value": 2.0}],
    )
    # 参数 patch（damping/kp/mass/friction）不改结构签名 → 允许。
    new_ref = backend.transplant_state(patched.new_model_ref, state_ref)
    assert new_ref.startswith("simsta_")
    snap = backend.store.get(new_ref)
    assert snap["transplanted_from"] == state_ref
    assert snap["model_ref"] == patched.new_model_ref


def test_transplant_rejects_structural_mismatch(backend) -> None:
    ref = backend.load_model("arm.xml")
    sibling = backend.load_model("sibling.xml")
    state_ref = backend.initial_state(ref.model_ref)

    # nq/nv/nu 完全相同（2/2/2），但 joint 名/类型映射不同 → 拒绝。
    inspection_a = backend.inspect_model(ref.model_ref)
    inspection_b = backend.inspect_model(sibling.model_ref)
    assert (inspection_a.nq, inspection_a.nv, inspection_a.nu) == (
        inspection_b.nq, inspection_b.nv, inspection_b.nu
    )
    with pytest.raises(ValueError, match="STATE_INCOMPATIBLE"):
        backend.transplant_state(sibling.model_ref, state_ref)


def test_transplant_rejects_dimension_mismatch(backend) -> None:
    ref = backend.load_model("arm.xml")
    state_ref = backend.initial_state(ref.model_ref)
    (backend._task_root / "three.xml").write_text(
        TINY_MJCF.replace(
            '<joint name="elbow" type="hinge" axis="0 1 0" damping="0.1"/>',
            '<joint name="elbow" type="hinge" axis="0 1 0" damping="0.1"/>'
            '<body name="extra" pos="0 0 0.2"><joint name="wrist" type="hinge" axis="0 1 0"/>'
            '<geom name="g2" type="sphere" size="0.02" mass="0.1"/></body>',
        ),
        encoding="utf-8",
    )
    three = backend.load_model("three.xml")
    # 2 关节状态 → 3 关节模型：维度即拒（fail closed）。
    with pytest.raises(ValueError, match="STATE_DIMENSION|STATE_INCOMPATIBLE"):
        backend.transplant_state(three.model_ref, state_ref)
