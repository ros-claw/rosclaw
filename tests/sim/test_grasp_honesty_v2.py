"""Grasp Honesty v2 测试（MH20-C，讨论总纲 §8-§11，红→绿）。

main 复核实证四处真值问题：
1. "接触证据"实为 geom 最小距离 ≤1cm（proximity 冒充 contact）；
2. weld rel_quat 存 world-frame body2 quat（body1 旋转即错）；
3. eq_data 布局错位（实测 3.13：anchor[0:3] / pos[3:6] /
   quat[6:10] / torquescale[10]，当前代码 [0:3]=pos [3:7]=identity
   [7:11]=quat 全错）；
4. release 证据看全局 max qvel（其他关节加速即假阳性）。
"""

from __future__ import annotations

import numpy as np
import pytest

WELD_WORLD = """<mujoco model="weld_world">
  <compiler autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="arm_base" pos="0 0 0.3">
      <joint name="lift" type="slide" axis="0 0 1"/>
      <geom name="palm" type="box" size="0.05 0.05 0.02" mass="1.0"/>
      <body name="finger" pos="0.05 0 0">
        <joint name="grip" type="slide" axis="1 0 0" range="-0.03 0.03"/>
        <geom name="fg" type="box" size="0.012 0.025 0.025" mass="0.05"/>
      </body>
    </body>
    <body name="cube" pos="0.05 0 0.25">
      <freejoint name="cube_free"/>
      <geom name="cube_g" type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
  </worldbody>
  <equality>
    <weld name="grip_weld" body1="finger" body2="cube" active="false"/>
  </equality>
  <actuator>
    <position name="grip_servo" joint="grip" kp="200"/>
    <position name="lift_servo" joint="lift" kp="200"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def runtime(tmp_path):
    from rosclaw.sim.runtime import SimulationRuntime

    (tmp_path / "weld.xml").write_text(WELD_WORLD, encoding="utf-8")
    return SimulationRuntime(tmp_path)


def _attach(runtime, model_ref, state_ref, **extra):
    payload = {"weld": "grip_weld", **extra}
    return runtime.interact(
        model_ref,
        state_ref,
        {"executor": "constraint_attach", "target": {"type": "equality", "name": "grip_weld"}},
        payload,
    )


def _restore(runtime, model_ref, state_ref):
    return runtime.backend.restore_state_v2(model_ref, state_ref)


# ---------------------------------------------------------------- 接触证据三层


def test_proximity_is_not_contact(runtime) -> None:
    """§8.2：cube 在 gripper 附近（<1cm）但无接触对 → 默认
    constraint_attach 必须拒绝（proximity ≠ contact）。"""
    loaded = runtime.load_model("weld.xml")
    snap = runtime.snapshot(loaded["model_ref"])
    # cube 在 finger 上方 0.25-0.02-0.012=0.218m？把 lift 上抬让
    # finger 接近但不接触 cube（间隙 ~5mm）。
    lifted = runtime.interact(
        loaded["model_ref"],
        snap["state_ref"],
        {"executor": "joint_target", "target": {"type": "joint", "name": "lift"}},
        {"target": -0.18, "duration_s": 1.0},
    )
    with pytest.raises(ValueError, match="INTERACTION_NO_CONTACT_EVIDENCE"):
        _attach(runtime, loaded["model_ref"], lifted["state_ref"])


def test_proximity_assisted_attach_is_honestly_named(runtime) -> None:
    """§8.3：proximity abstraction 必须显式降级命名——payload 声明
    evidence_level=PROXIMITY_ASSISTED_ATTACH 才允许，且 receipt
    不得标 constraint_assisted_grasp 的 CONTACT 级证据。"""
    loaded = runtime.load_model("weld.xml")
    snap = runtime.snapshot(loaded["model_ref"])
    lifted = runtime.interact(
        loaded["model_ref"],
        snap["state_ref"],
        {"executor": "joint_target", "target": {"type": "joint", "name": "lift"}},
        {"target": -0.18, "duration_s": 1.0},
    )
    attached = _attach(
        runtime,
        loaded["model_ref"],
        lifted["state_ref"],
        evidence_level="PROXIMITY_ASSISTED_ATTACH",
    )
    outcome = attached["outcome"]
    assert outcome["evidence_level"] == "PROXIMITY_ASSISTED_ATTACH"
    assert outcome["evidence_level"] != "CONTACT"


def test_real_contact_passes_with_evidence(runtime) -> None:
    """§8.2：真实接触（finger 压到 cube）→ attach 允许，且记录
    contact_count/min_penetration/normal_force 证据。"""
    loaded = runtime.load_model("weld.xml")
    snap = runtime.snapshot(loaded["model_ref"])
    # lift 上抬使 finger 顶面贴到 cube 底面并压出接触。
    pressed = runtime.interact(
        loaded["model_ref"],
        snap["state_ref"],
        {"executor": "joint_target", "target": {"type": "joint", "name": "lift"}},
        {"target": -0.19, "duration_s": 1.5},
    )
    attached = _attach(runtime, loaded["model_ref"], pressed["state_ref"])
    outcome = attached["outcome"]
    assert outcome["evidence_level"] == "CONTACT"
    assert outcome["contact_evidence"]["contact_count"] >= 1
    assert "normal_force" in outcome["contact_evidence"]


# ---------------------------------------------------------------- 相对位姿


def test_relative_body_pose_quaternion(runtime) -> None:
    """§9：relative_body_pose 必须返回 q_rel = inverse(q1) ⊗ q2——
    body1 yaw=90°、body2 yaw=120° 时 rel yaw = 30°（不是 120°）。"""
    from rosclaw.sim.backends.mujoco.interact import relative_body_pose

    def yaw_quat(deg: float) -> np.ndarray:
        half = np.deg2rad(deg) / 2
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

    pos, quat = relative_body_pose(
        pos1=np.array([1.0, 2.0, 3.0]),
        quat1=yaw_quat(90.0),
        pos2=np.array([1.5, 2.0, 3.0]),
        quat2=yaw_quat(120.0),
    )
    # 位置：R1^T (p2-p1)——yaw90 下 +x_world 变 -y_local。
    assert pos == pytest.approx([0.0, -0.5, 0.0], abs=1e-9)
    rel_yaw = 2 * np.degrees(np.arctan2(quat[3], quat[0]))
    assert rel_yaw == pytest.approx(30.0, abs=1e-6)


def test_attach_writes_correct_eq_layout_and_no_orientation_snap(runtime) -> None:
    """§9/§10：attach 后 eq_data 必须是
    anchor[0:3]/pos[3:6]/quat[6:10] 布局 + 正确相对四元数，
    且被焊 body 不发生 orientation snap。"""
    import mujoco

    loaded = runtime.load_model("weld.xml")
    snap = runtime.snapshot(loaded["model_ref"])
    pressed = runtime.interact(
        loaded["model_ref"],
        snap["state_ref"],
        {"executor": "joint_target", "target": {"type": "joint", "name": "lift"}},
        {"target": -0.19, "duration_s": 1.5},
    )
    # attach 前记录 cube 朝向。
    model_before, data_before = _restore(runtime, loaded["model_ref"], pressed["state_ref"])
    cube_id = mujoco.mj_name2id(model_before, mujoco.mjtObj.mjOBJ_BODY, "cube")
    quat_before = np.asarray(data_before.xquat[cube_id], dtype=float).copy()

    attached = _attach(runtime, loaded["model_ref"], pressed["state_ref"])
    model, data = _restore(runtime, loaded["model_ref"], attached["state_ref"])
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip_weld")

    # 布局验证：pos 在 [3:6]，quat 在 [6:10]（torquescale 在 [10]）。
    assert np.linalg.norm(np.asarray(model.eq_data[eq_id][3:6], dtype=float)) > 1e-9
    eq_quat = np.asarray(model.eq_data[eq_id][6:10], dtype=float)
    assert abs(np.linalg.norm(eq_quat) - 1.0) < 1e-9, f"quat 未归一: {eq_quat}"
    # 相对四元数 = inv(q_finger) ⊗ q_cube——相对测量一致。
    finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger")
    q1 = np.asarray(data.xquat[finger_id], dtype=float)
    q2 = np.asarray(data.xquat[cube_id], dtype=float)
    q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    expected = np.zeros(4)
    mujoco.mju_mulQuat(expected, q1_inv, q2)
    if np.dot(expected, eq_quat) < 0:  # 四元数符号等价
        expected = -expected
    assert eq_quat == pytest.approx(expected, abs=1e-6)
    # 无 orientation snap：attach 后 cube 朝向与 attach 前一致。
    quat_after = np.asarray(data.xquat[cube_id], dtype=float)
    if np.dot(quat_after, quat_before) < 0:
        quat_after = -quat_after
    assert quat_after == pytest.approx(quat_before, abs=1e-3)


def test_set_weld_relpose_uses_setconst(runtime) -> None:
    """§10：set_weld_relpose 必须经 mj_setConst——重复 attach/release
    语义稳定（不因残留 const 漂移）。"""
    import mujoco

    loaded = runtime.load_model("weld.xml")
    snap = runtime.snapshot(loaded["model_ref"])
    pressed = runtime.interact(
        loaded["model_ref"],
        snap["state_ref"],
        {"executor": "joint_target", "target": {"type": "joint", "name": "lift"}},
        {"target": -0.19, "duration_s": 1.5},
    )
    first = _attach(runtime, loaded["model_ref"], pressed["state_ref"])
    model, data = _restore(runtime, loaded["model_ref"], first["state_ref"])
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger")
    rel_before = np.asarray(data.xpos[cube_id]) - np.asarray(data.xpos[finger_id])
    # setConst 正确的 weld：step 后两 body 相对位置不漂。
    for _ in range(150):
        mujoco.mj_step(model, data)
    rel_after = np.asarray(data.xpos[cube_id]) - np.asarray(data.xpos[finger_id])
    drift = float(np.linalg.norm(rel_after - rel_before))
    assert drift < 0.005, f"weld 相对位置漂移 {drift}m（setConst/relpose 语义错误）"


# ---------------------------------------------------------------- release 证据


def test_release_evidence_payload_specific(runtime) -> None:
    """§11：release 证据只看 payload（cube）——机器人其他关节加速
    不得造成假阳性；掉落中 cube 必须有自身速度/位移证据。"""
    loaded = runtime.load_model("weld.xml")
    snap = runtime.snapshot(loaded["model_ref"])
    pressed = runtime.interact(
        loaded["model_ref"],
        snap["state_ref"],
        {"executor": "joint_target", "target": {"type": "joint", "name": "lift"}},
        {"target": -0.19, "duration_s": 1.5},
    )
    attached = _attach(runtime, loaded["model_ref"], pressed["state_ref"])
    released = runtime.interact(
        loaded["model_ref"],
        attached["state_ref"],
        {"executor": "constraint_release", "target": {"type": "equality", "name": "grip_weld"}},
        {"weld": "grip_weld", "duration_s": 0.4},
    )
    outcome = released["outcome"]
    # 证据必须指向 payload 自身。
    assert outcome["target_body"] == "cube"
    assert "payload_linear_velocity" in outcome or "payload_displacement" in outcome
    assert outcome["gravity_response"] is True
