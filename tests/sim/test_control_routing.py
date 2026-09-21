"""ControlSchema 统一路由测试（MH20-B，讨论总纲 §7/§十九，红→绿）。

P0（main 复核）：exec_joint_target/exec_gripper_motion 直接
`data.ctrl[actuator_id] = value`——把 actuator 序号当 ctrl 槽位。
PID 多输入执行器（一个 actuator 占 pos/vel 多槽）时槽位错位：
第一个执行器是 PID 时，第二个执行器的 joint_target 会写进
PID 的 **vel 槽**（ctrl[1]），物理语义全错。

纪律：所有执行路径只经 Canonical ControlSchema 写 ctrl；
业务代码不得出现 `data.ctrl[...]`（architecture test 锁定）。
"""

from __future__ import annotations

import pytest

#: 第一执行器 PID（占 ctrl[0]=pos, ctrl[1]=vel），第二执行器
#: 普通 position（应占 ctrl[2]）。
MIXED_MODEL = """<mujoco model="mixed_bot">
  <compiler autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 0.3">
      <joint name="j_pid" type="hinge" axis="0 1 0" damping="0.2"/>
      <geom name="g1" type="capsule" size="0.04 0.15" mass="0.8"/>
      <body name="link2" pos="0 0 0.2">
        <joint name="j_pos" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="g2" type="capsule" size="0.03 0.12" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <pid name="pid_servo" joint="j_pid" kp="10" kv="2"/>
    <position name="pos_servo" joint="j_pos" kp="10"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def runtime(tmp_path):
    from rosclaw.sim.runtime import SimulationRuntime

    (tmp_path / "mixed.xml").write_text(MIXED_MODEL, encoding="utf-8")
    return SimulationRuntime(tmp_path)


def test_joint_target_hits_correct_ctrl_slot(runtime) -> None:
    """红测试核心：j_pos 的 joint_target 必须写 ctrl[2]（pos_servo
    的 pos 槽），绝不能碰 ctrl[1]（pid_servo 的 vel 槽）。"""
    import numpy as np

    loaded = runtime.load_model("mixed.xml")
    snap = runtime.snapshot(loaded["model_ref"])
    result = runtime.interact(
        loaded["model_ref"],
        snap["state_ref"],
        {"executor": "joint_target", "target": {"type": "joint", "name": "j_pos"}},
        {"target": 0.3, "duration_s": 0.4},
    )
    # 恢复终态直接读 ctrl：vel 槽（ctrl[1]）必须保持 0——
    # 修复前 raw write 会把 0.3 写进 ctrl[1]（vel 槽错位）。
    model, data = runtime.backend.restore_state_v2(loaded["model_ref"], result["state_ref"])
    ctrl = np.asarray(data.ctrl, dtype=float)
    assert len(ctrl) == 3  # pid pos/vel + pos_servo pos
    assert ctrl[1] == pytest.approx(0.0, abs=1e-12), f"vel 槽被污染: ctrl={ctrl}"
    assert ctrl[2] == pytest.approx(0.3, abs=1e-12), f"pos_servo 槽没收到目标: ctrl={ctrl}"
    assert result["outcome"]["reached"] is True


def test_gripper_motion_schema_routed(runtime, tmp_path) -> None:
    """gripper_close 同样必须走 schema（与 joint_target 同族 bug）。"""
    import numpy as np

    (tmp_path / "gripper.xml").write_text(
        MIXED_MODEL.replace(
            '<position name="pos_servo" joint="j_pos" kp="10"/>',
            '<position name="pos_servo" joint="j_pos" kp="50" ctrlrange="-0.5 0.5"/>',
        ),
        encoding="utf-8",
    )
    loaded = runtime.load_model("gripper.xml")
    snap = runtime.snapshot(loaded["model_ref"])
    result = runtime.interact(
        loaded["model_ref"],
        snap["state_ref"],
        {"executor": "gripper_close", "target": {"type": "actuator", "name": "pos_servo"}},
        {"close_target": 0.4, "duration_s": 0.3},
    )
    model, data = runtime.backend.restore_state_v2(loaded["model_ref"], result["state_ref"])
    ctrl = np.asarray(data.ctrl, dtype=float)
    assert ctrl[1] == pytest.approx(0.0, abs=1e-12), f"vel 槽被污染: ctrl={ctrl}"
    assert ctrl[2] == pytest.approx(0.4, abs=1e-12), f"gripper 目标没落到 pos 槽: ctrl={ctrl}"


def test_joint_target_pid_joint_also_schema_routed(runtime) -> None:
    """PID 关节自身的 joint_target → pos 槽（ctrl[0]），不碰 vel。"""
    import numpy as np

    loaded = runtime.load_model("mixed.xml")
    snap = runtime.snapshot(loaded["model_ref"])
    result = runtime.interact(
        loaded["model_ref"],
        snap["state_ref"],
        {"executor": "joint_target", "target": {"type": "joint", "name": "j_pid"}},
        {"target": 0.2, "duration_s": 0.3},
    )
    model, data = runtime.backend.restore_state_v2(loaded["model_ref"], result["state_ref"])
    ctrl = np.asarray(data.ctrl, dtype=float)
    assert ctrl[0] == pytest.approx(0.2, abs=1e-12)
    assert ctrl[1] == pytest.approx(0.0, abs=1e-12)
