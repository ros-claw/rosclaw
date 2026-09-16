"""可执行交互运行时测试（MH12，0916 优化 §十二-§十四，红→绿）。

Grasp 诚实流：gripper_close → 接触证据 → constraint_attach（实测
relpose + constraint_assisted_grasp 标记）→ 提升 → constraint_release
（重力响应证据）。不允许 qpos teleport / hidden weld。
"""

from __future__ import annotations

import pytest

GRASP_WORLD = """<mujoco model="grasp_world">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="palm" pos="0 0 0.04">
      <joint name="lift" type="slide" axis="0 0 1" range="0 0.3"/>
      <geom name="palm_g" type="box" size="0.04 0.04 0.02" mass="0.5"/>
      <body name="finger" pos="0.04 0 0.02">
        <joint name="gripper" type="slide" axis="1 0 0" range="-0.02 0.02"/>
        <geom name="finger_g" type="box" size="0.01 0.03 0.03" mass="0.05"/>
      </body>
    </body>
    <body name="cube" pos="0.05 0 0.02">
      <freejoint name="cube_f"/>
      <geom name="cube_g" type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="lift_servo" joint="lift" kp="200" ctrlrange="0 0.3"/>
    <position name="gripper_servo" joint="gripper" kp="50" ctrlrange="-0.02 0.02"/>
  </actuator>
  <equality>
    <weld name="grasp_weld" body1="palm" body2="cube" active="false"/>
  </equality>
</mujoco>
"""


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "world.xml").write_text(GRASP_WORLD, encoding="utf-8")
    b = MujocoBackend(tmp_path)
    return b, b.load_model("world.xml")


def test_joint_target_executor(backend) -> None:
    b, ref = backend
    # kp=200 无阻尼 slide 会过冲振荡——先 patch 阻尼（参数实验的标准动作）。
    damped = b.patch_model(
        ref.model_ref,
        [{"op": "set", "target": {"type": "joint", "name": "lift"}, "field": "damping", "value": 50.0}],
    )
    state = b.transplant_state(damped.new_model_ref, b.initial_state_v2(ref.model_ref))
    result = b.interact(
        damped.new_model_ref,
        state,
        {"executor": "joint_target", "target": {"type": "joint", "name": "lift"}},
        {"target": 0.15, "duration_s": 1.5, "tolerance": 0.05},
    )
    assert result["ok"] is True
    assert result["outcome"]["reached"] is True
    assert result["outcome"]["after"] == pytest.approx(0.15, abs=0.05)
    assert result["state_ref"].startswith("simsta_")
    assert result["receipt_ref"].startswith("simexp_")
    assert result["usable_for_real_execution"] is False


def test_joint_target_without_actuator_rejected(backend) -> None:
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)
    with pytest.raises(ValueError, match="INTERACTION_TARGET_NOT_FOUND|CAPABILITY_UNAVAILABLE"):
        b.interact(
            ref.model_ref,
            state,
            {"executor": "joint_target", "target": {"type": "joint", "name": "cube_f"}},
            {"target": 1.0},
        )


def test_actuator_setpoint_executor(backend) -> None:
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)
    result = b.interact(
        ref.model_ref,
        state,
        {"executor": "actuator_setpoint", "target": {"type": "actuator", "name": "lift_servo"}},
        {"setpoints": {"ctrl": 0.1}, "duration_s": 0.5},
    )
    assert result["outcome"]["applied"] == {"ctrl": 0.1}


def test_gripper_close_contact_evidence(backend) -> None:
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)
    result = b.interact(
        ref.model_ref,
        state,
        {"executor": "gripper_close", "target": {"type": "actuator", "name": "gripper_servo"}},
        {"close_target": -0.02, "duration_s": 0.4},
    )
    assert result["ok"] is True
    assert "contact_evidence" in result["outcome"]


def test_grasp_honest_flow(backend) -> None:
    """§十四：contact evidence → measured relpose → weld 标记 → 提升 →
    释放重力响应。"""
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)

    # 1. 闭合夹爪（接触证据）。
    r1 = b.interact(
        ref.model_ref,
        state,
        {"executor": "gripper_close", "target": {"type": "actuator", "name": "gripper_servo"}},
        {"close_target": -0.02, "duration_s": 0.4},
    )
    # 2. attach：实测 relpose + constraint_assisted_grasp 标记。
    r2 = b.interact(
        ref.model_ref,
        r1["state_ref"],
        {"executor": "constraint_attach", "target": {"type": "body", "name": "palm"}, "weld": "grasp_weld"},
        {"attach_threshold_m": 0.05},
    )
    assert r2["constraint_assisted_grasp"] is True
    assert r2["outcome"]["measured_relpose"]["pos"]
    assert r2["outcome"]["min_distance_m"] < 0.05

    # 3. 提升 palm——cube 应被 weld 带着上升（不能偷偷 teleport）。
    damped = b.patch_model(
        ref.model_ref,
        [{"op": "set", "target": {"type": "joint", "name": "lift"}, "field": "damping", "value": 50.0}],
    )
    r3 = b.interact(
        damped.new_model_ref,
        b.transplant_state(damped.new_model_ref, r2["state_ref"]),
        {"executor": "joint_target", "target": {"type": "joint", "name": "lift"}},
        {"target": 0.15, "duration_s": 1.5},
    )
    model, data = b.restore_state_v2(damped.new_model_ref, r3["state_ref"])
    cube_id = model.body("cube").id
    assert float(data.xpos[cube_id][2]) > 0.08  # cube 被带起（原为 0.02）

    # 4. 释放——cube 必须受重力响应（下落）。
    r4 = b.interact(
        damped.new_model_ref,
        r3["state_ref"],
        {"executor": "constraint_release", "target": {"type": "body", "name": "palm"}, "weld": "grasp_weld"},
        {"duration_s": 0.4},
    )
    assert r4["outcome"]["gravity_response"] is True
    assert r4["outcome"]["z_after"] < r4["outcome"]["z_before"]


def test_attach_precondition_failed_without_contact(backend) -> None:
    """无接触证据不得 attach（GRASP_HONESTY）。"""
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)
    # 直接把 cube 移远（freejoint qpos x=0.5）——无接触。
    model, data = b.restore_state_v2(ref.model_ref, state)
    cube_joint = model.joint("cube_f").id
    data.qpos[model.jnt_qposadr[cube_joint]] = 0.5
    import mujoco

    mujoco.mj_forward(model, data)
    far_state = b.capture_and_store_v2(ref.model_ref, model, data)
    with pytest.raises(ValueError, match="INTERACTION_PRECONDITION_FAILED"):
        b.interact(
            ref.model_ref,
            far_state,
            {"executor": "constraint_attach", "target": {"type": "body", "name": "palm"}, "weld": "grasp_weld"},
            {"attach_threshold_m": 0.01},
        )


def test_attach_undeclared_weld_rejected(backend) -> None:
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)
    with pytest.raises(ValueError, match="INTERACTION_NO_WELD_DECLARED"):
        b.interact(
            ref.model_ref,
            state,
            {"executor": "constraint_attach", "target": {"type": "body", "name": "palm"}, "weld": "ghost_weld"},
            {},
        )


def test_unknown_executor_rejected(backend) -> None:
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)
    with pytest.raises(ValueError, match="INTERACTION_EXECUTOR_UNKNOWN"):
        b.interact(
            ref.model_ref,
            state,
            {"executor": "run_python", "target": {"type": "body", "name": "palm"}},
            {},
        )


def test_interact_receipt_stored(backend) -> None:
    b, ref = backend
    state = b.initial_state_v2(ref.model_ref)
    result = b.interact(
        ref.model_ref,
        state,
        {"executor": "joint_target", "target": {"type": "joint", "name": "lift"}},
        {"target": 0.05, "duration_s": 0.3},
    )
    receipt = b.store.get(result["receipt_ref"])
    assert receipt["kind"] == "interaction_receipt"
    assert receipt["trust_level"] == "SIMULATED"
    assert receipt["usable_for_real_execution"] is False
    assert receipt["initial_state_ref"] == state
    assert receipt["final_state_ref"] == result["state_ref"]


def test_runtime_and_catalog_surface(tmp_path) -> None:
    from rosclaw.agent.tool_catalog import MCP_TOOL_SAFETY_LEVELS, P0_AGENT_MCP_TOOLS, P0_SIM_TOOLS
    from rosclaw.mcp import tools as mcp_tools

    assert "sim_interact" in P0_SIM_TOOLS
    assert "sim_interact" in P0_AGENT_MCP_TOOLS
    assert MCP_TOOL_SAFETY_LEVELS["sim_interact"] == "S1_SIMULATION_ONLY"
    registered = {f.__name__ for f in mcp_tools.P0_TOOLS}
    assert "sim_interact" in registered
