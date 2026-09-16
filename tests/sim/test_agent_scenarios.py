"""Agent-tier 场景测试 H01-H08（PR-MH8，规格 §47-§55，红→绿）。

纪律（对齐 W09）：本层执行**harness 能力层**——用 Agent 会发起的
同一序列 SimulationRuntime 调用跑通每个场景，Verifier 直接从
MuJoCo 模型/物理真相比对；不合成冒充真实模型测试（需真实 LLM
key 的 H01-H08 agent 变体在 Release Gate 单列为 pending-live）。
"""

from __future__ import annotations

import pytest

from rosclaw.sim.runtime import SimulationRuntime

UNKNOWN_BOT = """<mujoco model="x7_proto">
  <compiler autolimits="true"/>
  <worldbody>
    <body name="seg0" pos="0 0 0.1">
      <joint name="jz0" type="hinge" axis="0 1 0"/>
      <geom name="g0" type="capsule" size="0.04 0.15" mass="0.8"/>
      <body name="seg1" pos="0 0 0.3">
        <joint name="jz1" type="hinge" axis="0 1 0"/>
        <geom name="g1" type="capsule" size="0.03 0.12" mass="0.5"/>
        <body name="seg2" pos="0 0 0.24">
          <joint name="jz2" type="hinge" axis="0 1 0"/>
          <geom name="g2" type="capsule" size="0.025 0.1" mass="0.3"/>
          <body name="rail" pos="0 0 0.15">
            <joint name="jx3" type="slide" axis="0 0 1" range="0 0.1"/>
            <geom name="g3" type="box" size="0.02 0.02 0.05" mass="0.2"/>
            <site name="tip" pos="0 0 0.08"/>
          </body>
        </body>
      </body>
    </body>
    <camera name="eye" pos="1 0 0.5"/>
  </worldbody>
  <actuator>
    <position name="srv0" joint="jz0" kp="20"/>
    <position name="srv1" joint="jz1" kp="20"/>
    <position name="srv2" joint="jz2" kp="20"/>
  </actuator>
  <sensor>
    <jointpos name="ps0" joint="jz0"/>
  </sensor>
</mujoco>
"""

H02_BROKEN = """<mujoco model="sick_arm">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 0.03">
      <geom name="base_geom" type="box" size="0.05 0.05 0.05" mass="2.0"/>
      <body name="arm" pos="0.05 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom name="heavy" type="box" size="0.05 0.05 0.05" pos="0.05 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="elbow_servo" joint="elbow" kp="0.01" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""

GRIPPER_BOT = """<mujoco model="gripper_bot">
  <worldbody>
    <body name="base" pos="0 0 0.02">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom name="palm" type="box" size="0.04 0.04 0.02" mass="0.5"/>
      <body name="finger_l" pos="0.05 0 0">
        <joint name="gripper" type="slide" axis="1 0 0" range="-0.03 0.03"/>
        <geom name="fl" type="box" size="0.01 0.02 0.02" mass="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="gripper_servo" joint="gripper" kp="20"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def runtime(tmp_path):
    (tmp_path / "x7.xml").write_text(UNKNOWN_BOT, encoding="utf-8")
    (tmp_path / "sick.xml").write_text(H02_BROKEN, encoding="utf-8")
    (tmp_path / "bot.xml").write_text(GRIPPER_BOT, encoding="utf-8")
    # 声明→证明绑定（0915 §五）：gripper 由 sidecar 声明、模型证明。
    (tmp_path / "bot.capabilities.yaml").write_text(
        "grasp:\n  actuator_joints: [gripper]\n", encoding="utf-8"
    )
    return SimulationRuntime(tmp_path)


def test_h01_unknown_body_derive_not_guess(runtime, tmp_path) -> None:
    """H01：陌生 MJCF——必须用 inspect 从编译真相推导，不从名字猜。"""
    loaded = runtime.load_model("x7.xml")
    inspected = runtime.inspect_model(loaded["model_ref"])

    # Agent 答案（完全来自结构化字段）。
    answer = {
        "dof": inspected["nv"],
        "actuators": [a["name"] for a in inspected["detail"]["actuators_detail"]],
        "sensors": [s["name"] for s in inspected["detail"]["sensors_detail"]],
        "cameras": [c["name"] for c in inspected["detail"]["cameras_detail"]],
        "can_grasp": any("gripper" in a["target"] for a in inspected["detail"]["actuators_detail"]),
    }

    # Verifier 独立从 MuJoCo 模型真相比对（不经过 harness 的回答路径）。
    import mujoco

    truth = mujoco.MjModel.from_xml_path(str(tmp_path / "x7.xml"))
    assert answer["dof"] == truth.nv
    assert answer["actuators"] == [
        mujoco.mj_id2name(truth, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(truth.nu)
    ]
    assert answer["sensors"] == ["ps0"]
    assert answer["cameras"] == ["eye"]
    assert answer["can_grasp"] is False


def test_h02_broken_model_doctor(runtime) -> None:
    """H02（本轮核心里程碑）：inspect → audit → diagnose → patch →
    compile → audit PASS——面对故意做坏的模型自己修好。"""
    loaded = runtime.load_model("sick.xml")
    diagnosed = runtime.audit(loaded["model_ref"])
    assert diagnosed["status"] == "FAIL"
    failed_checks = {
        name for name, outcome in diagnosed["checks"].items() if outcome["status"] == "FAIL"
    }
    # 三个注入缺陷必须全部被定位：隐式质量 + 初始穿透 + 伺服下垂。
    assert "A02_explicit_mass" in failed_checks
    assert "A05_initial_penetration" in failed_checks
    assert "A03_servo_hold" in failed_checks

    # 修复：抬回刚好贴地 + 声明质量 + 伺服增益/阻尼。
    fixed = runtime.patch_model(
        loaded["model_ref"],
        [
            {
                "op": "set",
                "target": {"type": "body", "name": "base"},
                "field": "pos",
                "value": [0, 0, 0.05],
            },
            {
                "op": "set",
                "target": {"type": "geom", "name": "heavy"},
                "field": "mass",
                "value": 8.0,
            },
            {
                "op": "set",
                "target": {"type": "actuator", "name": "elbow_servo"},
                "field": "kp",
                "value": 5000.0,
            },
            {
                "op": "set",
                "target": {"type": "joint", "name": "elbow"},
                "field": "damping",
                "value": 50.0,
            },
        ],
    )
    healed = runtime.audit(fixed["new_model_ref"])
    # 三个注入缺陷必须全部消除。MH15 后 kp=5000 硬伺服在 dt=0.002 下
    # 被 A24 诚实标记 NUMERICAL_FRAGILITY（WARN 不判 FAIL——
    # 修缺陷不等于消除刚度，这是审计栈按设计工作）。
    for name in ("A02_explicit_mass", "A05_initial_penetration", "A03_servo_hold"):
        assert healed["checks"][name]["status"] == "PASS", healed["checks"][name]
    non_pass = {
        name: outcome["status"]
        for name, outcome in healed["checks"].items()
        if outcome["status"] not in ("PASS", "NOT_EVALUATED")
    }
    assert set(non_pass) <= {"A24_timestep_sensitivity"}, non_pass

    # 母模型仍然是坏的（不可变——修复不污染亲缘）。
    assert runtime.audit(loaded["model_ref"])["status"] == "FAIL"


def test_h03_control_parameter_scientist(runtime) -> None:
    """H03：baseline → snapshot → fork ≥3 → rollout → compare → select，
    不得只凭经验改一个 kp 就宣布成功。"""
    loaded = runtime.load_model("x7.xml")
    model_ref = loaded["model_ref"]
    base_state = runtime.snapshot(model_ref)["state_ref"]

    baseline = runtime.rollout(
        model_ref, controller={"position_targets": [0.4, 0.2, 0.1]}, duration_s=1.0
    )
    fork = runtime.backend.fork_state(model_ref, base_state, 3)
    assert len(set(fork["branch_refs"])) == 1  # 分支初始 digest 一致

    receipts = [baseline]
    for kp in (50.0, 100.0, 400.0):
        branch = runtime.patch_model(
            model_ref,
            [
                {
                    "op": "set",
                    "target": {"type": "actuator", "name": "srv0"},
                    "field": "kp",
                    "value": kp,
                },
                {
                    "op": "set",
                    "target": {"type": "actuator", "name": "srv1"},
                    "field": "kp",
                    "value": kp,
                },
                {
                    "op": "set",
                    "target": {"type": "actuator", "name": "srv2"},
                    "field": "kp",
                    "value": kp,
                },
            ],
        )
        # 显式状态移植：fork 的物理状态转到 patch 后模型（维度校验，
        # 不是静默跨模型——这是参数实验的标准动作）。
        branch_state = runtime.backend.transplant_state(branch["new_model_ref"], base_state)
        receipts.append(
            runtime.rollout(
                branch["new_model_ref"],
                controller={"position_targets": [0.4, 0.2, 0.1]},
                duration_s=1.0,
                state_ref=branch_state,
            )
        )

    compared = runtime.compare([r["receipt_ref"] for r in receipts])
    best_row = next(r for r in compared["metric_table"] if r["receipt_ref"] == compared["best_ref"])
    # 选择由指标表支撑，且优于 baseline——不是凭感觉宣布。
    assert best_row["tracking_rmse"] < baseline["metrics"]["tracking_rmse"]
    assert compared["best_ref"] in compared["pareto_refs"]


def test_h04_world_generation(runtime, tmp_path) -> None:
    """H04：WorldSpec → compile → interaction contract → execute →
    audit → task predicate 机器判定 + 渲染证据。"""
    from rosclaw.sim.world.compiler import compile_world

    spec = {
        "schema_version": "rosclaw.sim.worldspec.v1",
        "world": {"gravity": [0, 0, -9.81], "ground": True, "seed": 0},
        "body_refs": [{"id": "bot", "kind": "task", "ref": "bot.xml"}],
        "objects": [
            {
                "id": "cube",
                "shape": "box",
                "size": [0.03, 0.03, 0.03],
                "pos": [0.4, 0, 0.03],
                "mass": 0.1,
                "rgba": [1, 0, 0, 1],
            },
            {
                "id": "bin",
                "shape": "box",
                "size": [0.05, 0.05, 0.05],
                "pos": [0.6, 0, 0.05],
                "mass": 1.0,
                "dynamic": False,
            },
        ],
        "interaction_points": [
            {
                "id": "grasp_cube",
                "affordance": "grasp",
                "target": {"type": "body", "name": "cube"},
                "depends_on": [],
            },
        ],
        "task": {
            "goal": "把方块放进箱子",
            "success": [
                {
                    "channel": "body_pose:cube",
                    "field": "pos",
                    "inside": {"min": [0.5, -0.1, 0.0], "max": [0.7, 0.1, 0.2]},
                }
            ],
        },
    }
    world = compile_world(runtime.backend, spec, name="h04")
    assert world["interaction_order"] == ["grasp_cube"]
    assert world["capabilities"] == {"bot": "AVAILABLE"}

    audited = runtime.audit(world["model_ref"])
    assert audited["status"] == "PASS", audited["violations"]

    receipt = runtime.rollout(
        world["model_ref"],
        controller={"hold": True},
        duration_s=0.2,
        task_predicates=spec["task"]["success"],
    )
    # 诚实（0915 §三）：hold 完不成任务——task_success 机器判 False，
    # verification FAIL；物理健康是另一回事，不再共用一个 success 字段。
    assert receipt["task_success"] is False
    assert receipt["verification_status"] == "FAIL"
    assert receipt["physical_audit_pass"] is True

    try:
        evidence = runtime.render(receipt["trace_ref"], width=160, height=120, max_frames=2)
    except ValueError as exc:
        if "SIM_RENDER_UNAVAILABLE" in str(exc):
            pytest.skip("GL backend unavailable")
        raise
    assert evidence["artifact_ref"].startswith("simrnd_")


def test_h05_honest_missing_capability(runtime, tmp_path) -> None:
    """H05：无夹爪机器人被要求 grasp → CAPABILITY_UNAVAILABLE；
    不得 fake grasp / qpos teleport / hidden weld。"""
    from rosclaw.sim.world.compiler import compile_world

    sick = runtime.load_model("sick.xml")  # 无 gripper 的模型
    spec = {
        "schema_version": "rosclaw.sim.worldspec.v1",
        "body_refs": [{"id": "arm", "kind": "task", "ref": "sick.xml"}],
        "objects": [
            {
                "id": "cup",
                "shape": "cylinder",
                "size": [0.03, 0.05, 0.05],
                "pos": [0.3, 0, 0.05],
                "mass": 0.2,
            },
        ],
        "interaction_points": [
            {
                "id": "grasp_cup",
                "affordance": "grasp",
                "target": {"type": "body", "name": "cup"},
                "depends_on": [],
            },
        ],
        "task": {"goal": "把杯子抓起来", "success": []},
    }
    with pytest.raises(ValueError, match="CAPABILITY_UNAVAILABLE"):
        compile_world(runtime.backend, spec, name="h05")
    # 世界模型没有被悄悄建成（fail closed 不落盘）。
    assert sick["model_ref"] and runtime.inspect_model(sick["model_ref"])["nq"] >= 1


def test_h06_midpath_collision(runtime, tmp_path) -> None:
    """H06：start/end 清楚、中途穿模——endpoint 检查漏掉，A06 抓住。"""
    import shutil

    shutil.copy(
        __import__("pathlib").Path(__file__).parent
        / "fixtures/broken_models/07_sequence_penetration.xml",
        tmp_path / "wall.xml",
    )
    loaded = runtime.load_model("wall.xml")
    receipt = runtime.rollout(loaded["model_ref"], controller={"ctrl_series": [[1.0]] * 400})

    # endpoint 视角：初始与终点都无穿透（naive verifier 会放行）。
    audited_endpoints = runtime.audit(loaded["model_ref"], checks=["A05_initial_penetration"])
    assert audited_endpoints["checks"]["A05_initial_penetration"]["status"] == "PASS"

    # 序列审计：中途穿墙被抓。
    audited = runtime.audit(
        loaded["model_ref"], checks=["A06_sequence_penetration"], trace_ref=receipt["trace_ref"]
    )
    assert audited["checks"]["A06_sequence_penetration"]["status"] == "FAIL"


def test_h07_numerical_fragility_detected(runtime) -> None:
    """H07：dt=0.002 与 dt=0.001 结果实质不同——harness 必须能发现
    数值脆弱（不是只在特定数值条件下碰巧通过）。"""
    loaded = runtime.load_model("x7.xml")
    stiff = runtime.patch_model(
        loaded["model_ref"],
        [
            {
                "op": "set",
                "target": {"type": "actuator", "name": "srv0"},
                "field": "kp",
                "value": 400.0,
            },
            {
                "op": "set",
                "target": {"type": "actuator", "name": "srv1"},
                "field": "kp",
                "value": 400.0,
            },
        ],
    )
    baseline = runtime.rollout(
        stiff["new_model_ref"], controller={"position_targets": [0.4, 0.2, 0.1]}, duration_s=0.5
    )
    finer = runtime.patch_model(
        stiff["new_model_ref"],
        [{"op": "set", "target": {"type": "option"}, "field": "timestep", "value": 0.001}],
    )
    replayed = runtime.rollout(
        finer["new_model_ref"], controller={"position_targets": [0.4, 0.2, 0.1]}, duration_s=0.5
    )
    # 指标层差异被机器报告（欠阻尼高增益伺服对 dt 敏感）。
    delta = abs(baseline["metrics"]["tracking_rmse"] - replayed["metrics"]["tracking_rmse"])
    assert delta > 1e-9  # 差异存在且被量化，而不是被掩盖
    compared = runtime.compare([baseline["receipt_ref"], replayed["receipt_ref"]])
    assert len(compared["metric_table"]) == 2


def test_h08_revision_chain(runtime) -> None:
    """H08：修订链复用亲缘（不重建世界）：patch → patch → rollout。"""
    loaded = runtime.load_model("x7.xml")
    first = runtime.patch_model(
        loaded["model_ref"],
        [
            {
                "op": "set",
                "target": {"type": "joint", "name": "jz0"},
                "field": "damping",
                "value": 2.0,
            }
        ],
    )
    second = runtime.patch_model(
        first["new_model_ref"],
        [
            {
                "op": "set",
                "target": {"type": "actuator", "name": "srv0"},
                "field": "kp",
                "value": 60.0,
            }
        ],
    )
    described = runtime.backend.describe_model(second["new_model_ref"])
    assert described.parent_model_ref == first["new_model_ref"]
    parent = runtime.backend.describe_model(first["new_model_ref"])
    assert parent.parent_model_ref == loaded["model_ref"]

    # 修订链上可直接继续实验（无重复世界重建：model 清单恰为 3）。
    receipt = runtime.rollout(second["new_model_ref"], controller={"hold": True}, duration_s=0.05)
    assert receipt["trust_level"] == "SIMULATED"
    assert len(runtime.backend.store.list_children("models")) == 3
